/**
 * ExLlamaV3 (EXL3, `quant_method: "exl3"`): trellis-coded weights at a fractional average bits per weight.
 *
 * Real configs (turboderp, community): `{ quant_method: "exl3", version, bits, head_bits, calibration,
 * out_scales, codebook, mtp_bits, vision_bits, original_quantization_config? }`. The average bpw key is
 * `bits` (exllamav3 converter `-b`), `bpw` or `avg_bits` (community optimizers that mix per-tensor bits);
 * `head_bits` is the LM head's integer bpw (converter default 6; 16 = unquantized). `mtp_bits` and
 * `vision_bits` are outside the IR. `original_quantization_config` records the source checkpoint's own scheme
 * (e.g. fp8) and is ignored. A community `scope: "…routed_experts_only"` quantizes only the routed experts.
 *
 * Scope: every Linear of the decoder layers (attention, dense MLP, shared and routed experts, indexer, linear
 * attention, n-gram projections) at `bits`; the LM head at `head_bits`; embeddings, routers, norms, hyper-
 * connection mixers and lookup tables stay in the model dtype (converter `--embed_bits` default 16). Storage:
 * `exl3Format` (value bits + fp16 `suh`/`svh` side vectors). Weight-only with fp16 activations (exllamav3 runs
 * in fp16). Per-tensor bit differences (optimizer recipes) are not in config.json; the average is used for all.
 * See docs/simulation-model.md §2 "EXL3".
 */

import { EXL3_DEFAULT_HEAD_BPW, exl3Format } from '../../../ir/exl3';
import type { ModuleQuant } from '../../../ir/model';
import type { Probe, SchemeContext, SchemeReading } from './common';
import { numAt, strAt } from './common';
import type { JsonObject } from '../types';

/** Average bpw assumed when a config states none. Unit: bits/weight. */
const FALLBACK_BPW = 4;

/** Read an EXL3 `quantization_config`. */
export function readExl3(qc: JsonObject, ctx: SchemeContext): SchemeReading {
  const notes: string[] = [];
  const stated = numAt(qc, 'bits') ?? numAt(qc, 'bpw') ?? numAt(qc, 'avg_bits');
  if (stated === undefined) notes.push(`exl3: no bits / bpw / avg_bits in quantization_config; assuming ${FALLBACK_BPW} bpw`);
  const bpw = stated ?? FALLBACK_BPW;
  const headBpw = numAt(qc, 'head_bits') ?? EXL3_DEFAULT_HEAD_BPW;
  const scope = strAt(qc, 'scope') ?? '';
  const expertsOnly = /routed_experts_only/i.test(scope);

  const body: ModuleQuant = { weights: exl3Format(bpw) };
  const head: ModuleQuant = { weights: exl3Format(headBpw) };
  const resolve = (p: Probe): ModuleQuant => {
    if (p.cls === 'lm_head') return head;
    if (p.kind !== 'linear' || p.layer === null || p.cls === 'residual_mixer') return ctx.native;
    if (expertsOnly && p.cls !== 'routed_experts') return ctx.native;
    return body;
  };
  notes.push(
    `exl3: ${bpw} bpw trellis bits + fp16 suh/svh vectors per matrix; head ${headBpw >= 16 ? 'fp16' : `${headBpw} bpw`}; ` +
      'dequantized to fp16 in the matmul (fp16 activations)',
  );
  if (expertsOnly) notes.push(`exl3: scope '${scope}': only routed experts are quantized`);
  const codebook = strAt(qc, 'codebook');
  return {
    label: `exl3 ${bpw} bpw${expertsOnly ? ' (routed experts)' : ''}, head ${headBpw >= 16 ? 'fp16' : `${headBpw} bpw`}${codebook ? `, ${codebook}` : ''}`,
    source: 'quantization_config',
    resolve,
    activations: 'fp16',
    ignoreLists: [],
    notes,
    exl3: { bpw, headBpw },
  };
}
