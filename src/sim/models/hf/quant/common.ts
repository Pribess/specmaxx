/**
 * Shared vocabulary of the quant-default readers: probes (synthetic module names), name patterns, format
 * constructors and scale-overhead arithmetic.
 *
 * How a reader works: every quantization format names modules in its own way (exact names, dotted prefixes,
 * fnmatch globs, `re:` regexes, torch class names). Instead of interpreting each name, the readers are asked
 * about a fixed set of {@link Probe}s, one synthetic module name per (weight class, layer, projection) in the
 * checkpoint's own naming (`moduleClass.ts` builds them). Each reader answers "what format does this module
 * get?"; `assemble.ts` then takes the majority per class. See docs/simulation-model.md §2 (quant overheads).
 */

import type { Dtype, TensorFormat } from '../../../ir/dtype';
import type { ModuleQuant, WeightModule } from '../../../ir/model';
import { isJsonObject } from '../reader';
import type { JsonObject, QuantDefaults } from '../types';

/**
 * What a module is, as far as the quant defaults care: a {@link WeightModule} class, the LM head, the
 * embedding table, or an n-gram / Engram lookup table. (Routers, norms, convs, MTP and vision modules are
 * never probed: they are not quantized by any reader, or not part of the IR.)
 */
export type ModuleClass = WeightModule | 'lm_head' | 'embeddings' | 'lookup_tables';

/**
 * The torch module type of a probe, which quantizers target by class name (`targets: ["Linear"]`).
 * `other` is a raw parameter (e.g. DeepSeek `hc_attn_fn`), which no class target selects.
 */
export type ProbeKind = 'linear' | 'embedding' | 'other';

/** One synthetic module name in the checkpoint's naming, with the class and layer it stands for. */
export interface Probe {
  /** Full dotted module name, e.g. `'model.language_model.layers.3.self_attn.q_proj'`. */
  name: string;
  /** The class the module belongs to. */
  cls: ModuleClass;
  /** Decoder layer index (0-based), or `null` for a module outside the layer stack (LM head, embeddings). */
  layer: number | null;
  /** Torch module type, for class-name targets. */
  kind: ProbeKind;
  /**
   * Relative size of the projection in majority votes inside one layer: 1 for a full projection, 0.1 for a
   * small one (GatedDeltaNet `in_proj_a/b`, KDA gate low-rank projections). Dimensionless.
   */
  weight: number;
  /** The leaf name relative to the layer, e.g. `'self_attn.q_proj'` (for notes). */
  leaf: string;
  /**
   * True when the native (unconverted) checkpoint keeps this module in the high-precision dtype even though
   * its quant config does not say so (DeepSeek-V4.1 compressor and indexer `wk` / `weights_proj`; Kimi K3's
   * latent-MoE `routed_expert_{down,up}_proj`, which K3's compressed-tensors `ignore` list misses; per the
   * safetensors headers). The native-fp8 and compressed-tensors readers honor it.
   */
  nativeHighPrecision?: boolean;
  /**
   * Storage dtype of the module in the unconverted checkpoint when it is not the model dtype (DeepSeek-V4.1
   * `hc_*_fn` are fp32 per the safetensors headers). Applies when the reader returns `ctx.native`.
   */
  nativeDtype?: Dtype;
}

/**
 * The result of reading one quantization format: a per-probe format plus the global facts.
 * Plain data plus one function, so `assemble.ts` stays independent of the format.
 */
export interface SchemeReading {
  /** Human-readable scheme label, e.g. `'fp8 e4m3, block 128×128 fp32 scales, dynamic fp8 activations'`. */
  label: string;
  /** Which file / key the scheme was read from. */
  source: QuantDefaults['source'];
  /** Format of one probe; the native format (`ctx.native`) when the scheme leaves the module alone. */
  resolve(probe: Probe): ModuleQuant;
  /** Activation dtype, when the scheme fixes it (bitsandbytes compute dtype); default: the model dtype. */
  activations?: Dtype;
  /** KV cache format, when the scheme quantizes it (`kv_cache_scheme`, `kv_cache_quant_algo`). */
  kvCache?: TensorFormat;
  /** Indexer key cache format, when the scheme sets it (Mistral `dsa.fp8_indexer`). */
  indexerKeyCache?: TensorFormat;
  /** EXL3 only: the average and LM-head bits per weight the config states (prefills the UI's bpw inputs). Unit: bits/weight. */
  exl3?: { bpw: number; headBpw: number };
  /** Ignore / exclude lists, checked afterwards for entries that match no known module (each gives a note). */
  ignoreLists: { label: string; patterns: string[] }[];
  /** Notes from reading (approximations, unknown fields). */
  notes: string[];
}

/** What every reader gets besides its own config object. */
export interface SchemeContext {
  /** The model's own dtype (`torch_dtype` / `dtype`), used for unconverted modules and default scales. */
  modelDtype: Dtype;
  /** The format of a module the scheme does not convert: `{ weights: { dtype: modelDtype } }`. */
  native: ModuleQuant;
  /** Language-model config (`text_config` or the flat config). */
  text: JsonObject;
  /** Top-level config.json. */
  top: JsonObject;
}

// ---------------------------------------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------------------------------------

/**
 * A module format: storage dtype, group-scale overhead outside the dtype (bits/element, omitted when 0) and
 * the GEMM-input dtype (omitted for weight-only quantization).
 */
export function moduleFormat(dtype: Dtype, extraBitsPerElem = 0, gemmInput?: Dtype): ModuleQuant {
  const weights: TensorFormat = extraBitsPerElem > 0 ? { dtype, extraBitsPerElem } : { dtype };
  return gemmInput === undefined ? { weights } : { weights, gemmInput };
}

/**
 * Amortized scale overhead of a 2-D block-scaled weight: one `scaleBits`-bit scale per `rows × cols` block.
 * Unit: bits/element. Example: fp8 128×128 with fp32 scales = 32 / 16384 ≈ 0.00195.
 */
export function blockScaleBits(scaleBits: number, rows: number, cols: number): number {
  // Formula: extra = scaleBits / (rows · cols)
  // See docs/simulation-model.md §2 (quant overheads)
  const elemsPerBlock = rows * cols;
  return elemsPerBlock > 0 ? scaleBits / elemsPerBlock : 0;
}

/**
 * Amortized overhead of group quantization along the input dim: one scale and (asymmetric) one zero point
 * per `groupSize` elements. Unit: bits/element. Example: int4 asym g32 with a bf16 scale and an int8 zero
 * point = (16 + 8) / 32 = 0.75.
 */
export function groupScaleBits(scaleBits: number, zeroPointBits: number, groupSize: number): number {
  // Formula: extra = (scaleBits + zeroPointBits) / groupSize
  // See docs/simulation-model.md §2 (quant overheads)
  return groupSize > 0 ? (scaleBits + zeroPointBits) / groupSize : 0;
}

/**
 * Bits of a scale / zero-point dtype named in a config (`'torch.float16'`, `'float32'`, `'float8_e8m0fnu'`,
 * `'ue8m0'`, `'torch.uint8'`, `'torch.int8'`, …), or `fallback` when absent or unknown. Unit: bits.
 * Quark's bare `'float'` means fp32.
 */
export function dtypeNameBits(name: unknown, fallback: number): number {
  if (typeof name !== 'string') return fallback;
  const n = name.toLowerCase().replace(/^torch\./, '');
  if (/^(float32|fp32|float)$/.test(n)) return 32;
  if (/^(float16|fp16|half|bfloat16|bf16)$/.test(n)) return 16;
  if (/e8m0|e4m3|e5m2|float8|fp8|^u?int8$/.test(n)) return 8;
  if (/^u?int4$/.test(n)) return 4;
  if (/^u?int32$/.test(n)) return 32;
  return fallback;
}

/** Storage dtype of a model `torch_dtype` / `dtype` string; `bf16` when absent or unknown. */
export function modelDtypeOf(...candidates: unknown[]): Dtype {
  for (const c of candidates) {
    if (typeof c !== 'string') continue;
    const n = c.toLowerCase().replace(/^torch\./, '');
    if (n === 'bfloat16' || n === 'bf16') return 'bf16';
    if (n === 'float16' || n === 'fp16' || n === 'half') return 'fp16';
    if (n === 'float32' || n === 'fp32' || n === 'float') return 'fp32';
  }
  return 'bf16';
}

/** Integer storage dtype for a bit width; widths without their own dtype round up (with `approx` set). */
export function intDtypeFor(bits: number): { dtype: Dtype; approx: boolean } {
  if (bits === 4) return { dtype: 'int4', approx: false };
  if (bits === 8) return { dtype: 'int8', approx: false };
  if (bits < 4) return { dtype: 'int4', approx: true };
  return { dtype: 'int8', approx: true };
}

/** A canonical key of a module format, for equality and majority votes. */
export function formatKey(mq: ModuleQuant): string {
  const extra = mq.weights.extraBitsPerElem ?? 0;
  return `${mq.weights.dtype}|${extra.toFixed(8)}|${mq.gemmInput ?? '-'}|${mq.weights.valueBitsPerElem ?? '-'}|${mq.weights.perDimBits ?? '-'}`;
}

/** Short human-readable form of a module format, e.g. `'fp8_e4m3 +0.0020 bit/elem, fp8_e4m3 GEMM input'`. */
export function describeFormat(mq: ModuleQuant): string {
  const extra = mq.weights.extraBitsPerElem ?? 0;
  const value = mq.weights.valueBitsPerElem !== undefined ? ` ${mq.weights.valueBitsPerElem} bpw` : '';
  const scale = value + (extra > 0 ? ` +${extra.toFixed(4)} bit/elem` : '');
  const input = mq.gemmInput === undefined ? '' : `, ${mq.gemmInput} GEMM input`;
  return `${mq.weights.dtype}${scale}${input}`;
}

// ---------------------------------------------------------------------------------------------------------
// JSON helpers (quant configs are untyped; these never throw)
// ---------------------------------------------------------------------------------------------------------

/** `o[key]` as an object, or undefined. */
export function objAt(o: unknown, key: string): JsonObject | undefined {
  if (!isJsonObject(o)) return undefined;
  const v = o[key];
  return isJsonObject(v) ? v : undefined;
}

/** `o[key]` as a finite number, or undefined. */
export function numAt(o: unknown, key: string): number | undefined {
  if (!isJsonObject(o)) return undefined;
  const v = o[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

/** `o[key]` as a string, or undefined. */
export function strAt(o: unknown, key: string): string | undefined {
  if (!isJsonObject(o)) return undefined;
  const v = o[key];
  return typeof v === 'string' ? v : undefined;
}

/** `o[key]` as a list of strings (non-strings dropped); `[]` when absent. */
export function strListAt(o: unknown, key: string): string[] {
  if (!isJsonObject(o)) return [];
  const v = o[key];
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

/** `o[key]` as a list of finite numbers, or undefined when absent or not all numbers. */
export function numListAt(o: unknown, key: string): number[] | undefined {
  if (!isJsonObject(o)) return undefined;
  const v = o[key];
  if (!Array.isArray(v) || !v.every((x) => typeof x === 'number' && Number.isFinite(x))) return undefined;
  return v as number[];
}

/** Compact list of integers as ranges, e.g. `[0, 1, 2, 5, 7, 8]` → `'0–2, 5, 7–8'`. */
export function formatRanges(values: readonly number[]): string {
  const sorted = [...new Set(values)].sort((a, b) => a - b);
  const parts: string[] = [];
  let start = sorted[0];
  let prev = sorted[0];
  for (let i = 1; i <= sorted.length; i++) {
    const v = sorted[i];
    if (v !== undefined && v === prev + 1) {
      prev = v;
      continue;
    }
    if (start !== undefined) parts.push(start === prev ? `${start}` : `${start}–${prev}`);
    start = v;
    prev = v;
  }
  return parts.join(', ');
}
