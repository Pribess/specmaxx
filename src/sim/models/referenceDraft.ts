/**
 * Reference models WITH a speculative-decoding drafter (hand-written dims, like `reference.ts`). Kept out of
 * `reference.ts` because the deployment identity pin (engine/identity.test.ts) enumerates that module's exports as
 * its exact case set; with speculative decoding off these models simulate exactly like their base model.
 */

import { decoderLayer } from '../ir/builders';
import type { ModelSpec } from '../ir/model';
import { REF_DENSE_8B } from './reference';

/**
 * REF_DENSE_8B with a synthetic DeepSeek-V3-style MTP module (design doc §8.5, worked example W2): one
 * target-shaped decoder layer (32/8 GQA heads of 128, q/k-norm, SwiGLU 12288; 192,946,432 params, of which
 * 192,937,984 in GEMM weights) behind `eh_proj` (2h → h, 33,554,432 params) and the `enorm`, `hnorm` and
 * `final_norm` RMSNorms (3·4096); shared embedding and LM head. Single-step head (`maxReliableDepth 1`).
 * Made-up module: Qwen3-8B ships no MTP head.
 */
export const REF_DENSE_8B_MTP: ModelSpec = {
  ...REF_DENSE_8B,
  name: 'REF_DENSE_8B_MTP (Qwen3-8B shape + synthetic MTP module)',
  draft: {
    kind: 'mtp',
    layers: [
      decoderLayer({
        name: 'mtp.0',
        hidden: 4096,
        mixer: { kind: 'gqa', heads: 32, kvHeads: 8, headDim: 128, qkNorm: true },
        ffn: { kind: 'dense', intermediate: 12288 },
      }),
    ],
    hiddenProj: true,
    maxReliableDepth: 1,
  },
};
