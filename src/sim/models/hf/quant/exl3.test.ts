import { describe, expect, it } from 'vitest';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { mapHfModel } from '../index';

/** Real EXL3 repo configs (see ../../__fixtures__/quant/README.md): [repo, bits, head_bits, routed-experts-only]. */
const CASES: [string, number, number, boolean][] = [
  ['turboderp/Qwen3.8-27B-exl3', 4, 5, false],
  ['turboderp/GLM-5.3-Flash-exl3', 3.09, 6, false],
  ['bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw', 2.52, 6, false],
  ['Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw', 2.9, 6, false], // key: avg_bits
  ['doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw', 4, 6, false],
  ['Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold', 4, 16, true], // scope: routed experts only, fp16 head
];

describe('EXL3 quantization_config detection (real repos)', () => {
  it.each(CASES)('%s', (repo, bpw, headBpw, expertsOnly) => {
    const r = mapHfModel(fixtureRepoFiles(repo, { dir: 'quant' }));
    if (r.kind !== 'ok') throw new Error(`${repo}: ${r.kind}`);
    const d = r.quant;
    expect(d.exl3).toEqual({ bpw, headBpw });
    expect(d.source).toBe('quantization_config');
    expect(d.scheme).toMatch(/^exl3 /);
    expect(d.quant.activations).toBe('fp16');
    const body = d.quant.modules?.routed_experts?.weights ?? d.quant.weights;
    expect(body).toEqual({ dtype: 'fp16', valueBitsPerElem: bpw, perDimBits: 16 });
    // LM head: its own bpw (or unquantized fp16 for head_bits 16) — never silently the body's.
    const head = d.quant.lmHead ?? d.quant.embeddings;
    if (headBpw >= 16) expect(head?.valueBitsPerElem).toBeUndefined();
    else expect(head).toEqual({ dtype: 'fp16', valueBitsPerElem: headBpw, perDimBits: 16 });
    if (expertsOnly) {
      expect(d.quant.modules?.attention?.weights.valueBitsPerElem).toBeUndefined();
    } else {
      expect(d.quant.modules?.attention?.weights.valueBitsPerElem ?? bpw).toBe(bpw);
    }
  });

  it('a config without a bits key falls back to 4 bpw with a note', () => {
    const files = fixtureRepoFiles('turboderp/Qwen3.8-27B-exl3', { dir: 'quant' });
    const config = JSON.parse(JSON.stringify(files.config)) as Record<string, Record<string, unknown>>;
    config.quantization_config = { quant_method: 'exl3' };
    const r = mapHfModel({ ...files, config });
    if (r.kind !== 'ok') throw new Error(r.kind);
    expect(r.quant.exl3).toEqual({ bpw: 4, headBpw: 6 });
    expect(r.quant.notes.join('\n')).toMatch(/assuming 4 bpw/);
  });
});
