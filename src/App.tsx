import { REF_DENSE_8B, RTX_5090, simulate, singleDevice, type SimResult } from './sim';

/** The demo configuration: REF_DENSE_8B on 1×RTX 5090, bf16 everywhere, batch 1, prompt 1024, decode ctx 1024. */
function demoResult(): SimResult {
  return simulate({
    model: REF_DENSE_8B,
    quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 1024, decodeContextLen: 1024 },
    hardware: { device: RTX_5090, topology: singleDevice(RTX_5090) },
  });
}

/** Formats a rate in tokens/s with thousands separators and one decimal. */
function formatRate(tokensPerSec: number): string {
  return tokensPerSec.toLocaleString('en-US', { maximumFractionDigits: 1, minimumFractionDigits: 1 });
}

/**
 * Placeholder page with one demo line computed by simulate(): REF_DENSE_8B on 1×RTX 5090, bf16, batch 1,
 * prompt 1024, decode context 1024. The real UI is a follow-up task.
 */
export function App() {
  const result = demoResult();
  return (
    <main>
      <h1>specmaxx</h1>
      <p>Simulator under construction</p>
      <p>
        Demo (REF_DENSE_8B on 1×RTX 5090, bf16, batch 1, prompt 1024, decode ctx 1024): prefill{' '}
        {formatRate(result.prefill.tokensPerSec)} tok/s, decode {formatRate(result.decode.tokensPerSec)} tok/s
        (theoretical maximum).
      </p>
    </main>
  );
}
