import { validDegrees, applyQuantOverride } from '../state';
import { Panel } from './Panel';
import { NumberField } from './NumberField';
import type { PanelProps as Props } from './types';

const HELP = {
  tp: "Tensor parallelism: splits each layer's matrices across devices, so every device computes part of every layer and needs an all-reduce per layer. Matters locally when one GPU cannot hold the weights or you want lower latency; needs a fast link (NVLink or PCIe P2P).",
  pp: 'Pipeline parallelism: splits the layers into consecutive stages on different devices. Only activations cross stage boundaries, so it tolerates slow links (Ethernet, Thunderbolt) and suits multi-machine local clusters. One sequence does not run faster.',
  dp: 'Data parallelism: runs independent full replicas of the model for more aggregate throughput. Each replica needs the whole model in memory, so it matters when the model fits on a subset of your devices.',
  ep: 'Expert parallelism: spreads the experts of a mixture-of-experts model across devices, with an all-to-all of tokens. Matters for large MoE models on fast links; no effect on dense models.',
} as const;

export function ParallelPanel({ state, dispatch, loaded, built }: Props) {
  return (
    <Panel
      title="Parallelism"
      value={`TP ${state.parallel.tp}, PP ${state.parallel.pp}, DP ${state.parallel.dp}, EP ${state.parallel.ep}`}
      open
    >
      <div className="fields two">
        {(['tp', 'pp', 'dp', 'ep'] as const).map((field) => (
          <NumberField
            key={field}
            label={field.toUpperCase()}
            help={HELP[field]}
            value={state.parallel[field]}
            onChange={(v) => dispatch({ type: 'setParallel', parallel: { [field]: v } })}
            issues={built.issues.filter((i) => i.field === field)}
            hint={
              loaded && built.hardware
                ? `Valid ${field.toUpperCase()}: ${validDegrees(field, loaded.model, applyQuantOverride(loaded.quant.quant, state.quant, loaded.quant.exl3), state.parallel, state.workload, built.hardware.topology).join(', ')}`
                : undefined
            }
          />
        ))}
      </div>
    </Panel>
  );
}
