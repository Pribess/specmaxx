import { DEFAULT_EP_IMBALANCE } from '../../sim/lower/moe';
import {
  validDegrees,
  dpAttentionEpSplits,
  applyQuantOverride,
  chooseDegrees,
  effectiveParallel,
  hasMoe,
  type ParallelLayout,
} from '../state';
import { Panel } from './Panel';
import { NumberField } from './NumberField';
import { ParallelAdvancedFields } from './ParallelAdvancedFields';
import { Segmented } from './Segmented';
import { helpLines } from './helpLines';
import { poolTitle, type PanelProps as Props } from './types';

/** Help text per field: short sentences, each shown on its own line in the tooltip. */
const HELP = {
  tp: [
    'Splits every layer across GPUs.',
    'Each GPU holds a slice of the weights.',
    'Needs a fast link: GPUs sync after every layer.',
  ],
  pp: [
    'Splits the layers into stages, one per GPU.',
    'Only small activations cross between stages, so slow links are fine.',
    'A single sequence does not get faster.',
  ],
  dp: [
    'Runs full copies of the model side by side.',
    'Each copy serves its own share of the batch.',
    'More total throughput; a single sequence is not faster.',
  ],
  ep: [
    'Spreads the MoE experts across GPUs.',
    'Each GPU holds a share of the experts, not a slice of each.',
    'EP must divide TP, or with DP be a multiple of TP that divides TP × DP.',
  ],
  layout: [
    'Custom: set TP, PP, DP and EP yourself.',
    'DP attention + EP: each GPU runs attention on its own sequences; experts are spread over all of them.',
  ],
  depEp: [
    'GPUs per pipeline stage.',
    'Each runs attention on its own share of the batch and holds a share of the experts.',
  ],
  dense: [
    'Dense FFN layers and the LM head.',
    'DP: whole on every GPU, no extra traffic.',
    'TP: split over the EP GPUs; tokens are gathered before and scattered after.',
  ],
  epImbalance: [
    "Busiest GPU's expert load ÷ the average GPU's (1 = perfectly even).",
    "Scales that GPU's expert compute, which the step waits for.",
    'About 1.5 on 8 GPUs and 2.5 on 32 without a load balancer; about 1.1 with EPLB.',
  ],
  redundantExperts: [
    'Extra copies of hot experts each GPU stores in every MoE layer (EPLB).',
    "Adds their weights to each GPU's memory; the load they even out is the EP imbalance you set.",
    '1 per GPU (32 at EP 32) brought DeepSeek-R1 to about 1.1.',
  ],
  epImbalanceAllToAll: [
    'On: the busiest GPU also moves EP imbalance × the average all-to-all traffic.',
    'Off (default): every GPU moves the average amount.',
    'Only matters when EP spans several DP copies.',
  ],
} as const;

/** The layout control's segments. */
const LAYOUTS: { value: ParallelLayout; label: string }[] = [
  { value: 'custom', label: 'Custom' },
  { value: 'dpAttentionEp', label: 'DP attention + EP' },
];

/** The dense-layers control's segments: DP (`denseTpGroup` absent) or TP over the EP group (`'ep'`). */
const DENSE = [
  { value: 'dp', label: 'DP' },
  { value: 'tp', label: 'TP' },
];

/** An off / on switch's segments. */
const ON_OFF = [
  { value: 'off', label: 'Off' },
  { value: 'on', label: 'On' },
];

export function ParallelPanel({ state, dispatch, loaded, built, pool }: Props) {
  // EP and the layouts are offered only for a loaded MoE model; a dense model always runs with EP 1.
  const moe = loaded !== null && hasMoe(loaded.model);
  const dep = moe && state.parallelLayout === 'dpAttentionEp';
  const fields = moe ? (['tp', 'pp', 'dp', 'ep'] as const) : (['tp', 'pp', 'dp'] as const);
  const { tp, pp, dp, ep, dcp } = state.parallel;
  const denseTp = state.parallel.denseTpGroup === 'ep';
  const value = dep
    ? `DP attention + EP ${ep}${pp > 1 ? `, PP ${pp}` : ''}${denseTp ? ', dense TP' : ''}`
    : `TP ${tp}, PP ${pp}, DP ${dp}${moe ? `, EP ${ep}` : ''}${dcp !== undefined && dcp > 1 ? `, DCP ${dcp}` : ''}`;
  const topology = built.hardware?.topology;
  const devices = state.hw.devicesPerNode * state.hw.nodes;
  const checked =
    loaded && topology
      ? {
          model: loaded.model,
          quant: applyQuantOverride(loaded.quant.quant, state.quant, loaded.quant.exl3),
          parallel: effectiveParallel(state.parallel, loaded.model, state.parallelLayout),
          topology,
        }
      : undefined;
  const issuesOf = (...names: string[]) => built.issues.filter((i) => names.includes(i.field));
  // The EP balance knobs only where the engine reads them: a MoE model at EP > 1 (effectiveParallel drops them otherwise).
  const balance = moe && ep > 1 && (
    <>
      <div className="fields two">
        <NumberField
          label="EP imbalance"
          help={helpLines(HELP.epImbalance)}
          value={state.parallel.epImbalance ?? DEFAULT_EP_IMBALANCE}
          step="0.01"
          min={1}
          onChange={(v) => dispatch({ type: 'setParallel', parallel: { epImbalance: v } })}
          issues={issuesOf('epImbalance')}
        />
        <NumberField
          label="Redundant experts"
          help={helpLines(HELP.redundantExperts)}
          value={state.parallel.redundantExpertsPerRank ?? 0}
          min={0}
          onChange={(v) =>
            dispatch({ type: 'setParallel', parallel: { redundantExpertsPerRank: v } })
          }
          issues={issuesOf('redundantExperts')}
        />
      </div>
      <Segmented
        label="Imbalance on all-to-all"
        value={state.parallel.epImbalanceScalesAllToAll === true ? 'on' : 'off'}
        segments={ON_OFF}
        onChange={(v) =>
          dispatch({
            type: 'setParallel',
            parallel: { epImbalanceScalesAllToAll: v === 'on' ? true : undefined },
          })
        }
        help={helpLines(HELP.epImbalanceAllToAll)}
      />
    </>
  );
  // The splits EP × PP = devices the layout's EP and PP fields move between; their hints list exactly these.
  const splits =
    dep && checked
      ? dpAttentionEpSplits(
          checked.model,
          checked.quant,
          checked.parallel,
          state.workload,
          checked.topology,
        )
      : undefined;
  /** EP = DP = n, and PP = devices / n when n divides the device count (else the device-count error shows). */
  const stageOf = (n: number) =>
    n >= 1 && Number.isInteger(n) && devices % n === 0
      ? { ep: n, dp: n, pp: devices / n }
      : { ep: n, dp: n };

  const setLayout = (layout: ParallelLayout) => {
    dispatch({ type: 'setParallelLayout', layout });
    // Into DP attention + EP: a valid size at the current PP when there is one (chooseDegrees keeps the shape).
    if (layout === 'dpAttentionEp' && checked) {
      dispatch({
        type: 'setParallel',
        parallel: chooseDegrees(checked.model, checked.quant, state.workload, checked.topology, {
          pp,
          layout,
        }),
      });
    }
  };

  return (
    <Panel title={poolTitle('Parallelism', pool)} value={value} open>
      {moe && (
        <Segmented
          label="Layout"
          value={dep ? 'dpAttentionEp' : 'custom'}
          segments={LAYOUTS}
          onChange={(v) => setLayout(v as ParallelLayout)}
          help={helpLines(HELP.layout)}
        />
      )}
      {dep ? (
        <>
          <div className="fields two">
            <NumberField
              label="EP"
              help={helpLines(HELP.depEp)}
              value={ep}
              // EP and DP move together (N GPUs per stage, each its own DP rank); PP follows: devices / N.
              onChange={(v) => dispatch({ type: 'setParallel', parallel: stageOf(v) })}
              issues={issuesOf('ep', 'dp', 'tp')}
              hint={
                splits
                  ? `Valid EP: ${splits
                      .map((s) => s.n)
                      .sort((a, b) => a - b)
                      .join(', ')}`
                  : undefined
              }
            />
            <NumberField
              label="PP"
              help={helpLines(HELP.pp)}
              value={pp}
              // The stage's GPUs follow: EP = DP = devices / PP when PP divides the device count.
              onChange={(v) =>
                dispatch({
                  type: 'setParallel',
                  parallel:
                    v >= 1 && Number.isInteger(v) && devices % v === 0
                      ? { pp: v, ep: devices / v, dp: devices / v }
                      : { pp: v },
                })
              }
              issues={issuesOf('pp')}
              hint={splits ? `Valid PP: ${splits.map((s) => s.pp).join(', ')}` : undefined}
            />
          </div>
          <Segmented
            label="Dense layers"
            value={denseTp ? 'tp' : 'dp'}
            segments={DENSE}
            onChange={(v) =>
              dispatch({
                type: 'setParallel',
                parallel: { denseTpGroup: v === 'tp' ? 'ep' : undefined },
              })
            }
            help={helpLines(HELP.dense)}
            issues={issuesOf('denseTp')}
          />
          {balance}
        </>
      ) : (
        <>
          {/* Three fields in a row; four (with EP) as two rows of two, so the hints keep room. */}
          <div className={`fields ${moe ? 'two' : 'three'}`}>
            {fields.map((field) => (
              <NumberField
                key={field}
                label={field.toUpperCase()}
                help={helpLines(HELP[field])}
                value={state.parallel[field]}
                onChange={(v) => dispatch({ type: 'setParallel', parallel: { [field]: v } })}
                issues={issuesOf(field)}
                hint={
                  checked
                    ? `Valid ${field.toUpperCase()}: ${validDegrees(field, checked.model, checked.quant, checked.parallel, state.workload, checked.topology).join(', ')}`
                    : undefined
                }
              />
            ))}
          </div>
          {balance}
        </>
      )}
      {/* DCP splits the KV cache over TP ranks: none in DP attention + EP (TP 1). */}
      {!dep && (
        <details className="advanced">
          <summary>Advanced</summary>
          <ParallelAdvancedFields
            state={state}
            dispatch={dispatch}
            loaded={loaded}
            built={built}
            hint={
              checked
                ? `Valid DCP: ${validDegrees('dcp', checked.model, checked.quant, checked.parallel, state.workload, checked.topology).join(', ')}`
                : undefined
            }
          />
        </details>
      )}
    </Panel>
  );
}
