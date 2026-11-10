import {
  DEFAULT_OUTPUT_LEN,
  KV_LINK_PRESETS,
  formatContextTokens,
  kvLinkPreset,
  kvLinkIdOf,
  type AppAction,
  type AppState,
  type FieldIssue,
  type KvEditPath,
  type PoolId,
} from '../state';
import { Panel } from './Panel';
import { Segmented } from './Segmented';
import { NumberField } from './NumberField';
import { Picker } from './Picker';
import { helpLines } from './helpLines';

/** The KV link presets as picker options. */
const KV_LINK_OPTIONS = KV_LINK_PRESETS.map((l) => ({ value: kvLinkIdOf(l), label: l.name }));

/**
 * Prefill/decode disaggregation: Off | On; when on, which pool the Hardware, Parallelism and batch fields edit, the
 * output length, the instance counts and the P→D KV link (preset, bandwidth, latency, overlap).
 */
export function DisaggregationPanel({
  state,
  dispatch,
  editingPool,
  onEditingPool,
  issues,
}: {
  state: AppState;
  dispatch(action: AppAction): void;
  /** The pool the per-pool panels edit (React state of the app, not in the URL). */
  editingPool: PoolId;
  onEditingPool(pool: PoolId): void;
  /** The disaggregated build's issues (only the panel's own fields are shown here). */
  issues: FieldIssue[];
}) {
  const d = state.disaggregation;
  const instances = d?.instances ?? { prefill: 1, decode: 1 };
  const outputLen = d?.outputLen ?? DEFAULT_OUTPUT_LEN;
  const field = (f: FieldIssue['field']) => issues.filter((i) => i.field === f);
  const preset = d ? kvLinkPreset(d) : undefined;
  const bw =
    d?.kvEdits?.bandwidthPerDirectionBytesPerSec ?? preset?.bandwidthPerDirectionBytesPerSec;
  const lat = d?.kvEdits?.latencySec ?? preset?.latencySec;
  /** Sets one KV link edit; the preset's own value clears it. */
  const setKvEdit = (path: KvEditPath, value: number) =>
    dispatch({
      type: 'setKvEdit',
      path,
      value: preset !== undefined && value === preset[path] ? undefined : value,
    });
  return (
    <Panel
      title="Disaggregation"
      value={
        d
          ? `${instances.prefill} : ${instances.decode} instances, output ${formatContextTokens(outputLen)}`
          : 'Off'
      }
      open
    >
      <Segmented
        label="Prefill/decode disaggregation"
        value={d ? 'on' : 'off'}
        segments={[
          { value: 'off', label: 'Off' },
          { value: 'on', label: 'On' },
        ]}
        onChange={(v) => {
          dispatch({ type: 'setDisaggregation', on: v === 'on' });
          if (v === 'off') onEditingPool('prefill');
        }}
        help={helpLines([
          'Runs prefill and decode on separate groups of GPUs (pools), each with its own hardware, parallelism and batch.',
          'Finished prompts send their KV cache from a prefill GPU to a decode GPU.',
          'Off: one group runs both, as before.',
        ])}
      />
      {d && (
        <>
          <Segmented
            label="Edit pool"
            value={editingPool}
            segments={[
              { value: 'prefill', label: 'Prefill' },
              { value: 'decode', label: 'Decode' },
            ]}
            onChange={(v) => onEditingPool(v as PoolId)}
            help={helpLines([
              'Which pool the Hardware, Parallelism and batch size fields change.',
              'The model, formats and context lengths are shared by both pools.',
            ])}
          />
          <div className="fields">
            <NumberField
              label="Output length (tokens)"
              value={outputLen}
              onChange={(v) => dispatch({ type: 'setDisaggregationParams', outputLen: v })}
              issues={field('outputLen')}
              help={helpLines([
                'Tokens each request generates.',
                'The decode pool makes them all at the decode context, so set that near prompt + half of this.',
              ])}
            />
            <NumberField
              label="Prefill instances"
              value={instances.prefill}
              min={1}
              onChange={(v) =>
                dispatch({
                  type: 'setDisaggregationParams',
                  instances: { prefill: v, decode: instances.decode },
                })
              }
              issues={field('instances')}
              help={helpLines([
                'Copies of the prefill pool in the deployment.',
                'With the decode instances it sets the system rate and which side limits it.',
              ])}
            />
            <NumberField
              label="Decode instances"
              value={instances.decode}
              min={1}
              onChange={(v) =>
                dispatch({
                  type: 'setDisaggregationParams',
                  instances: { prefill: instances.prefill, decode: v },
                })
              }
              help={helpLines([
                'Copies of the decode pool in the deployment.',
                'The balanced ratio shows how many keep both sides busy.',
              ])}
            />
          </div>
          <Picker
            label="KV link"
            value={preset ? kvLinkIdOf(preset) : undefined}
            options={KV_LINK_OPTIONS}
            onChange={(v) => dispatch({ type: 'setKvLink', linkId: v })}
            help={helpLines([
              'The network card each GPU uses to send KV cache from prefill to decode.',
              'Its speed can slow prefill and cap how many requests decode takes in.',
            ])}
          />
          <div className="fields">
            <NumberField
              label="KV bandwidth (GB/s)"
              value={bw === undefined ? NaN : bw / 1e9}
              step="any"
              min={0}
              onChange={(v) => setKvEdit('bandwidthPerDirectionBytesPerSec', v * 1e9)}
              issues={field('kvBandwidth')}
              help={helpLines([
                'Speed of the KV link per GPU, in one direction.',
                'The same speed is used at the sending and the receiving GPU.',
              ])}
            />
            <NumberField
              label="KV latency (µs)"
              value={lat === undefined ? NaN : Math.round(lat * 1e9) / 1e3}
              step="any"
              min={0}
              onChange={(v) => setKvEdit('latencySec', Math.round(v * 1e3) / 1e9)}
              issues={field('kvLatency')}
              help={helpLines(['Delay before the first byte of a transfer arrives.'])}
            />
          </div>
          <Segmented
            label="KV transfer"
            value={d.kvOverlap === 'none' ? 'none' : 'layerwise'}
            segments={[
              { value: 'layerwise', label: 'Overlapped' },
              { value: 'none', label: 'Not overlapped' },
            ]}
            onChange={(v) =>
              dispatch({ type: 'setKvOverlap', overlap: v === 'none' ? 'none' : 'layerwise' })
            }
            help={helpLines([
              'Overlapped: each layer is sent while the next one computes.',
              'Not overlapped: the prefill GPU waits until its batch has arrived before the next one.',
            ])}
          />
        </>
      )}
    </Panel>
  );
}
