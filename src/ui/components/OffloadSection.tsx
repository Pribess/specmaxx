import { DEFAULT_PREFETCH_ACCURACY } from '../../sim/engine/transfer';
import { HOST_PCIE_LINKS, HOST_RAM_PRESETS, SSD_PRESETS, hostPcieLink } from '../../sim/hw/presets';
import {
  defaultScaleLayout,
  formatBytesPerSec,
  hasLookupTables,
  hasMoe,
  offloadsAnything,
  resolveHost,
  tablesOffDeviceByDesign,
  type HitRatioChoice,
  type HostChoice,
  type OffloadChoice,
  type ScaleLayoutChoice,
  type TierChoice,
} from '../state';
import { NumberField } from './NumberField';
import { Picker } from './Picker';
import { helpLines } from './helpLines';
import type { PickerOption } from './fuzzy';
import type { PanelProps } from './types';

/** Picker value of "on the device". */
const DEVICE = 'device';

/** "?" help of the prefetch accuracy (docs/simulation-model.md §12 "Expert prefetch"). */
const PREFETCH_HELP = helpLines([
  'How often needed experts are guessed one layer early, so they load in advance.',
  'Missed experts load late and slow the step.',
  '100 means perfect guesses, 0 means no prefetching.',
]);

/** Tier labels. */
const TIER_LABELS: Record<TierChoice | typeof DEVICE, string> = {
  device: 'Device',
  host_ram: 'Host RAM',
  nvme: 'SSD',
};

/** Hit-ratio model labels (custom is a number in the state). */
const HIT_RATIO_OPTIONS: PickerOption[] = [
  { value: 'uniform', label: 'Uniform (h = f)' },
  { value: 'skewed_classic', label: 'Skewed, ≤ 64 experts' },
  { value: 'skewed_fine_grained', label: 'Skewed, ≥ 128 experts' },
  { value: 'custom', label: 'Custom' },
];

/** Scale layout labels. */
const SCALE_LAYOUT_OPTIONS: PickerOption[] = [
  { value: 'separate', label: 'Separate scale reads' },
  { value: 'pinned_host', label: 'Scales pinned in host RAM' },
  { value: 'interleaved', label: 'Interleaved rows' },
];

/** Scale layout labels on unified memory: the pinned scales stay in the device's own pool. */
const UNIFIED_SCALE_LAYOUT_OPTIONS: PickerOption[] = SCALE_LAYOUT_OPTIONS.map((o) =>
  o.value === 'pinned_host' ? { ...o, label: 'Scales pinned in memory' } : o,
);

/** Picker value of the default host link (the device's own interface). */
const DEFAULT_LINK = 'default';

/** `<gen>x<lanes>` of a host-link spec name `PCIe 5.0 x16 (host link)`. */
function linkKey(name: string): string {
  const m = /PCIe (\d)\.0 x(\d+)/.exec(name);
  return m ? `${m[1]}x${m[2]}` : name;
}

/** "PCIe 5.0 x16, 63.0 GB/s" of a host-link spec. */
function linkLabel(link: { name: string; bandwidthPerDirectionBytesPerSec: number }): string {
  return `${link.name.replace(' (host link)', '')}, ${formatBytesPerSec(link.bandwidthPerDirectionBytesPerSec)}`;
}

/** Host-RAM picker rows: grouped by DDR generation (DDR5 first, then DDR4), fastest first within a group. */
const HOST_RAM_OPTIONS: PickerOption[] = [...HOST_RAM_PRESETS]
  .map((r) => ({ r, generation: r.id.split('-')[0].toUpperCase() }))
  .sort(
    (a, b) =>
      (a.generation < b.generation ? 1 : a.generation > b.generation ? -1 : 0) ||
      b.r.bandwidthBytesPerSec - a.r.bandwidthBytesPerSec,
  )
  .map(({ r, generation }) => ({
    value: r.id,
    label: r.name,
    group: generation,
    description: `${(r.bandwidthBytesPerSec / 1e9).toFixed(1)} GB/s`,
  }));

/**
 * Shared state of the offload controls (the main section and its Advanced subsection): the model and device, the
 * current offload choice and its setter, and the tier picker rows. `undefined` before a model and device resolve.
 */
function offloadControls({ state, dispatch, loaded, built }: PanelProps) {
  const model = loaded?.model;
  const device = built.hardware?.device.device;
  if (model === undefined || device === undefined) return undefined;
  const unified = device.memory.kind === 'unified';
  const offload = state.deployment.offload ?? {};
  const designOffDevice = tablesOffDeviceByDesign(model);
  const setOffload = (next: OffloadChoice) =>
    dispatch({
      type: 'setOffload',
      offload: next,
      tablesOffDeviceByDesign: designOffDevice && !unified,
    });
  const tierOptions: PickerOption[] = (
    unified ? [DEVICE, 'nvme'] : [DEVICE, 'host_ram', 'nvme']
  ).map((t) => ({
    value: t,
    label: TIER_LABELS[t as TierChoice | typeof DEVICE],
  }));
  const tier = (v: string): TierChoice | undefined =>
    v === DEVICE ? undefined : (v as TierChoice);
  return {
    model,
    device,
    unified,
    offload,
    designOffDevice,
    setOffload,
    tierOptions,
    tier,
    anyOff: offloadsAnything(offload),
  };
}

/**
 * Hardware → "Offload" (dedicated memory) / "Storage" (unified memory), design doc §6.2: where routed experts and
 * whole layers live, the expert cache, and the host's RAM, SSD and PCIe link. On unified memory host RAM is the
 * device pool, so only the SSD is a tier and there is no host link. The less common knobs (lookup-table placement,
 * scale layout, queue depth, prefetch accuracy) are {@link OffloadAdvancedFields}, in the panel's Advanced.
 */
export function OffloadSection(props: PanelProps) {
  const { state, dispatch, built } = props;
  const controls = offloadControls(props);
  if (controls === undefined) return null;
  const { model, device, unified, offload, setOffload, tierOptions, tier, anyOff } = controls;
  const host = state.hw.host ?? {};
  const setHost = (next: Partial<HostChoice>) =>
    dispatch({ type: 'setHost', host: { ...host, ...next } });
  const resolvedHost = resolveHost(state.hw, device);
  const layerCount = model.layers.length;
  const issues = built.issues.filter((i) => i.field === 'offload');
  const experts = offload.experts;
  const hitRatio: HitRatioChoice = experts?.hitRatio ?? 'uniform';
  return (
    <fieldset className="deployment-section" data-testid="offload-section">
      <legend>{unified ? 'Storage' : 'Offload'}</legend>
      {hasMoe(model) && (
        <>
          <Picker
            label="Routed experts"
            value={experts?.tier ?? DEVICE}
            options={tierOptions}
            onChange={(v) => {
              const t = tier(v);
              setOffload({
                ...offload,
                experts: t === undefined ? undefined : { ...experts, tier: t },
              });
            }}
            help={helpLines([
              'Where the experts (the specialist sub-networks of an MoE model) are stored.',
              'Experts not on the GPU stream in over PCIe.',
              'The GPU still does the compute.',
            ])}
          />
          {experts !== undefined && (
            <div className="fields two">
              <NumberField
                label="From layer"
                value={experts.fromLayer ?? 0}
                min={0}
                max={layerCount - 1}
                onChange={(v) =>
                  setOffload({
                    ...offload,
                    experts: { ...experts, fromLayer: Number.isInteger(v) && v >= 0 ? v : 0 },
                  })
                }
                help={helpLines([
                  'Experts from this layer onward are offloaded.',
                  'Earlier layers keep their experts on the device.',
                ])}
              />
              <NumberField
                label="Resident %"
                value={Math.round((experts.residentFraction ?? 0) * 1000) / 10}
                min={0}
                max={100}
                step="1"
                onChange={(v) =>
                  setOffload({
                    ...offload,
                    experts: { ...experts, residentFraction: Math.min(1, Math.max(0, v / 100)) },
                  })
                }
                help={helpLines([
                  'Share of each offloaded layer’s experts kept in device memory.',
                  'More resident experts mean fewer slow transfers.',
                  'They count toward memory use.',
                ])}
              />
            </div>
          )}
          {experts !== undefined && (
            <>
              <Picker
                label="Hit-ratio model"
                value={typeof hitRatio === 'number' ? 'custom' : hitRatio}
                options={HIT_RATIO_OPTIONS}
                onChange={(v) =>
                  setOffload({
                    ...offload,
                    experts: {
                      ...experts,
                      hitRatio:
                        v === 'custom' ? (experts.residentFraction ?? 0) : (v as HitRatioChoice),
                    },
                  })
                }
                help={helpLines([
                  'How often a needed expert is already on the device.',
                  'Uniform assumes experts are used evenly.',
                  'Skewed options assume some experts are used more often.',
                ])}
              />
              {typeof hitRatio === 'number' && (
                <NumberField
                  label="Hit ratio h"
                  value={hitRatio}
                  step="0.01"
                  min={0}
                  max={1}
                  onChange={(v) =>
                    setOffload({
                      ...offload,
                      experts: { ...experts, hitRatio: Math.min(1, Math.max(0, v)) },
                    })
                  }
                  help={helpLines([
                    'How often a needed expert is already on the device.',
                    '1 means always, 0 means never.',
                    'Higher is faster.',
                  ])}
                />
              )}
            </>
          )}
        </>
      )}
      <Picker
        label="Dense layers"
        value={offload.dense?.tier ?? DEVICE}
        options={tierOptions}
        onChange={(v) => {
          const t = tier(v);
          setOffload({
            ...offload,
            dense:
              t === undefined
                ? undefined
                : {
                    tier: t,
                    layersOnDevice: offload.dense?.layersOnDevice ?? Math.floor(layerCount / 2),
                  },
          });
        }}
        help={helpLines([
          'Where whole layers beyond the on-device count are stored.',
          'They stream in from there every step.',
        ])}
      />
      {offload.dense !== undefined && (
        <NumberField
          label="Layers on device"
          value={offload.dense.layersOnDevice}
          min={0}
          max={layerCount}
          onChange={(v) =>
            setOffload({
              ...offload,
              dense: { ...offload.dense!, layersOnDevice: Number.isInteger(v) && v >= 0 ? v : 0 },
            })
          }
          help={helpLines([
            `How many of the ${layerCount} layers stay on the device.`,
            'The rest stream in from the tier every step.',
          ])}
        />
      )}
      {issues.length > 0 && (
        <span className="hint" role="status">
          {issues.map((i, n) => (
            <span className={i.level === 'error' ? 'error' : ''} key={n}>
              {i.message}
            </span>
          ))}
        </span>
      )}
      {anyOff && !unified && (
        <>
          <Picker
            label="Host RAM"
            value={resolvedHost.ram.id}
            options={HOST_RAM_OPTIONS}
            searchThreshold={4}
            onChange={(v) => setHost({ ramId: v, ramCapacityBytes: undefined })}
            help={helpLines([
              'Speed of the machine’s system memory (RAM), from its datasheet.',
              'All GPUs in a machine share it.',
            ])}
          />
          <NumberField
            label="Host RAM capacity (GB)"
            value={resolvedHost.ram.capacityBytes / 1e9}
            min={1}
            onChange={(v) => setHost({ ramCapacityBytes: v > 0 ? Math.round(v * 1e9) : undefined })}
            help={helpLines([
              'Installed system RAM per machine.',
              'Offloaded weights must fit in it.',
            ])}
          />
        </>
      )}
      {anyOff && (
        <Picker
          label="SSD"
          value={resolvedHost.ssd.id}
          options={SSD_PRESETS.map((s) => ({ value: s.id, label: s.name }))}
          searchThreshold={4}
          onChange={(v) => setHost({ ssdId: v })}
          help={helpLines([
            'Speed of the NVMe SSD, from its datasheet.',
            'All GPUs in a machine share it.',
          ])}
        />
      )}
      {anyOff && !unified && (
        <Picker
          label="Host link"
          value={host.link ? `${host.link.generation}x${host.link.lanes}` : DEFAULT_LINK}
          options={[
            {
              value: DEFAULT_LINK,
              label: device.hostInterface
                ? `${linkLabel(hostPcieLink(device.hostInterface.generation, device.hostInterface.lanes))} (device)`
                : 'Not specified by the device',
            },
            ...HOST_PCIE_LINKS.map((l) => ({ value: linkKey(l.name), label: linkLabel(l) })),
          ]}
          onChange={(v) => {
            if (v === DEFAULT_LINK) setHost({ link: undefined });
            else {
              const [g, lanes] = v.split('x').map(Number);
              setHost({ link: { generation: g as 3 | 4 | 5, lanes: lanes as 4 | 8 | 16 } });
            }
          }}
          help={helpLines([
            'The PCIe link between the GPU and system memory.',
            'Offloaded data cannot stream faster than this link or the RAM allows.',
          ])}
        />
      )}
    </fieldset>
  );
}

/**
 * Hardware → Advanced → "Offload" / "Storage": the lookup-table placement (models with n-gram tables), the scale
 * layout of off-device tables, the queue depth and the prefetch accuracy (with any offload on). Renders nothing when
 * none of them applies.
 */
export function OffloadAdvancedFields(props: PanelProps) {
  const controls = offloadControls(props);
  if (controls === undefined) return null;
  const { model, unified, offload, designOffDevice, setOffload, tierOptions, tier, anyOff } =
    controls;
  const tables = offload.tables;
  if (!hasLookupTables(model) && !anyOff) return null;
  const placeTablesOnHost = () =>
    setOffload({ ...offload, tables: { tier: unified ? 'nvme' : 'host_ram' } });
  return (
    <fieldset className="unlock advanced-fields" data-testid="offload-advanced">
      <legend>{unified ? 'Storage' : 'Offload'}</legend>
      {hasLookupTables(model) && (
        <div className="tables-field">
          <Picker
            label="Lookup tables"
            value={tables?.tier ?? DEVICE}
            options={tierOptions}
            onChange={(v) => {
              const t = tier(v);
              setOffload({ ...offload, tables: t === undefined ? undefined : { tier: t } });
            }}
            help={helpLines([
              'Where the lookup tables are stored.',
              designOffDevice
                ? 'This model is built to keep its tables off the GPU.'
                : 'Only a few rows are read per token, so an SSD delay matters.',
            ])}
          />
          {designOffDevice && tables === undefined && (
            <button
              type="button"
              className="chip"
              onClick={placeTablesOnHost}
              data-testid="tables-hint"
            >
              {unified ? 'SSD ↗' : 'Host RAM ↗'}
            </button>
          )}
        </div>
      )}
      {tables !== undefined && (
        <Picker
          label="Scale layout"
          value={tables.scaleLayout ?? defaultScaleLayout(tables.tier)}
          options={unified ? UNIFIED_SCALE_LAYOUT_OPTIONS : SCALE_LAYOUT_OPTIONS}
          onChange={(v) =>
            setOffload({
              ...offload,
              tables: { ...tables, scaleLayout: v as ScaleLayoutChoice },
            })
          }
          help={helpLines([
            'Separate: each row and its scales are two reads.',
            unified
              ? 'Pinned in memory: scales stay there, so only rows are read from the SSD.'
              : 'Pinned in host RAM: scales stay there, so only rows are read from the SSD.',
            'Interleaved: one read per row.',
          ])}
        />
      )}
      {anyOff && (
        <NumberField
          label="Queue depth"
          value={offload.queueDepth ?? NaN}
          min={1}
          onChange={(v) =>
            setOffload({
              ...offload,
              queueDepth: Number.isInteger(v) && v >= 1 ? v : undefined,
            })
          }
          help={helpLines([
            'How many reads are in flight at once.',
            '1 means every read waits for the one before.',
            'Empty uses the drive’s rated maximum.',
          ])}
        />
      )}
      {anyOff && (
        <NumberField
          label="Prefetch accuracy %"
          value={Math.round((offload.prefetchAccuracy ?? DEFAULT_PREFETCH_ACCURACY) * 100)}
          min={0}
          max={100}
          step="1"
          onChange={(v) =>
            setOffload({
              ...offload,
              prefetchAccuracy: Number.isFinite(v) ? Math.min(1, Math.max(0, v / 100)) : undefined,
            })
          }
          help={PREFETCH_HELP}
        />
      )}
    </fieldset>
  );
}
