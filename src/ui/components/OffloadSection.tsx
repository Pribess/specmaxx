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
import type { PickerOption } from './fuzzy';
import type { PanelProps } from './types';

/** Picker value of "on the device". */
const DEVICE = 'device';

/** "?" help of the prefetch accuracy (docs/simulation-model.md §12 "Expert prefetch"). */
const PREFETCH_HELP = (
  <>
    <span className="help-line">
      Share of the needed experts the predictor picks one layer ahead (recall). Measured 67–90 %.
    </span>
    <span className="help-line">
      Predicted experts are copied while the previous layer computes; the rest after the router, on
      the critical path.
    </span>
    <span className="help-line">100 = perfect prediction, 0 = no prefetching. Default 80.</span>
  </>
);

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
            help="Where the routed expert weights live. Misses stream to the GPU over PCIe at the hardware ceiling; the GPU computes."
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
                help="Experts of layers ≥ this index are offloaded (llama.cpp --n-cpu-moe keeps the first layers on the device)."
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
                help="Share f of each offloaded layer's experts kept in device memory (pinned or cache), counted in the memory fit."
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
                help="Probability that a needed expert is already on the device, per unique expert per layer per step at batch 1. Uniform is exact for balanced routing; skewed presets fit measured LRU locality (h = f^0.55, f^0.2). Bounded by the resident count."
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
                  help="Probability that a needed expert is resident, per unique expert per layer per step at batch 1."
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
        help="Where whole decoder layers beyond the on-device count live; they are streamed from that tier every step (llama.cpp -ngl)."
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
          help={`Layers beyond this count (of ${layerCount}) are streamed from the tier every step.`}
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
            help="Host memory: theoretical channel bandwidth and capacity (datasheet values); shared by every GPU of the node."
          />
          <NumberField
            label="Host RAM capacity (GB)"
            value={resolvedHost.ram.capacityBytes / 1e9}
            min={1}
            onChange={(v) => setHost({ ramCapacityBytes: v > 0 ? Math.round(v * 1e9) : undefined })}
            help="Installed host-RAM capacity of the node; offloaded weights count against it."
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
          help="NVMe drive: sequential read, QD1 latency and max random-read IOPS from the datasheet; shared by every GPU of the node."
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
          help="GPU-to-host PCIe link per direction; with host RAM bandwidth, the streaming ceiling."
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
            help={`Where the hashed n-gram tables live. Only the rows a token needs are gathered: ~1 µs per step from RAM, latency-bound from SSD. Counted against that tier's capacity.${designOffDevice ? ' This model ships its table to be kept off-device.' : ''}`}
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
          options={SCALE_LAYOUT_OPTIONS}
          onChange={(v) =>
            setOffload({
              ...offload,
              tables: { ...tables, scaleLayout: v as ScaleLayoutChoice },
            })
          }
          help="Block-scaled tables only. Separate: the value row and its scales are two reads on the tier. Scales pinned in host RAM: the rows on the SSD, the scales in host RAM (counted there). Interleaved: one read per row."
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
          help="Outstanding read requests the deployment keeps in flight; 1 serializes every request behind its latency. Empty: the tier's rated maximum."
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
