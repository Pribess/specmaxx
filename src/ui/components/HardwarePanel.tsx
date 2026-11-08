import {
  DEVICE_PRESET_GROUPS,
  SYSTEM_PRESETS,
  deviceUnlocks,
  LINK_PRESETS,
  linkPresetId,
} from '../../sim/hw/presets';
import { topologyPresetsFor, matchingSystem, systemChoice } from '../state';
import { Panel } from './Panel';
import { Picker } from './Picker';
import type { PickerOption } from './fuzzy';
import { NumberField } from './NumberField';
import { CustomSpecEditor } from './CustomSpecEditor';
import type { PanelProps as Props } from './types';

/** Picker options: every device preset by vendor group, then the multi-device setups. */
const deviceOptions: PickerOption[] = [
  ...DEVICE_PRESET_GROUPS.flatMap((g) =>
    g.devices.map((d) => ({ value: `device:${d.id}`, label: d.name, group: g.name })),
  ),
  ...SYSTEM_PRESETS.map((p) => ({
    value: `system:${p.id}`,
    label: p.name,
    group: 'Multi-device setups',
  })),
];

export function HardwarePanel(props: Props) {
  const { state, dispatch, built } = props;
  const device = built.hardware?.device;
  const system = matchingSystem(state.hw);
  const selected = system ? `system:${system.id}` : `device:${state.hw.deviceId}`;
  const unit = device?.device.memory.kind === 'unified' ? 'Devices' : 'GPUs';
  const total = state.hw.devicesPerNode * state.hw.nodes;
  return (
    <Panel
      title="Hardware"
      value={system?.name ?? `${total} × ${device?.base.name ?? state.hw.deviceId}`}
      open
    >
      <Picker
        label="Device"
        value={selected}
        options={deviceOptions}
        onChange={(value) => {
          if (value.startsWith('system:')) {
            const preset = SYSTEM_PRESETS.find((p) => `system:${p.id}` === value);
            if (preset) dispatch({ type: 'setHardware', hw: systemChoice(preset) });
          } else dispatch({ type: 'setDevice', deviceId: value.slice('device:'.length) });
        }}
        help="A single device, or a ready-made multi-device setup that also sets the memory option, device counts and links."
      />
      {device && device.device.memory.capacityOptionsBytes.length > 1 && (
        <Picker
          label="Memory capacity"
          value={String(state.hw.capacityIndex)}
          options={device.device.memory.capacityOptionsBytes.map((v, i) => ({
            value: String(i),
            label: `${v / 1e9}GB`,
          }))}
          onChange={(v) => dispatch({ type: 'setCapacity', index: Number(v) })}
          help="Memory per device, for the SKU variants this device ships with."
        />
      )}
      <details className="advanced">
        <summary>Advanced</summary>
      {device &&
        deviceUnlocks(device.base).map((axis) => (
          <fieldset className="unlock" key={axis.id}>
            <legend>{axis.name}</legend>
            {axis.options.length > 1 && (
              <label className="check">
                <input
                  type="radio"
                  name={axis.id}
                  checked={!state.hw.unlocks[axis.id]}
                  onChange={() =>
                    dispatch({ type: 'setUnlock', axisId: axis.id, optionId: undefined })
                  }
                />
                Stock
              </label>
            )}
            {axis.options.map((o) => {
              const community = [...o.deviceModifiers, ...(o.intraNodeLinkModifiers ?? [])].some(
                (m) => m.sources?.some((s) => s.kind === 'community'),
              );
              return (
                <label className="check" key={o.id}>
                  <input
                    type={axis.options.length === 1 ? 'checkbox' : 'radio'}
                    name={axis.id}
                    checked={state.hw.unlocks[axis.id] === o.id}
                    onChange={(e) =>
                      dispatch({
                        type: 'setUnlock',
                        axisId: axis.id,
                        optionId: e.target.checked ? o.id : undefined,
                      })
                    }
                  />
                  <span>
                    {o.name}
                    {community && <small className="tag">community-measured</small>}
                  </span>
                </label>
              );
            })}
            <details className="hint">
              <summary>What this changes</summary>
              {axis.description}
            </details>
          </fieldset>
        ))}
      {device && device.device.engines.length > 1 && (
        <Picker
          label="Compute engine"
          value={state.hw.engineId ?? device.device.engines[0].id}
          options={device.device.engines.map((e) => ({ value: e.id, label: e.name }))}
          onChange={(engineId) => dispatch({ type: 'setEngine', engineId })}
          help="The matrix engine whose peak rates are used (for example tensor cores versus shader cores)."
        />
      )}
      <div className="fields two">
        <NumberField
          label={`${unit} in one machine`}
          value={state.hw.devicesPerNode}
          onChange={(devicesPerNode) => dispatch({ type: 'setCounts', devicesPerNode })}
          issues={built.issues.filter((i) => i.field === 'devices')}
          help={`How many ${unit.toLowerCase()} sit in one machine. They talk to each other over the fast local link (NVLink, PCIe or the unified fabric).`}
        />
        <NumberField
          label="Machines"
          value={state.hw.nodes}
          onChange={(nodes) => dispatch({ type: 'setCounts', nodes })}
          help="How many machines are connected. Machines talk to each other over the network link (Ethernet, InfiniBand, Thunderbolt or ConnectX), which is much slower than the local link."
        />
      </div>
      <p className="total" data-testid="device-total">
        = {total} {unit} total
      </p>
      {(['intra', 'inter'] as const).map(
        (level) =>
          (level === 'intra' ? state.hw.devicesPerNode : state.hw.nodes) > 1 && (
            <Picker
              key={level}
              label={level === 'intra' ? 'Local link (within a machine)' : 'Network link (between machines)'}
              value={state.hw[level === 'intra' ? 'intraLinkId' : 'interLinkId'] ?? 'default'}
              options={[
                { value: 'default', label: 'Device default' },
                ...LINK_PRESETS.map((l) => ({ value: linkPresetId(l), label: l.name })),
              ]}
              onChange={(v) =>
                dispatch({ type: 'setLink', level, linkId: v === 'default' ? undefined : v })
              }
              help={
                level === 'intra'
                  ? 'The link between devices inside one machine.'
                  : 'The link between machines.'
              }
            />
          ),
      )}
      {device && (
        <details className="quick-picks" open>
          <summary>Topology quick picks</summary>
          {topologyPresetsFor(device.base).map((t) => (
            <button
              className="quiet"
              key={t.id}
              onClick={() => {
                dispatch({ type: 'setCounts', devicesPerNode: t.devicesPerNode, nodes: t.nodes });
                if (t.intraNode)
                  dispatch({ type: 'setLink', level: 'intra', linkId: linkPresetId(t.intraNode) });
                if (t.interNode)
                  dispatch({ type: 'setLink', level: 'inter', linkId: linkPresetId(t.interNode) });
              }}
            >
              {t.name}
            </button>
          ))}
        </details>
      )}
      <CustomSpecEditor {...props} />
      </details>
    </Panel>
  );
}
