import {
  DEVICE_PRESET_GROUPS,
  deviceUnlocks,
  LINK_PRESETS,
  linkPresetId,
} from '../../sim/hw/presets';
import { topologyPresetsFor } from '../state';
import { Panel } from './Panel';
import { NumberField } from './NumberField';
import { CustomSpecEditor } from './CustomSpecEditor';
import type { PanelProps as Props } from './types';

export function HardwarePanel(props: Props) {
  const { state, dispatch, built } = props;
  const device = built.hardware?.device;
  return (
    <Panel
      title="Hardware"
      value={`${state.hw.devicesPerNode * state.hw.nodes} × ${device?.base.name ?? state.hw.deviceId}`}
      open
    >
      <label className="field">
        <span>Device preset</span>
        <select
          value={state.hw.deviceId}
          onChange={(e) => dispatch({ type: 'setDevice', deviceId: e.target.value })}
        >
          {DEVICE_PRESET_GROUPS.map((g) => (
            <optgroup label={g.name} key={g.name}>
              {g.devices.map((d) => (
                <option value={d.id} key={d.id}>
                  {d.name}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
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
        <label className="field">
          <span>Compute engine</span>
          <select
            value={state.hw.engineId ?? device.device.engines[0].id}
            onChange={(e) => dispatch({ type: 'setEngine', engineId: e.target.value })}
          >
            {device.device.engines.map((e) => (
              <option value={e.id} key={e.id}>
                {e.name}
              </option>
            ))}
          </select>
        </label>
      )}
      {device && device.device.memory.capacityOptionsBytes.length > 1 && (
        <label className="field">
          <span>Memory capacity</span>
          <select
            value={state.hw.capacityIndex}
            onChange={(e) => dispatch({ type: 'setCapacity', index: +e.target.value })}
          >
            {device.device.memory.capacityOptionsBytes.map((v, i) => (
              <option value={i} key={i}>
                {v / 1e9} GB
              </option>
            ))}
          </select>
        </label>
      )}
      <div className="fields two">
        <NumberField
          label="Devices per node"
          value={state.hw.devicesPerNode}
          onChange={(devicesPerNode) => dispatch({ type: 'setCounts', devicesPerNode })}
          issues={built.issues.filter((i) => i.field === 'devices')}
        />
        <NumberField
          label="Nodes"
          value={state.hw.nodes}
          onChange={(nodes) => dispatch({ type: 'setCounts', nodes })}
        />
      </div>
      {(['intra', 'inter'] as const).map(
        (level) =>
          (level === 'intra' ? state.hw.devicesPerNode : state.hw.nodes) > 1 && (
            <label className="field" key={level}>
              <span>{level === 'intra' ? 'Intra-node' : 'Inter-node'} interconnect</span>
              <select
                value={state.hw[level === 'intra' ? 'intraLinkId' : 'interLinkId'] ?? ''}
                onChange={(e) =>
                  dispatch({ type: 'setLink', level, linkId: e.target.value || undefined })
                }
              >
                <option value="">Device default</option>
                {LINK_PRESETS.map((l) => (
                  <option key={linkPresetId(l)} value={linkPresetId(l)}>
                    {l.name}
                  </option>
                ))}
              </select>
            </label>
          ),
      )}
      {device && (
        <details className="quick-picks">
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
    </Panel>
  );
}
