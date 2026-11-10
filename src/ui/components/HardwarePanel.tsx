import { DEVICE_PRESET_GROUPS, deviceUnlocks } from '../../sim/hw/presets';
import { POPULAR_SETUPS, setupLabel } from '../../sim/hw/presets/catalog';
import {
  matchingPopularSetup,
  popularSetupChoice,
  reduce,
  resolveHardware,
  setupChoiceOf,
  chooseDegrees,
  applyQuantOverride,
  type AppAction,
} from '../state';
import { Panel } from './Panel';
import { Picker } from './Picker';
import { helpLines } from './helpLines';
import type { PickerOption } from './fuzzy';
import { CustomSpecEditor } from './CustomSpecEditor';
import { SetupRow } from './SetupRow';
import { OffloadAdvancedFields, OffloadSection } from './OffloadSection';
import { AllToAllFields } from './AllToAllFields';
import { MemoryFitFields } from './MemoryFitFields';
import { PerOpCostFields } from './PerOpCostFields';
import type { PanelProps as Props } from './types';

/** The group of the curated multi-device setups, at the top of the device picker. */
export const POPULAR_GROUP = 'Popular setups';

/** Picker options: the popular setups first, then every single device by vendor group (newest first). */
const deviceOptions: PickerOption[] = [
  ...POPULAR_SETUPS.map((p) => ({ value: `setup:${p.id}`, label: p.name, group: POPULAR_GROUP })),
  ...DEVICE_PRESET_GROUPS.flatMap((g) =>
    g.devices.map((d) => ({ value: `device:${d.id}`, label: d.name, group: g.name })),
  ),
];

export function HardwarePanel(props: Props) {
  const { state, built, loaded } = props;
  /**
   * Dispatches hardware actions, then picks TP and PP that use all devices of the resulting rig
   * (TP × PP × DP must equal the device count), keeping the user's DP and EP while they stay valid.
   */
  const dispatch = (...actions: AppAction[]) => {
    for (const a of actions) props.dispatch(a);
    // Capacity, unlock, engine, link, cabling and edit changes keep the user's TP and PP.
    const resets = actions.some(
      (a) => a.type === 'setHardware' || a.type === 'setDevice' || a.type === 'setCounts',
    );
    if (!loaded || !resets) return;
    try {
      const next = actions.reduce(reduce, state);
      const topology = resolveHardware(next.hw).topology;
      const quant = applyQuantOverride(loaded.quant.quant, next.quant, loaded.quant.exl3);
      props.dispatch({
        type: 'setParallel',
        parallel: chooseDegrees(loaded.model, quant, next.workload, topology, {
          dp: next.parallel.dp,
          ep: next.parallel.ep,
          pp: next.parallel.pp,
          layout: next.parallelLayout,
          denseTpGroup: next.parallel.denseTpGroup,
          // Kept only when it still fits the chosen TP; else the result's `dcp: undefined` turns DCP off.
          dcp: next.parallel.dcp,
        }),
      });
    } catch {
      // Unresolvable hardware: the build step reports it.
    }
  };
  const device = built.hardware?.device;
  const setup = matchingPopularSetup(state.hw);
  const selected = setup ? `setup:${setup.id}` : `device:${state.hw.deviceId}`;
  const total = state.hw.devicesPerNode * state.hw.nodes;
  const label = (() => {
    if (setup) return setup.name;
    try {
      const name = setupLabel(setupChoiceOf(state.hw, built.hardware?.memCapacityBytes));
      return built.hardware?.edited ? `${name} (edited)` : name;
    } catch {
      return `${total} × ${device?.base.name ?? state.hw.deviceId}`;
    }
  })();
  return (
    <Panel title="Hardware" value={label} open>
      <Picker
        label="Device"
        value={selected}
        options={deviceOptions}
        onChange={(value) => {
          if (value.startsWith('setup:')) {
            const preset = POPULAR_SETUPS.find((p) => `setup:${p.id}` === value);
            if (preset) dispatch({ type: 'setHardware', hw: popularSetupChoice(preset) });
          } else dispatch({ type: 'setDevice', deviceId: value.slice('device:'.length) });
        }}
        help={helpLines([
          'Pick a single chip or a ready-made setup.',
          'Setups also fill in memory, machine count, links and cabling.',
        ])}
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
          help={helpLines(['Memory per device.', 'Lists the sizes this device is sold with.'])}
        />
      )}
      <SetupRow state={state} built={built} dispatch={dispatch} />
      <OffloadSection {...props} />
      <details className="advanced">
        <summary>Advanced</summary>
        <OffloadAdvancedFields {...props} />
        <MemoryFitFields {...props} />
        <AllToAllFields {...props} />
        <PerOpCostFields {...props} />
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
            help={helpLines([
              'Which units do the matrix math, such as tensor cores or shader cores.',
              'Their peak speed sets the compute limit.',
            ])}
          />
        )}
        <CustomSpecEditor {...props} />
      </details>
    </Panel>
  );
}
