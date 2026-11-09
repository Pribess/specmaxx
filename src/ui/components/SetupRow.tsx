import type { ReactNode } from 'react';
import { linkPresetId } from '../../sim/hw/presets';
import {
  cablingName,
  cablingOptions,
  effectiveCabling,
  effectiveLinkIds,
  findLink,
  localLinkChoices,
  setupOptionsOf,
  type AppAction,
  type AppState,
  type BuildResult,
} from '../state';
import { needsHostStaging } from '../../sim/engine/staging';
import { HelpIcon } from './HelpIcon';
import { helpLines } from './helpLines';
import { networkLinkHelp } from './linkHelp';
import { Picker } from './Picker';
import type { PickerOption } from './fuzzy';
import { Segmented, type Segment } from './Segmented';

/** Help texts of the setup row ("?" icons). */
const HELP = {
  machines: helpLines([
    'How many machines are linked together.',
    'They talk over the network, which is much slower than memory.',
  ]),
  network: helpLines([
    'The cable or network between machines.',
    'Only links this device supports are listed.',
  ]),
  local: helpLines([
    'The link between devices inside one machine.',
    'Only links this device supports are listed.',
  ]),
  cabling: helpLines([
    'Switch: every machine plugs into one switch.',
    'Full mesh: a cable between every pair of machines.',
    'Ring: each machine links to two neighbours; a daisy chain is a ring missing one cable.',
  ]),
};

/** Segments for the offered counts, plus the current count when it is not offered (an older link). */
function countSegments(offered: number[], current: number): Segment[] {
  const counts = offered.includes(current) ? offered : [...offered, current].sort((a, b) => a - b);
  return counts.map((n) => ({ value: String(n), label: String(n) }));
}

/** A label with its "?" help and a value as plain text: a field with exactly one possibility. */
function StaticField({
  label,
  help,
  children,
  testId,
}: {
  label: string;
  help: ReactNode;
  children: ReactNode;
  testId?: string;
}) {
  return (
    <div className="field">
      <div className="field-head">
        <span className="field-label">{label}</span>
        <HelpIcon term={label}>{help}</HelpIcon>
      </div>
      <p className="field-static" data-testid={testId}>
        {children}
      </p>
    </div>
  );
}

/**
 * Picker options for a link level: the catalog's links for the device (full name in the list, the catalog's short
 * name as the first keyword, shown in the trigger), plus the current link under "Other" when the catalog does not
 * list it (an older URL).
 */
function linkOptions(
  catalog: { link: { name: string }; shortName: string }[],
  current: string | undefined,
): PickerOption[] {
  const options: PickerOption[] = catalog.map((o) => ({
    value: linkPresetId(o.link),
    label: o.link.name,
    keywords: [o.shortName],
  }));
  if (current !== undefined && !options.some((o) => o.value === current)) {
    options.push({ value: current, label: findLink(current)?.name ?? current, group: 'Other' });
  }
  return options;
}

/**
 * The setup row under the device picker: devices per machine (multi-GPU devices only) and the local link, the
 * machine count, and between machines the network link and the cabling. Every control offers only what the
 * device's catalog entry (`setupOptionsFor`) and its ports allow; the cabling picker is the fabric shape picker.
 */
export function SetupRow({
  state,
  built,
  dispatch,
}: {
  state: AppState;
  built: BuildResult;
  dispatch(...actions: AppAction[]): void;
}) {
  const hw = state.hw;
  const options = setupOptionsOf(hw);
  const device = built.hardware?.device;
  const unit = device?.device.memory.kind === 'unified' ? 'Devices' : 'GPUs';
  const total = hw.devicesPerNode * hw.nodes;
  const links = effectiveLinkIds(hw);
  const showPerMachine = options.devicesPerNodeCounts.length > 1 || hw.devicesPerNode !== 1;
  // Only the local links that join this many devices (the 2-card NVLink bridge only at 2).
  const local = linkOptions(localLinkChoices(hw), links.intra);
  const network = linkOptions(options.networks, links.inter);
  const cablings = cablingOptions(hw);
  const cabling = effectiveCabling(hw);
  // The Network link's "?": the TCP links explain their latency (and host staging for a GPU), others the generic text.
  const interLink = built.hardware?.topology.interNode;
  const staged =
    device !== undefined &&
    built.hardware !== undefined &&
    needsHostStaging(device.device, built.hardware.topology);
  const networkHelp =
    (interLink !== undefined ? networkLinkHelp(interLink, staged) : undefined) ?? HELP.network;
  const totalText = (
    <span data-testid="device-total">
      = {total} {unit === 'GPUs' ? 'GPUs' : 'devices'} total
    </span>
  );
  const linkField = (level: 'intra' | 'inter', opts: PickerOption[], value: string | undefined) => {
    const label = level === 'intra' ? 'Local link' : 'Network link';
    const help = level === 'intra' ? HELP.local : networkHelp;
    if (opts.length === 1 && value === opts[0].value) {
      return (
        <StaticField label={label} help={help}>
          {opts[0].keywords?.[0] ?? opts[0].label}
        </StaticField>
      );
    }
    const defaultId =
      level === 'intra'
        ? linkPresetId(options.localLinks[0]?.link ?? { name: '' })
        : linkPresetId(options.networks[0].link);
    return (
      <Picker
        label={label}
        value={value}
        options={opts}
        renderValue={(o) => o?.keywords?.[0] ?? o?.label}
        onChange={(v) =>
          dispatch({ type: 'setLink', level, linkId: v === defaultId ? undefined : v })
        }
        help={help}
      />
    );
  };
  return (
    <div className="setup-row" data-testid="setup-row">
      {showPerMachine && (
        <div className="setup-level">
          <Segmented
            label={`${unit} in one machine`}
            value={String(hw.devicesPerNode)}
            segments={countSegments(options.devicesPerNodeCounts, hw.devicesPerNode)}
            onChange={(v) => dispatch({ type: 'setCounts', devicesPerNode: Number(v) })}
            help={helpLines([
              `How many ${unit === 'GPUs' ? 'GPUs' : 'devices'} sit in one machine.`,
              'They talk over the local link, which is faster than the network.',
            ])}
          />
          {hw.devicesPerNode > 1 && local.length > 0 && linkField('intra', local, links.intra)}
        </div>
      )}
      <div className="setup-level">
        <Segmented
          label="Machines"
          value={String(hw.nodes)}
          segments={countSegments(options.nodeCounts, hw.nodes)}
          onChange={(v) => dispatch({ type: 'setCounts', nodes: Number(v) })}
          help={HELP.machines}
          aside={totalText}
          issues={built.issues.filter((i) => i.field === 'devices')}
        />
        {hw.nodes > 1 && (
          <div className="fields two setup-network">
            {linkField('inter', network, links.inter)}
            {cablings.length === 1 || cabling === undefined ? (
              <StaticField label="Cabling" help={HELP.cabling} testId="cabling">
                {cablingName(cabling ?? 'switched', hw.nodes)}
              </StaticField>
            ) : (
              <Picker
                label="Cabling"
                value={cabling}
                options={cablings.map((c) => ({ value: c, label: cablingName(c, hw.nodes) }))}
                onChange={(v) =>
                  dispatch({ type: 'setCabling', cabling: cablings.find((c) => c === v) })
                }
                help={HELP.cabling}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
