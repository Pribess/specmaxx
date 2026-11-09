import type { ReactNode } from 'react';
import { linkPresetId } from '../../sim/hw/presets';
import type { LinkSpec } from '../../sim/hw/types';
import { helpLines } from './helpLines';

/** The message delay in µs as shown in the help: at most one decimal, no trailing zero (10, 45, 1.5). */
function micros(latencySec: number): string {
  return String(Math.round(latencySec * 1e7) / 10);
}

/** What each TCP link preset is (first help line), by `linkPresetId`. */
const KINDS: Record<string, string> = {
  '10gbe-tcp': 'Ethernet between machines, talking over TCP.',
  '5gbe-tcp': 'Ethernet between machines, talking over TCP.',
  'strix-halo-usb4-tcp-20-gb-s-negotiated': 'A USB4 cable between machines, talking over TCP.',
  'thunderbolt-4-bridge-tcp': 'A Thunderbolt cable between machines, talking over TCP.',
  'thunderbolt-4-bridge-tcp-20-gb-s-negotiated': 'A Thunderbolt cable that linked at 20 Gb/s instead of 40, talking over TCP.',
  'thunderbolt-5-bridge-tcp': 'A Thunderbolt cable between machines, talking over TCP.',
};

/** The extra line for a dedicated-memory GPU whose inter-node messages are staged through host memory. */
const STAGED_LINE = 'GPUs with their own memory add about 2 µs per message.';

/** The link preset ids that have their own Network link help (the TCP links). */
export const LINK_HELP_IDS: string[] = Object.keys(KINDS);

/**
 * The "?" help of the Network link for `link`: what the link is, and the message delay the engine uses (formatted
 * from `link.latencySec`) with its effect on the estimate, plus a host-staging line when `staged` (`needsHostStaging`
 * of the device and topology). At most 3 lines. Undefined for links without their own help (the caller keeps its
 * generic text).
 */
export function networkLinkHelp(link: LinkSpec, staged: boolean): ReactNode | undefined {
  const kind = KINDS[linkPresetId(link)];
  if (kind === undefined) return undefined;
  return helpLines([
    kind,
    `Each message takes about ${micros(link.latencySec)} µs one way, which slows every sync between machines.`,
    ...(staged ? [STAGED_LINE] : []),
  ]);
}
