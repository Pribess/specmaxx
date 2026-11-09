import type { ReactNode } from 'react';
import { HALO_TCP_URL, linkPresetId, MLX_JACCL_PR_URL } from '../../sim/hw/presets';
import type { LinkSpec } from '../../sim/hw/types';

/** Socket collectives measured between two nodes (NCCL over sockets: GH200 on Slingshot TCP; H100 on AWS ENA). */
const SOCKET_COLLECTIVE_URLS = [
  'https://github.com/eth-cscs/cray-network-stack/issues/6',
  'https://github.com/dstackai/dstack/issues/3069',
];

/** One help line: its text and the sources it links to. */
interface HelpLine {
  text: string;
  sources: string[];
}

/** α in µs as shown in the help: at most one decimal, no trailing zero (10, 45, 1.5). */
function micros(latencySec: number): string {
  return String(Math.round(latencySec * 1e7) / 10);
}

const SOCKET_LINE: HelpLine = {
  text: 'Collective libraries over sockets measure about 50–130 µs for a small all-reduce between two nodes.',
  sources: SOCKET_COLLECTIVE_URLS,
};

/** The lines of each TCP link preset, by `linkPresetId`; α comes from the link itself. */
const LINES: Record<string, (alpha: string) => HelpLine[]> = {
  '10gbe-tcp': (a) => [
    {
      text: `Latency ${a} µs: one-way TCP floor between two machines over a direct cable, half a tuned ping-pong round trip (estimated from 5GbE; about 15 µs through a switch).`,
      sources: [HALO_TCP_URL],
    },
    SOCKET_LINE,
  ],
  '5gbe-tcp': (a) => [
    {
      text: `Latency ${a} µs: one-way TCP floor over a direct cable, half the 20 µs tuned ping-pong round trip; about 15 µs through a switch.`,
      sources: [HALO_TCP_URL],
    },
    SOCKET_LINE,
  ],
  'strix-halo-usb4-tcp-20-gb-s-negotiated': (a) => [
    {
      text: `Latency ${a} µs: one-way TCP floor over USB4, about half the 17–20 µs tuned ping-pong round trip.`,
      sources: [HALO_TCP_URL],
    },
    SOCKET_LINE,
  ],
  'thunderbolt-4-bridge-tcp': (a) => thunderboltLines(a),
  'thunderbolt-5-bridge-tcp': (a) => thunderboltLines(a),
};

function thunderboltLines(a: string): HelpLine[] {
  return [
    {
      text: `Latency ${a} µs: one-way time over the Thunderbolt bridge, estimated from MLX's TCP ring (about 48 µs per step).`,
      sources: [MLX_JACCL_PR_URL],
    },
    {
      text: "MLX's TCP ring measures 290–300 µs for a 4-Mac all-reduce; RDMA over Thunderbolt 5 is 9–12 µs.",
      sources: [MLX_JACCL_PR_URL],
    },
  ];
}

/** The extra line for a dedicated-memory GPU whose inter-node messages are staged through host memory. */
const STAGED_LINE =
  'GPUs with their own memory stage every message through host memory: about 2 µs more per message.';

/** The link preset ids that have their own Network link help (the TCP links). */
export const LINK_HELP_IDS: string[] = Object.keys(LINES);

/**
 * The "?" help of the Network link for `link`: the latency the engine uses (formatted from `link.latencySec`) and what
 * software over the same transport measures, each with its source, plus a host-staging line when `staged`
 * (`needsHostStaging` of the device and topology). Undefined for links without their own help (the caller keeps its
 * generic text).
 */
export function networkLinkHelp(link: LinkSpec, staged: boolean): ReactNode | undefined {
  const lines = LINES[linkPresetId(link)];
  if (lines === undefined) return undefined;
  return (
    <>
      {lines(micros(link.latencySec)).map((l) => (
        <span className="help-line" key={l.text}>
          {l.text}{' '}
          {l.sources.map((url, i) => (
            <span key={url}>
              {i > 0 ? ', ' : ''}
              <a href={url} target="_blank" rel="noreferrer">
                {i === 0 ? 'source' : `source ${i + 1}`}
              </a>
            </span>
          ))}
        </span>
      ))}
      {staged && <span className="help-line">{STAGED_LINE}</span>}
    </>
  );
}
