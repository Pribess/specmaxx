/**
 * Identity pin for fabric shapes (docs/simulation-model.md §6 "Fabric shapes"). Originally generated before
 * fabric shapes; deliberately regenerated after latency-audit G1–G4/G7/G8 (design §4.3 records deltas).
 * It now pins absent-fabric/switched results under the corrected engine, not the historical pre-audit timings.
 * Regenerated again for the RDMA vs TCP latency fix (one α definition; docs §4 "Links and topologies") and the TCP
 * socket copy (docs §6 "TCP socket copy"). Moved keys, and only these: collectives on `thunderbolt-5-rdma-macos-26-2`
 * (α 50 → 7 µs) and `strix-halo-usb4-tcp-20-gb-s-negotiated` (α 10 → 9 µs), 84 keys each (the collective level has
 * no staging, so no copy); simulate / ui rigs on TB5 RDMA (decode +0.3 % … +568 %, the α) and on TCP links (the copy:
 * TB4 TCP, 5 / 10GbE decode −0.0 … −1.1 %; USB4 TCP α + copy −0.02 … +1.4 %). Every non-TCP, non-TB5-RDMA key and
 * the hierarchical collectives are unchanged bit for bit.
 * Regenerated for the PCIe payload efficiency (every PCIe link × 256/280; links.ts `PCIE_PAYLOAD_EFFICIENCY`, docs §4
 * "PCIe payload efficiency"). Moved keys, and only these: flat collectives on the PCIe presets whose PCIe bandwidth
 * binds (all 84 of each direct-P2P preset; host bounce only where B < hostMem / (2·n): 12 of Gen4 x16 and Gen5 x8, 48
 * of Gen4 x8; Gen5 x16 host bounce is host-memory bound and unchanged); simulate rigs on 4090 / 3090 / B60 PCIe boxes
 * (comm β and PP send × 280/256 = +9.4 %, decode − 0.00 %). 84 keys added for the new
 * `thunderbolt-4-bridge-tcp-20-gb-s-negotiated`. Every other key is unchanged bit for bit.
 * Regenerated for the MoE routing review fixes (docs/review/practicality-review.md F5, F12): only the 12 `simulate`
 * keys of the tiny-hybrid MoE at ep > 1 (batch 4) moved: decode −0.01 … +3.7 % (expected distinct experts instead of
 * min(E_loc, A)), prefill 0 … −4.2 % (EP load imbalance ρ = 1.2). Collectives, UI rigs and every dense key unchanged.
 * Regenerated for the attention softmax (docs §7 "Attention softmax") and the Apple GEMV path (docs §4 "GEMV
 * path"). Moved keys, 44, and only these: `hgx-b200-1` / `hgx-b200-2` dense8b prefill (softmax binds on B200 at
 * hd 128: tok/s − 0.6 … − 1.3 %, with the exposed-comm and best-algorithm split that follows from the longer compute);
 * `m3-ultra-tb5-4` tiny-hybrid decode (GEMV path: + 0.8 … + 3.4 %; exposedCommSec in the last ulp); `synth-cluster`
 * (H100 devices) tiny-hybrid prefill exposedCommSec only (the tiny model's small heads let the softmax bind, − 0.4 …
 * − 3.0 % exposed comm).
 * Decode outside Apple, and every collective key, are unchanged bit for bit.
 * Numbers are compared with exact equality (Object.is; JSON round-trips doubles exactly).
 *
 * Three levels:
 * - collectives: every link preset × every collective kind × n ∈ {2, 3, 4, 8} × S ∈ {1e3, 1e6, 1e8} B on a flat path,
 *   plus hierarchical paths;
 * - simulate: every topology preset of the time, built by explicit builder calls (not the preset constants, so a
 *   preset refactor does not change what is pinned), with REF_DENSE_8B at tp = all, pp = all and one mixed split, and
 *   a MoE with ep > 1 on 4-node clusters (the all-to-all);
 * - ui: the multi-device setups of the UI as `HardwareChoice` literals → `resolveHardware` → `simulate`.
 *
 * `uiDeliberate` holds "before" values of UI defaults that fabric shapes change ON PURPOSE (3- and 4-machine
 * Thunderbolt 5 / USB4 setups were implicitly switched, which no switch exists for). They are not asserted; the test
 * prints their deltas so a submission can quote them.
 *
 * Regenerate (only on purpose; an edit of the fixture is a review red flag):
 *   UPDATE_FABRIC_IDENTITY=1 bun run test src/sim/engine/fabric-identity.test.ts
 */
import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_IB, SYNTH_NVLINK } from '../__fixtures__/synthetic';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import { REF_DENSE_8B, TINY_HYBRID } from '../models/reference';
import type { DeviceSpec, LinkSpec, TopologySpec } from '../hw/types';
import { INSTINCT_MI300X, RADEON_AI_PRO_R9700, RYZEN_AI_MAX_PLUS_395 } from '../hw/presets/amd';
import { M3_ULTRA_80C, M4_MAX_40C, M5_MAX_40C, M5_ULTRA_80C } from '../hw/presets/apple';
import { ARC_PRO_B60 } from '../hw/presets/intel';
import { RTX_3090, RTX_4090, RTX_5090 } from '../hw/presets/nvidia';
import { B200, B300, DGX_SPARK, H100_SXM } from '../hw/presets/nvidia-datacenter';
import {
  ETHERNET_10G_TCP,
  ETHERNET_5G_TCP,
  IB_NDR_400,
  LINK_PRESETS,
  NVLINK_4,
  PCIE_4_X16,
  PCIE_5_X16,
  PCIE_5_X8_DIRECT,
  RTX_3090_NVLINK,
  SPARK_CONNECTX_7,
  TB4_TCP,
  TB5_RDMA,
  USB4_HALO_RDMA,
  USB4_HALO_TCP,
  XGMI_MI300X_8_GPU,
} from '../hw/presets/links';
import {
  hgxCluster,
  linkPresetId,
  nodeTopology,
  pcieBox,
  socCluster,
} from '../hw/presets/topologies';
import type { HardwareChoice } from '../../ui/state/appState';
import { resolveHardware } from '../../ui/state/hardware';
import {
  collectiveTime,
  type AllReduceAlgorithm,
  type CollectiveKind,
  type CollectivePath,
} from './collectives';
import { simulate } from './simulate';
import type { PhaseResult, SimResult } from './types';

/** A pinned number set: field → value (numbers exact; strings for algorithm names and errors). */
type Pin = Record<string, number | string>;

/** The fixture file's shape. */
interface Fixture {
  /** `linkPresetId`s of the links the collective level covers (renaming one breaks old URLs too). */
  linkIds: string[];
  collectives: Record<string, Pin>;
  simulate: Record<string, Pin>;
  ui: Record<string, Pin>;
  uiDeliberate: Record<string, Pin>;
}

const FIXTURE_URL = new URL('./__fixtures__/fabric-identity.json', import.meta.url);
const RAW = import.meta.glob<string>('./__fixtures__/fabric-identity.json', {
  query: '?raw',
  import: 'default',
  eager: true,
});
const UPDATE =
  (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env
    .UPDATE_FABRIC_IDENTITY === '1';

// ---------------------------------------------------------------------------------------------------------------
// Collective level

const KINDS: { kind: CollectiveKind; algorithm: AllReduceAlgorithm; label: string }[] = [
  { kind: 'all_reduce', algorithm: 'best', label: 'all_reduce:best' },
  { kind: 'all_reduce', algorithm: 'ring', label: 'all_reduce:ring' },
  { kind: 'all_reduce', algorithm: 'tree', label: 'all_reduce:tree' },
  { kind: 'all_gather', algorithm: 'best', label: 'all_gather' },
  { kind: 'reduce_scatter', algorithm: 'best', label: 'reduce_scatter' },
  { kind: 'all_to_all', algorithm: 'best', label: 'all_to_all' },
  { kind: 'send_recv', algorithm: 'best', label: 'send_recv' },
];
const GROUP_SIZES = [2, 3, 4, 8];
const MESSAGE_BYTES = [1e3, 1e6, 1e8];

/** The four numbers of one collective. */
function collectivePin(
  kind: CollectiveKind,
  s: number,
  n: number,
  path: CollectivePath,
  a: AllReduceAlgorithm,
): Pin {
  const t = collectiveTime(kind, s, n, path, a);
  return {
    latencySec: t.latencySec,
    bandwidthSec: t.bandwidthSec,
    totalSec: t.totalSec,
    algorithm: t.algorithm,
  };
}

/** Flat paths over `links`, plus hierarchical paths (send_recv is flat only). */
function collectivePins(links: LinkSpec[]): Record<string, Pin> {
  const out: Record<string, Pin> = {};
  for (const link of links) {
    const path: CollectivePath = { kind: 'flat', level: 'inter', link };
    for (const { kind, algorithm, label } of KINDS) {
      for (const n of GROUP_SIZES) {
        for (const s of MESSAGE_BYTES)
          out[`flat|${linkPresetId(link)}|${label}|n${n}|S${s}`] = collectivePin(
            kind,
            s,
            n,
            path,
            algorithm,
          );
      }
    }
  }
  const hier: [string, LinkSpec, LinkSpec, number[], number[]][] = [
    ['synth', SYNTH_NVLINK, SYNTH_IB, [SYNTH_CLUSTER.devicesPerNode], [SYNTH_CLUSTER.nodes]],
    ['nvlink4-ndr', NVLINK_4, IB_NDR_400, [2, 8], [2, 4]],
    ['pcie5-hostbounce-10gbe', PCIE_5_X16, ETHERNET_10G_TCP, [2, 4], [2, 4]],
  ];
  for (const [name, intra, inter, n1s, n2s] of hier) {
    for (const n1 of n1s) {
      for (const n2 of n2s) {
        const path: CollectivePath = {
          kind: 'hierarchical',
          devicesPerNode: n1,
          nodes: n2,
          intra,
          inter,
        };
        for (const { kind, algorithm, label } of KINDS) {
          if (kind === 'send_recv') continue;
          for (const s of MESSAGE_BYTES)
            out[`hier|${name}|${label}|${n1}x${n2}|S${s}`] = collectivePin(
              kind,
              s,
              n1 * n2,
              path,
              algorithm,
            );
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Simulate level

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

/** One pinned rig: a device on a topology built by a builder call. */
interface Rig {
  name: string;
  device: DeviceSpec;
  topology: TopologySpec;
}

/**
 * Every entry of `TOPOLOGY_PRESETS` before fabric shapes, as builder calls, plus 8-node SoC clusters (N = 8). The
 * named constants with hand-written ids (RTX 3090 NVLink pair, MI300X) are rebuilt with `nodeTopology`; ids and names
 * do not enter any result.
 */
function rigs(): Rig[] {
  const out: Rig[] = [
    { name: 'rtx-5090-pcie-2', device: RTX_5090, topology: pcieBox(RTX_5090, 2) },
    { name: 'rtx-5090-pcie-4', device: RTX_5090, topology: pcieBox(RTX_5090, 4) },
    { name: 'rtx-4090-pcie-2', device: RTX_4090, topology: pcieBox(RTX_4090, 2, PCIE_4_X16) },
    { name: 'rtx-4090-pcie-4', device: RTX_4090, topology: pcieBox(RTX_4090, 4, PCIE_4_X16) },
    { name: 'rtx-3090-pcie-2', device: RTX_3090, topology: pcieBox(RTX_3090, 2, PCIE_4_X16) },
    { name: 'rtx-3090-pcie-4', device: RTX_3090, topology: pcieBox(RTX_3090, 4, PCIE_4_X16) },
    {
      name: 'rtx-3090-nvlink-2',
      device: RTX_3090,
      topology: nodeTopology({
        devicesPerNode: 2,
        nodes: 1,
        intraNode: RTX_3090_NVLINK,
        device: RTX_3090,
      }),
    },
    {
      name: 'arc-pro-b60-pcie-2',
      device: ARC_PRO_B60,
      topology: pcieBox(ARC_PRO_B60, 2, PCIE_5_X8_DIRECT),
    },
    {
      name: 'arc-pro-b60-pcie-4',
      device: ARC_PRO_B60,
      topology: pcieBox(ARC_PRO_B60, 4, PCIE_5_X8_DIRECT),
    },
    {
      name: 'r9700-pcie-4',
      device: RADEON_AI_PRO_R9700,
      topology: pcieBox(RADEON_AI_PRO_R9700, 4, PCIE_5_X16),
    },
    {
      name: 'mi300x-xgmi-8',
      device: INSTINCT_MI300X,
      topology: nodeTopology({
        devicesPerNode: 8,
        nodes: 1,
        intraNode: XGMI_MI300X_8_GPU,
        device: INSTINCT_MI300X,
      }),
    },
  ];
  const soc: [string, DeviceSpec, LinkSpec, number[]][] = [
    ['dgx-spark-cx7', DGX_SPARK, SPARK_CONNECTX_7, [2, 4, 8]],
    ['m3-ultra-tb5', M3_ULTRA_80C, TB5_RDMA, [2, 3, 4, 8]],
    ['m4-max-tb5', M4_MAX_40C, TB5_RDMA, [2, 3, 4]],
    ['m5-max-tb5', M5_MAX_40C, TB5_RDMA, [2, 3, 4]],
    ['m5-ultra-tb5', M5_ULTRA_80C, TB5_RDMA, [2, 3, 4]],
    ['m3-ultra-tb4-tcp', M3_ULTRA_80C, TB4_TCP, [2, 3, 4]],
    ['strix-halo-usb4-tcp', RYZEN_AI_MAX_PLUS_395, USB4_HALO_TCP, [2, 3, 4]],
    ['strix-halo-usb4-rdma', RYZEN_AI_MAX_PLUS_395, USB4_HALO_RDMA, [2, 3, 4]],
    ['strix-halo-5gbe', RYZEN_AI_MAX_PLUS_395, ETHERNET_5G_TCP, [2, 3, 4]],
    ['strix-halo-10gbe', RYZEN_AI_MAX_PLUS_395, ETHERNET_10G_TCP, [2, 3, 4]],
  ];
  for (const [name, device, link, counts] of soc) {
    for (const n of counts)
      out.push({ name: `${name}-${n}`, device, topology: socCluster(device, n, link) });
  }
  const hgx: ['h100' | 'b200' | 'b300', DeviceSpec][] = [
    ['h100', H100_SXM],
    ['b200', B200],
    ['b300', B300],
  ];
  for (const [gen, device] of hgx) {
    for (const nodes of [1, 2])
      out.push({ name: `hgx-${gen}-${nodes}`, device, topology: hgxCluster(gen, nodes) });
  }
  return out;
}

/** Σ over stages of a numeric field of a phase. Unit: as the field (seconds). */
function stageSum(p: PhaseResult, f: (s: PhaseResult['stages'][number]) => number): number {
  return p.stages.reduce((sum, s) => sum + f(s), 0);
}

/** The pinned numbers of one phase, prefixed with its name. */
function phasePin(p: PhaseResult): Pin {
  const pre = p.phase;
  return {
    [`${pre}.tokensPerSec`]: p.tokensPerSec,
    [`${pre}.latencySec`]: p.latencySec,
    [`${pre}.cycleSec`]: p.cycleSec,
    [`${pre}.commLatencySec`]: stageSum(p, (s) => s.commLatencySec),
    [`${pre}.commBandwidthSec`]: stageSum(p, (s) => s.commBandwidthSec),
    [`${pre}.exposedCommSec`]: stageSum(p, (s) => s.exposedCommSec),
    [`${pre}.sendSec`]: stageSum(p, (s) => s.sendToNext?.sec ?? 0),
  };
}

/** Runs one simulation; an error message is pinned too (it must stay an error, with the same text). */
function simPin(run: () => SimResult): Pin {
  try {
    const r = run();
    return { ...phasePin(r.prefill), ...phasePin(r.decode) };
  } catch (e) {
    return { error: (e as Error).message };
  }
}

/** Parallel splits of `devices`: tp = all, pp = all, and tp = devices per node (or 2) × pp = the rest. */
function splits(devices: number, devicesPerNode: number): [string, ParallelConfig][] {
  const out: [string, ParallelConfig][] = [
    [`tp${devices}`, { tp: devices, pp: 1, dp: 1, ep: 1 }],
    [`pp${devices}`, { tp: 1, pp: devices, dp: 1, ep: 1 }],
  ];
  const tp = devicesPerNode > 1 ? devicesPerNode : 2;
  const pp = devices / tp;
  if (Number.isInteger(pp) && pp >= 2) out.push([`tp${tp}pp${pp}`, { tp, pp, dp: 1, ep: 1 }]);
  return out;
}

function simulatePins(): Record<string, Pin> {
  const out: Record<string, Pin> = {};
  const run = (model: ModelSpec, rig: Rig, parallel: ParallelConfig, batch: number) => () =>
    simulate({
      model,
      quant: BF16,
      parallel,
      workload: { batch, promptLen: 4096, decodeContextLen: 8192 },
      hardware: { device: rig.device, topology: rig.topology },
    });
  for (const rig of rigs()) {
    const devices = rig.topology.devicesPerNode * rig.topology.nodes;
    for (const [split, par] of splits(devices, rig.topology.devicesPerNode)) {
      for (const batch of [1, 8])
        out[`${rig.name}|dense8b|${split}|b${batch}`] = simPin(run(REF_DENSE_8B, rig, par, batch));
    }
  }
  // MoE with ep > 1 (mode B: the EP group spans replicas), so the all-to-all runs on the inter-node link.
  const moeRigs: Rig[] = [
    {
      name: 'dgx-spark-cx7-4',
      device: DGX_SPARK,
      topology: socCluster(DGX_SPARK, 4, SPARK_CONNECTX_7),
    },
    {
      name: 'm3-ultra-tb5-4',
      device: M3_ULTRA_80C,
      topology: socCluster(M3_ULTRA_80C, 4, TB5_RDMA),
    },
    {
      name: 'strix-halo-usb4-rdma-4',
      device: RYZEN_AI_MAX_PLUS_395,
      topology: socCluster(RYZEN_AI_MAX_PLUS_395, 4, USB4_HALO_RDMA),
    },
    { name: 'synth-cluster', device: H100_SXM, topology: SYNTH_CLUSTER },
  ];
  const moeSplits: [string, ParallelConfig, number][] = [
    ['tp1dp4ep4', { tp: 1, pp: 1, dp: 4, ep: 4 }, 4],
    ['tp2dp2ep4', { tp: 2, pp: 1, dp: 2, ep: 4 }, 4],
    ['tp1pp2dp2ep2', { tp: 1, pp: 2, dp: 2, ep: 2 }, 4],
  ];
  for (const rig of moeRigs) {
    for (const [split, par, batch] of moeSplits)
      out[`${rig.name}|tiny-hybrid|${split}|b${batch}`] = simPin(run(TINY_HYBRID, rig, par, batch));
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// UI level

/** A stock `HardwareChoice` (no unlocks, no edits). */
function choice(
  c: Pick<
    HardwareChoice,
    'deviceId' | 'capacityIndex' | 'devicesPerNode' | 'nodes' | 'intraLinkId' | 'interLinkId'
  >,
): HardwareChoice {
  return { unlocks: {}, deviceEdits: {}, intraEdits: {}, interEdits: {}, ...c };
}

/** The UI's multi-device setups before fabric shapes (`systemChoice` of each `SYSTEM_PRESETS` entry), by setup id. */
const UI_SETUPS: Record<string, HardwareChoice> = {
  'rtx-5090-x2-pcie5': choice({
    deviceId: 'rtx-5090',
    capacityIndex: 0,
    devicesPerNode: 2,
    nodes: 1,
    intraLinkId: 'pcie-5-0-x16-host-bounce',
  }),
  'rtx-5090-x4-pcie5': choice({
    deviceId: 'rtx-5090',
    capacityIndex: 0,
    devicesPerNode: 4,
    nodes: 1,
    intraLinkId: 'pcie-5-0-x16-host-bounce',
  }),
  'rtx-3090-x2-nvlink': choice({
    deviceId: 'rtx-3090',
    capacityIndex: 0,
    devicesPerNode: 2,
    nodes: 1,
    intraLinkId: 'rtx-3090-nvlink-bridge',
  }),
  'dgx-spark-x2': choice({
    deviceId: 'dgx-spark',
    capacityIndex: 0,
    devicesPerNode: 1,
    nodes: 2,
    interLinkId: 'dgx-spark-connectx-7-200gbe-roce',
  }),
  'dgx-spark-x4': choice({
    deviceId: 'dgx-spark',
    capacityIndex: 0,
    devicesPerNode: 1,
    nodes: 4,
    interLinkId: 'dgx-spark-connectx-7-200gbe-roce',
  }),
  'm5-ultra-80c-256gb-x2-tb5': choice({
    deviceId: 'apple-m5-ultra-80c',
    capacityIndex: 1,
    devicesPerNode: 1,
    nodes: 2,
    interLinkId: 'thunderbolt-5-rdma-macos-26-2',
  }),
  'strix-halo-128gb-x2-usb4': choice({
    deviceId: 'ryzen-ai-max-plus-395',
    capacityIndex: 0,
    devicesPerNode: 1,
    nodes: 2,
    interLinkId: 'strix-halo-usb4-native-rdma-20-gb-s-negotiated',
  }),
  'r9700-x4-pcie5': choice({
    deviceId: 'radeon-ai-pro-r9700',
    capacityIndex: 0,
    devicesPerNode: 4,
    nodes: 1,
    intraLinkId: 'pcie-5-0-x16-host-bounce',
  }),
};

/**
 * UI defaults (no link id: the device's default link) that fabric shapes change ON PURPOSE: 3 and 4 machines on
 * Thunderbolt 5 / USB4 were implicitly switched before; no such switch exists, so the default cabling becomes a mesh
 * or a ring. Pinned as "before" values only.
 */
const UI_DELIBERATE: Record<string, HardwareChoice> = {
  'm3-ultra-80c-x3-default': choice({
    deviceId: 'apple-m3-ultra-80c',
    capacityIndex: 0,
    devicesPerNode: 1,
    nodes: 3,
  }),
  'm3-ultra-80c-x4-default': choice({
    deviceId: 'apple-m3-ultra-80c',
    capacityIndex: 0,
    devicesPerNode: 1,
    nodes: 4,
  }),
  'strix-halo-x3-default': choice({
    deviceId: 'ryzen-ai-max-plus-395',
    capacityIndex: 0,
    devicesPerNode: 1,
    nodes: 3,
  }),
  'strix-halo-x4-default': choice({
    deviceId: 'ryzen-ai-max-plus-395',
    capacityIndex: 0,
    devicesPerNode: 1,
    nodes: 4,
  }),
};

/** tp = all and pp = all (the UI fixes dp = ep = 1) for each choice. */
function uiPins(choices: Record<string, HardwareChoice>): Record<string, Pin> {
  const out: Record<string, Pin> = {};
  for (const [id, hw] of Object.entries(choices)) {
    const devices = hw.devicesPerNode * hw.nodes;
    for (const [split, parallel] of splits(devices, hw.devicesPerNode).slice(0, 2)) {
      out[`${id}|${split}`] = simPin(() =>
        simulate({
          model: REF_DENSE_8B,
          quant: BF16,
          parallel,
          workload: { batch: 1, promptLen: 4096, decodeContextLen: 8192 },
          hardware: resolveHardware(hw).stack,
        }),
      );
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------

/** Every (key, field) whose value differs (exact `Object.is`), or is missing on either side. */
function mismatches(expected: Record<string, Pin>, actual: Record<string, Pin>): string[] {
  const out: string[] = [];
  for (const key of new Set([...Object.keys(expected), ...Object.keys(actual)])) {
    const e = expected[key];
    const a = actual[key];
    if (e === undefined || a === undefined) {
      out.push(`${key}: ${e === undefined ? 'not in the fixture' : 'no longer computed'}`);
      continue;
    }
    for (const field of new Set([...Object.keys(e), ...Object.keys(a)])) {
      if (!Object.is(e[field], a[field]))
        out.push(`${key} ${field}: expected ${String(e[field])}, got ${String(a[field])}`);
    }
  }
  return out;
}

/** The fixture as generated now. */
function generate(linkIds: string[]): Fixture {
  const links = linkIds.map((id) => {
    const link = LINK_PRESETS.find((l) => linkPresetId(l) === id);
    if (link === undefined)
      throw new Error(`link preset '${id}' of the identity fixture no longer exists`);
    return link;
  });
  return {
    linkIds,
    collectives: collectivePins(links),
    simulate: simulatePins(),
    ui: uiPins(UI_SETUPS),
    uiDeliberate: uiPins(UI_DELIBERATE),
  };
}

describe('fabric identity pin: absent fabric = the pre-fabric results, bit for bit', () => {
  if (UPDATE) {
    it('regenerates the fixture (UPDATE_FABRIC_IDENTITY=1)', async () => {
      const fs = (await import(/* @vite-ignore */ 'node:fs' as string)) as {
        mkdirSync(p: URL, o: { recursive: boolean }): void;
        writeFileSync(p: URL, d: string): void;
      };
      fs.mkdirSync(new URL('./', FIXTURE_URL), { recursive: true });
      fs.writeFileSync(
        FIXTURE_URL,
        `${JSON.stringify(generate(LINK_PRESETS.map(linkPresetId)), null, 1)}\n`,
      );
    });
    return;
  }
  const raw = Object.values(RAW)[0];
  if (raw === undefined)
    throw new Error(
      'missing __fixtures__/fabric-identity.json; generate it with UPDATE_FABRIC_IDENTITY=1',
    );
  const fixture = JSON.parse(raw) as Fixture;
  const now = generate(fixture.linkIds);

  it('pins enough cases to mean something', () => {
    expect(Object.keys(fixture.collectives).length).toBeGreaterThan(2000);
    expect(Object.keys(fixture.simulate).length).toBeGreaterThan(200);
    expect(Object.keys(fixture.ui).length).toBe(16);
  });
  it('collectives on every link preset (flat) and hierarchical paths', () => {
    expect(mismatches(fixture.collectives, now.collectives)).toEqual([]);
  });
  it('simulate on every pre-fabric topology preset, dense and MoE (ep > 1)', () => {
    expect(mismatches(fixture.simulate, now.simulate)).toEqual([]);
  });
  it('the UI setups through resolveHardware', () => {
    expect(mismatches(fixture.ui, now.ui)).toEqual([]);
  });
  it('reports the deliberate UI changes (3- and 4-machine switchless defaults; not asserted)', () => {
    const deltas = Object.entries(fixture.uiDeliberate).flatMap(([key, before]) =>
      ['prefill.tokensPerSec', 'decode.tokensPerSec'].map((f) => {
        const b = before[f];
        const a = now.uiDeliberate[key]?.[f];
        return typeof b === 'number' && typeof a === 'number'
          ? `${key} ${f}: ${b} → ${a} (${((a / b - 1) * 100).toFixed(2)}%)`
          : `${key} ${f}: ${String(b)} → ${String(a)}`;
      }),
    );
    console.log(`fabric identity, deliberate UI deltas:\n${deltas.join('\n')}`);
    expect(Object.keys(fixture.uiDeliberate).length).toBeGreaterThan(0);
  });
});
