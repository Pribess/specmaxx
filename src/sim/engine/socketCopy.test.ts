/**
 * The TCP socket copy (docs/simulation-model.md §6 "TCP socket copy"): on a `transport: 'tcp'` link each end copies
 * the message between the user buffer and kernel socket buffers (a memcpy, 2 DRAM accesses per byte), as extra
 * stages of the host-staging pipeline (`stagedLink`). RDMA, single-node and intra-node paths never pay it, and the
 * PCIe-only staging path computes exactly what it did before the copy was modelled.
 */

import { describe, expect, it } from 'vitest';
import {
  DGX_SPARK,
  ETHERNET_10G_TCP,
  H100_SXM,
  HGX_H100_16_NDR,
  hostPcieLink,
  HOST_STAGING_STEP_BYTES,
  IB_NDR_400,
  M3_ULTRA_80C,
  RTX_5090,
  RTX_5090_PCIE_2,
  RYZEN_AI_MAX_PLUS_395,
  SOCKET_COPY_STEP_BYTES,
  SPARK_CONNECTX_7,
  TB5_RDMA,
  TB5_TCP,
  USB4_HALO_RDMA,
  USB4_HALO_TCP,
} from '../hw/presets';
import { DDR5_6000_2CH } from '../hw/presets/storage';
import type { DeviceSpec, LinkSpec, TopologySpec } from '../hw/types';
import { REF_DENSE_8B } from '../models/reference';
import {
  collectiveTime,
  stagedLink,
  stagedStageRates,
  stagingBandwidthCapBytesPerSec,
  type CollectivePath,
  type EffectiveLink,
} from './collectives';
import { fabricLinkParams } from './fabricCollectives';
import { collectivePath, groupSpan } from './placement';
import { simulate } from './simulate';
import { resolveHostStaging, type HostStaging } from './staging';
import type { SimInput } from './types';

function expectClose(actual: number, expected: number, rel = 1e-12): void {
  expect(Math.abs(actual - expected), `${actual} vs ${expected}`).toBeLessThanOrEqual(
    rel * Math.abs(expected),
  );
}

/** `nodes` machines of `devicesPerNode` devices, `link` between them (PCIe 5.0 x16 host bounce inside a node). */
function cluster(link: LinkSpec, nodes = 2, devicesPerNode = 1): TopologySpec {
  return {
    id: `test-${nodes}x${devicesPerNode}`,
    name: 'test',
    devicesPerNode,
    nodes,
    ...(devicesPerNode > 1 ? { intraNode: RTX_5090_PCIE_2.intraNode } : {}),
    interNode: link,
    sources: [],
  };
}

const PAR = { tp: 2, pp: 1, dp: 1, ep: 1 };
const GEN5_X16 = hostPcieLink(5, 16);
const B_PCIE = GEN5_X16.bandwidthPerDirectionBytesPerSec;

/** Socket copy alone (unified memory over TCP), synthetic B_mem = 800 GB/s, one device per node. */
const SOCK_800: HostStaging = {
  socketCopy: true,
  hostMemBandwidthBytesPerSec: 800e9,
  stepBytes: SOCKET_COPY_STEP_BYTES,
  activeDevicesPerNode: 1,
};
/** A 10 GB/s TCP link, α 45 µs, before staging. */
const WIRE_10G: EffectiveLink = { bandwidthBytesPerSec: 10e9, latencySec: 45e-6 };

/** PCIe staging of an RTX 5090 (PCIe 5.0 x16, 96 GB/s host): the staging of before, no socket copy. */
const PCIE_ONLY: HostStaging = {
  pcie: { bandwidthBytesPerSec: B_PCIE, latencySec: 1e-6 },
  socketCopy: false,
  hostMemBandwidthBytesPerSec: 96e9,
  stepBytes: HOST_STAGING_STEP_BYTES,
  activeDevicesPerNode: 1,
};

describe('resolveHostStaging: when the socket copy applies', () => {
  const resolve = (device: DeviceSpec, topology: TopologySpec) =>
    resolveHostStaging(device, topology, { device, topology }, PAR);

  it('unified memory over TCP: the socket copy alone, B_host = the unified pool peak, c = 64 KiB, no warning', () => {
    for (const [soc, link] of [
      [M3_ULTRA_80C, TB5_TCP],
      [RYZEN_AI_MAX_PLUS_395, USB4_HALO_TCP],
      [DGX_SPARK, ETHERNET_10G_TCP],
    ] as const) {
      expect(resolve(soc, cluster(link)), `${soc.id} × ${link.name}`).toEqual({
        staging: {
          socketCopy: true,
          hostMemBandwidthBytesPerSec: soc.memory.bandwidthBytesPerSec,
          stepBytes: SOCKET_COPY_STEP_BYTES,
          activeDevicesPerNode: 1,
        },
        warnings: [],
      });
    }
    expect(SOCKET_COPY_STEP_BYTES).toBe(65_536);
  });

  it('RDMA, a single node, and a GPUDirect pair stage nothing', () => {
    for (const [soc, link] of [
      [M3_ULTRA_80C, TB5_RDMA],
      [RYZEN_AI_MAX_PLUS_395, USB4_HALO_RDMA],
      [DGX_SPARK, SPARK_CONNECTX_7],
    ] as const) {
      expect(resolve(soc, cluster(link)), `${soc.id} × ${link.name}`).toEqual({
        staging: undefined,
        warnings: [],
      });
    }
    expect(resolve(M3_ULTRA_80C, cluster(TB5_TCP, 1)).staging).toBeUndefined();
    expect(resolve(H100_SXM, HGX_H100_16_NDR).staging).toBeUndefined();
    expect(resolve(H100_SXM, cluster(IB_NDR_400)).staging).toBeUndefined();
  });

  it('a dedicated GPU: TCP adds the socket copy to the PCIe staging; RDMA without GPUDirect keeps PCIe only', () => {
    const tcp = resolve(RTX_5090, cluster(ETHERNET_10G_TCP)).staging;
    expect(tcp).toEqual({
      ...PCIE_ONLY,
      socketCopy: true,
      hostMemBandwidthBytesPerSec: DDR5_6000_2CH.bandwidthBytesPerSec,
    });
    const ib = resolve(RTX_5090, cluster(IB_NDR_400)).staging;
    expect(ib).toEqual({
      ...PCIE_ONLY,
      hostMemBandwidthBytesPerSec: DDR5_6000_2CH.bandwidthBytesPerSec,
    });
  });
});

describe('stagedLink with the socket copy (docs §6 "TCP socket copy")', () => {
  it('unified TCP, small message (m = 4 KiB): α gains 2·m/r_sock, β unchanged', () => {
    // r_sock = 800/2 = 400 GB/s; r_net = min(10, 400) = 10 GB/s; B_hj = 800/6 = 133.3 GB/s; B_eff = 10 GB/s
    const l = stagedLink(WIRE_10G, SOCK_800, 4096);
    expect(l.bandwidthBytesPerSec).toBe(10e9);
    expectClose(l.latencySec - 45e-6, 8192 / 400e9, 1e-9);
    expectClose(l.latencySec - 45e-6, 0.02048e-6, 1e-9);
  });

  it('unified TCP, large message (8 MiB): β = m/B, the fill is one 64 KiB step through both copies', () => {
    const m = 8 * 1024 * 1024;
    const l = stagedLink(WIRE_10G, SOCK_800, m);
    expectClose(m / l.bandwidthBytesPerSec, 838.8608e-6, 1e-12);
    expectClose(l.latencySec - 45e-6, (2 * 65_536) / 400e9, 1e-9);
    expectClose(l.latencySec - 45e-6, 0.32768e-6, 1e-9);
  });

  it('host memory binds (B_mem 30 GB/s, B 10 GB/s): B_eff = B_mem/6 = 5 GB/s, fill ≥ 0', () => {
    const slow: HostStaging = { ...SOCK_800, hostMemBandwidthBytesPerSec: 30e9 };
    const rates = stagedStageRates(10e9, slow);
    expect(rates).toEqual({
      pcieBytesPerSec: Infinity,
      sockBytesPerSec: 15e9,
      netBytesPerSec: 10e9,
      hostJointBytesPerSec: 5e9,
    });
    const m = 8 * 1024 * 1024;
    const l = stagedLink(WIRE_10G, slow, m);
    expect(l.bandwidthBytesPerSec).toBe(5e9);
    expectClose(m / l.bandwidthBytesPerSec, 1677.7216e-6, 1e-12);
    // fill = c·(2/r_sock + 1/r_net − 1/B_eff) = 65536·(2/15 + 1/10 − 1/5) / 1e9 s = 2.18453 µs
    expectClose(l.latencySec - 45e-6, 65_536 * (2 / 15e9 + 1 / 10e9 - 1 / 5e9), 1e-9);
    expectClose(l.latencySec - 45e-6, 2.184533e-6, 1e-6);
  });

  it('dedicated TCP (10GbE, PCIe 5.0 x16, 96 GB/s): the PCIe staging plus 2·m/r_sock, β unchanged', () => {
    const tcp: EffectiveLink = { bandwidthBytesPerSec: 1.25e9, latencySec: 10e-6 };
    const both: HostStaging = { ...PCIE_ONLY, socketCopy: true };
    // r_sock = 96/2 = 48 GB/s; B_hj = 96/8 = 12 GB/s > B = 1.25 GB/s
    const withCopy = stagedLink(tcp, both, 4096);
    const pcieOnly = stagedLink(tcp, PCIE_ONLY, 4096);
    expect(withCopy.bandwidthBytesPerSec).toBe(1.25e9);
    expect(pcieOnly.bandwidthBytesPerSec).toBe(1.25e9);
    expectClose(withCopy.latencySec - pcieOnly.latencySec, 8192 / 48e9, 1e-9);
    expectClose(withCopy.latencySec - pcieOnly.latencySec, 0.170667e-6, 1e-5);
    expect(stagingBandwidthCapBytesPerSec(both)).toBe(12e9);
  });

  it('PCIe-only staging is bit-identical to the formula before the socket copy', () => {
    /** The pre-change `stagedLink`, copied verbatim (flat HostStaging fields). */
    function oldStagedLink(unstaged: EffectiveLink, s: HostStaging, m: number): EffectiveLink {
      const bPcie = s.pcie?.bandwidthBytesPerSec ?? NaN;
      const n = s.activeDevicesPerNode;
      const host = s.hostMemBandwidthBytesPerSec;
      const copy = Math.min(bPcie, host / n);
      const net = Math.min(unstaged.bandwidthBytesPerSec, host / (2 * n));
      const joint = host / (4 * n);
      const bandwidthBytesPerSec = Math.min(copy, net, joint);
      const stage = 2 * (s.pcie?.latencySec ?? NaN);
      const step = Math.min(m, s.stepBytes);
      const all = (2 * step) / copy + step / net;
      const fill = all - step / bandwidthBytesPerSec;
      return { bandwidthBytesPerSec, latencySec: unstaged.latencySec + stage + fill };
    }
    const nic: EffectiveLink = {
      bandwidthBytesPerSec: IB_NDR_400.bandwidthPerDirectionBytesPerSec,
      latencySec: IB_NDR_400.latencySec,
    };
    const c = HOST_STAGING_STEP_BYTES;
    for (const staging of [PCIE_ONLY, { ...PCIE_ONLY, activeDevicesPerNode: 8 }]) {
      for (const m of [0, 1, 1000, 4096, c - 1, c, c + 1, 3 * c, 64 * 1024 * 1024]) {
        const now = stagedLink(nic, staging, m);
        const before = oldStagedLink(nic, staging, m);
        expect(Object.is(now.latencySec, before.latencySec), `α at m = ${m}`).toBe(true);
        expect(
          Object.is(now.bandwidthBytesPerSec, before.bandwidthBytesPerSec),
          `β at m = ${m}`,
        ).toBe(true);
      }
      const n = staging.activeDevicesPerNode;
      expect(
        Object.is(stagingBandwidthCapBytesPerSec(staging), Math.min(B_PCIE, 96e9 / (4 * n))),
      ).toBe(true);
    }
  });
});

describe('collectives on a unified TCP path: the copy rides every algorithm step', () => {
  const path = (staging?: HostStaging): CollectivePath => ({
    kind: 'flat',
    level: 'inter',
    link: TB5_TCP,
    ...(staging === undefined ? {} : { interStaging: staging }),
  });
  // r_sock = 400 GB/s ≫ 10 GB/s: the wire sets B_eff, so each step's fill is exactly 2·m/r_sock.
  const rSock = 400e9;

  it('ring all-reduce, n = 4: 2(n−1) steps of 2·(S/4)/r_sock; β unchanged', () => {
    const S = 16 * 1024;
    const t = collectiveTime('all_reduce', S, 4, path(SOCK_800), 'ring');
    const u = collectiveTime('all_reduce', S, 4, path(), 'ring');
    expectClose(t.latencySec - u.latencySec, 6 * ((2 * (S / 4)) / rSock), 1e-6);
    expect(t.bandwidthSec).toBe(u.bandwidthSec);
  });

  it('tree all-reduce, n = 4: 2⌈log2 n⌉ = 4 steps of 2·S/r_sock (S ≤ c)', () => {
    const S = 16 * 1024;
    const t = collectiveTime('all_reduce', S, 4, path(SOCK_800), 'tree');
    const u = collectiveTime('all_reduce', S, 4, path(), 'tree');
    expectClose(t.latencySec - u.latencySec, 4 * ((2 * S) / rSock), 1e-6);
  });

  it('send/recv copies the whole message (m = S)', () => {
    const t = collectiveTime('send_recv', 1000, 2, path(SOCK_800), 'best');
    expectClose(t.latencySec, TB5_TCP.latencySec + 2000 / rSock, 1e-12);
  });

  it('a 4-node ring or mesh: every hop gets the fill, and C ≤ B_mem/(6·n_stg)', () => {
    for (const shape of [
      { kind: 'ring' } as const,
      { kind: 'full_mesh', cablesPerPair: 1 } as const,
    ]) {
      const p = fabricLinkParams(TB5_TCP, shape, 4, SOCK_800, 4096);
      expectClose(p.hopLatencySec, TB5_TCP.latencySec + 8192 / rSock, 1e-12);
      expect(p.allPortsCapBytesPerSec).toBe(800e9 / 6);
      expect(p.portBandwidthBytesPerSec).toBe(TB5_TCP.bandwidthPerDirectionBytesPerSec);
      expect(p.forwardingLatencySec).toBe(TB5_TCP.forwardingLatencySec);
    }
    const slow = fabricLinkParams(TB5_TCP, { kind: 'ring' }, 4, {
      ...SOCK_800,
      hostMemBandwidthBytesPerSec: 30e9,
    });
    expect(slow.allPortsCapBytesPerSec).toBe(5e9);
  });

  it('intra-node groups never carry the copy; the inter-node group does', () => {
    const topo = cluster(ETHERNET_10G_TCP, 2, 2);
    expect(collectivePath(groupSpan(2, topo), topo, SOCK_800)).not.toHaveProperty('interStaging');
    expect(collectivePath(groupSpan(4, topo), topo, SOCK_800)).toHaveProperty(
      'interStaging',
      SOCK_800,
    );
  });
});

describe('simulate: 2 × M3 Ultra over the Thunderbolt 5 bridge', () => {
  const input = (link: LinkSpec): SimInput => ({
    model: REF_DENSE_8B,
    quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
    parallel: PAR,
    workload: { batch: 1, promptLen: 1024, decodeContextLen: 1024 },
    hardware: { device: M3_ULTRA_80C, topology: cluster(link) },
  });

  it('TCP pays a little more comm α than the same link numbers over RDMA; nothing else moves, no new warning', () => {
    const tcp = simulate(input(TB5_TCP));
    const zeroCopy = simulate(input({ ...TB5_TCP, transport: 'rdma' }));
    expect(tcp.decode.comm.latencySec).toBeGreaterThan(zeroCopy.decode.comm.latencySec);
    expect(tcp.decode.comm.bandwidthSec).toBe(zeroCopy.decode.comm.bandwidthSec);
    const rel = (tcp.decode.cycleSec - zeroCopy.decode.cycleSec) / zeroCopy.decode.cycleSec;
    expect(rel).toBeGreaterThan(0);
    expect(rel).toBeLessThan(1e-3);
    expect(tcp.warnings).toEqual(zeroCopy.warnings);
  });

  it('TB5 RDMA decodes faster than TB5 TCP (α 7 vs 45 µs, and no copy)', () => {
    expect(simulate(input(TB5_RDMA)).decode.tokensPerSec).toBeGreaterThan(
      simulate(input(TB5_TCP)).decode.tokensPerSec,
    );
  });
});
