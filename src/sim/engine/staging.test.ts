import { describe, expect, it } from 'vitest';
import { SYNTH_GPU, SYNTH_IB } from '../__fixtures__/synthetic';
import {
  DEVICE_PRESETS,
  DGX_SPARK,
  ETHERNET_10G_TCP,
  ETHERNET_5G_TCP,
  H100_SXM,
  HGX_H100_16_NDR,
  hostPcieLink,
  HOST_STAGING_STEP_BYTES,
  IB_NDR_400,
  LINK_PRESETS,
  M3_ULTRA_80C,
  RTX_5090,
  RTX_5090_PCIE_2,
  RYZEN_AI_MAX_PLUS_395,
  SPARK_CONNECTX_7,
  TB4_TCP,
  TB5_RDMA,
  USB4_HALO_TCP,
} from '../hw/presets';
import { DDR5_6000_2CH } from '../hw/presets/storage';
import type { DeviceSpec, LinkSpec, TopologySpec } from '../hw/types';
import {
  collectiveTime,
  effectiveLink,
  stagedLink,
  type CollectivePath,
  type EffectiveLink,
} from './collectives';
import { collectivePath, groupSpan, ppSendPath } from './placement';
import { simulate } from './simulate';
import { needsHostStaging, resolveHostStaging, type HostStaging } from './staging';
import { REF_DENSE_8B } from '../models/reference';
import type { SimInput } from './types';

function expectClose(actual: number, expected: number, rel = 1e-12): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
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

const PAR = { tp: 1, pp: 1, dp: 1, ep: 1 };
const GEN5_X16 = hostPcieLink(5, 16);
const GEN3_X16 = hostPcieLink(3, 16);

/** PCIe staging of an RTX 5090 (PCIe 5.0 x16, DDR5-6000 2ch): B_pcie 63.015 GB/s, α_pcie 1 µs, B_host 96 GB/s. */
const STAGING: HostStaging = {
  pcie: { bandwidthBytesPerSec: GEN5_X16.bandwidthPerDirectionBytesPerSec, latencySec: 1e-6 },
  socketCopy: false,
  hostMemBandwidthBytesPerSec: 96e9,
  stepBytes: HOST_STAGING_STEP_BYTES,
  activeDevicesPerNode: 1,
};
const B_PCIE = GEN5_X16.bandwidthPerDirectionBytesPerSec;

describe('needsHostStaging (docs §6 "Inter-node host staging")', () => {
  it('stages a dedicated GPU without a GPUDirect path', () => {
    expect(needsHostStaging(RTX_5090, cluster(ETHERNET_10G_TCP))).toBe(true);
    // The NIC is RDMA-capable, the GeForce is not: still staged.
    expect(needsHostStaging(RTX_5090, cluster(IB_NDR_400))).toBe(true);
    // TCP never reaches device memory, even on a data-center GPU.
    expect(needsHostStaging(H100_SXM, cluster(ETHERNET_10G_TCP))).toBe(true);
  });

  it('does not stage a GPUDirect pair, unified memory, or a single node', () => {
    expect(needsHostStaging(H100_SXM, cluster(IB_NDR_400))).toBe(false);
    expect(needsHostStaging(H100_SXM, HGX_H100_16_NDR)).toBe(false);
    for (const soc of [DGX_SPARK, M3_ULTRA_80C, RYZEN_AI_MAX_PLUS_395]) {
      for (const link of [
        ETHERNET_10G_TCP,
        ETHERNET_5G_TCP,
        SPARK_CONNECTX_7,
        TB4_TCP,
        TB5_RDMA,
        USB4_HALO_TCP,
      ]) {
        expect(needsHostStaging(soc, cluster(link)), `${soc.id} × ${link.name}`).toBe(false);
      }
    }
    expect(needsHostStaging(RTX_5090, RTX_5090_PCIE_2)).toBe(false);
    expect(needsHostStaging(RTX_5090, cluster(ETHERNET_10G_TCP, 1))).toBe(false);
  });

  it('treats absent fields as GPUDirect-capable (the convention before staging)', () => {
    expect(SYNTH_GPU.gpuDirectRdma).toBeUndefined();
    expect(SYNTH_IB.gpuDirectRdma).toBeUndefined();
    expect(needsHostStaging(SYNTH_GPU, cluster(SYNTH_IB))).toBe(false);
  });
});

describe('resolveHostStaging', () => {
  it('takes B_pcie and α_pcie from the device host interface and B_host from the default host RAM; TCP adds the socket copy', () => {
    const r = resolveHostStaging(
      RTX_5090,
      cluster(ETHERNET_10G_TCP, 2, 2),
      { device: RTX_5090, topology: cluster(ETHERNET_10G_TCP) },
      { ...PAR, tp: 4 },
    );
    expect(r.staging).toEqual({
      ...STAGING,
      socketCopy: true,
      hostMemBandwidthBytesPerSec: DDR5_6000_2CH.bandwidthBytesPerSec,
      activeDevicesPerNode: 2,
    });
    expect(r.staging?.stepBytes).toBe(524_288);
    expect(r.warnings).toEqual([
      `inter-node: ${RTX_5090.name} has no GPUDirect path over ${ETHERNET_10G_TCP.name}; each message is staged through host memory`,
    ]);
  });

  it('prefers hardware.host; falls back to PCIe 5.0 x16 with a warning for an SXM module', () => {
    const topo = cluster(ETHERNET_10G_TCP);
    const given = resolveHostStaging(
      RTX_5090,
      topo,
      { device: RTX_5090, topology: topo, host: { link: GEN3_X16 } },
      PAR,
    );
    expect(given.staging?.pcie?.bandwidthBytesPerSec).toBe(
      GEN3_X16.bandwidthPerDirectionBytesPerSec,
    );
    const sxm = resolveHostStaging(H100_SXM, topo, { device: H100_SXM, topology: topo }, PAR);
    expect(sxm.staging?.pcie?.bandwidthBytesPerSec).toBe(B_PCIE);
    expect(sxm.warnings[0]).toMatch(/no host link known for .*H100.*PCIe 5\.0 x16/);
  });

  it('is absent when nothing stages', () => {
    const topo = HGX_H100_16_NDR;
    expect(resolveHostStaging(H100_SXM, topo, { device: H100_SXM, topology: topo }, PAR)).toEqual({
      staging: undefined,
      warnings: [],
    });
  });
});

describe('effectiveLink with host staging', () => {
  const TCP: EffectiveLink = { bandwidthBytesPerSec: 1.25e9, latencySec: 10e-6 };

  it('small message: α + 2·α_pcie + 2·m/B_pcie, bandwidth unchanged', () => {
    const l = stagedLink(TCP, STAGING, 2048);
    expectClose(l.latencySec, 10e-6 + 2e-6 + (2 * 2048) / B_PCIE);
    expect(l.bandwidthBytesPerSec).toBe(1.25e9);
  });

  it('large message: one step of fill, 2·c/B_pcie', () => {
    const l = stagedLink(TCP, STAGING, 32 * 1024 * 1024);
    expectClose(l.latencySec, 10e-6 + 2e-6 + (2 * 524_288) / B_PCIE);
  });

  it('a binding PCIe link: 50 GB/s NIC on Gen3 x16 ⇒ B_eff = 15.754 GB/s', () => {
    const nic: EffectiveLink = { bandwidthBytesPerSec: 50e9, latencySec: 1e-6 };
    const gen3 = {
      ...STAGING,
      pcie: { latencySec: 1e-6, bandwidthBytesPerSec: GEN3_X16.bandwidthPerDirectionBytesPerSec },
    };
    expectClose(stagedLink(nic, gen3, 1e6).bandwidthBytesPerSec, 15.753846153846155e9);
  });

  it('a binding host share: 8 devices of a node on 96 GB/s ⇒ 96e9/(4·8) = 3 GB/s', () => {
    const nic: EffectiveLink = { bandwidthBytesPerSec: 50e9, latencySec: 1e-6 };
    expectClose(
      stagedLink(nic, { ...STAGING, activeDevicesPerNode: 8 }, 1e6).bandwidthBytesPerSec,
      3e9,
    );
  });

  it('without staging, effectiveLink is exactly the unstaged result', () => {
    for (const link of LINK_PRESETS) {
      expect(effectiveLink(link, 2, undefined, 4096)).toEqual(effectiveLink(link, 2));
    }
  });
});

describe('collectiveTime on staged paths', () => {
  const staged = (link: LinkSpec, fabric?: CollectivePath & { kind: 'flat' }): CollectivePath => ({
    kind: 'flat',
    level: 'inter',
    link,
    ...(fabric?.fabric !== undefined ? { fabric: fabric.fabric } : {}),
    interStaging: STAGING,
  });
  const plain = (link: LinkSpec): CollectivePath => ({ kind: 'flat', level: 'inter', link });

  it('4 KiB all-reduce, n = 2: latency 2·(α + 2 µs + 2·2048/B_pcie), bandwidth unchanged', () => {
    const t = collectiveTime('all_reduce', 4096, 2, staged(ETHERNET_10G_TCP), 'best');
    const u = collectiveTime('all_reduce', 4096, 2, plain(ETHERNET_10G_TCP), 'best');
    expectClose(t.latencySec, 2 * (10e-6 + 2e-6 + (2 * 2048) / B_PCIE));
    expect(t.bandwidthSec).toBe(u.bandwidthSec);
  });

  it('64 MiB all-reduce, n = 2: each step pays one step of fill, 2·c/B_pcie', () => {
    const t = collectiveTime('all_reduce', 64 * 1024 * 1024, 2, staged(ETHERNET_10G_TCP), 'ring');
    expectClose(t.latencySec, 2 * (10e-6 + 2e-6 + (2 * 524_288) / B_PCIE));
  });

  it('send/recv stages the whole message (m = S)', () => {
    const t = collectiveTime('send_recv', 1000, 2, staged(ETHERNET_10G_TCP), 'best');
    expectClose(t.latencySec, 10e-6 + 2e-6 + (2 * 1000) / B_PCIE);
  });

  it('all-to-all keeps one α (staging is not host bounce)', () => {
    const t = collectiveTime('all_to_all', 4096, 4, staged(ETHERNET_10G_TCP), 'best');
    expectClose(t.latencySec, 10e-6 + 2e-6 + (2 * 1024) / B_PCIE);
  });

  it('a full mesh of 3 adds the staging α on every hop', () => {
    const mesh: CollectivePath & { kind: 'flat' } = {
      kind: 'flat',
      level: 'inter',
      link: ETHERNET_10G_TCP,
      fabric: { kind: 'full_mesh', cablesPerPair: 1 },
    };
    const S = 3 * 4096;
    const u = collectiveTime('all_reduce', S, 3, mesh, 'ring');
    const t = collectiveTime('all_reduce', S, 3, { ...mesh, interStaging: STAGING }, 'ring');
    const hopExtraSec = 2e-6 + (2 * 4096) / B_PCIE;
    // Formula: every α term of the unstaged cost gains α_stage + fill.
    const hops = u.latencySec / ETHERNET_10G_TCP.latencySec;
    expect(Math.round(hops)).toBeGreaterThanOrEqual(1);
    expectClose(t.latencySec, u.latencySec + Math.round(hops) * hopExtraSec, 1e-9);
  });

  it('a fabric tree all-reduce stages S per step', () => {
    const mesh: CollectivePath & { kind: 'flat' } = {
      kind: 'flat',
      level: 'inter',
      link: ETHERNET_10G_TCP,
      fabric: { kind: 'full_mesh', cablesPerPair: 1 },
      interStaging: STAGING,
    };
    const S = 3 * 4096;
    const t = collectiveTime('all_reduce', S, 3, mesh, 'tree');
    // 2·⌈log2 3⌉ = 4 steps, each α + 2 µs + 2·S/B_pcie
    expectClose(t.latencySec, 4 * (10e-6 + 2e-6 + (2 * S) / B_PCIE), 1e-9);
  });

  it('hierarchical: only the inter-node phase stages', () => {
    const topo = cluster(ETHERNET_10G_TCP, 2, 2);
    const withS = collectivePath(groupSpan(4, topo), topo, STAGING);
    const without = collectivePath(groupSpan(4, topo), topo);
    expect(withS).toEqual({ ...without, interStaging: STAGING });
    const t = collectiveTime('all_reduce', 8192, 4, withS, 'ring');
    const u = collectiveTime('all_reduce', 8192, 4, without, 'ring');
    // inter phase: ring all-reduce of S/n1 over n2 = 2: 2 α steps, m = S/n = 2048.
    expectClose(t.latencySec - u.latencySec, 2 * (2e-6 + (2 * 2048) / B_PCIE), 1e-9);
    // inter phase as a tree: 2·⌈log2 2⌉ = 2 steps, each moving the whole node shard S/n1 = 4096.
    const tt = collectiveTime('all_reduce', 8192, 4, withS, 'tree');
    const tu = collectiveTime('all_reduce', 8192, 4, without, 'tree');
    expectClose(tt.latencySec - tu.latencySec, 2 * (2e-6 + (2 * 4096) / B_PCIE), 1e-9);
  });
});

describe('staged totals: the fill excludes the stage that sets β (docs §6 "Inter-node host staging")', () => {
  /** A 50 GB/s, α 1 µs RDMA-class NIC path staged with `staging`. */
  const nic = (staging: HostStaging): CollectivePath => ({
    kind: 'flat',
    level: 'inter',
    link: {
      ...ETHERNET_10G_TCP,
      name: 'test NIC',
      bandwidthPerDirectionBytesPerSec: 50e9,
      latencySec: 1e-6,
    },
    interStaging: staging,
  });
  const send = (S: number, path: CollectivePath) =>
    collectiveTime('send_recv', S, 2, path, 'best').totalSec;
  const c = HOST_STAGING_STEP_BYTES;

  it('PCIe binds (50 GB/s NIC, PCIe 3.0 x16, host memory not binding)', () => {
    const pcie3: HostStaging = {
      ...STAGING,
      pcie: { latencySec: 1e-6, bandwidthBytesPerSec: GEN3_X16.bandwidthPerDirectionBytesPerSec },
      hostMemBandwidthBytesPerSec: 1e15,
    };
    const bPcie = GEN3_X16.bandwidthPerDirectionBytesPerSec;
    // Small (m ≤ c), serial: α + 2·α_pcie + m/B + 2·m/B_pcie = 3.60192 µs
    expectClose(send(4096, nic(pcie3)), 1e-6 + 2e-6 + 4096 / 50e9 + (2 * 4096) / bPcie, 1e-12);
    expectClose(send(4096, nic(pcie3)), 3.60192e-6, 1e-6);
    // Large: α + 2·α_pcie + m/B_pcie + (c/B + c/B_pcie): one step through the NIC and the OTHER PCIe copy = 579.24576 µs
    const m = 8 * 1024 * 1024;
    expectClose(send(m, nic(pcie3)), 3e-6 + m / bPcie + c / 50e9 + c / bPcie, 1e-12);
    expectClose(send(m, nic(pcie3)), 579.24576e-6, 1e-6);
  });

  it('host memory binds (50 GB/s NIC, PCIe 5.0 x16, 96 GB/s DRAM, one device)', () => {
    // r_copy = min(63.015, 96) GB/s; r_net = min(50, 96/2) = 48 GB/s; B_eff = 96/4 = 24 GB/s
    const rNet = 48e9;
    const bEff = 24e9;
    expectClose(send(4096, nic(STAGING)), 3e-6 + (2 * 4096) / B_PCIE + 4096 / rNet, 1e-12);
    const m = 8 * 1024 * 1024;
    const t = collectiveTime('send_recv', m, 2, nic(STAGING), 'best');
    expectClose(t.bandwidthSec, m / bEff, 1e-12);
    expectClose(t.totalSec, 3e-6 + m / bEff + ((2 * c) / B_PCIE + c / rNet - c / bEff), 1e-12);
  });
});

describe('staged message size per all-reduce algorithm', () => {
  const path: CollectivePath = {
    kind: 'flat',
    level: 'inter',
    link: ETHERNET_10G_TCP,
    interStaging: STAGING,
  };

  it('explicit tree stages S per step: S = 64 KiB, n = 4 ⇒ α = 4·(10 + 2 + 2·S/B_pcie) µs = 56.32 µs', () => {
    const S = 64 * 1024;
    const t = collectiveTime('all_reduce', S, 4, path, 'tree');
    expectClose(t.latencySec, 4 * (12e-6 + (2 * S) / B_PCIE), 1e-12);
    expectClose(t.latencySec, 56.32e-6, 1e-4);
  });

  it('best compares ring and tree each staged with its own message, crossing over where the totals meet', () => {
    // n = 4: t_tree − t_ring = 4·(12 µs + 2S/B_p) − 6·(12 µs + 2(S/4)/B_p) + (2 − 3/2)·S/B
    //                      = −24 µs + 5·S/B_p + 0.5·S/B  ⇒  S* = 24 µs / (5/B_p + 0.5/B)
    const sStar = 24e-6 / (5 / B_PCIE + 0.5 / ETHERNET_10G_TCP.bandwidthPerDirectionBytesPerSec);
    for (const [S, algorithm] of [
      [0.99 * sStar, 'tree'],
      [1.01 * sStar, 'ring'],
    ] as const) {
      const best = collectiveTime('all_reduce', S, 4, path, 'best');
      const ring = collectiveTime('all_reduce', S, 4, path, 'ring');
      const tree = collectiveTime('all_reduce', S, 4, path, 'tree');
      expect(best.algorithm, `S = ${S}`).toBe(algorithm);
      expect(best.totalSec).toBe(Math.min(ring.totalSec, tree.totalSec));
    }
  });
});

describe('placement attaches staging to inter-node paths only', () => {
  it('intra-node groups and PP sends carry no staging', () => {
    const topo = cluster(ETHERNET_10G_TCP, 2, 2);
    expect(collectivePath(groupSpan(2, topo), topo, STAGING)).not.toHaveProperty('interStaging');
    const par = { tp: 2, pp: 2, dp: 1, ep: 1 };
    expect(ppSendPath(par, topo, 0, 1, STAGING)).toHaveProperty('interStaging', STAGING);
    const intra = { tp: 1, pp: 2, dp: 1, ep: 1 };
    expect(ppSendPath(intra, topo, 0, 1, STAGING)).not.toHaveProperty('interStaging');
  });
});

describe('simulate with host staging', () => {
  const input = (device: DeviceSpec, topology: TopologySpec): SimInput => ({
    model: REF_DENSE_8B,
    quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
    parallel: { tp: 2, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 1024, decodeContextLen: 1024 },
    hardware: { device, topology },
  });

  it('warns once and slows decode for 2 × RTX 5090 over 10GbE; GPUDirect RDMA undoes it exactly', () => {
    const topo = cluster(ETHERNET_10G_TCP);
    const r = simulate(input(RTX_5090, topo));
    const gdr = simulate(
      input(
        { ...RTX_5090, gpuDirectRdma: true },
        cluster({ ...ETHERNET_10G_TCP, gpuDirectRdma: true, transport: 'rdma' }),
      ),
    );
    expect(r.warnings.filter((w) => w.includes('staged through host memory'))).toHaveLength(1);
    expect(gdr.warnings.some((w) => w.includes('staged through host memory'))).toBe(false);
    expect(r.decode.comm.latencySec).toBeGreaterThan(gdr.decode.comm.latencySec);
    expect(r.decode.comm.bandwidthSec).toBe(gdr.decode.comm.bandwidthSec);
    expect(r.decode.tokensPerSec).toBeLessThan(gdr.decode.tokensPerSec);
  });

  it('no warning when no message crosses nodes (tp 1 on a 2-node topology)', () => {
    const r = simulate({ ...input(RTX_5090, cluster(ETHERNET_10G_TCP)), parallel: PAR });
    expect(r.warnings.some((w) => w.includes('staged through host memory'))).toBe(false);
  });
});

describe('preset audit: GPUDirect capability is set everywhere it matters', () => {
  it('every inter-node link preset sets gpuDirectRdma', () => {
    const interKinds = new Set<LinkSpec['kind']>([
      'ethernet',
      'roce',
      'infiniband',
      'thunderbolt',
      'usb4',
    ]);
    for (const l of LINK_PRESETS.filter((x) => interKinds.has(x.kind))) {
      expect(typeof l.gpuDirectRdma, l.name).toBe('boolean');
      const cited =
        l.sources.some((s) => s.fields.includes('gpuDirectRdma')) ||
        l.estimated.some((e) => e.field === 'gpuDirectRdma');
      expect(cited, l.name).toBe(true);
    }
    for (const l of LINK_PRESETS.filter((x) => x.kind === 'ethernet' || x.name.includes('TCP'))) {
      expect(l.gpuDirectRdma, l.name).toBe(false);
    }
  });

  it('every dedicated device preset sets gpuDirectRdma (cited); unified ones leave it absent', () => {
    for (const d of DEVICE_PRESETS) {
      if (d.memory.kind === 'unified') {
        expect(d.gpuDirectRdma, d.id).toBeUndefined();
        continue;
      }
      expect(typeof d.gpuDirectRdma, d.id).toBe('boolean');
      const cited =
        d.sources.some((s) => s.fields.includes('gpuDirectRdma')) ||
        d.estimated.some((e) => e.field === 'gpuDirectRdma');
      expect(cited, d.id).toBe(true);
    }
  });
});
