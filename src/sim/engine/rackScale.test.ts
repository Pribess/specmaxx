/** Rack-scale preset conventions, CPU tiers, and DeepSeek EP64 placement (docs/simulation-model.md §4–6). */
import { describe, expect, it } from 'vitest';
import {
  B200,
  GB200,
  GB300,
  RTX_5090,
  GB200_NVL72,
  GB300_NVL72,
  GB200_NVL4_X18,
  MI325X_XGMI_8,
  MI355X_XGMI_8,
  MI455X_HELIOS_72,
  NVLINK_5_NVL72,
  NVLINK_5_NVL4,
  NVLINK_C2C,
  UALOE_MI455X,
  XGMI_MI325X_PER_PEER,
  XGMI_MI325X_8_GPU,
  XGMI_MI355X_PER_PEER,
  XGMI_MI355X_8_GPU,
  IB_NDR_400,
  IB_XDR_800,
  DEFAULT_HOST_RAM,
  DDR5_5600_2CH,
  composeTopology,
  defaultHostRamFor,
  deviceHostLink,
  graceHostRam,
  hostPcieLink,
  linkPresetId,
  nvl72Rack,
  nvl4Trays,
  setupErrors,
  setupOptionsFor,
  singleDevice,
} from '../hw/presets';
import type { DeviceSpec, TopologySpec } from '../hw/types';
import {
  validateDevice,
  validateLink,
  validateStorageTier,
  validateTopology,
} from '../hw/validate';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf';
import { REF_MOE_30B_A3B } from '../models/reference';
import { collectiveTime } from './collectives';
import { resolveHost } from './host';
import { collectivePath, groupSpan, groupSpanErrors } from './placement';
import { simulate } from './simulate';
import { resolveHostStaging } from './staging';
import type { SimInput } from './types';
import { validateInput } from './validate';

const rack64 = composeTopology({ deviceId: 'gb200', devicesPerNode: 64, nodes: 1 });
const trays64 = composeTopology({
  deviceId: 'gb200',
  devicesPerNode: 4,
  nodes: 16,
  localLinkId: linkPresetId(NVLINK_5_NVL4),
});
const par = { tp: 1, pp: 1, dp: 64, ep: 64 };
const bf16 = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
} as const;

function moeInput(device: DeviceSpec, topology: TopologySpec): SimInput {
  return {
    model: REF_MOE_30B_A3B,
    quant: bf16,
    hardware: { device, topology },
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 128, decodeContextLen: 8192 },
  };
}

describe('rack-scale hardware conventions', () => {
  it.each([
    [GB200, 186, 10000, 5000, 40],
    [GB300, 279, 15000, 165, 1.3],
  ] as const)(
    '%s: per-GPU memory and dense/sparse peaks, not rack or superchip totals',
    (device, capacityGB, fp4TF, int8TOPS, fp64TF) => {
      expect(validateDevice(device)).toEqual([]);
      expect(device.memory.capacityOptionsBytes).toEqual([capacityGB * 1e9]);
      expect(device.memory.bandwidthBytesPerSec).toBe(8e12);
      const e = device.engines[0];
      const peak = (input: string, sparsity = 'dense') =>
        e.matmulPeaks.find((p) => p.input === input && p.sparsity === sparsity)?.flopsPerSec;
      expect(peak('fp4')).toBe(fp4TF * 1e12);
      expect(peak('fp4', '2:4')).toBe(20000e12);
      expect(peak('int8')).toBe(int8TOPS * 1e12);
      expect(peak('fp8')).toBe(5000e12);
      expect(peak('fp6')).toBe(5000e12);
      expect(peak('bf16')).toBe(2500e12);
      expect(peak('fp16')).toBe(2500e12);
      expect(peak('tf32')).toBe(1250e12);
      expect(e.vectorFlopsPerSec).toEqual({ fp32: 80e12, fp64: fp64TF * 1e12 });
      expect(e.sfuOpsPerSec).toBe((80e12 / 256) * (device.id === 'gb300' ? 32 : 16));
      expect(device.hostLink).toBe(NVLINK_C2C);
    },
  );

  it.each([
    [NVLINK_5_NVL72, 900e9],
    [NVLINK_5_NVL4, 900e9],
    [NVLINK_C2C, 450e9],
    [XGMI_MI325X_PER_PEER, 64e9],
    [XGMI_MI325X_8_GPU, 448e9],
    [XGMI_MI355X_PER_PEER, 72e9],
    [XGMI_MI355X_8_GPU, 504e9],
    [UALOE_MI455X, 1.8e12],
  ] as const)('%s: per-direction injection bandwidth in B/s', (link, bandwidth) => {
    expect(validateLink(link)).toEqual([]);
    expect(link.bandwidthPerDirectionBytesPerSec).toBe(bandwidth);
    expect(link.p2p).toBe('direct');
  });

  it('validates nested host-link rates and provenance on the link, without duplicated device citations', () => {
    expect(validateDevice(GB200)).toEqual([]);
    expect(
      validateDevice({
        ...GB200,
        hostLink: { ...NVLINK_C2C, bandwidthPerDirectionBytesPerSec: -1 },
      }).some((e) => e.includes('bandwidthPerDirectionBytesPerSec')),
    ).toBe(true);
    expect(
      validateDevice({ ...GB200, hostLink: { ...NVLINK_C2C, sources: [], estimated: [] } }).some(
        (e) => e.includes('not covered'),
      ),
    ).toBe(true);
  });

  it('keeps unpublished C2C, NVL4 mesh, and UALoE latency assumptions visible', () => {
    expect(
      NVLINK_C2C.estimated.find((e) => e.field === 'bandwidthPerDirectionBytesPerSec')?.reason,
    ).toMatch(/ambiguous/);
    expect(NVLINK_C2C.estimated.some((e) => e.field === 'latencySec')).toBe(true);
    expect(NVLINK_5_NVL4.estimated[0].reason).toMatch(/mesh inferred/);
    expect(UALOE_MI455X.estimated[0].reason).toMatch(/unpublished/);
  });

  it.each([
    GB200_NVL72,
    GB300_NVL72,
    GB200_NVL4_X18,
    MI325X_XGMI_8,
    MI355X_XGMI_8,
    MI455X_HELIOS_72,
  ])('%s validates', (topology) => {
    expect(validateTopology(topology)).toEqual([]);
  });

  it('keeps NVLink inside each rack/board; uses the right per-GPU NIC between domains', () => {
    expect(nvl72Rack('gb200').interNode).toBeUndefined();
    expect(nvl72Rack('gb200', 2).interNode).toBe(IB_NDR_400);
    expect(nvl72Rack('gb300', 2).interNode).toBe(IB_XDR_800);
    expect(nvl4Trays('gb200', 16)).toMatchObject({
      devicesPerNode: 4,
      nodes: 16,
      intraNode: NVLINK_5_NVL4,
      interNode: IB_NDR_400,
    });
    expect(() => nvl72Rack('gb200', 0)).toThrow();
    expect(() => nvl4Trays('gb200', 1.5)).toThrow();
  });

  it.each(['gb200', 'gb300'])(
    '%s supports active 64/72 GPU domains and eighteen machines',
    (deviceId) => {
      const opts = setupOptionsFor(deviceId);
      expect(opts.devicesPerNodeCounts).toEqual([1, 2, 4, 8, 16, 32, 36, 64, 72]);
      expect(opts.nodeCounts).toContain(18);
      for (const devicesPerNode of opts.devicesPerNodeCounts) {
        expect(setupErrors({ deviceId, devicesPerNode, nodes: 18 })).toEqual([]);
        expect(composeTopology({ deviceId, devicesPerNode, nodes: 18 }).intraNode).toBe(
          devicesPerNode === 1 ? undefined : NVLINK_5_NVL72,
        );
      }
      expect(setupOptionsFor('b200').devicesPerNodeCounts).toEqual([1, 2, 4, 8]);
      expect(setupOptionsFor('b200').nodeCounts).toContain(32);
      expect(setupOptionsFor('instinct-mi455x').devicesPerNodeCounts).toContain(72);
      if (deviceId === 'gb200')
        expect(
          setupErrors({
            deviceId,
            devicesPerNode: 8,
            nodes: 1,
            localLinkId: linkPresetId(NVLINK_5_NVL4),
          }).length,
        ).toBeGreaterThan(0);
    },
  );
});

describe('Grace host RAM and C2C', () => {
  it.each(['gb200', 'gb300'] as const)(
    '%s pools ceil(GPUs/2) Grace CPUs without doubling rack RAM',
    (gen) => {
      for (const count of [1, 2, 3, 4, 64, 72]) {
        const ram = graceHostRam(gen, count);
        expect(validateStorageTier(ram)).toEqual([]);
        expect(ram.capacityBytes).toBe(Math.ceil(count / 2) * 480e9);
        expect(ram.bandwidthBytesPerSec).toBe(Math.ceil(count / 2) * 384e9);
      }
      expect(graceHostRam(gen, 72).capacityBytes).toBe(17.28e12);
      expect(graceHostRam(gen, 72).bandwidthBytesPerSec).toBe(13.824e12);
      expect(() => graceHostRam(gen, 0)).toThrow();
    },
  );

  it('uses C2C and node-sized Grace RAM by default, while explicit stack host specs still win', () => {
    expect(deviceHostLink(GB200)).toBe(NVLINK_C2C);
    expect(deviceHostLink({ ...GB200, hostInterface: { generation: 5, lanes: 16 } })).toBe(
      NVLINK_C2C,
    );
    expect(defaultHostRamFor(GB200, 64)).toEqual(graceHostRam('gb200', 64));
    const host = resolveHost({ device: GB200, topology: rack64 }, GB200, new Set(['host_ram']));
    expect(host.link).toBe(NVLINK_C2C);
    expect(host.ram).toEqual(graceHostRam('gb200', 64));
    const link = hostPcieLink(4, 16);
    const explicit = resolveHost(
      { device: GB200, topology: rack64, host: { link, ram: DDR5_5600_2CH } },
      GB200,
      new Set(['host_ram']),
    );
    expect(explicit.link).toBe(link);
    expect(explicit.ram).toBe(DDR5_5600_2CH);
    const input = moeInput(GB200, rack64);
    input.deployment = {
      offload: {
        routedExperts: { tier: 'host_ram', residentFraction: 0, hitRatio: { kind: 'uniform' } },
      },
    };
    expect(validateInput(input)).toEqual([]);
    expect(
      simulate(input)
        .decode.stages.flatMap((s) => s.layers.flatMap((l) => l.ops))
        .some((o) => o.op.kind === 'transfer'),
    ).toBe(true);
  });

  it('uses C2C and pooled Grace RAM even for a non-GPUDirect network staging path', () => {
    const device = { ...GB200, gpuDirectRdma: false };
    const host = resolveHostStaging(device, trays64, { device, topology: trays64 }, par);
    expect(host.staging?.pcie?.bandwidthBytesPerSec).toBe(450e9);
    expect(host.staging?.hostMemBandwidthBytesPerSec).toBe(2 * 384e9);
  });

  it('no hostLink: legacy PCIe host defaults and offload results are bit-identical to explicit pre-change defaults', () => {
    expect(RTX_5090.hostLink).toBeUndefined();
    expect(deviceHostLink(RTX_5090)).toEqual(hostPcieLink(5, 16));
    expect(deviceHostLink(B200)).toBeUndefined();
    expect(defaultHostRamFor(RTX_5090, 8)).toBe(DEFAULT_HOST_RAM);
    expect(() =>
      resolveHost({ device: B200, topology: singleDevice(B200) }, B200, new Set(['host_ram'])),
    ).toThrow(/no host link/);
    const input = moeInput(RTX_5090, singleDevice(RTX_5090));
    input.deployment = {
      offload: {
        routedExperts: { tier: 'host_ram', residentFraction: 0, hitRatio: { kind: 'uniform' } },
      },
    };
    const oldHost = { link: hostPcieLink(5, 16), ram: DEFAULT_HOST_RAM };
    const implicit = simulate(input);
    const explicit = simulate({ ...input, hardware: { ...input.hardware, host: oldHost } });
    expect(implicit.prefill).toEqual(explicit.prefill);
    expect(implicit.decode).toEqual(explicit.decode);
    expect(implicit.memory).toEqual(explicit.memory);
  });
});

describe('DeepSeek-V4-Flash-0731 EP64 @ 8k within an NVL72 domain', () => {
  const mapped = mapHfModel(fixtureRepoFiles('deepseek-ai/DeepSeek-V4-Flash-0731'));
  if (mapped.kind !== 'ok') throw new Error(JSON.stringify(mapped));
  const input: SimInput = {
    model: mapped.model,
    quant: mapped.quant.quant,
    parallel: par,
    workload: { batch: 64, promptLen: 128, decodeContextLen: 8192 },
    hardware: { device: GB200, topology: rack64 },
  };

  it('EP all-to-all uses NVL72 NVLink, not the inter-node link, through simulate', () => {
    expect(validateInput(input)).toEqual([]);
    const path = collectivePath(groupSpan(par.ep, rack64), rack64);
    expect(path).toEqual({ kind: 'flat', level: 'intra', link: NVLINK_5_NVL72 });
    const sim = simulate(input);
    expect(sim.decode.tokensPerSec).toBeGreaterThan(0);
    const epOps = sim.decode.stages
      .flatMap((s) => s.layers.flatMap((l) => l.ops))
      .filter((o) => o.op.kind === 'collective' && o.op.collective === 'all_to_all');
    expect(epOps.length).toBeGreaterThan(0);
    for (const result of epOps) {
      const op = result.op;
      if (op.kind !== 'collective') throw new Error('expected collective');
      expect(op.group).toBe('ep');
      expect(op.groupSize).toBe(64);
      const expected = collectiveTime(op.collective, op.messageBytes, op.groupSize, path, 'best');
      expect(result.commBandwidthSec).toBe(expected.bandwidthSec);
      expect(result.commLatencySec).toBe(expected.latencySec);
    }
    const trays = simulate({ ...input, hardware: { device: GB200, topology: trays64 } });
    expect(trays.decode.tokensPerSec).toBeLessThan(sim.decode.tokensPerSec);
  });

  it('small-batch all-to-all is latency-dominated; larger batches expose the NIC bandwidth gap @ 8k', () => {
    const measurements = [];
    for (const batch of [64, 2048]) {
      const runs = [rack64, trays64].map((topology) =>
        simulate({
          ...input,
          workload: { ...input.workload, batch },
          hardware: { device: GB200, topology },
        }),
      );
      const times = runs.map((run) => {
        const layer = run.decode.stages[0].layers.find((l) =>
          l.ops.some((o) => o.op.kind === 'collective' && o.op.collective === 'all_to_all'),
        );
        if (layer === undefined) throw new Error('no MoE layer');
        const messages = layer.ops.filter(
          (o) => o.op.kind === 'collective' && o.op.collective === 'all_to_all',
        );
        return {
          messages: messages.map((o) => ({
            bytes: o.op.kind === 'collective' ? o.op.messageBytes : 0,
            alpha: o.commLatencySec,
            beta: o.commBandwidthSec,
          })),
          alpha: messages.reduce((sum, o) => sum + o.commLatencySec, 0),
          beta: messages.reduce((sum, o) => sum + o.commBandwidthSec, 0),
        };
      });
      expect(times[1].alpha + times[1].beta).toBeGreaterThan(times[0].alpha + times[0].beta);
      const noOverlap = [rack64, trays64].map(
        (topology) =>
          simulate({
            ...input,
            options: { overlap: 'none' },
            workload: { ...input.workload, batch },
            hardware: { device: GB200, topology },
          }).decode.tokensPerSec,
      );
      measurements.push({
        batch,
        rackTokPerSec: runs[0].decode.tokensPerSec,
        traysTokPerSec: runs[1].decode.tokensPerSec,
        noOverlap,
        times,
      });
    }
    // α stays on the critical path; ideal overlap can hide the larger NIC β behind compute.
    const slowdown = (rates: number[]) => 1 - rates[1] / rates[0];
    expect(slowdown(measurements[1].noOverlap)).toBeGreaterThan(
      slowdown(measurements[0].noOverlap),
    );
    expect(measurements[0].times[0].alpha).toBe(1.6e-6);
    expect(measurements[0].times[1].alpha).toBe(2e-6);
    expect(measurements[0].times[0].beta).toBe(0.10752e-6);
    expect(measurements[0].times[1].beta).toBe(1.8432e-6);
    expect(measurements[1].times[1].beta / measurements[0].times[1].beta).toBe(32);
    expect(measurements[0].times[0].alpha).toBeGreaterThan(measurements[0].times[0].beta);
    expect(measurements[1].times[1].beta).toBeGreaterThan(measurements[1].times[1].alpha);
  });

  it('EP64 across sixteen NVL4 boards is hierarchical over NDR, while EP64 cannot tile all 72 GPUs', () => {
    expect(collectivePath(groupSpan(par.ep, trays64), trays64)).toEqual({
      kind: 'hierarchical',
      devicesPerNode: 4,
      nodes: 16,
      intra: NVLINK_5_NVL4,
      inter: IB_NDR_400,
    });
    expect(groupSpanErrors(64, GB200_NVL72).length).toBeGreaterThan(0);
  });
});
