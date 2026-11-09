/**
 * The link transports and the RDMA ≤ TCP latency invariant (docs/simulation-model.md §4 "Links and topologies").
 * The pairs come from `LINK_MEDIA` (data), and every preset with a `transport` must be listed there, so a new preset
 * joins the check without editing this file.
 */

import { describe, expect, it } from 'vitest';
import {
  A100_NVLINK_BRIDGE,
  IB_HDR_200,
  NVLINK_3,
  NVLINK_2_HCM_8,
  NVLINK_2_MESH_4,
  NVLINK_2_PAIR,
  PCIE_3_X16_DIRECT,
  PCIE_3_X16,
  pcieBandwidthPerDirectionBytesPerSec,
} from './links';
import type { LinkSpec } from '../types';
import { validateLink } from '../validate';
import {
  ETHERNET_10G_TCP,
  IB_NDR_400,
  LINK_MEDIA,
  LINK_PRESETS,
  TB4_TCP,
  TB5_RDMA,
  TB5_TCP,
} from './links';

/** Link kinds that only ever connect machines (the inter-node presets). */
const INTER_NODE_KINDS = new Set<LinkSpec['kind']>([
  'ethernet',
  'roce',
  'infiniband',
  'thunderbolt',
  'usb4',
]);

describe('link transports', () => {
  it('every inter-node preset sets transport (cited); intra-node presets leave it absent', () => {
    for (const l of LINK_PRESETS) {
      if (!INTER_NODE_KINDS.has(l.kind)) {
        expect(l.transport, l.name).toBeUndefined();
        continue;
      }
      expect(['rdma', 'tcp'], l.name).toContain(l.transport);
      expect(
        l.sources.some((s) => s.fields.includes('transport')),
        l.name,
      ).toBe(true);
      if (l.transport === 'tcp') expect(l.gpuDirectRdma, l.name).toBe(false);
    }
  });

  it("validateLink rejects transport 'tcp' with a GPUDirect path", () => {
    expect(validateLink({ ...ETHERNET_10G_TCP, gpuDirectRdma: true })).toEqual([
      `link ${ETHERNET_10G_TCP.name}: transport 'tcp' needs gpuDirectRdma: false (a socket takes host pointers only)`,
    ]);
    expect(validateLink({ ...ETHERNET_10G_TCP, gpuDirectRdma: true, transport: 'rdma' })).toEqual(
      [],
    );
    expect(validateLink(IB_NDR_400)).toEqual([]);
  });
});

describe('LINK_MEDIA: RDMA ≤ TCP on one physical medium', () => {
  it('lists every preset with a transport exactly once, under its own transport, and only presets', () => {
    const listed = LINK_MEDIA.flatMap((m) => [...m.rdma, ...m.tcp]);
    for (const l of LINK_PRESETS.filter((x) => x.transport !== undefined)) {
      expect(
        listed.filter((x) => x === l),
        l.name,
      ).toHaveLength(1);
    }
    for (const l of listed) expect(LINK_PRESETS, l.name).toContain(l);
    for (const m of LINK_MEDIA) {
      for (const l of m.rdma) expect(l.transport, `${m.medium}: ${l.name}`).toBe('rdma');
      for (const l of m.tcp) expect(l.transport, `${m.medium}: ${l.name}`).toBe('tcp');
    }
    expect(new Set(LINK_MEDIA.map((m) => m.medium)).size).toBe(LINK_MEDIA.length);
  });

  it('every RDMA preset has latencySec (and forwardingLatencySec, where both have one) ≤ every TCP preset of its medium', () => {
    let pairs = 0;
    for (const m of LINK_MEDIA) {
      for (const r of m.rdma) {
        for (const t of m.tcp) {
          pairs++;
          expect(r.latencySec, `${m.medium}: α of ${r.name} vs ${t.name}`).toBeLessThanOrEqual(
            t.latencySec,
          );
          if (r.forwardingLatencySec !== undefined && t.forwardingLatencySec !== undefined) {
            expect(
              r.forwardingLatencySec,
              `${m.medium}: α_f of ${r.name} vs ${t.name}`,
            ).toBeLessThanOrEqual(t.forwardingLatencySec);
          }
        }
      }
    }
    // Thunderbolt 5 and Strix Halo USB4 today; the check grows with LINK_MEDIA.
    expect(pairs).toBeGreaterThanOrEqual(2);
  });

  it('cross-medium: Thunderbolt 5 RDMA is also no slower than the Thunderbolt 4 bridge', () => {
    expect(TB5_RDMA.latencySec).toBeLessThanOrEqual(TB4_TCP.latencySec);
    expect(TB5_RDMA.latencySec).toBeLessThan(TB5_TCP.latencySec);
  });
});

describe('Ampere and Volta links', () => {
  it('records directional bandwidth, not bidirectional headlines', () => {
    expect(NVLINK_3.bandwidthPerDirectionBytesPerSec).toBe(300e9);
    expect(A100_NVLINK_BRIDGE.bandwidthPerDirectionBytesPerSec).toBe(300e9);
    expect(A100_NVLINK_BRIDGE.kind).toBe('nvlink_bridge');
    expect(NVLINK_2_HCM_8.bandwidthPerDirectionBytesPerSec).toBe(150e9);
    expect(NVLINK_2_MESH_4.bandwidthPerDirectionBytesPerSec).toBe(150e9);
    expect(NVLINK_2_PAIR.bandwidthPerDirectionBytesPerSec).toBe(50e9);
    expect(IB_HDR_200.bandwidthPerDirectionBytesPerSec).toBe(25e9);
    expect(PCIE_3_X16_DIRECT.bandwidthPerDirectionBytesPerSec).toBe(
      pcieBandwidthPerDirectionBytesPerSec(3, 16),
    );
    expect(PCIE_3_X16.bandwidthPerDirectionBytesPerSec).toBe(
      PCIE_3_X16_DIRECT.bandwidthPerDirectionBytesPerSec,
    );
    expect(PCIE_3_X16_DIRECT.p2p).toBe('direct');
    expect(PCIE_3_X16.p2p).toBe('host_bounce');
  });
});
