/**
 * Hardware preset registry: the device presets the UI offers, and topology helpers.
 * The full preset library (more devices, multi-device topologies, unlocked variants) is a follow-up task.
 */

import type { DeviceSpec, TopologySpec } from '../types';
import { RTX_5090 } from './nvidia';

export { RTX_5090 } from './nvidia';

/** Every device preset, in display order. Each must pass `validateDevice` (enforced by presets.test.ts). */
export const DEVICE_PRESETS: DeviceSpec[] = [RTX_5090];

/**
 * A topology of exactly one device: 1 device per node × 1 node, no links (any parallelism > 1 is a
 * validation error). Its numbers are configuration counts, not vendor numbers, so it cites none.
 */
export function singleDevice(device: DeviceSpec): TopologySpec {
  return { id: `single-${device.id}`, name: `1 × ${device.name}`, devicesPerNode: 1, nodes: 1, sources: [] };
}
