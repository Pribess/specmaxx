/** Full hardware resolver and unlock citations, loaded only when Advanced is opened. */
export type HardwareProvenance = typeof import('virtual:hardware-provenance');

let pending: Promise<HardwareProvenance> | undefined;

/**
 * Load the original, cited preset graph. Failed requests can be retried; simultaneous hardware panels
 * share one request. Dev and Vitest also use the original sources (the build transform is disabled).
 * See docs/deploy.md ("Preset provenance sidecar").
 */
export function loadHardwareProvenance(): Promise<HardwareProvenance> {
  pending ??= import('virtual:hardware-provenance').catch((error: unknown) => {
    pending = undefined;
    throw error;
  });
  return pending;
}
