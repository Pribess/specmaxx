/** Virtual modules provided by vite/stripProvenance.ts in dev, test and production builds. */
declare module 'virtual:hardware-provenance' {
  /** Rebuilds edited/unlocked hardware from full cited presets, without changing any physical numbers. */
  export const resolveHardware: typeof import('./ui/state/hardware').resolveHardware;
  /** Original unlock options, including their complete citations. */
  export const DEVICE_UNLOCKS: typeof import('./sim/hw/presets').DEVICE_UNLOCKS;
}

declare module 'virtual:community-unlocks' {
  /** Keys `device/axis/option` whose modifiers cite community measurements; no citation payload. */
  export const COMMUNITY_UNLOCKS: Set<string>;
}
