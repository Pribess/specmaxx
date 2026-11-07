/**
 * Public API of the simulation core (pure TypeScript, no React, no DOM). The UI imports from here.
 * See docs/simulation-model.md for the model; README "Layout" for the module map.
 */

// Model IR: dtypes, ops, metrics, architecture spec, builders, parameter counts.
export * from './ir/dtype';
export * from './ir/ops';
export * from './ir/metrics';
export * from './ir/model';
export * from './ir/builders';
export * from './ir/params';

// Lowering: model + workload step + parallelism → per-device ops.
export * from './lower/lower';

// Reference models.
export * from './models/reference';

// Hardware: schema, units, peaks, shape efficiency, modifiers, unlock axes, validation, presets (devices, links,
// topologies, unlock axes per device).
export * from './hw/types';
export * from './hw/units';
export * from './hw/peaks';
export * from './hw/efficiency';
export * from './hw/modifiers';
export * from './hw/unlocks';
export * from './hw/validate';
export * from './hw/presets/index';

// Engine: collectives, placement, per-op time, pipeline, memory, validation, simulate.
export * from './engine/types';
export * from './engine/collectives';
export * from './engine/placement';
export * from './engine/opTime';
export * from './engine/pipeline';
export * from './engine/memory';
export * from './engine/validate';
export * from './engine/simulate';
