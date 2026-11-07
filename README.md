# specmaxx

A simulator runtime for LLM inference: given a model and a hardware stack, estimate performance numbers (throughput, latency, memory).

## Development

Requires Node.js 22.12+ on the 22 line, 24.x, or 26+ (`^22.12.0 || ^24.0.0 || >=26.0.0`, the intersection of
what Vitest 5, Vite 8 and @vitejs/plugin-react 6 support; Node 20 and 23 are not supported).

```sh
npm ci            # install the locked dependencies
npm run dev       # Vite dev server
npm run test      # unit tests (vitest), once; npm run test:watch to watch
npm run check     # typecheck + lint + test + build: must pass before submitting
```

Layout:

- `src/sim/`: the simulation core, pure TypeScript (no React, no DOM; enforced by `tsconfig.sim.json` and ESLint).
  `src/sim/index.ts` is its public API.
  - `ir/`: dtypes, the op catalogue, FLOP/byte metrics, the model architecture spec, builders, parameter counts.
  - `lower/`: model spec + workload step + quantization + parallelism → per-device ops (one file per block kind).
  - `models/`: hand-written reference models (`REF_DENSE_8B` and tiny fixtures for every block kind), and
    `models/hf/`: the Hugging Face config.json → model IR mapping (families, quant defaults, param check) with
    real-config fixtures in `models/__fixtures__/`.
  - `hw/`: hardware spec schema, units, peak lookup, shape efficiency, device/link variant modifiers, unlock axes
    (`unlocks.ts`), validation, presets (`hw/presets/`: one file per vendor family (`nvidia.ts`,
    `nvidia-datacenter.ts`, `apple.ts`, `amd.ts`, `intel.ts`), `links.ts`, `topologies.ts`, and the registry
    `index.ts`; see docs/simulation-model.md §4 "Preset library").
  - `engine/`: collectives, placement, per-op time, stage/pipeline schedule, memory fit, input validation,
    `simulate()`.
  - `__fixtures__/`: synthetic hardware for tests.
- `docs/simulation-model.md`: the simulation model (formulas, assumptions, exclusions, extension checklists);
  the code links to its sections and it links back to the functions.
- `docs/hf-loader.md`: the Hugging Face loader (fetch, token, caching, dispatch, adding a family, quant defaults,
  param check, the supported model shortlist).
- `src/hf/`: browser-side Hub fetch (`loadHfModel`: api + config.json + sidecars, token, cache, errors).
- `src/App.tsx`, `src/main.tsx`: the UI (a placeholder page with one demo line for now).
