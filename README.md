# specmaxx

A frontend-only estimator of the theoretical maximum prefill and decode tokens/s for a Hugging Face model on your hardware. Compare rigs, quantization and parallelism, with per-op limits and a memory-fit verdict. These are hardware ceilings, not inference-engine benchmark predictions.

## Development

Requires Node.js 22.12+ on the 22 line, 24.x, or 26+ (`^22.12.0 || ^24.0.0 || >=26.0.0`, the intersection of
what Vitest 5, Vite 8 and @vitejs/plugin-react 6 support; Node 20 and 23 are not supported).

```sh
bun install --frozen-lockfile  # install the locked dependencies (Bun; see bun.lock)
bun run dev       # Vite dev server
bun run test      # unit tests (vitest), once; bun run test:watch to watch
bun run check     # typecheck + lint + unit/component tests + build + worker smoke test
bun run e2e       # Playwright smoke against the production build (run bun run build first)
bun run screenshots # rebuild and regenerate docs/screenshots/*.png
bun run format -- src/ui/components/MemoryFit.tsx # format only the files you changed
```

The browser tests need Chromium: `bunx playwright install chromium --only-shell`. HF responses are mocked
from checked-in fixtures, so the tests do not call the live Hub.
Screenshots include the 1440×900 and 390×844 light/dark views, extra tall desktop views showing all
input controls (`desktop-full-*.png`), and a viewport-sized About page.

The app starts with a built-in 8B reference, without a network request. Load an HF model ID to read its
config directly from the Hub. Gated models can use an optional HF token stored in this browser's
localStorage; it is sent only to Hugging Face, never included in the shareable URL. Use **Forget token**
to remove it. Every simulation input (including custom spec edits) is encoded in the URL. Copy the
address to share an estimate. The app follows your system's light/dark preference.

**How it works** opens `about.html`, which renders the simulation-model documentation (and the sanity
check document when present). Its back-link restores your estimate.

## Deployment

Pushing the tag `release-pages` deploys the build as a Cloudflare Worker with static assets, served at https://cro.sh/specmaxx/; see
[docs/deploy.md](docs/deploy.md).

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
  - `__sanity__/`: the sanity check against community benchmarks (harness, sibling model specs, reports as data,
    the `simulated ≥ 0.9 × reported` test); see docs/sanity-check.md.
- `docs/simulation-model.md`: the simulation model (formulas, assumptions, exclusions, extension checklists);
  the code links to its sections and it links back to the functions.
- `docs/hf-loader.md`: the Hugging Face loader (fetch, token, caching, dispatch, adding a family, quant defaults,
  param check, the supported model shortlist).
- `docs/sanity-check.md`: simulated maxima vs 43 community benchmark reports (method, ratios, flagged cases,
  follow-ups).
- `src/hf/`: browser-side Hub fetch (`loadHfModel`: api + config.json + sidecars, token, cache, errors).
- `src/ui/`: React controls, roofline and throughput charts, memory fit and op breakdown.
  - `state/`: pure state/reducer, URL codec, hardware resolution, validation, summaries and batch sweep.
  - `sim/`: the real web-worker runner (32-entry LRU) and an injectable inline runner for component tests.
- `src/styles/tokens.css`: design tokens (Newsreader + Schibsted Grotesk, the deep-blue light/dark palette,
  type, spacing and control sizes); the design plan is `docs/design/redesign/plan.md`.
- `src/about/`, `about.html`: the readable methodology page with working documentation anchors.
- `src/main.tsx`: React entry point.
- `e2e/`: fixture-backed production-worker smoke test and desktop/phone screenshot generation.
