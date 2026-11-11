# Deploying

The app is static. [.github/workflows/deploy.yml](../.github/workflows/deploy.yml) builds it (typecheck, lint, unit
tests, build; the Playwright e2e runs locally in `bun run check` only), stages `dist/` as `deploy/specmaxx/`
(`bun run deploy:assets`), and publishes it with `wrangler deploy` as the Cloudflare Worker `specmaxx` using
[Workers Static Assets](https://developers.cloudflare.com/workers/static-assets/) (no Worker script,
[wrangler.jsonc](../wrangler.jsonc)). It runs on a push of the tag `release-pages`, or by hand (Actions → Deploy to
Cloudflare Workers → Run workflow).

The only public URL is https://cro.sh/specmaxx/ (route `cro.sh/specmaxx*` in zone `cro.sh`). `workers_dev` and
`preview_urls` are off, so no `*.workers.dev` or preview URL exists, and there is no Pages project.

- Assets are matched on the full request path, hence the `specmaxx/` directory inside `deploy/`.
- The build uses relative asset URLs (`base: './'` in vite.config.ts), so it works under the `/specmaxx/` prefix.
- `/specmaxx` redirects to `/specmaxx/` and `/specmaxx/about.html` to `/specmaxx/about`
  (`html_handling: "auto-trailing-slash"`). Unknown paths return 404 (`not_found_handling: "none"`, no SPA fallback).

## Caching

`bun run deploy:assets` ([scripts/stage-deploy.mjs](../scripts/stage-deploy.mjs)) also writes `deploy/_headers`, which
Workers Static Assets reads from the root of the assets directory (so it cannot live in `public/`):

- `/specmaxx/assets/*` (hashed JS, CSS, fonts, and the About page's reference documents): `Cache-Control: public,
  max-age=31536000, immutable`. The script checks
  that every file in `dist/assets/` has a content hash in its name; if one does not, it lists the hashed files one by
  one instead of the splat, so the unhashed file keeps revalidating.
- HTML (`/specmaxx/`, `index.html`, `about`, `about.html`): `max-age=0, must-revalidate`.

A warm repeat visit then revalidates only the HTML. `index.html` also preconnects to `https://huggingface.co` (with
`crossorigin`, since the fetches are CORS) and has a `dns-prefetch` fallback.

After deploying, check:

```sh
curl -sI https://cro.sh/specmaxx/assets/<file> | grep -i cache-control   # ... max-age=31536000, immutable
curl -sI https://cro.sh/specmaxx/ | grep -i cache-control                # ... max-age=0, must-revalidate
```

## Bundle-size budget

`bun run check:bundle` ([scripts/check-bundle-size.mjs](../scripts/check-bundle-size.mjs)) runs after `bun run build`
in `bun run check` and in CI. It prints raw, gzip and brotli sizes of every JS chunk and every prerendered About
document (`.htm`) in `dist/assets/` and checks raw **and gzip** limits independently (decimal kB):

| Group                                               |  Raw | Gzip |
| --------------------------------------------------- | ---: | ---: |
| `main` (estimator entry)                            |  620 |  190 |
| `worker` (simulation worker)                        |  175 |   52 |
| `about` (About page entry, with the method doc)     |  100 |   35 |
| `doc:simulation-model` (fetched on demand)          |  446 |  153 |
| `doc:sanity-check` (fetched on demand; optional)    |   56 |   19 |
| `total` (all JS and documents, including lazy ones) | 1890 |  600 |

Main/worker have tighter limits to protect initial loading; the other limits leave about 5–10% headroom. To raise one on
purpose, edit `LIMITS_KB` in the script in the same change and explain why. `doc:sanity-check` is skipped when
`docs/sanity-check.md` is not in the repository.

`vite.config.ts` marks `src/sim/` and `src/ui/state/` as free of side effects (`treeshake.moduleSideEffects`, for the
app and for the worker build), so the bundler drops the unused parts of barrel files. Without it the worker also
bundled the whole preset catalog, which it never reads. A module there that must run for its side effects (a registry
filled by importing it) has to be taken out of that rule. (`package.json` `sideEffects` is not used: it also drops
CSS imports unless every pattern matches.)

## About page documents

The About page ships no Markdown parser. [vite/prerenderDocs.ts](../vite/prerenderDocs.ts) renders the docs with
`marked` at build time (helpers in [src/about/markdown.ts](../src/about/markdown.ts)) and provides
`virtual:about-docs`: the method's HTML and table of contents, inlined in the About chunk, and per reference document
its URL and the list of its element ids. The reference documents are emitted as hashed assets
(`assets/simulation-model-<hash>.htm`, `.htm` rather than `.html` so `auto-trailing-slash` does not redirect them) and
fetched when the reference opens or when the URL hash names one of their ids. The dev server serves them from
`/__about-docs/<slug>.htm`, rendered per request.

For a deep link (`about.html#4-hardware`) the plugin also puts a small inline script in about.html's `<head>`, right
after `<meta charset>`. It holds the ids of each reference document (about 3 kB, 1.6 kB gzip) and adds
`<link rel="preload" as="fetch" crossorigin="anonymous">` for the document the hash points into, so its download
starts before the page's JS runs; the page's `fetch` (mode cors, credentials same-origin) reuses it, and fetches the
other document only after that one has rendered. The site sets no Content-Security-Policy; one that blocks inline
scripts would need a hash for this script (it is stable for a given set of docs).

## Preset provenance sidecar

Preset files in `src/sim/hw/presets/` remain the reviewed, fully cited source of truth. The small
[Vite plugin](../vite/stripProvenance.ts) runs only during a production build, in **both** the main and worker
module graphs. It parses TypeScript object literals, empties `sources` and `estimated`, and clears
`estimatedReason`. Standalone citation records (`url` + `title`) and helper estimate `reason` strings are emptied
too, so private SFU/source helpers do not retain their payloads. Dead citation constants then disappear through tree
shaking (some unused helper URL arguments may survive conservative optimization). It does not rewrite physical
numbers or other data. Modifier `note` and unlock descriptions stay because they are user-visible labels/warnings.
Its pure data equivalent is [`strip(full)`](../src/sim/hw/provenance.ts).

An explicit `provenanceStripped: true` marker distinguishes trusted build-generated presets from unsourced custom
specs. The shared validators still check **all physical numbers and structural constraints**; only citation coverage
checks wait for the full data. Source tests, including `presets.test.ts`, run on the original files and still enforce
cite-every-number. The build integration test compares actual bundled exports with `strip(full)`, including generated
host links, and compares the worker runner's complete output for default and PD-disaggregated + offload scenarios.

Opening Hardware → Advanced dynamically imports `virtual:hardware-provenance` through
[`loadHardwareProvenance`](../src/ui/state/provenance.ts). The plugin serves a separate full-query preset graph,
including its original factory functions, so the editor rebuilds hardware with full citations **before** validating
it. Unlock modifiers, generated host links and user-edit estimates therefore retain their original provenance;
uncovered editor fields are still errors. Requests are shared, and a failed request offers a retry. Simulation never
waits for the sidecar. A generated tiny set of community-option flags preserves the community badge without loading
citations. Dev/Vitest do not strip anything; the same on-demand editor path remains usable there.

The sidecar deliberately includes the full preset data and necessary resolver code, rather than a hand-maintained
citation-only file. This duplicates some numeric data in the lazy chunk, but avoids splitting any reviewed presets
or changing helper-generated provenance. It is not preloaded on the default path; its bytes count toward `total`.

## Release

```sh
git tag -f release-pages <commit>
git push -f origin release-pages
```

Or run the workflow by hand from the Actions tab.

## One-time setup

1. Create an API token (Cloudflare dashboard → My Profile → API Tokens → Create Token → Custom token) with:
   - **Account → Workers Scripts → Edit** (scoped to the account)
   - **Zone → Workers Routes → Edit** (scoped to the zone `cro.sh`)
   - **Zone → Zone → Read** (scoped to the zone `cro.sh`)

   That is the minimum for `wrangler deploy` with an explicit account ID and a route: Workers Scripts Edit deploys
   the Worker and its assets, Zone Read lets Wrangler resolve `zone_name: "cro.sh"` to its zone ID
   ([List Zones](https://developers.cloudflare.com/api/resources/zones/methods/list/)), and Workers Routes Edit
   attaches the route
   ([Workers roles and permissions](https://developers.cloudflare.com/workers/authorization/workers/#routes-and-custom-domains),
   [API token permissions](https://developers.cloudflare.com/fundamentals/api/reference/permissions/)). Because
   `CLOUDFLARE_ACCOUNT_ID` is set, Wrangler does not need to look up memberships or user details, so no User or
   Account Settings read permission is required. (Cloudflare's "Edit Cloudflare Workers" template adds those and more;
   it works too but is broader.)

2. In the GitHub repository (Settings → Secrets and variables → Actions), add the secrets `CLOUDFLARE_API_TOKEN` and
   `CLOUDFLARE_ACCOUNT_ID`.
3. In DNS for `cro.sh`, the record for the root (the CNAME to GitHub Pages) must be **proxied** (orange cloud), or
   requests never reach the Worker route.

## Updating wrangler

The version is pinned twice: the `wrangler` devDependency in package.json and `wranglerVersion` in the workflow.
Change both together.
