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
