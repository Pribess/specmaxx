# Deploying

The app is static. [.github/workflows/deploy-pages.yml](../.github/workflows/deploy-pages.yml) builds it
(typecheck, lint, unit tests, build; the Playwright e2e runs locally in `bun run check` only) and uploads `dist/` to
the Cloudflare Pages project `specmaxx` as a production deployment. It runs on a push of the tag
`release-pages`, or by hand (Actions → Deploy to Cloudflare Pages → Run workflow).

The build uses relative asset URLs (`base: './'` in vite.config.ts), so the same `dist/` works at
https://specmaxx.pages.dev/ and under https://cro.sh/specmaxx/.

## Release

```sh
git tag -f release-pages <commit>
git push -f origin release-pages
```

## One-time setup

1. Create the Pages project (Direct Upload):
   `bunx wrangler pages project create specmaxx --production-branch=main`
2. Create an API token (My Profile → API Tokens → Custom token) with **Account → Cloudflare Pages → Edit**.
3. In the GitHub repository (Settings → Secrets and variables → Actions), add the secrets
   `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`. Optional: the repository variable `CF_PAGES_PROJECT` to
   deploy to a project not named `specmaxx`.

## Serving at cro.sh/specmaxx

A redirect rule alone would send visitors to pages.dev; to keep the cro.sh URL, proxy with a Worker on the routes
`cro.sh/specmaxx` and `cro.sh/specmaxx/*` that strips the prefix:

- `/specmaxx` without the trailing slash must 301 to `/specmaxx/`, or the page's relative URLs resolve against `/`.
- Pages redirects `/about.html` to `/about` with a root-relative `Location`; the Worker puts the prefix back.

```js
const PREFIX = '/specmaxx';
const ORIGIN = 'https://specmaxx.pages.dev';

export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === PREFIX)
      return Response.redirect(`${url.origin}${PREFIX}/${url.search}`, 301);
    const upstream = new URL(url.pathname.slice(PREFIX.length) + url.search, ORIGIN);
    const response = await fetch(new Request(upstream, request), { redirect: 'manual' });
    let location = response.headers.get('Location');
    if (location?.startsWith(ORIGIN)) location = location.slice(ORIGIN.length);
    if (!location?.startsWith('/')) return response;
    const headers = new Headers(response.headers);
    headers.set('Location', PREFIX + location);
    return new Response(response.body, { status: response.status, headers });
  },
};
```
