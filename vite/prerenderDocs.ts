import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { Plugin } from 'vite';
import {
  elementIds,
  renderMarkdown,
  tableOfContents,
  type TocEntry,
} from '../src/about/markdown.ts';

/** The virtual module the About page imports; its types are in src/about-docs.d.ts. */
const VIRTUAL = 'virtual:about-docs';
/**
 * Extension of a prerendered document file. Not `.html`: Cloudflare's `html_handling: auto-trailing-slash` (see
 * wrangler.jsonc) answers `/x.html` with a 307 to `/x`, an extra round trip; `.htm` is served directly as text/html.
 */
const DOC_EXT = '.htm';
/** Dev-server path prefix that serves the lazily loaded documents (production emits hashed assets instead). */
const DEV_PREFIX = '/__about-docs/';

/** One document rendered on demand: fetched when the reference opens or a URL hash points into it. */
export interface LazyDocSpec {
  /** File name stem, also the id of its `<article>` and of cross-document links (`#simulation-model`). */
  slug: string;
  /** Accessible name of its article and its label in the reference navigation. */
  label: string;
  /** Path relative to the project root. */
  file: string;
  /** Optional documents are left out (with their links rendered as text) when the file is missing. */
  optional: boolean;
}

/** The method, rendered eagerly into the page. */
const METHOD_FILE = 'docs/modeling-method.md';
/** The full reference, in page order; sanity-check is optional, as it is not always in the repository. */
export const LAZY_DOCS: readonly LazyDocSpec[] = [
  {
    slug: 'simulation-model',
    label: 'Simulation model',
    file: 'docs/simulation-model.md',
    optional: false,
  },
  { slug: 'sanity-check', label: 'Sanity checks', file: 'docs/sanity-check.md', optional: true },
];

/** One lazy document after prerendering. */
export interface PrerenderedLazyDoc {
  slug: string;
  label: string;
  /** The document's HTML, as the old runtime `renderMarkdown` produced it. */
  html: string;
  /** Every element id inside it, so a hash maps to this document before it is fetched. */
  ids: string[];
}

/** Everything the About page needs from the docs, before URLs are assigned. */
export interface PrerenderedDocs {
  methodHtml: string;
  /** The method's `## ` sections, for the table of contents. */
  methodSections: TocEntry[];
  lazyDocs: PrerenderedLazyDoc[];
}

/**
 * Renders the method and the reference documents with the same options the page used at runtime: links to
 * a reference document resolve only when that document is on the page, heading ids are deduplicated per doc.
 * `sources` maps a file (relative to the root) to its Markdown; a missing optional file is skipped.
 */
export function prerenderDocs(
  sources: Readonly<Record<string, string | undefined>>,
): PrerenderedDocs {
  const present = LAZY_DOCS.filter((doc) => {
    if (sources[doc.file] !== undefined) return true;
    if (!doc.optional) throw new Error(`Missing ${doc.file}`);
    return false;
  });
  const renderedDocs = present.map((doc) => `${doc.slug}.md`);
  const methodSource = sources[METHOD_FILE];
  if (methodSource === undefined) throw new Error(`Missing ${METHOD_FILE}`);
  const methodHtml = renderMarkdown(methodSource, renderedDocs);
  return {
    methodHtml,
    methodSections: tableOfContents(methodHtml),
    lazyDocs: present.map((doc) => {
      const html = renderMarkdown(sources[doc.file]!, renderedDocs);
      return { slug: doc.slug, label: doc.label, html, ids: elementIds(html) };
    }),
  };
}

/** Reads the method and every reference document present under `root`. */
function readSources(root: string): Record<string, string | undefined> {
  const sources: Record<string, string | undefined> = {};
  for (const file of [METHOD_FILE, ...LAZY_DOCS.map((doc) => doc.file)]) {
    const path = resolve(root, file);
    if (existsSync(path)) sources[file] = readFileSync(path, 'utf8');
  }
  return sources;
}

/** One reference document as the preload script sees it: its URL and the hashes that point into it. */
export interface PreloadTarget {
  url: string;
  /** Fragment ids that load this document: its slug and the ids inside it, minus ids the eager page has too. */
  ids: readonly string[];
}

/**
 * The hashes that load each reference document before the page's JS runs: the slug and every id inside it, minus
 * ids the eager page also has (the page resolves those without loading anything, so preloading would waste bytes).
 */
export function preloadTargets(
  docs: PrerenderedDocs,
  urlOf: (slug: string) => string,
): PreloadTarget[] {
  const eager = new Set(elementIds(docs.methodHtml));
  return docs.lazyDocs.map((doc) => ({
    url: urlOf(doc.slug),
    ids: [doc.slug, ...doc.ids].filter((id) => !eager.has(id)),
  }));
}

/**
 * A tiny classic script for about.html's `<head>`: when the URL hash names an id inside a reference document, it
 * adds `<link rel="preload" as="fetch" crossorigin="anonymous">` for that document, so the download starts while
 * the page's JS is still loading. `crossorigin="anonymous"` gives the preload the same request mode (cors) and
 * credentials mode (same-origin) as the page's plain `fetch(url)`, so the fetch reuses the preloaded response.
 * Ids are joined with spaces (ids never contain whitespace; checked here), which is smaller than a JSON array.
 */
export function hashPreloadScript(targets: readonly PreloadTarget[]): string {
  for (const target of targets)
    for (const id of target.ids)
      if (/\s/.test(id)) throw new Error(`Whitespace in element id "${id}"`);
  const data = JSON.stringify(targets.map((t) => [t.url, t.ids.join(' ')])).replace(
    /</g,
    '\\u003c',
  );
  return (
    '(function(){var h;try{h=decodeURIComponent(location.hash.slice(1))}catch(e){return}' +
    `if(!h||/\\s/.test(h))return;var d=${data};` +
    'for(var i=0;i<d.length;i++)if((" "+d[i][1]+" ").indexOf(" "+h+" ")>=0){' +
    'var l=document.createElement("link"),a={rel:"preload",as:"fetch",crossorigin:"anonymous",href:d[i][0]};' +
    'for(var k in a)l.setAttribute(k,a[k]);document.head.appendChild(l);return}})()'
  );
}

/**
 * Puts the preload script right after `<meta charset>`: after it, so the charset stays in the first 1024 bytes;
 * before the stylesheets, because a classic inline script waits for earlier stylesheets before it runs.
 */
export function injectPreloadScript(html: string, script: string): string {
  const charset = /<meta charset="[^"]*"\s*\/?>/i.exec(html);
  if (!charset) throw new Error('about.html has no <meta charset> to put the preload script after');
  const at = charset.index + charset[0].length;
  return `${html.slice(0, at)}\n    <script>${script}</script>${html.slice(at)}`;
}

/**
 * Prerenders the About page's Markdown with `marked` at build time, so the page ships HTML and no parser.
 * `virtual:about-docs` exports the method's HTML and table of contents (inlined in the About chunk), and per
 * reference document its label, URL and id list. A build emits each reference document as a hashed asset
 * (`assets/simulation-model-<hash>.htm`); the dev server (and Vitest) serves it from `/__about-docs/`. about.html
 * gets an inline script that preloads the document a URL hash points into (see `hashPreloadScript`).
 */
export function prerenderAboutDocs(): Plugin {
  let building = false;
  let root = '';
  let base = '/';
  /** Dev-server URL of a reference document (a relative `base` is served at the root in dev). */
  const devUrl = (slug: string) =>
    `${(base.startsWith('/') ? base : '/').replace(/\/$/, '')}${DEV_PREFIX}${slug}${DOC_EXT}`;
  return {
    name: 'prerender-about-docs',
    configResolved(config) {
      building = config.command === 'build';
      root = config.root;
      base = config.base;
    },
    transformIndexHtml: {
      order: 'post',
      handler(html, context) {
        if (context.path !== '/about.html') return html;
        const docs = prerenderDocs(readSources(root));
        const urlOf = (slug: string) => {
          if (!context.bundle) return devUrl(slug);
          const name = new RegExp(`^assets/${slug}-[\\w-]{8}\\${DOC_EXT}$`);
          const asset = Object.values(context.bundle).find((file) => name.test(file.fileName));
          if (!asset) throw new Error(`Missing prerendered ${slug}${DOC_EXT} asset`);
          return `./${asset.fileName}`;
        };
        return injectPreloadScript(html, hashPreloadScript(preloadTargets(docs, urlOf)));
      },
    },
    configureServer(server) {
      // Rendered per request, so an edited doc shows on reload; the virtual module tracks the files too.
      server.middlewares.use((request, response, next) => {
        const path = request.url?.split('?')[0] ?? '';
        const at = path.indexOf(DEV_PREFIX);
        if (at === -1 || !path.endsWith(DOC_EXT)) return next();
        const slug = path.slice(at + DEV_PREFIX.length, -DOC_EXT.length);
        const doc = prerenderDocs(readSources(root)).lazyDocs.find((d) => d.slug === slug);
        if (!doc) return next();
        response.setHeader('Content-Type', 'text/html; charset=utf-8');
        response.setHeader('Cache-Control', 'no-cache');
        response.end(doc.html);
      });
    },
    resolveId(source) {
      return source === VIRTUAL ? `\0${VIRTUAL}` : null;
    },
    load(id) {
      if (id !== `\0${VIRTUAL}`) return null;
      for (const file of [METHOD_FILE, ...LAZY_DOCS.map((doc) => doc.file)])
        this.addWatchFile(resolve(root, file));
      const docs = prerenderDocs(readSources(root));
      const lazy = docs.lazyDocs.map((doc) => {
        let url: string;
        if (building) {
          const ref = this.emitFile({
            type: 'asset',
            name: `${doc.slug}${DOC_EXT}`,
            source: doc.html,
          });
          url = `import.meta.ROLLUP_FILE_URL_${ref}`;
        } else {
          url = JSON.stringify(devUrl(doc.slug));
        }
        return `{ slug: ${JSON.stringify(doc.slug)}, label: ${JSON.stringify(doc.label)}, url: ${url}, ids: ${JSON.stringify(doc.ids)} }`;
      });
      return [
        `export const methodHtml = ${JSON.stringify(docs.methodHtml)};`,
        `export const methodSections = ${JSON.stringify(docs.methodSections)};`,
        `export const referenceDocs = [${lazy.join(', ')}];`,
      ].join('\n');
    },
  };
}
