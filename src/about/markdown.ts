// Build-time only: vite/prerenderDocs.ts and the tests import this module; the About page never does (it ships
// prerendered HTML, so `marked` stays out of its bundle). The runtime half is src/about/segments.ts.
import { Marked, Renderer } from 'marked';

/** GitHub-style fragment for a heading; duplicate allocation happens per document. */
export function githubSlug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s_-]/gu, '')
    .replace(/\s/g, '-');
}

/** Render trusted repository Markdown, resolving links only to documents present on this page. */
export function renderMarkdown(
  source: string,
  renderedDocs: readonly string[] = ['simulation-model.md'],
): string {
  const used = new Set<string>();
  const defaultRenderer = new Renderer();
  const markdown = new Marked({
    gfm: true,
    renderer: {
      link(token) {
        const documentLink = /^(?:\.\/)?(simulation-model|sanity-check)\.md(#.*)?$/.exec(
          token.href,
        );
        if (documentLink && renderedDocs.includes(`${documentLink[1]}.md`)) {
          const href = documentLink[2] || `#${documentLink[1]}`;
          return defaultRenderer.link.call(this, { ...token, href });
        }
        if (token.href.startsWith('#') || /^(?:https?:|mailto:|\/\/)/i.test(token.href)) {
          return false; // Keep the default renderer's attribute escaping for live web links.
        }
        // No repository remote is available: preserve the label, not a broken relative URL.
        return this.parser.parseInline(token.tokens);
      },
      heading({ tokens, depth, text }) {
        const base = githubSlug(text);
        let id = base;
        let suffix = 0;
        while (used.has(id)) id = `${base}-${++suffix}`;
        used.add(id);
        return `<h${depth} id="${id}">${this.parser.parseInline(tokens)}</h${depth}>\n`;
      },
      table(token) {
        const header = token.header.map((cell) => this.tablecell(cell)).join('');
        const rows = token.rows
          .map((row) => `<tr>${row.map((cell) => this.tablecell(cell)).join('')}</tr>`)
          .join('');
        // AboutPage wraps each table in a themed horizontal ScrollArea (see splitTables).
        return `<table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table>`;
      },
    },
  });
  return markdown.parse(source, { async: false });
}

/** One entry of a table of contents: a heading's fragment id and its plain-text title. */
export interface TocEntry {
  id: string;
  title: string;
}

/** Decodes the few entities marked emits in heading text, so a title can be shown as plain text. */
function decodeEntities(text: string): string {
  return text
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&');
}

/** The `## ` sections of rendered Markdown, in document order (the About page's table of contents). */
export function tableOfContents(html: string): TocEntry[] {
  return Array.from(html.matchAll(/<h2 id="([^"]+)">(.*?)<\/h2>/g), (match) => ({
    id: match[1],
    title: decodeEntities(match[2].replace(/<[^>]+>/g, '')),
  }));
}

/**
 * Every element id in rendered Markdown (headings, plus any raw HTML ids), in document order. The About page
 * ships these per lazily loaded document, so a URL hash tells which document to fetch before it is loaded.
 */
export function elementIds(html: string): string[] {
  return Array.from(html.matchAll(/<[a-zA-Z][^>]*?\sid="([^"]+)"/g), (match) => match[1]);
}
