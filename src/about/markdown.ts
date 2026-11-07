import { Marked, Renderer } from 'marked';

/** GitHub-style fragment for a heading; duplicate allocation happens per document. */
export function githubSlug(text: string): string {
  return text.toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
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
        const documentLink = /^(?:\.\/)?(simulation-model|sanity-check)\.md(#.*)?$/.exec(token.href);
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
        const header = token.header.map(cell => this.tablecell(cell)).join('');
        const rows = token.rows.map(row =>
          `<tr>${row.map(cell => this.tablecell(cell)).join('')}</tr>`
        ).join('');
        return `<div class="table-scroll" tabindex="0" role="region" aria-label="Data table"><table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table></div>`;
      },
    },
  });
  return markdown.parse(source, { async: false });
}
