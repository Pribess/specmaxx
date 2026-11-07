import { Marked } from 'marked';

/** GitHub-style fragment for a heading; duplicate allocation happens per document. */
export function githubSlug(text: string): string {
  return text.toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
}

/** Render trusted repository Markdown, never model metadata or other user input. */
export function renderMarkdown(source: string): string {
  const used = new Set<string>();
  const markdown = new Marked({
    gfm: true,
    renderer: {
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
