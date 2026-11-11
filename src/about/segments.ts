/**
 * Runtime half of the About page's document handling: no Markdown parser here, so `marked` stays out of the
 * About bundle. The HTML it splits is prerendered at build time (vite/prerenderDocs.ts, src/about/markdown.ts).
 */

/** One piece of rendered Markdown: prose HTML, or one table to show in its own scroll region. */
export interface HtmlSegment {
  kind: 'html' | 'table';
  html: string;
}

/**
 * Splits rendered Markdown at its tables, so each table can sit in a horizontally scrolling ScrollArea
 * (wide tables scroll sideways with a themed scrollbar instead of a native one). GFM tables never nest.
 */
export function splitTables(html: string): HtmlSegment[] {
  const segments: HtmlSegment[] = [];
  let at = 0;
  for (const match of html.matchAll(/<table>[\s\S]*?<\/table>/g)) {
    if (match.index > at) segments.push({ kind: 'html', html: html.slice(at, match.index) });
    segments.push({ kind: 'table', html: match[0] });
    at = match.index + match[0].length;
  }
  if (at < html.length) segments.push({ kind: 'html', html: html.slice(at) });
  return segments;
}
