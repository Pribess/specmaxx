import { useEffect, useState } from 'react';
import modelingMethod from '../../docs/modeling-method.md?raw';
import simulationModel from '../../docs/simulation-model.md?raw';
import { ScrollArea } from '../ui/components/ScrollArea';
import { renderMarkdown, splitTables } from './markdown';

const optionalDocs = import.meta.glob<string>('../../docs/sanity-check.md', {
  query: '?raw',
  import: 'default',
  eager: true,
});
const sanitySource = optionalDocs['../../docs/sanity-check.md'];
const renderedDocs = sanitySource
  ? ['simulation-model.md', 'sanity-check.md']
  : ['simulation-model.md'];
const simulationHtml = renderMarkdown(simulationModel, renderedDocs);
const methodHtml = renderMarkdown(modelingMethod, renderedDocs);
const sanityHtml = sanitySource ? renderMarkdown(sanitySource, renderedDocs) : '';
/** The method sections, for the table of contents (`## ` headings of docs/modeling-method.md). */
const methodSections = Array.from(
  methodHtml.matchAll(/<h2 id="([^"]+)">(.*?)<\/h2>/g),
  (match) => ({ id: match[1], title: match[2].replace(/<[^>]+>/g, '') }),
);

/** Id of the collapsible block holding the full reference. */
const REFERENCE_ID = 'full-reference';

/** Restore the estimate's query without trusting a cross-origin referrer or stored URL. */
export function estimateHref(): string {
  const current = new URL(window.location.href);
  if (current.search) return `./${current.search}`;
  try {
    const query = sessionStorage.getItem('specmaxx:last-query');
    if (query?.startsWith('?')) return `./${query}`;
  } catch {
    // Storage can be disabled; same-origin navigation still restores the last estimate.
  }
  try {
    const referrer = new URL(document.referrer);
    if (referrer.origin === current.origin) return `./${referrer.search}`;
  } catch {
    // Direct visits have no referrer.
  }
  return './';
}

/** The estimator's plain-language method, with the full design reference collapsed below it. */
export function AboutPage() {
  const [referenceOpen, setReferenceOpen] = useState(false);

  // A hash that points into the (collapsed) reference opens it, then scrolls to the target.
  useEffect(() => {
    const reveal = () => {
      const id = decodeURIComponent(window.location.hash.slice(1));
      if (!id) return;
      const target = document.getElementById(id);
      if (!target) return;
      const inReference = target.closest(`#${REFERENCE_ID}`) !== null;
      if (inReference) setReferenceOpen(true);
      // The next frame has the reference laid out, so the heading has a position to scroll to.
      requestAnimationFrame(() => target.scrollIntoView?.());
    };
    reveal();
    window.addEventListener('hashchange', reveal);
    return () => window.removeEventListener('hashchange', reveal);
  }, []);

  return (
    <>
      <header className="about-header">
        <a className="about-wordmark" href={estimateHref()}>
          specmaxx
        </a>
        <a href={estimateHref()}>Back to the estimate</a>
      </header>
      <main className="about-main">
        <section className="about-intro" aria-labelledby="about-title">
          <h1 id="about-title">How the estimate works</h1>
          <p>Hardware and math set a ceiling. The inference engine decides how close you get.</p>
          <nav aria-label="Documents">
            <a href="#modeling-method">Modeling method</a>
            <a href={`#${REFERENCE_ID}`}>Full reference</a>
          </nav>
          <nav className="about-toc" aria-label="Modeling method contents">
            <ol>
              {methodSections.map((section) => (
                <li key={section.id}>
                  <a href={`#${section.id}`}>{section.title}</a>
                </li>
              ))}
            </ol>
          </nav>
        </section>
        <Prose id="modeling-method" label="Modeling method" html={methodHtml} />
        <details
          className="about-reference"
          id={REFERENCE_ID}
          open={referenceOpen}
          onToggle={(event) => setReferenceOpen(event.currentTarget.open)}
        >
          <summary>Full reference</summary>
          <nav aria-label="Reference documents">
            <a href="#simulation-model">Simulation model</a>
            {sanitySource && <a href="#sanity-check">Sanity checks</a>}
          </nav>
          <Prose id="simulation-model" label="Simulation model" html={simulationHtml} />
          {sanitySource && <Prose id="sanity-check" label="Sanity checks" html={sanityHtml} />}
        </details>
        <footer className="about-footer">
          <a href={estimateHref()}>Back to the estimate</a>
        </footer>
      </main>
    </>
  );
}

/**
 * One rendered document. Only trusted, build-time repository Markdown enters these HTML containers. Each
 * table gets a keyboard-focusable horizontal ScrollArea (themed scrollbar); the prose between tables is
 * inserted as-is, in `display: contents` wrappers so the article's layout is unchanged.
 */
function Prose({ id, label, html }: { id: string; label: string; html: string }) {
  return (
    <article className="about-prose" id={id} aria-label={label}>
      {splitTables(html).map((segment, n) =>
        segment.kind === 'table' ? (
          <ScrollArea
            key={n}
            axis="horizontal"
            className="table-scroll"
            viewportProps={{ tabIndex: 0, role: 'region', 'aria-label': 'Data table' }}
          >
            <div dangerouslySetInnerHTML={{ __html: segment.html }} />
          </ScrollArea>
        ) : (
          <div
            key={n}
            className="prose-segment"
            dangerouslySetInnerHTML={{ __html: segment.html }}
          />
        ),
      )}
    </article>
  );
}
