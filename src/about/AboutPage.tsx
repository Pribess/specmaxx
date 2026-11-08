import simulationModel from '../../docs/simulation-model.md?raw';
import { ScrollArea } from '../ui/components/ScrollArea';
import { renderMarkdown, splitTables } from './markdown';

const optionalDocs = import.meta.glob<string>('../../docs/sanity-check.md', {
  query: '?raw', import: 'default', eager: true,
});
const sanitySource = optionalDocs['../../docs/sanity-check.md'];
const renderedDocs = sanitySource
  ? ['simulation-model.md', 'sanity-check.md']
  : ['simulation-model.md'];
const simulationHtml = renderMarkdown(simulationModel, renderedDocs);

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

/** The estimator's readable, bundled simulation design reference. */
export function AboutPage() {
  return (
    <>
      <header className="about-header">
        <a className="about-wordmark" href={estimateHref()}>specmaxx</a>
        <a href={estimateHref()}>Back to the estimate</a>
      </header>
      <main className="about-main">
        <section className="about-intro" aria-labelledby="about-title">
          <h1 id="about-title">How the estimate works</h1>
          <p>Hardware and math set a ceiling. The inference engine decides how close you get.</p>
          <p>These are theoretical maximums, not benchmark predictions. The reference below spells out the operations, bandwidth limits, memory accounting and deliberate exclusions behind each number.</p>
          <nav aria-label="Documents">
            <a href="#simulation-model">Simulation model</a>
            {sanitySource && <a href="#sanity-check">Sanity checks</a>}
          </nav>
        </section>
        <Prose id="simulation-model" label="Simulation model" html={simulationHtml} />
        {sanitySource && (
          <Prose id="sanity-check" label="Sanity checks"
            html={renderMarkdown(sanitySource, renderedDocs)} />
        )}
        <footer className="about-footer"><a href={estimateHref()}>Back to the estimate</a></footer>
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
          <ScrollArea key={n} axis="horizontal" className="table-scroll"
            viewportProps={{ tabIndex: 0, role: 'region', 'aria-label': 'Data table' }}>
            <div dangerouslySetInnerHTML={{ __html: segment.html }} />
          </ScrollArea>
        ) : (
          <div key={n} className="prose-segment" dangerouslySetInnerHTML={{ __html: segment.html }} />
        ),
      )}
    </article>
  );
}
