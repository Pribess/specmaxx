import simulationModel from '../../docs/simulation-model.md?raw';
import { renderMarkdown } from './markdown';

const optionalDocs = import.meta.glob<string>('../../docs/sanity-check.md', {
  query: '?raw', import: 'default', eager: true,
});
const simulationHtml = renderMarkdown(simulationModel);
const sanitySource = optionalDocs['../../docs/sanity-check.md'];

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
        {/* Only trusted, build-time repository Markdown enters these HTML containers. */}
        <article className="about-prose" id="simulation-model" aria-label="Simulation model"
          dangerouslySetInnerHTML={{ __html: simulationHtml }} />
        {sanitySource && <article className="about-prose" id="sanity-check" aria-label="Sanity checks"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(sanitySource) }} />}
        <footer className="about-footer"><a href={estimateHref()}>Back to the estimate</a></footer>
      </main>
    </>
  );
}
