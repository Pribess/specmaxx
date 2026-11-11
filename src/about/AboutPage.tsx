import { useEffect, useRef, useState, type ReactNode } from 'react';
import { methodHtml, methodSections, referenceDocs } from 'virtual:about-docs';
import { ScrollArea } from '../ui/components/ScrollArea';
import { splitTables, type HtmlSegment } from './segments';

/** Id of the collapsible block holding the full reference. */
const REFERENCE_ID = 'full-reference';

/** The method, split once at module load (it is prerendered HTML, so this is a cheap regex pass). */
const methodSegments = splitTables(methodHtml);

/** Load state of one reference document; absent from the map while it is loading (or not yet asked for). */
type DocState = { status: 'ready'; segments: HtmlSegment[] } | { status: 'error' };

/** The reference document whose prerendered HTML holds `id`, known from the build's id lists. */
function referenceDocOf(id: string) {
  return referenceDocs.find((doc) => doc.slug === id || doc.ids.includes(id));
}

/** What a reader does to move the page; any of them cancels a pending re-scroll. */
const READER_INPUT = ['wheel', 'touchstart', 'keydown', 'pointerdown'] as const;

/** The element id the URL hash names now, decoded; null for no hash or a malformed escape (which names nothing). */
function currentHashId(): string | null {
  try {
    return decodeURIComponent(window.location.hash.slice(1)) || null;
  } catch {
    return null;
  }
}

/**
 * Scrolls to the hash target `id` in the next frame, once the reference is laid out. Web fonts that the new text
 * starts loading reflow everything above the target when they swap in, so it scrolls again once they are ready,
 * unless the reader has scrolled, touched or typed since. Each scroll first checks that `id` is still the URL's
 * hash: a reader who has followed another link meanwhile stays where that link took them.
 */
function scrollToTarget(target: Element, id: string) {
  requestAnimationFrame(() => {
    if (currentHashId() !== id) return;
    target.scrollIntoView?.();
    const fonts = document.fonts as FontFaceSet | undefined;
    if (!fonts || fonts.status === 'loaded') return;
    let moved = false;
    const stop = () => {
      moved = true;
    };
    for (const type of READER_INPUT) window.addEventListener(type, stop, { passive: true });
    void fonts.ready.then(() => {
      for (const type of READER_INPUT) window.removeEventListener(type, stop);
      if (!moved && currentHashId() === id) target.scrollIntoView?.();
    });
  });
}

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

/**
 * The estimator's plain-language method, with the full design reference collapsed below it. The reference
 * documents are fetched when the reference opens, or when the URL hash points into one of them.
 */
export function AboutPage() {
  const [referenceOpen, setReferenceOpen] = useState(false);
  const [docs, setDocs] = useState<Record<string, DocState>>({});
  /** Bumped by "Retry", to fetch failed documents again. */
  const [attempt, setAttempt] = useState(0);
  /** One request per document, shared by StrictMode's repeated effects; failed ones are dropped. */
  const requests = useRef(new Map<string, Promise<string>>());
  /** A hash target inside a reference document that is not rendered yet: scrolled to once it is. */
  const pendingTarget = useRef<string | null>(null);
  /**
   * The reference document a URL hash points into: it is fetched alone first (about.html's head script has
   * already preloaded it), and the others only once it has rendered or failed, so they do not compete for bandwidth.
   */
  const [first, setFirst] = useState<string | null>(null);
  const firstSettled = first === null || docs[first] !== undefined;

  // A hash that points into the (collapsed) reference opens it, then scrolls to the target.
  useEffect(() => {
    const reveal = () => {
      // Every hash replaces the previous one's pending target, also one that resolves at once or names nothing.
      pendingTarget.current = null;
      const id = currentHashId();
      if (!id) return;
      const target = document.getElementById(id);
      const doc = referenceDocOf(id);
      if (target) {
        if (target.closest(`#${REFERENCE_ID}`) !== null) {
          if (doc) setFirst(doc.slug);
          setReferenceOpen(true);
        }
        scrollToTarget(target, id);
      } else if (doc) {
        pendingTarget.current = id;
        setFirst(doc.slug);
        setReferenceOpen(true);
      }
    };
    reveal();
    window.addEventListener('hashchange', reveal);
    return () => window.removeEventListener('hashchange', reveal);
  }, []);

  // Opening the reference fetches its documents (once each; failed ones again after "Retry"), the one a hash points
  // into first. A plain same-origin fetch (mode cors, credentials same-origin) reuses the head script's preload.
  useEffect(() => {
    if (!referenceOpen) return;
    let live = true;
    const wave = firstSettled ? referenceDocs : referenceDocs.filter((doc) => doc.slug === first);
    for (const doc of wave) {
      let request = requests.current.get(doc.slug);
      if (!request) {
        request = fetch(doc.url).then((response) => {
          if (!response.ok) throw new Error(`${response.status} ${doc.url}`);
          return response.text();
        });
        requests.current.set(doc.slug, request);
      }
      request.then(
        (html) => {
          if (live)
            setDocs((now) =>
              now[doc.slug]?.status === 'ready'
                ? now
                : { ...now, [doc.slug]: { status: 'ready', segments: splitTables(html) } },
            );
        },
        () => {
          requests.current.delete(doc.slug);
          if (live) setDocs((now) => ({ ...now, [doc.slug]: { status: 'error' } }));
        },
      );
    }
    return () => {
      live = false;
    };
  }, [referenceOpen, attempt, first, firstSettled]);

  // Once the document holding a pending hash target renders, scroll to the target, if it is still the hash.
  useEffect(() => {
    const id = pendingTarget.current;
    if (!id) return;
    const target = document.getElementById(id);
    if (!target) return;
    pendingTarget.current = null;
    scrollToTarget(target, id);
  }, [docs]);

  const retry = () => {
    setDocs((now) =>
      Object.fromEntries(Object.entries(now).filter(([, state]) => state.status !== 'error')),
    );
    setAttempt((n) => n + 1);
  };

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
        <Prose id="modeling-method" label="Modeling method" segments={methodSegments} />
        <details
          className="about-reference"
          id={REFERENCE_ID}
          open={referenceOpen}
          onToggle={(event) => setReferenceOpen(event.currentTarget.open)}
        >
          <summary>Full reference</summary>
          <nav aria-label="Reference documents">
            {referenceDocs.map((doc) => (
              <a key={doc.slug} href={`#${doc.slug}`}>
                {doc.label}
              </a>
            ))}
          </nav>
          {referenceDocs.map((doc) => {
            const state = docs[doc.slug];
            return (
              <Prose
                key={doc.slug}
                id={doc.slug}
                label={doc.label}
                segments={state?.status === 'ready' ? state.segments : []}
              >
                {referenceOpen && !state && (
                  <p className="about-status" role="status">
                    Loading…
                  </p>
                )}
                {state?.status === 'error' && (
                  <p className="about-status" role="alert">
                    Could not load.{' '}
                    <button type="button" onClick={retry}>
                      Retry
                    </button>
                  </p>
                )}
              </Prose>
            );
          })}
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
 * inserted as-is, in `display: contents` wrappers so the article's layout is unchanged. `children` is a
 * loading or error state shown in place of a document that is not rendered yet.
 */
function Prose({
  id,
  label,
  segments,
  children,
}: {
  id: string;
  label: string;
  segments: readonly HtmlSegment[];
  children?: ReactNode;
}) {
  return (
    <article className="about-prose" id={id} aria-label={label}>
      {children}
      {segments.map((segment, n) =>
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
