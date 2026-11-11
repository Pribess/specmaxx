import { describe, expect, it } from 'vitest';
import modelingMethod from '../../docs/modeling-method.md?raw';
import simulationModel from '../../docs/simulation-model.md?raw';
import { methodHtml, methodSections, referenceDocs } from 'virtual:about-docs';
import { LAZY_DOCS, prerenderDocs } from '../../vite/prerenderDocs';
import { elementIds, renderMarkdown, tableOfContents } from './markdown';

const sanitySource = import.meta.glob<string>('../../docs/sanity-check.md', {
  query: '?raw',
  import: 'default',
  eager: true,
})['../../docs/sanity-check.md'];
const allSources = {
  'docs/modeling-method.md': modelingMethod,
  'docs/simulation-model.md': simulationModel,
  'docs/sanity-check.md': sanitySource,
};
/** The documents the page linked to before prerendering (sanity-check only when present). */
const renderedDocs = sanitySource
  ? ['simulation-model.md', 'sanity-check.md']
  : ['simulation-model.md'];

describe('prerendered About documents', () => {
  it('match what the page rendered at runtime before, byte for byte', () => {
    const docs = prerenderDocs(allSources);
    expect(docs.methodHtml).toBe(renderMarkdown(modelingMethod, renderedDocs));
    expect(docs.lazyDocs[0].html).toBe(renderMarkdown(simulationModel, renderedDocs));
    if (sanitySource)
      expect(docs.lazyDocs[1].html).toBe(renderMarkdown(sanitySource, renderedDocs));
    expect(docs.lazyDocs.map((d) => d.slug)).toEqual(
      renderedDocs.map((file) => file.replace(/\.md$/, '')),
    );
  });

  it('are what the virtual module exports', () => {
    const docs = prerenderDocs(allSources);
    expect(methodHtml).toBe(docs.methodHtml);
    expect(methodSections).toEqual(docs.methodSections);
    expect(referenceDocs.map(({ slug, label, ids }) => ({ slug, label, ids }))).toEqual(
      docs.lazyDocs.map(({ slug, label, ids }) => ({ slug, label, ids })),
    );
    for (const doc of referenceDocs) expect(doc.url).toMatch(new RegExp(`/${doc.slug}\\.htm$`));
  });

  it('build the table of contents from the method sections, as plain text', () => {
    const { methodSections: toc } = prerenderDocs(allSources);
    expect(toc.length).toBeGreaterThanOrEqual(8);
    expect(toc[0]).toEqual({ id: 'what-the-numbers-mean', title: 'What the numbers mean' });
    expect(toc).toEqual(
      Array.from(modelingMethod.matchAll(/^## (.+)$/gm), (m, n) => ({
        id: toc[n].id,
        title: m[1].replace(/`/g, ''),
      })),
    );
    expect(tableOfContents('<h2 id="a-b">A <code>&amp;</code> B</h2>\n<h3 id="c">C</h3>')).toEqual([
      { id: 'a-b', title: 'A & B' },
    ]);
  });

  it('list every id of each reference document, including duplicate suffixes', () => {
    const docs = prerenderDocs(allSources);
    const simulation = docs.lazyDocs[0];
    expect(simulation.ids).toContain('4-hardware');
    expect(simulation.ids).toContain('memory-fit');
    expect(new Set(simulation.ids).size).toBe(simulation.ids.length);
    expect(simulation.ids.length).toBe(simulation.html.match(/<h[1-6] id="/g)?.length);
    const dup = renderMarkdown('# Same\n## Same\n### Same-1\n<div id="raw">x</div>');
    expect(elementIds(dup)).toEqual(['same', 'same-1', 'same-1-1', 'raw']);
  });

  it('leave the optional sanity checks out, with their links as text, when the file is missing', () => {
    const docs = prerenderDocs({
      'docs/modeling-method.md': '[Checks](sanity-check.md) [Model](simulation-model.md#gemm)',
      'docs/simulation-model.md': '# Model\n## GEMM\n[Back](sanity-check.md#method)',
    });
    expect(docs.lazyDocs.map((d) => d.slug)).toEqual(['simulation-model']);
    expect(docs.methodHtml).toBe('<p>Checks <a href="#gemm">Model</a></p>\n');
    expect(docs.lazyDocs[0].html).toContain('<p>Back</p>');
    expect(docs.lazyDocs[0].ids).toEqual(['model', 'gemm']);
  });

  it('refuse to build without a required document', () => {
    expect(() => prerenderDocs({ 'docs/modeling-method.md': '# M' })).toThrow(
      'Missing docs/simulation-model.md',
    );
    expect(LAZY_DOCS.filter((d) => !d.optional).map((d) => d.slug)).toEqual(['simulation-model']);
  });
});
