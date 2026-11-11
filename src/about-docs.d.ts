/** Prerendered About page documents, provided by vite/prerenderDocs.ts in dev, test and production builds. */
declare module 'virtual:about-docs' {
  /** docs/modeling-method.md as HTML (trusted repository content, rendered at build time). */
  export const methodHtml: string;
  /** The method's `## ` sections: heading id and plain-text title, for the table of contents. */
  export const methodSections: readonly { id: string; title: string }[];
  /** The full reference documents, fetched on demand, in page order. Optional ones may be absent. */
  export const referenceDocs: readonly {
    /** Id of the document's article, e.g. `simulation-model`. */
    slug: string;
    /** Accessible name of its article and its navigation label. */
    label: string;
    /** Absolute URL of its prerendered HTML (a hashed asset in production). */
    url: string;
    /** Every element id inside the document, so a URL hash tells which document to load. */
    ids: readonly string[];
  }[];
}
