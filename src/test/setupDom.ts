/**
 * Browser APIs jsdom lacks that the Radix primitives (Popover, ScrollArea) and cmdk call. Runs before every
 * test file (vite.config.ts `test.setupFiles`); a no-op in the node environment.
 */
if (typeof window !== 'undefined') {
  if (typeof window.ResizeObserver === 'undefined') {
    /** Never reports a size change: jsdom does no layout, so there is nothing to observe. */
    class ResizeObserverStub {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    window.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
  }
  // cmdk scrolls its active item into view; jsdom has no scrolling.
  Element.prototype.scrollIntoView ??= function scrollIntoView() {};
  Element.prototype.hasPointerCapture ??= () => false;
  Element.prototype.releasePointerCapture ??= () => {};
}
