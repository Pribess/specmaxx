import { useLayoutEffect, useState, type CSSProperties, type RefObject } from 'react';

/** Which side of its anchor a floating panel opens on. */
export type FloatingSide = 'below' | 'above';

/** The anchor's box in viewport coordinates, in CSS px (as `getBoundingClientRect` gives it). */
export interface AnchorBox {
  top: number;
  left: number;
  bottom: number;
  width: number;
}

/** The result of {@link placeFloating}: a `position: fixed` box in viewport px. */
export interface FloatingPlacement {
  /** Top edge in viewport px. */
  top: number;
  /** Left edge in viewport px. */
  left: number;
  /** Width in px. */
  width: number;
  /** Maximum height in px; the panel scrolls beyond it. */
  maxHeight: number;
  /** The side actually used (it can differ from the preferred side when that side is too small). */
  side: FloatingSide;
}

/** Gap in px between the anchor and the panel. */
const GAP_PX = 6;
/** Margin in px kept between the panel and the viewport edges. */
const EDGE_PX = 8;

/**
 * Where a fixed-position panel goes next to its anchor, in viewport px.
 *
 * The panel opens on `preferred` unless that side has less room than the panel needs (its height, capped by
 * `maxHeight`) and the other side has more room. It is left-aligned with the anchor and clamped into the
 * viewport horizontally. All sizes are px; `height` is the panel's natural (unscrolled) height.
 */
export function placeFloating(
  anchor: AnchorBox,
  panel: { width: number; height: number },
  viewport: { width: number; height: number },
  opts: { preferred: FloatingSide; maxHeight: number },
): FloatingPlacement {
  const spaceBelow = Math.max(0, viewport.height - anchor.bottom - GAP_PX - EDGE_PX);
  const spaceAbove = Math.max(0, anchor.top - GAP_PX - EDGE_PX);
  const wantHeight = Math.min(panel.height, opts.maxHeight);
  let side = opts.preferred;
  if (side === 'below' && spaceBelow < wantHeight && spaceAbove > spaceBelow) side = 'above';
  else if (side === 'above' && spaceAbove < wantHeight && spaceBelow > spaceAbove) side = 'below';
  const space = side === 'below' ? spaceBelow : spaceAbove;
  const maxHeight = Math.min(opts.maxHeight, space);
  const height = Math.min(panel.height, maxHeight);
  const width = Math.min(panel.width, viewport.width - 2 * EDGE_PX);
  const top = side === 'below' ? anchor.bottom + GAP_PX : anchor.top - GAP_PX - height;
  const left = Math.min(Math.max(anchor.left, EDGE_PX), viewport.width - EDGE_PX - width);
  return { top, left, width, maxHeight, side };
}

/** Options of {@link useFloating}. Primitive values only, so the hook's effect does not re-run each render. */
export interface FloatingOptions {
  /** The side to open on when it fits. */
  preferred: FloatingSide;
  /** Maximum panel height in px. */
  maxHeight: number;
  /** Panel width in px, or 'anchor' to match the anchor's width. */
  width: 'anchor' | number;
}

/**
 * Keeps a `position: fixed` panel next to its anchor while it is open. Fixed panels escape the clipping of
 * scrolling ancestors (the sidebar scrolls), so the panel is never cut off; its position is recomputed on
 * every scroll (capture, so nested scrollers count) and on resize.
 *
 * Returns the panel's style while `open`, or undefined. The first placement runs in a layout effect, before
 * paint, so the panel never shows at a stale place.
 */
export function useFloating(
  anchorRef: RefObject<HTMLElement | null>,
  panelRef: RefObject<HTMLElement | null>,
  open: boolean,
  opts: FloatingOptions,
): CSSProperties | undefined {
  const { preferred, maxHeight, width } = opts;
  const [style, setStyle] = useState<CSSProperties | undefined>(undefined);
  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const anchor = anchorRef.current;
      const panel = panelRef.current;
      if (!anchor || !panel) return;
      const r = anchor.getBoundingClientRect();
      const p = placeFloating(
        { top: r.top, left: r.left, bottom: r.bottom, width: r.width },
        { width: width === 'anchor' ? r.width : width, height: panel.scrollHeight },
        { width: document.documentElement.clientWidth, height: window.innerHeight },
        { preferred, maxHeight },
      );
      setStyle({
        position: 'fixed',
        top: p.top,
        left: p.left,
        width: p.width,
        maxHeight: p.maxHeight,
      });
    };
    place();
    window.addEventListener('scroll', place, true);
    window.addEventListener('resize', place);
    return () => {
      window.removeEventListener('scroll', place, true);
      window.removeEventListener('resize', place);
    };
  }, [open, preferred, maxHeight, width, anchorRef, panelRef]);
  return open ? style : undefined;
}
