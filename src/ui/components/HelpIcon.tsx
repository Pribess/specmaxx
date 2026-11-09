import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type FocusEvent,
  type ReactNode,
} from 'react';
import { useFloating } from './floating';
import { ScrollArea } from './ScrollArea';

/** How long a hover-opened popup stays open after the pointer leaves, so the pointer can cross into it. Unit: ms. */
const HOVER_CLOSE_DELAY_MS = 150;

/**
 * A "?" toggletip showing its children as a definition next to an input or heading.
 *
 * - Mouse: opens on hover, stays open after a click (click again to close), closes shortly after the pointer leaves
 *   the icon and the popup unless pinned by a click. The pointer may move into the popup (e.g. to follow a link).
 * - Keyboard: opens on focus; Tab moves on into the popup's links; closes when focus leaves the icon and the popup,
 *   and on Escape (which returns focus to the icon when it was inside the popup).
 * - Touch: a tap opens it (the click pins it); a tap anywhere else closes it.
 *
 * The popup is always in the DOM (hidden while closed), so an input can reference it with aria-describedby
 * through `id`. It is `position: fixed`, so the sidebar's scrolling does not clip it.
 */
export function HelpIcon({
  term,
  children,
  id: externalId,
}: {
  /** The name of the thing explained; the button reads "About <term>". */
  term: string;
  /** The definition shown in the popup. */
  children: ReactNode;
  /** Id of the popup, for aria-describedby on the field the icon explains. Generated when omitted. */
  id?: string;
}) {
  const generated = useId();
  const popupId = externalId ?? `${generated}-tip`;
  const wrapRef = useRef<HTMLSpanElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const popupRef = useRef<HTMLSpanElement>(null);
  /** True while a mouse pointer is over the icon or the popup. */
  const hoverRef = useRef(false);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  /** Set while Escape returns focus to the icon, so that focus does not reopen the popup. */
  const refocusing = useRef(false);
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const floating = useFloating(buttonRef, popupRef, open, {
    preferred: 'below',
    maxHeight: 320,
    width: 260,
  });
  // The popup is placed by useFloating; its height cap goes on the ScrollArea viewport, which scrolls.
  const { maxHeight, ...style } = floating ?? {};

  const cancelClose = useCallback(() => {
    if (closeTimer.current !== undefined) clearTimeout(closeTimer.current);
    closeTimer.current = undefined;
  }, []);
  const close = useCallback(() => {
    cancelClose();
    setOpen(false);
    setPinned(false);
  }, [cancelClose]);
  const inside = useCallback(
    (node: EventTarget | null): boolean =>
      node instanceof Node && (wrapRef.current?.contains(node) ?? false),
    [],
  );

  // A tap or click anywhere outside closes a pinned or touch-opened popup (iOS does not focus a tapped button,
  // so blur alone is not enough).
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!inside(e.target)) close();
    };
    document.addEventListener('pointerdown', onDown);
    return () => document.removeEventListener('pointerdown', onDown);
  }, [open, close, inside]);

  useEffect(() => cancelClose, [cancelClose]);

  // Focus leaving the icon and the popup closes it. Focus moving between them (Tab into a source link, or a mouse
  // press on a link, which some browsers do not focus: relatedTarget null while the pointer is inside) keeps it.
  const onBlur = (e: FocusEvent) => {
    if (inside(e.relatedTarget)) return;
    if (e.relatedTarget === null && hoverRef.current) return;
    close();
  };

  return (
    <span
      className="help"
      ref={wrapRef}
      onBlur={onBlur}
      onPointerEnter={(e) => {
        if (e.pointerType !== 'mouse') return;
        hoverRef.current = true;
        cancelClose();
      }}
      onPointerLeave={(e) => {
        if (e.pointerType !== 'mouse') return;
        hoverRef.current = false;
        if (pinned || inside(document.activeElement)) return;
        cancelClose();
        closeTimer.current = setTimeout(() => setOpen(false), HOVER_CLOSE_DELAY_MS);
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape' && open) {
          e.stopPropagation();
          const wasInPopup = popupRef.current?.contains(document.activeElement) ?? false;
          close();
          if (wasInPopup) {
            refocusing.current = true;
            buttonRef.current?.focus({ preventScroll: true });
            refocusing.current = false;
          }
        }
      }}
    >
      <button
        ref={buttonRef}
        type="button"
        className="help-icon"
        aria-label={`About ${term}`}
        aria-expanded={open}
        aria-controls={popupId}
        onFocus={() => {
          if (!refocusing.current) setOpen(true);
        }}
        onPointerEnter={(e) => {
          if (e.pointerType === 'mouse') setOpen(true);
        }}
        onClick={() => {
          if (pinned) close();
          else {
            setPinned(true);
            setOpen(true);
          }
        }}
      >
        ?
      </button>
      <span ref={popupRef} className="help-popup" hidden={!open} style={style}>
        {/* The tooltip is the text itself, so its description excludes the ScrollArea's own markup. */}
        <ScrollArea viewportProps={{ style: { maxHeight } }}>
          <span id={popupId} role="tooltip" className="help-text" hidden={!open}>
            {children}
          </span>
        </ScrollArea>
      </span>
    </span>
  );
}
