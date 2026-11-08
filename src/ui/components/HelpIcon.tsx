import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { useFloating } from './floating';

/**
 * A "?" toggletip showing its children as a definition next to an input or heading.
 *
 * - Mouse: opens on hover, stays open after a click (click again to close), closes when the pointer leaves
 *   unless pinned by a click.
 * - Keyboard: opens on focus, closes on blur and on Escape.
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
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const style = useFloating(buttonRef, popupRef, open, {
    preferred: 'below',
    maxHeight: 320,
    width: 260,
  });

  const close = () => {
    setOpen(false);
    setPinned(false);
  };

  // A tap or click anywhere outside closes a pinned or touch-opened popup (iOS does not focus a tapped button,
  // so blur alone is not enough).
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) close();
    };
    document.addEventListener('pointerdown', onDown);
    return () => document.removeEventListener('pointerdown', onDown);
  }, [open]);

  return (
    <span className="help" ref={wrapRef}>
      <button
        ref={buttonRef}
        type="button"
        className="help-icon"
        aria-label={`About ${term}`}
        aria-expanded={open}
        aria-controls={popupId}
        onFocus={() => setOpen(true)}
        onBlur={close}
        onPointerEnter={(e) => {
          if (e.pointerType === 'mouse') setOpen(true);
        }}
        onPointerLeave={(e) => {
          if (e.pointerType === 'mouse' && !pinned) setOpen(false);
        }}
        onClick={() => {
          if (pinned) close();
          else {
            setPinned(true);
            setOpen(true);
          }
        }}
        onKeyDown={(e) => {
          if (e.key === 'Escape' && open) {
            e.stopPropagation();
            close();
          }
        }}
      >
        ?
      </button>
      <span
        ref={popupRef}
        id={popupId}
        role="tooltip"
        className="help-popup"
        hidden={!open}
        style={style}
      >
        {children}
      </span>
    </span>
  );
}
