import {
  useCallback,
  useEffect,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type RefObject,
} from 'react';

/** localStorage key of the inputs pane's width, stored as a fraction of the workspace width (0–1). */
export const SPLIT_STORAGE_KEY = 'specmaxx:split';
/** Default inputs pane width: one third of the workspace (output 2/3 | inputs 1/3). */
export const DEFAULT_INPUTS_FRACTION = 1 / 3;
/** Narrowest inputs pane, as a fraction of the workspace. */
export const MIN_INPUTS_FRACTION = 0.22;
/** Widest inputs pane, as a fraction of the workspace. */
export const MAX_INPUTS_FRACTION = 0.5;
/** Narrowest inputs pane in CSS px, whatever the fraction says. */
export const MIN_INPUTS_PX = 300;
/** Narrowest output pane in CSS px. */
export const MIN_OUTPUT_PX = 480;
/** Keyboard step (arrow keys), and with Shift held, as fractions of the workspace. */
const STEP = 0.02;
const BIG_STEP = 0.1;

/** The allowed [min, max] inputs fraction for a workspace `widthPx` wide (0: unknown, fraction limits only). */
export function splitBounds(widthPx: number): [number, number] {
  const min = Math.max(MIN_INPUTS_FRACTION, widthPx > 0 ? MIN_INPUTS_PX / widthPx : 0);
  const max = Math.min(MAX_INPUTS_FRACTION, widthPx > 0 ? 1 - MIN_OUTPUT_PX / widthPx : 1);
  return [min, Math.max(min, max)];
}

/** `fraction` clamped to {@link splitBounds}. */
export function clampSplit(fraction: number, widthPx: number): number {
  const [min, max] = splitBounds(widthPx);
  return Math.min(max, Math.max(min, fraction));
}

function readStored(): number {
  try {
    const raw = localStorage.getItem(SPLIT_STORAGE_KEY);
    const value = raw === null ? NaN : Number(raw);
    return Number.isFinite(value) && value > 0 && value < 1 ? value : DEFAULT_INPUTS_FRACTION;
  } catch {
    return DEFAULT_INPUTS_FRACTION;
  }
}

function writeStored(fraction: number | undefined) {
  try {
    if (fraction === undefined) localStorage.removeItem(SPLIT_STORAGE_KEY);
    else localStorage.setItem(SPLIT_STORAGE_KEY, fraction.toFixed(4));
  } catch {
    /* storage may be disabled */
  }
}

/**
 * The inputs pane's width as a fraction of the workspace `container`: read from localStorage, clamped to the
 * container's current width (re-clamped when it resizes), and saved on every change; `reset` returns to 1/3 and
 * forgets the stored value.
 */
export function useSplit(container: RefObject<HTMLElement | null>) {
  const [stored, setStored] = useState(readStored);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = container.current;
    if (!el || typeof ResizeObserver !== 'function') return;
    // The observer reports the current width on observe, then on every resize.
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(el);
    return () => observer.disconnect();
  }, [container]);
  const set = useCallback(
    (fraction: number) => {
      const next = clampSplit(fraction, width);
      setStored(next);
      writeStored(next);
    },
    [width],
  );
  const reset = useCallback(() => {
    setStored(DEFAULT_INPUTS_FRACTION);
    writeStored(undefined);
  }, []);
  return { fraction: clampSplit(stored, width), width, set, reset };
}

/**
 * The thin vertical divider between the output pane (left) and the inputs pane (right): an ARIA `separator`
 * whose value is the inputs pane's width in percent.
 *
 * - Pointer: drag (pointer capture) to resize.
 * - Keyboard: ←/→ widen/narrow the inputs pane by 2 % (Shift: 10 %), Home/End jump to the narrowest/widest,
 *   Enter resets.
 * - Double-click resets to 1/3.
 */
export function Splitter({
  fraction,
  bounds,
  controls,
  onChange,
  onReset,
}: {
  /** The inputs pane's width as a fraction of the workspace (0–1). */
  fraction: number;
  /** Allowed [min, max] fractions, from {@link splitBounds}. */
  bounds: [number, number];
  /** Id of the inputs pane. */
  controls: string;
  onChange(fraction: number): void;
  onReset(): void;
}) {
  const [dragging, setDragging] = useState(false);
  const [min, max] = bounds;
  const percent = Math.round(fraction * 100);
  const fromPointer = (event: PointerEvent<HTMLDivElement>) => {
    const workspace = event.currentTarget.parentElement?.getBoundingClientRect();
    if (!workspace || workspace.width <= 0) return;
    onChange((workspace.right - event.clientX) / workspace.width);
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? BIG_STEP : STEP;
    let next: number | undefined;
    switch (event.key) {
      case 'ArrowLeft':
        next = fraction + step; // the divider moves left: the inputs pane (right) widens
        break;
      case 'ArrowRight':
        next = fraction - step;
        break;
      case 'Home':
        next = min;
        break;
      case 'End':
        next = max;
        break;
      case 'Enter':
        event.preventDefault();
        onReset();
        return;
      default:
        return;
    }
    event.preventDefault();
    onChange(next);
  };
  return (
    <div
      className="splitter"
      role="separator"
      tabIndex={0}
      aria-orientation="vertical"
      aria-label="Resize inputs"
      aria-controls={controls}
      aria-valuenow={percent}
      aria-valuemin={Math.round(min * 100)}
      aria-valuemax={Math.round(max * 100)}
      aria-valuetext={`Inputs ${percent}% of width`}
      title="Drag to resize, double-click to reset"
      data-dragging={dragging || undefined}
      onKeyDown={onKeyDown}
      onDoubleClick={onReset}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.currentTarget.setPointerCapture?.(event.pointerId);
        setDragging(true);
      }}
      onPointerMove={(event) => {
        if (dragging) fromPointer(event);
      }}
      onPointerUp={(event) => {
        if (!dragging) return;
        event.currentTarget.releasePointerCapture?.(event.pointerId);
        setDragging(false);
      }}
      onPointerCancel={() => setDragging(false)}
    />
  );
}
