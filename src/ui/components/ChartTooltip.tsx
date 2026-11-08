import {
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
} from 'react';

/** Shared focus, pointer and touch selection for a chart's ordered points. */
export function useChartPoint(xs: number[], viewWidth: number) {
  const [active, setActive] = useState<number | null>(null);
  const [tabStop, setTabStop] = useState(0);
  const id = useId();
  const focused = useRef<number | null>(null);
  function select(index: number) {
    setActive(index);
    setTabStop(index);
  }
  function onKeyDown(event: KeyboardEvent<SVGElement | HTMLElement>, index: number) {
    let next: number;
    switch (event.key) {
      case 'Enter':
      case ' ':
        event.preventDefault();
        select(index);
        return;
      case 'ArrowRight':
        next = Math.min(xs.length - 1, index + 1);
        break;
      case 'ArrowLeft':
        next = Math.max(0, index - 1);
        break;
      case 'Home':
        next = 0;
        break;
      case 'End':
        next = xs.length - 1;
        break;
      case 'Escape':
        setActive(null);
        return;
      default:
        return;
    }
    event.preventDefault();
    select(next);
    const root = event.currentTarget.closest('[data-chart]');
    root?.querySelectorAll<SVGElement | HTMLElement>('[data-chart-point]')[next]?.focus();
  }
  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    if (!xs.length) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    if (!bounds.width) return;
    const px = ((event.clientX - bounds.left) / bounds.width) * viewWidth;
    select(xs.reduce((best, x, i) => (Math.abs(x - px) < Math.abs(xs[best] - px) ? i : best), 0));
  }
  return {
    active,
    id,
    chartProps: {
      onPointerMove,
      onPointerDown: onPointerMove,
      onPointerLeave: () => setActive(focused.current),
    },
    pointProps: (index: number, label: string) => ({
      'data-chart-point': true,
      role: 'button',
      tabIndex: tabStop === index ? 0 : -1,
      'aria-label': label,
      'aria-describedby': active === index ? id : undefined,
      onFocus: () => {
        focused.current = index;
        select(index);
      },
      onBlur: () => {
        focused.current = null;
        setActive(null);
      },
      onPointerEnter: () => select(index),
      onPointerDown: () => select(index),
      onClick: () => select(index),
      onKeyDown: (event: KeyboardEvent<SVGElement | HTMLElement>) => onKeyDown(event, index),
    }),
  };
}

/** HTML tooltip stays within the chart on narrow screens; points supply the precise x/y values. */
export function ChartTooltip({ id, children }: { id: string; children: ReactNode }) {
  return (
    <div className="chart-tooltip" id={id} role="tooltip" aria-live="polite">
      {children}
    </div>
  );
}
