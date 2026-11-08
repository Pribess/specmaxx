// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { useRef } from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { Splitter, SPLIT_STORAGE_KEY, clampSplit, splitBounds, useSplit } from './Splitter';

/** The workspace as App builds it: a container whose width the split is a fraction of, with the splitter. */
function Workspace() {
  const ref = useRef<HTMLDivElement>(null);
  const split = useSplit(ref);
  return (
    <div ref={ref}>
      <Splitter
        fraction={split.fraction}
        bounds={splitBounds(split.width)}
        controls="inputs-pane"
        onChange={split.set}
        onReset={split.reset}
      />
    </div>
  );
}
const separator = () => screen.getByRole('separator', { name: 'Resize inputs' });
const value = () => Number(separator().getAttribute('aria-valuenow'));
const stored = () => localStorage.getItem(SPLIT_STORAGE_KEY);

beforeEach(() => localStorage.clear());
afterEach(cleanup);

describe('splitter', () => {
  it('is a vertical ARIA separator for the inputs pane, at 1/3 by default', () => {
    render(<Workspace />);
    const el = separator();
    expect(el.getAttribute('aria-orientation')).toBe('vertical');
    expect(el.getAttribute('aria-controls')).toBe('inputs-pane');
    expect(el.tabIndex).toBe(0);
    expect(value()).toBe(33);
    expect(el.getAttribute('aria-valuemin')).toBe('22');
    expect(el.getAttribute('aria-valuemax')).toBe('50');
    expect(el.getAttribute('aria-valuetext')).toBe('Inputs 33% of width');
    expect(stored()).toBeNull();
  });

  it('resizes with the keyboard within its bounds and saves each step', () => {
    render(<Workspace />);
    fireEvent.keyDown(separator(), { key: 'ArrowLeft' });
    expect(value()).toBe(35);
    expect(Number(stored())).toBeCloseTo(1 / 3 + 0.02, 4);
    fireEvent.keyDown(separator(), { key: 'ArrowRight' });
    fireEvent.keyDown(separator(), { key: 'ArrowRight' });
    expect(value()).toBe(31);
    fireEvent.keyDown(separator(), { key: 'ArrowLeft', shiftKey: true });
    expect(value()).toBe(41);
    fireEvent.keyDown(separator(), { key: 'End' });
    expect(value()).toBe(50);
    fireEvent.keyDown(separator(), { key: 'ArrowLeft' });
    expect(value()).toBe(50);
    fireEvent.keyDown(separator(), { key: 'Home' });
    expect(value()).toBe(22);
    expect(Number(stored())).toBeCloseTo(0.22, 4);
  });

  it('resets to 1/3 on double-click and on Enter, forgetting the stored width', () => {
    render(<Workspace />);
    fireEvent.keyDown(separator(), { key: 'End' });
    fireEvent.doubleClick(separator());
    expect(value()).toBe(33);
    expect(stored()).toBeNull();
    fireEvent.keyDown(separator(), { key: 'Home' });
    fireEvent.keyDown(separator(), { key: 'Enter' });
    expect(value()).toBe(33);
    expect(stored()).toBeNull();
  });

  it('restores a stored width, clamps an out-of-range one and ignores garbage', () => {
    localStorage.setItem(SPLIT_STORAGE_KEY, '0.4500');
    render(<Workspace />);
    expect(value()).toBe(45);
    cleanup();
    localStorage.setItem(SPLIT_STORAGE_KEY, '0.9');
    render(<Workspace />);
    expect(value()).toBe(50);
    cleanup();
    localStorage.setItem(SPLIT_STORAGE_KEY, 'wide');
    render(<Workspace />);
    expect(value()).toBe(33);
  });

  it('keeps the inputs pane ≥ 300px and the output pane ≥ 480px', () => {
    expect(splitBounds(0)).toEqual([0.22, 0.5]);
    const [min900, max900] = splitBounds(900);
    expect(min900).toBeCloseTo(300 / 900);
    expect(max900).toBeCloseTo(1 - 480 / 900);
    expect(clampSplit(0.25, 1440)).toBeCloseTo(0.25);
    expect(clampSplit(0.1, 1440)).toBeCloseTo(0.22);
    expect(clampSplit(0.2, 1000)).toBeCloseTo(0.3);
  });
});
