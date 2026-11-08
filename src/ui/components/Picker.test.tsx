// @vitest-environment jsdom
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { Picker } from './Picker';
import type { PickerOption } from './fuzzy';

afterEach(cleanup);

/**
 * Radix arms its outside-press listener a tick after the popup opens (so the opening click is not one), and
 * acts on a primary-button press at its click.
 */
const tick = () => act(() => new Promise((resolve) => setTimeout(resolve, 0)));

const SMALL: PickerOption[] = [
  { value: 'a', label: 'Alpha' },
  { value: 'b', label: 'Beta' },
  { value: 'c', label: 'Gamma' },
];

const LONG: PickerOption[] = [
  { value: 'device:rtx-5090', label: 'RTX 5090', group: 'NVIDIA' },
  { value: 'device:rtx-4090', label: 'RTX 4090', group: 'NVIDIA' },
  { value: 'device:rtx-3090', label: 'RTX 3090', group: 'NVIDIA' },
  { value: 'device:m5-max', label: 'M5 Max 40C', group: 'Apple' },
  { value: 'device:m4-max', label: 'M4 Max 40C', group: 'Apple' },
  { value: 'device:m3-ultra', label: 'M3 Ultra 80C', group: 'Apple' },
  { value: 'device:mi300x', label: 'MI300X', group: 'AMD' },
  { value: 'device:mi355x', label: 'MI355X', group: 'AMD' },
  { value: 'device:h100', label: 'H100 SXM', group: 'Datacenter' },
  { value: 'device:h200', label: 'H200 SXM', group: 'Datacenter' },
];

/** A controlled picker, so the selected value updates after a pick. */
function Harness({
  options,
  initial,
  onChangeSpy,
  label = 'Device',
}: {
  options: PickerOption[];
  initial?: string;
  onChangeSpy?: (v: string) => void;
  label?: string;
}) {
  const [value, setValue] = useState<string | undefined>(initial);
  return (
    <Picker
      label={label}
      value={value}
      options={options}
      onChange={(v) => {
        onChangeSpy?.(v);
        setValue(v);
      }}
      help={<>Pick the device.</>}
    />
  );
}

const trigger = (name = 'Device') => screen.getByRole('combobox', { name });

describe('Picker', () => {
  it('names the trigger by its label and shows the selected label', () => {
    render(<Harness options={SMALL} initial="b" />);
    const t = trigger();
    expect(t.getAttribute('aria-haspopup')).toBe('listbox');
    expect(t.getAttribute('aria-expanded')).toBe('false');
    expect(t.textContent).toContain('Beta');
  });

  it('opens on click with a listbox of options and checks the selected one', () => {
    render(<Harness options={SMALL} initial="b" />);
    fireEvent.click(trigger());
    const list = screen.getByRole('listbox', { name: 'Device' });
    const options = within(list).getAllByRole('option');
    expect(options.map((o) => o.querySelector('.listbox-label')?.textContent)).toEqual([
      'Alpha',
      'Beta',
      'Gamma',
    ]);
    expect(options.map((o) => o.hasAttribute('data-checked'))).toEqual([false, true, false]);
    // The selected option is also the active (aria-selected) one when the list opens.
    expect(options[1].getAttribute('aria-selected')).toBe('true');
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
    expect(trigger().getAttribute('aria-controls')).toBe(list.id);
  });

  it('picks on click and closes', () => {
    const spy = vi.fn();
    render(<Harness options={SMALL} onChangeSpy={spy} />);
    fireEvent.click(trigger());
    fireEvent.click(screen.getByRole('option', { name: 'Gamma' }));
    expect(spy).toHaveBeenCalledWith('c');
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
    expect(trigger().textContent).toContain('Gamma');
  });

  it('moves the active option with ArrowDown and ArrowUp, and points aria-activedescendant at it', () => {
    render(<Harness options={SMALL} initial="a" />);
    const t = trigger();
    t.focus();
    fireEvent.keyDown(t, { key: 'ArrowDown' });
    expect(t.getAttribute('aria-expanded')).toBe('true');
    const alpha = screen.getByRole('option', { name: 'Alpha' });
    expect(t.getAttribute('aria-activedescendant')).toBe(alpha.id);
    fireEvent.keyDown(t, { key: 'ArrowDown' });
    fireEvent.keyDown(t, { key: 'ArrowDown' });
    expect(t.getAttribute('aria-activedescendant')).toBe(
      screen.getByRole('option', { name: 'Gamma' }).id,
    );
    fireEvent.keyDown(t, { key: 'ArrowDown' });
    expect(t.getAttribute('aria-activedescendant')).toBe(
      screen.getByRole('option', { name: 'Gamma' }).id,
    );
    fireEvent.keyDown(t, { key: 'ArrowUp' });
    expect(t.getAttribute('aria-activedescendant')).toBe(
      screen.getByRole('option', { name: 'Beta' }).id,
    );
  });

  it('jumps with Home and End', () => {
    render(<Harness options={SMALL} initial="b" />);
    const t = trigger();
    t.focus();
    fireEvent.keyDown(t, { key: 'ArrowDown' });
    fireEvent.keyDown(t, { key: 'End' });
    expect(t.getAttribute('aria-activedescendant')).toBe(
      screen.getByRole('option', { name: 'Gamma' }).id,
    );
    fireEvent.keyDown(t, { key: 'Home' });
    expect(t.getAttribute('aria-activedescendant')).toBe(
      screen.getByRole('option', { name: 'Alpha' }).id,
    );
  });

  it('picks the active option with Enter and returns focus to the trigger', () => {
    const spy = vi.fn();
    render(<Harness options={SMALL} onChangeSpy={spy} />);
    const t = trigger();
    t.focus();
    fireEvent.keyDown(t, { key: 'ArrowDown' });
    fireEvent.keyDown(t, { key: 'ArrowDown' });
    fireEvent.keyDown(t, { key: 'Enter' });
    expect(spy).toHaveBeenCalledWith('b');
    expect(t.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(t);
  });

  it('picks the active option with Space, and moves ten rows with PageDown / PageUp', () => {
    const spy = vi.fn();
    const many = Array.from({ length: 14 }, (_, i) => ({ value: `v${i}`, label: `Item ${i}` }));
    render(<Harness options={many} onChangeSpy={spy} />);
    fireEvent.click(trigger());
    const search = screen.getByRole('combobox', { name: 'Search Device' });
    fireEvent.keyDown(search, { key: 'PageDown' });
    expect(search.getAttribute('aria-activedescendant')).toBe(
      screen.getByRole('option', { name: 'Item 10' }).id,
    );
    fireEvent.keyDown(search, { key: 'PageUp' });
    fireEvent.keyDown(search, { key: 'ArrowDown' });
    fireEvent.keyDown(search, { key: 'Enter' });
    expect(spy).toHaveBeenCalledWith('v1');
    const t = trigger();
    render(<Harness options={SMALL} onChangeSpy={spy} label="Small" />);
    const small = trigger('Small');
    small.focus();
    fireEvent.keyDown(small, { key: 'ArrowDown' });
    fireEvent.keyDown(small, { key: 'ArrowDown' });
    fireEvent.keyDown(small, { key: ' ' });
    expect(spy).toHaveBeenLastCalledWith('b');
    expect(t.getAttribute('aria-expanded')).toBe('false');
  });

  it('opens with Enter, Space and ArrowUp when closed', () => {
    render(<Harness options={SMALL} />);
    const t = trigger();
    t.focus();
    fireEvent.keyDown(t, { key: 'Enter' });
    expect(t.getAttribute('aria-expanded')).toBe('true');
    fireEvent.keyDown(t, { key: 'Escape' });
    fireEvent.keyDown(t, { key: ' ' });
    expect(t.getAttribute('aria-expanded')).toBe('true');
    fireEvent.keyDown(t, { key: 'Escape' });
    fireEvent.keyDown(t, { key: 'ArrowUp' });
    expect(t.getAttribute('aria-expanded')).toBe('true');
  });

  it('closes on Escape without changing the value and returns focus to the trigger', () => {
    const spy = vi.fn();
    render(<Harness options={SMALL} initial="a" onChangeSpy={spy} />);
    const t = trigger();
    t.focus();
    fireEvent.keyDown(t, { key: 'ArrowDown' });
    fireEvent.keyDown(t, { key: 'ArrowDown' });
    fireEvent.keyDown(t, { key: 'Escape' });
    expect(spy).not.toHaveBeenCalled();
    expect(t.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(t);
    expect(t.textContent).toContain('Alpha');
  });

  it('closes on Tab and leaves focus where the browser puts it', () => {
    render(<Harness options={SMALL} />);
    const t = trigger();
    t.focus();
    fireEvent.keyDown(t, { key: 'ArrowDown' });
    fireEvent.keyDown(t, { key: 'Tab' });
    expect(t.getAttribute('aria-expanded')).toBe('false');
  });

  it('closes on an outside pointer press without changing the value', async () => {
    const spy = vi.fn();
    render(
      <>
        <button type="button">outside</button>
        <Harness options={SMALL} onChangeSpy={spy} />
      </>,
    );
    fireEvent.click(trigger());
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
    await tick();
    fireEvent.pointerDown(screen.getByRole('button', { name: 'outside' }));
    fireEvent.click(screen.getByRole('button', { name: 'outside' }));
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
    expect(spy).not.toHaveBeenCalled();
  });

  it('renders option groups as labelled groups in first-appearance order', () => {
    render(<Harness options={LONG} />);
    fireEvent.click(trigger());
    const groups = screen.getAllByRole('group');
    expect(
      groups.map(
        (g) =>
          g.getAttribute('aria-labelledby') &&
          document.getElementById(g.getAttribute('aria-labelledby')!)?.textContent,
      ),
    ).toEqual(['NVIDIA', 'Apple', 'AMD', 'Datacenter']);
  });

  it('shows a search box above the threshold, focuses it on open and filters the list', () => {
    render(<Harness options={LONG} initial="device:rtx-4090" />);
    fireEvent.click(trigger());
    const search = screen.getByRole('combobox', { name: 'Search Device' });
    expect(document.activeElement).toBe(search);
    fireEvent.change(search, { target: { value: 'max' } });
    const options = screen.getAllByRole('option').map((o) => o.textContent);
    expect(options).toEqual(['M5 Max 40C', 'M4 Max 40C']);
    expect(search.getAttribute('aria-activedescendant')).toBe(
      screen.getByRole('option', { name: 'M5 Max 40C' }).id,
    );
  });

  it('picks the best search match with Enter', () => {
    const spy = vi.fn();
    render(<Harness options={LONG} onChangeSpy={spy} />);
    fireEvent.click(trigger());
    const search = screen.getByRole('combobox', { name: 'Search Device' });
    fireEvent.change(search, { target: { value: 'h200' } });
    fireEvent.keyDown(search, { key: 'Enter' });
    expect(spy).toHaveBeenCalledWith('device:h200');
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
  });

  it('navigates the filtered list with arrows from the search box and reports no matches', () => {
    const spy = vi.fn();
    render(<Harness options={LONG} onChangeSpy={spy} />);
    fireEvent.click(trigger());
    const search = screen.getByRole('combobox', { name: 'Search Device' });
    fireEvent.change(search, { target: { value: 'amd' } });
    fireEvent.change(search, { target: { value: 'mi' } });
    fireEvent.keyDown(search, { key: 'ArrowDown' });
    expect(search.getAttribute('aria-activedescendant')).toBe(
      screen.getByRole('option', { name: 'MI355X' }).id,
    );
    fireEvent.keyDown(search, { key: 'ArrowUp' });
    fireEvent.keyDown(search, { key: 'Enter' });
    expect(spy).toHaveBeenCalledWith('device:mi300x');
    fireEvent.click(trigger());
    fireEvent.change(screen.getByRole('combobox', { name: 'Search Device' }), {
      target: { value: 'zzzz' },
    });
    expect(screen.getByRole('status').textContent).toBe('No matches');
    expect(screen.queryAllByRole('option')).toHaveLength(0);
  });

  it('leaves Home and End to the caret inside the search box', () => {
    render(<Harness options={LONG} initial="device:h100" />);
    fireEvent.click(trigger());
    const search = screen.getByRole('combobox', { name: 'Search Device' });
    const before = search.getAttribute('aria-activedescendant');
    fireEvent.keyDown(search, { key: 'Home' });
    expect(search.getAttribute('aria-activedescendant')).toBe(before);
  });

  it('keeps the search header outside the scrolling viewport, which holds the whole listbox', () => {
    render(<Harness options={LONG} />);
    fireEvent.click(trigger());
    const search = screen.getByRole('combobox', { name: 'Search Device' });
    const list = screen.getByRole('listbox', { name: 'Device' });
    const viewport = list.closest('[data-radix-scroll-area-viewport]');
    expect(viewport).not.toBeNull();
    expect(viewport!.contains(search)).toBe(false);
    // Header and scroll area are siblings in one popup, so rows scroll beneath the header.
    const header = search.closest('.picker-search-head')!;
    expect(header.parentElement).toBe(viewport!.closest('.scroll-area')!.parentElement);
    expect(
      header.compareDocumentPosition(viewport!) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    // Every option scrolls inside the themed viewport, never in a native scroller.
    for (const option of within(list).getAllByRole('option')) {
      expect(viewport!.contains(option)).toBe(true);
    }
  });

  it('shows the help icon next to the label', () => {
    render(<Harness options={SMALL} />);
    expect(screen.getByRole('button', { name: 'About Device' })).toBeTruthy();
  });

  it('does not show a search box at or below the threshold', () => {
    render(<Harness options={SMALL} />);
    fireEvent.click(trigger());
    expect(screen.queryByRole('combobox', { name: 'Search Device' })).toBeNull();
  });

  it('honours searchThreshold', () => {
    render(
      <Picker
        label="Device"
        value={undefined}
        options={SMALL}
        searchThreshold={2}
        onChange={() => {}}
      />,
    );
    fireEvent.click(trigger());
    expect(screen.getByRole('combobox', { name: 'Search Device' })).toBeTruthy();
  });
});
