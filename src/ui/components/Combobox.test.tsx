// @vitest-environment jsdom
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { Combobox } from './Combobox';
import type { PickerOption } from './fuzzy';

afterEach(cleanup);

const MODELS: PickerOption[] = [
  { value: 'Qwen/Qwen3.8-27B', label: 'Qwen3.8-27B' },
  { value: 'Qwen/Qwen3.8-Flash-Next', label: 'Qwen3.8-Flash-Next' },
  { value: 'zai-org/GLM-5.3', label: 'GLM-5.3' },
];

/** A controlled combobox that records what was committed. */
function Harness({ onCommit, initial = '' }: { onCommit: (v: string) => void; initial?: string }) {
  const [text, setText] = useState(initial);
  return (
    <Combobox
      label="Hugging Face model ID"
      value={text}
      onValueChange={setText}
      options={MODELS}
      onCommit={(v) => {
        onCommit(v);
        setText(v);
      }}
      placeholder="org/model"
      help={<>A Hugging Face repository ID.</>}
    />
  );
}

const input = () =>
  screen.getByRole('combobox', { name: 'Hugging Face model ID' }) as HTMLInputElement;

describe('Combobox', () => {
  it('is a combobox named by its label, with the typed text as value', () => {
    render(<Harness onCommit={() => {}} initial="Qwen/Q" />);
    expect(input().value).toBe('Qwen/Q');
    expect(input().getAttribute('aria-autocomplete')).toBe('list');
    expect(input().getAttribute('aria-expanded')).toBe('false');
  });

  it('opens the suggestions on focus and on typing, filtered by fuzzy search', () => {
    render(<Harness onCommit={() => {}} />);
    fireEvent.focus(input());
    expect(screen.getAllByRole('option')).toHaveLength(3);
    fireEvent.change(input(), { target: { value: 'flash' } });
    expect(screen.getAllByRole('option').map((o) => o.textContent)).toEqual(['Qwen3.8-Flash-Next']);
    expect(input().getAttribute('aria-expanded')).toBe('true');
  });

  it('commits the typed text on Enter when no suggestion is active', () => {
    const spy = vi.fn();
    render(<Harness onCommit={spy} />);
    fireEvent.change(input(), { target: { value: 'my-org/my-model' } });
    fireEvent.keyDown(input(), { key: 'Enter' });
    expect(spy).toHaveBeenCalledWith('my-org/my-model');
    expect(input().getAttribute('aria-expanded')).toBe('false');
  });

  it('commits the active suggestion value on Enter after ArrowDown', () => {
    const spy = vi.fn();
    render(<Harness onCommit={spy} />);
    fireEvent.change(input(), { target: { value: 'qwen' } });
    fireEvent.keyDown(input(), { key: 'ArrowDown' });
    expect(input().getAttribute('aria-activedescendant')).toBe(screen.getAllByRole('option')[0].id);
    fireEvent.keyDown(input(), { key: 'ArrowDown' });
    fireEvent.keyDown(input(), { key: 'Enter' });
    expect(spy).toHaveBeenCalledWith('Qwen/Qwen3.8-Flash-Next');
  });

  it('moves up from no active suggestion to the last one', () => {
    render(<Harness onCommit={() => {}} />);
    fireEvent.focus(input());
    fireEvent.keyDown(input(), { key: 'ArrowUp' });
    expect(input().getAttribute('aria-activedescendant')).toBe(screen.getAllByRole('option')[2].id);
  });

  it('commits a clicked suggestion value', () => {
    const spy = vi.fn();
    render(<Harness onCommit={spy} />);
    fireEvent.focus(input());
    fireEvent.click(screen.getByRole('option', { name: 'GLM-5.3' }));
    expect(spy).toHaveBeenCalledWith('zai-org/GLM-5.3');
  });

  it('closes on Escape, and a second Escape does nothing harmful', () => {
    render(<Harness onCommit={() => {}} initial="qwen" />);
    fireEvent.focus(input());
    fireEvent.keyDown(input(), { key: 'Escape' });
    expect(screen.queryAllByRole('option')).toHaveLength(0);
    fireEvent.keyDown(input(), { key: 'Escape' });
    expect(input().value).toBe('qwen');
  });

  it('closes on an outside pointer press and keeps the typed text, but not on a press in the input', async () => {
    render(<Harness onCommit={() => {}} initial="qwen" />);
    fireEvent.focus(input());
    expect(screen.getAllByRole('option').length).toBeGreaterThan(0);
    // Radix arms its outside-press listener a tick after the popup opens, and acts on the press's click.
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)));
    fireEvent.pointerDown(input());
    fireEvent.click(input());
    expect(screen.getAllByRole('option').length).toBeGreaterThan(0);
    fireEvent.pointerDown(document.body);
    fireEvent.click(document.body);
    expect(screen.queryAllByRole('option')).toHaveLength(0);
    expect(input().value).toBe('qwen');
  });

  it('commits the typed text on Enter even while suggestions match, until a key moves into the list', () => {
    const spy = vi.fn();
    render(<Harness onCommit={spy} />);
    fireEvent.change(input(), { target: { value: 'Qwen/Qwen3.8' } });
    expect(screen.getAllByRole('option').length).toBeGreaterThan(0);
    expect(screen.queryAllByRole('option', { selected: true })).toHaveLength(0);
    expect(input().getAttribute('aria-activedescendant')).toBeNull();
    fireEvent.keyDown(input(), { key: 'Home' });
    fireEvent.keyDown(input(), { key: 'Enter' });
    expect(spy).toHaveBeenCalledWith('Qwen/Qwen3.8');
  });

  it('shows the help icon next to the label', () => {
    render(<Harness onCommit={() => {}} />);
    expect(screen.getByRole('button', { name: 'About Hugging Face model ID' })).toBeTruthy();
  });
});
