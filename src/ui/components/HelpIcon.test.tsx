// @vitest-environment jsdom
import { useState } from 'react';
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { HelpIcon } from './HelpIcon';
import { NumberField } from './NumberField';

afterEach(cleanup);

/** jsdom has no reliable PointerEvent, so pointer events are plain events with a pointerType property. */
function pointer(
  el: Element,
  type: 'pointerover' | 'pointerout' | 'pointerdown',
  pointerType: string,
) {
  const ev = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(ev, { pointerType, relatedTarget: null });
  fireEvent(el, ev);
}

const DEFINITION = 'Storage format of the weights.';
const icon = () => screen.getByRole('button', { name: 'About Weights' });
const tip = () => document.getElementById(icon().getAttribute('aria-controls')!)!;

describe('HelpIcon', () => {
  it('is a button named "About <term>" whose popup is a tooltip, hidden at first', () => {
    render(<HelpIcon term="Weights">{DEFINITION}</HelpIcon>);
    expect(icon().getAttribute('aria-expanded')).toBe('false');
    expect(tip().getAttribute('role')).toBe('tooltip');
    expect(tip().hidden).toBe(true);
    expect(tip().textContent).toBe(DEFINITION);
  });

  it('opens on mouse hover and closes when the pointer leaves', () => {
    render(<HelpIcon term="Weights">{DEFINITION}</HelpIcon>);
    pointer(icon(), 'pointerover', 'mouse');
    expect(icon().getAttribute('aria-expanded')).toBe('true');
    expect(tip().hidden).toBe(false);
    pointer(icon(), 'pointerout', 'mouse');
    expect(icon().getAttribute('aria-expanded')).toBe('false');
  });

  it('opens on focus and closes on blur', () => {
    render(<HelpIcon term="Weights">{DEFINITION}</HelpIcon>);
    fireEvent.focus(icon());
    expect(tip().hidden).toBe(false);
    fireEvent.blur(icon());
    expect(tip().hidden).toBe(true);
  });

  it('opens on a tap (touch pointer, then click) and stays open after the pointer leaves', () => {
    render(<HelpIcon term="Weights">{DEFINITION}</HelpIcon>);
    pointer(icon(), 'pointerover', 'touch');
    expect(tip().hidden).toBe(true);
    fireEvent.click(icon());
    expect(tip().hidden).toBe(false);
    pointer(icon(), 'pointerout', 'mouse');
    expect(tip().hidden).toBe(false);
  });

  it('a click toggles a pinned popup closed again', () => {
    render(<HelpIcon term="Weights">{DEFINITION}</HelpIcon>);
    fireEvent.click(icon());
    expect(tip().hidden).toBe(false);
    fireEvent.click(icon());
    expect(tip().hidden).toBe(true);
  });

  it('closes on Escape and keeps focus on the icon', () => {
    render(<HelpIcon term="Weights">{DEFINITION}</HelpIcon>);
    icon().focus();
    fireEvent.click(icon());
    fireEvent.keyDown(icon(), { key: 'Escape' });
    expect(tip().hidden).toBe(true);
    expect(document.activeElement).toBe(icon());
  });

  it('closes on an outside tap or press', () => {
    render(
      <>
        <p>elsewhere</p>
        <HelpIcon term="Weights">{DEFINITION}</HelpIcon>
      </>,
    );
    fireEvent.click(icon());
    expect(tip().hidden).toBe(false);
    pointer(screen.getByText('elsewhere'), 'pointerdown', 'touch');
    expect(tip().hidden).toBe(true);
  });

  it('uses a given id for the popup', () => {
    render(
      <HelpIcon term="Weights" id="custom-tip">
        {DEFINITION}
      </HelpIcon>,
    );
    expect(tip().id).toBe('custom-tip');
  });
});

describe('NumberField', () => {
  function Field({ help }: { help?: string }) {
    const [n, setN] = useState(2);
    return <NumberField label="TP" value={n} onChange={setN} help={help} hint="Valid TP: 1, 2" />;
  }

  it('names the input exactly by its label, with the help popup and the hint as description', () => {
    render(<Field help="Splits every layer." />);
    const input = screen.getByRole('spinbutton', { name: 'TP' });
    const describedBy = (input.getAttribute('aria-describedby') ?? '').split(' ');
    expect(describedBy).toHaveLength(2);
    expect(document.getElementById(describedBy[0])?.textContent).toBe('Splits every layer.');
    expect(document.getElementById(describedBy[1])?.textContent).toContain('Valid TP: 1, 2');
  });

  it('has no help icon and only the hint when help is absent', () => {
    render(<Field />);
    const input = screen.getByRole('spinbutton', { name: 'TP' });
    expect(input.getAttribute('aria-describedby')?.split(' ')).toHaveLength(1);
    expect(screen.queryByRole('button', { name: 'About TP' })).toBeNull();
  });
});
