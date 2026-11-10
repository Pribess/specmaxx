// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { CountField, countChoices } from './CountField';

afterEach(cleanup);

describe('countChoices', () => {
  it('adds an unoffered current count in order', () => {
    expect(countChoices([1, 2, 8], 4)).toEqual([1, 2, 4, 8]);
    expect(countChoices([1, 2, 8], 2)).toEqual([1, 2, 8]);
  });
});

describe('CountField', () => {
  it('is a segmented control with up to 6 choices', () => {
    render(<CountField label="Machines" value={2} counts={[1, 2, 4, 8]} onChange={vi.fn()} />);
    expect(screen.getByRole('radiogroup', { name: 'Machines' })).toBeTruthy();
    expect(screen.getAllByRole('radio')).toHaveLength(4);
  });
  it('is a picker beyond 6 choices, keeping the aside and issues', () => {
    render(
      <CountField
        label="GPUs in one machine"
        value={72}
        counts={[1, 2, 4, 8, 16, 32, 36, 64, 72]}
        onChange={vi.fn()}
        aside={<span>= 72 GPUs total</span>}
        issues={[{ field: 'devices', level: 'error', message: 'too many' }]}
      />,
    );
    expect(screen.queryByRole('radiogroup')).toBeNull();
    expect(screen.getByRole('combobox', { name: 'GPUs in one machine' }).textContent).toContain(
      '72',
    );
    expect(screen.getByText('= 72 GPUs total')).toBeTruthy();
    expect(screen.getByText('too many')).toBeTruthy();
  });
});
