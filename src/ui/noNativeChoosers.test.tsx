// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { App } from './App';
import { inlineRunner } from './sim/runner';

const sources = import.meta.glob('/src/**/*.tsx', { query: '?raw', import: 'default', eager: true }) as Record<
  string,
  string
>;

afterEach(cleanup);

describe('no OS-native choosers', () => {
  it('renders no select, datalist or native color/date/time input, even with every details opened', async () => {
    const { container } = render(<App runner={inlineRunner} loader={async () => ({ kind: 'network_error', message: 'offline' }) as never} />);
    await screen.findByTestId('decode-tps');
    container.querySelectorAll('details').forEach((d) => (d.open = true));
    await screen.findByText(/Edit spec numbers/);
    expect(container.querySelectorAll('select, datalist')).toHaveLength(0);
    expect(
      container.querySelectorAll('input[type=color], input[type=date], input[type=time], input[type=datetime-local], input[type=month], input[type=week]'),
    ).toHaveLength(0);
  });
  it('has no <select or <datalist element in any component source', () => {
    const files = Object.entries(sources).filter(([path]) => !/\.test\.tsx$/.test(path));
    expect(files.length).toBeGreaterThan(5);
    for (const [path, raw] of files) {
      const source = raw.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
      expect(source, path).not.toMatch(/<select[\s>]/);
      expect(source, path).not.toMatch(/<datalist[\s>]/);
    }
  });
});
