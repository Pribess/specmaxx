// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import {
  buildSimInput,
  decodeState,
  loadedFromHf,
  REFERENCE_MODEL,
  type AppState,
  type LoadedModel,
} from '../state';
import { ParallelPanel } from './ParallelPanel';

afterEach(cleanup);

const ID = 'zai-org/GLM-5.3-Flash';
const r = mapHfModel(fixtureRepoFiles(ID));
if (r.kind !== 'ok') throw new Error(r.kind);
const GLM = loadedFromHf({ kind: 'hf', id: ID }, r);

function panel(query: string, loaded: LoadedModel = GLM) {
  const state: AppState = decodeState(query).state;
  const dispatch = vi.fn();
  render(
    <ParallelPanel
      state={state}
      dispatch={dispatch}
      loaded={loaded}
      built={buildSimInput(state, loaded)}
    />,
  );
  return dispatch;
}

describe('ParallelPanel', () => {
  it('a MoE model offers the layout; custom shows TP, PP, DP and EP', () => {
    panel(`m=${ID}&d=gb200&n=72&b=72&dp=72&ep=72`);
    const layout = screen.getByRole('radiogroup', { name: 'Layout' });
    expect(layout.textContent).toContain('Custom');
    expect(layout.textContent).toContain('DP attention + EP');
    for (const name of ['TP', 'PP', 'DP', 'EP'])
      expect(screen.getByRole('spinbutton', { name })).toBeTruthy();
    expect(screen.queryByRole('radiogroup', { name: 'Dense layers' })).toBeNull();
  });

  it('DP attention + EP shows EP, PP and the dense-layers choice; the value names the layout', () => {
    panel(`m=${ID}&d=gb200&n=72&b=72&pp=3&dp=24&ep=24&lay=dep&dn=tp`);
    expect(screen.getByText('DP attention + EP 24, PP 3, dense TP')).toBeTruthy();
    expect(screen.queryByRole('spinbutton', { name: 'TP' })).toBeNull();
    expect(screen.queryByRole('spinbutton', { name: 'DP' })).toBeNull();
    expect((screen.getByRole('spinbutton', { name: 'EP' }) as HTMLInputElement).value).toBe('24');
    expect(screen.getByRole('spinbutton', { name: 'PP' })).toBeTruthy();
    const dense = screen.getByRole('radiogroup', { name: 'Dense layers' });
    expect((dense.querySelector('input[value="tp"]') as HTMLInputElement).checked).toBe(true);
    // The hints list exactly the coupled splits EP × PP = 72 that keep dense TP valid (EP divides 12288).
    expect(screen.getByText('Valid EP: 2, 3, 4, 6, 8, 12, 24')).toBeTruthy();
    expect(screen.getByText('Valid PP: 3, 6, 9, 12, 18, 24, 36')).toBeTruthy();
  });

  it('dense TP at EP 72 shows the error on the dense-layers control', () => {
    panel(`m=${ID}&d=gb200&n=72&b=72&dp=72&ep=72&lay=dep&dn=tp`);
    expect(
      screen.getByText('Dense layers on TP need EP 72 to divide their width 12288. Use DP.'),
    ).toBeTruthy();
    expect(
      screen.getByRole('radiogroup', { name: 'Dense layers' }).getAttribute('aria-invalid'),
    ).toBe('true');
  });

  it('picking the layout dispatches it, then a valid size; EP edits move DP and PP; PP refills the stage', () => {
    const dispatch = panel(`m=${ID}&d=gb200&n=72&b=72`);
    fireEvent.click(screen.getByRole('radio', { name: 'DP attention + EP' }));
    expect(dispatch.mock.calls.map((c) => c[0])).toEqual([
      { type: 'setParallelLayout', layout: 'dpAttentionEp' },
      { type: 'setParallel', parallel: { tp: 1, pp: 1, dp: 72, ep: 72 } },
    ]);
    cleanup();
    const dep = panel(`m=${ID}&d=gb200&n=72&b=72&dp=72&ep=72&lay=dep`);
    fireEvent.change(screen.getByRole('spinbutton', { name: 'EP' }), { target: { value: '36' } });
    fireEvent.change(screen.getByRole('spinbutton', { name: 'PP' }), { target: { value: '3' } });
    fireEvent.click(screen.getByRole('radio', { name: 'TP' }));
    expect(dep.mock.calls.map((c) => c[0])).toEqual([
      { type: 'setParallel', parallel: { ep: 36, dp: 36, pp: 2 } },
      { type: 'setParallel', parallel: { pp: 3, ep: 24, dp: 24 } },
      { type: 'setParallel', parallel: { denseTpGroup: 'ep' } },
    ]);
  });

  it('a dense model shows no layout control, even with lay=dep in the URL', () => {
    panel('dp=1&lay=dep', REFERENCE_MODEL);
    expect(screen.queryByRole('radiogroup', { name: 'Layout' })).toBeNull();
    expect(screen.getByRole('spinbutton', { name: 'TP' })).toBeTruthy();
    expect(screen.queryByRole('spinbutton', { name: 'EP' })).toBeNull();
  });

  it('EP imbalance and redundant experts show only at EP > 1, in both layouts, with the defaults 1.2 and 0', () => {
    panel(`m=${ID}&d=gb200&n=72&b=72`);
    expect(screen.queryByRole('spinbutton', { name: 'EP imbalance' })).toBeNull();
    expect(screen.queryByRole('spinbutton', { name: 'Redundant experts' })).toBeNull();
    expect(screen.queryByRole('radiogroup', { name: 'Imbalance on all-to-all' })).toBeNull();
    cleanup();
    for (const q of [
      `m=${ID}&d=gb200&n=72&b=72&dp=72&ep=72`,
      `m=${ID}&d=gb200&n=72&b=72&dp=72&ep=72&lay=dep`,
    ]) {
      const dispatch = panel(q);
      const rho = screen.getByRole('spinbutton', { name: 'EP imbalance' }) as HTMLInputElement;
      const replicas = screen.getByRole('spinbutton', {
        name: 'Redundant experts',
      }) as HTMLInputElement;
      expect([rho.value, rho.step, replicas.value]).toEqual(['1.2', '0.01', '0']);
      fireEvent.change(rho, { target: { value: '1.55' } });
      fireEvent.change(replicas, { target: { value: '1' } });
      const a2a = screen.getByRole('radiogroup', { name: 'Imbalance on all-to-all' });
      expect((a2a.querySelector('input[value="off"]') as HTMLInputElement).checked).toBe(true);
      fireEvent.click(screen.getByRole('radio', { name: 'On' }));
      expect(dispatch.mock.calls.map((c) => c[0])).toEqual([
        { type: 'setParallel', parallel: { epImbalance: 1.55 } },
        { type: 'setParallel', parallel: { redundantExpertsPerRank: 1 } },
        { type: 'setParallel', parallel: { epImbalanceScalesAllToAll: true } },
      ]);
      cleanup();
    }
    panel(`m=${ID}&d=gb200&n=72&b=72&dp=72&ep=72&epi=1.01&epa=1&rex=1`);
    expect(
      (screen.getByRole('spinbutton', { name: 'EP imbalance' }) as HTMLInputElement).value,
    ).toBe('1.01');
    const on = screen.getByRole('radiogroup', { name: 'Imbalance on all-to-all' });
    expect((on.querySelector('input[value="on"]') as HTMLInputElement).checked).toBe(true);
  });

  it('a dense model never shows the EP balance fields', () => {
    panel('n=2&dp=2&ep=2&epi=1.5&rex=2&b=2', REFERENCE_MODEL);
    expect(screen.queryByRole('spinbutton', { name: 'EP imbalance' })).toBeNull();
    expect(screen.queryByRole('spinbutton', { name: 'Redundant experts' })).toBeNull();
    expect(screen.queryByRole('radiogroup', { name: 'Imbalance on all-to-all' })).toBeNull();
  });
});
