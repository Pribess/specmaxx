// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { TINY_CSA, TINY_MLA_SPARSE } from '../../sim/models/reference';
import {
  buildSimInput,
  decodeState,
  BUNDLED_DEFAULT_MODEL,
  type AppState,
  type LoadedModel,
} from '../state';
import { OffloadAdvancedFields } from './OffloadSection';

afterEach(cleanup);

function fields(query: string, loaded: LoadedModel = BUNDLED_DEFAULT_MODEL) {
  const state: AppState = decodeState(query).state;
  const dispatch = vi.fn();
  render(
    <OffloadAdvancedFields
      state={state}
      dispatch={dispatch}
      loaded={loaded}
      built={buildSimInput(state, loaded)}
    />,
  );
  return dispatch;
}

const spin = (name: string) =>
  screen.queryByRole('spinbutton', { name }) as HTMLInputElement | null;

describe('OffloadAdvancedFields: KV cache', () => {
  it('off: only the GPU share, at 100; typing a share turns it on in host RAM', () => {
    const dispatch = fields('');
    expect(spin('KV cache on GPU %')?.value).toBe('100');
    expect(screen.queryByText('KV cache tier')).toBeNull();
    expect(spin('Prefix KV on GPU %')).toBeNull();
    fireEvent.change(spin('KV cache on GPU %')!, { target: { value: '25' } });
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'setOffload',
        offload: expect.objectContaining({
          kvCache: { tier: 'host_ram', deviceFraction: 0.25 },
        }) as unknown,
      }),
    );
  });

  it('on: the tier picker; a split adds the host-RAM share; prefix hits add the prefix share', () => {
    fields('okt=split&okd=25&okr=50&ph=50');
    expect(screen.getByText('KV cache tier')).toBeTruthy();
    expect(spin('KV in host RAM %')?.value).toBe('50');
    expect(spin('Prefix KV on GPU %')?.value).toBe('25');
    // A dense model has no top-k layers: no hit model.
    expect(screen.queryByText('KV hit model')).toBeNull();
  });

  it('top-k models get the hit model; a custom one shows its ratio', () => {
    const sparse = { ...BUNDLED_DEFAULT_MODEL, model: TINY_MLA_SPARSE };
    fields('okt=ram&okd=50&okh=80', sparse);
    expect(screen.getByText('KV hit model')).toBeTruthy();
    expect(spin('KV hit ratio %')?.value).toBe('80');
  });

  it('a model without a growing KV cache shows no KV fields', () => {
    fields('', { ...BUNDLED_DEFAULT_MODEL, model: TINY_CSA });
    expect(screen.queryByTestId('kv-offload')).toBeNull();
  });
});
