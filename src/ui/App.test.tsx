// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { App, type ModelLoader } from './App';
import { inlineRunner } from './sim/runner';
import { mapHfModel } from '../sim/models/hf';
import { fixtureRepoFiles } from '../sim/models/__fixtures__/fixtures.testutil';
import { HF_TOKEN_KEY } from '../hf/token';
import { decodeState, encodeState } from './state';

const fixtureLoader: ModelLoader = async (id) => mapHfModel(fixtureRepoFiles(id));
function mount(loader = fixtureLoader) {
  return render(<App loader={loader} runner={inlineRunner} />);
}
beforeEach(() => {
  history.replaceState(null, '', '/');
  localStorage.clear();
});
afterEach(cleanup);

describe('estimator inputs', () => {
  it('rejects TP, EP, world-size and batch / DP errors without computing a result', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    mount();
    await screen.findByText(/qwen:/);
    await screen.findByTestId('prefill-tps');
    fireEvent.change(screen.getByLabelText(/^TP/), { target: { value: '3' } });
    await waitFor(() => expect(screen.queryByTestId('prefill-tps')).toBeNull());
    expect(screen.getByLabelText(/^TP/).getAttribute('aria-invalid')).toBe('true');
    expect(screen.getByText(/Valid TP:/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText(/^TP/), { target: { value: '1' } });
    fireEvent.change(screen.getByLabelText(/^EP/), { target: { value: '2' } });
    expect(screen.getByLabelText(/^EP/).getAttribute('aria-invalid')).toBe('true');
    fireEvent.change(screen.getByLabelText(/^EP/), { target: { value: '1' } });
    fireEvent.change(screen.getByLabelText(/^DP/), { target: { value: '2' } });
    expect(screen.getByLabelText(/^Batch size/).getAttribute('aria-invalid')).toBe('true');
    expect(screen.queryByTestId('decode-tps')).toBeNull();
    fireEvent.change(screen.getByLabelText(/^Batch size/), { target: { value: '2' } });
    expect(screen.getByLabelText(/^Devices per node/).getAttribute('aria-invalid')).toBe('true');
    expect(screen.getAllByText(/needs 2 devices; the rig has 1/).length).toBeGreaterThan(0);
  });
  it('restores the URL and writes canonical state after a change', async () => {
    history.replaceState(
      null,
      '',
      '/?m=Qwen/Qwen3.8-27B&d=rtx-4090&u=vram-mod.48gb&n=2&tp=2&kv=fp8_e4m3&b=4',
    );
    mount();
    await screen.findByText(/qwen:/);
    expect((screen.getByLabelText('Device preset') as HTMLSelectElement).value).toBe('rtx-4090');
    expect((screen.getByLabelText('KV cache') as HTMLSelectElement).value).toBe('fp8_e4m3');
    expect((screen.getByLabelText(/^TP/) as HTMLInputElement).value).toBe('2');
    expect((screen.getByLabelText(/^Batch size/) as HTMLInputElement).value).toBe('4');
    fireEvent.change(screen.getByLabelText(/^Batch size/), { target: { value: '8' } });
    await waitFor(() => expect(new URLSearchParams(location.search).get('b')).toBe('8'));
    expect(location.search.slice(1)).toBe(encodeState(decodeState(location.search).state));
    expect(sessionStorage.getItem('specmaxx:last-query')).toBe(location.search);
    history.replaceState(null, '', '/?b=2');
    fireEvent.popState(window);
    await waitFor(() =>
      expect((screen.getByLabelText(/^Batch size/) as HTMLInputElement).value).toBe('2'),
    );
  });
  it('prompts for a gated token, stores it, retries and forgets it without leaking into the URL', async () => {
    const loader = vi
      .fn<ModelLoader>()
      .mockResolvedValueOnce({
        kind: 'unauthorized',
        message: 'Gated model. Add a token.',
        status: 401,
      })
      .mockImplementation(fixtureLoader);
    mount(loader);
    fireEvent.change(screen.getByLabelText('Hugging Face model ID'), {
      target: { value: 'Qwen/Qwen3.8-27B' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    fireEvent.change(await screen.findByLabelText('Hugging Face token'), {
      target: { value: 'hf_test_secret' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save token and retry' }));
    await screen.findByText(/qwen:/);
    expect(localStorage.getItem(HF_TOKEN_KEY)).toBe('hf_test_secret');
    expect(loader.mock.calls.at(-1)?.[1]?.token).toBe('hf_test_secret');
    expect(location.search).not.toContain('secret');
    fireEvent.click(screen.getByRole('button', { name: 'Forget token' }));
    expect(localStorage.getItem(HF_TOKEN_KEY)).toBeNull();
  });
  it('resets quantization overrides to the model defaults', async () => {
    mount();
    fireEvent.change(screen.getByLabelText('Weights'), { target: { value: 'int4' } });
    expect((screen.getByLabelText('Weights') as HTMLSelectElement).value).toBe('int4');
    fireEvent.click(screen.getByRole('button', { name: 'Reset to model default' }));
    expect((screen.getByLabelText('Weights') as HTMLSelectElement).value).toBe('');
    expect(screen.queryByRole('button', { name: 'Reset to model default' })).toBeNull();
    await screen.findByTestId('decode-tps');
  });
  it('ignores stale HF responses', async () => {
    let resolve!: (value: Awaited<ReturnType<ModelLoader>>) => void;
    const loader: ModelLoader = () =>
      new Promise((r) => {
        resolve = r;
      });
    mount(loader);
    fireEvent.change(screen.getByLabelText('Hugging Face model ID'), {
      target: { value: 'Qwen/Qwen3.8-27B' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    fireEvent.click(screen.getByRole('button', { name: 'Use built-in reference' }));
    resolve(mapHfModel(fixtureRepoFiles('Qwen/Qwen3.8-27B')));
    await screen.findByTestId('decode-tps');
    expect(screen.getByText('Built-in dense reference')).toBeTruthy();
    expect(screen.queryByText(/qwen:/)).toBeNull();
  });
  it('keeps revision and quant overrides when reloading the same model', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&rev=test-revision&w=int4');
    const loader = vi.fn<ModelLoader>().mockImplementation(fixtureLoader);
    mount(loader);
    await screen.findByText(/qwen:/);
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    await waitFor(() => expect(loader).toHaveBeenCalledTimes(2));
    expect(loader.mock.calls.at(-1)?.[1]?.revision).toBe('test-revision');
    expect((screen.getByLabelText('Weights') as HTMLSelectElement).value).toBe('int4');
    await waitFor(() =>
      expect(new URLSearchParams(location.search).get('rev')).toBe('test-revision'),
    );
  });
  it('explains field errors in the results pane without assuming the input layout', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&n=2&tp=3');
    const { container } = mount();
    await screen.findByText(/qwen:/);
    const output = container.querySelector('.output')!;
    expect(output.textContent).toMatch(/needs 3 devices; the rig has 2/);
    expect(output.textContent).not.toMatch(/inputs below/);
  });
});
