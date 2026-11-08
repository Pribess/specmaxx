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

describe('quant picker defaults', () => {
  it('labels the default option with the resolved format and updates when the model changes', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    mount(async (id) =>
      mapHfModel(fixtureRepoFiles(id, id.includes('EXL3') ? { dir: 'quant' } : {})),
    );
    await screen.findByText(/qwen:/);
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).toMatch(
        /Model default \(bf16\)/,
      ),
    );
    expect(screen.getByRole('combobox', { name: 'KV cache' }).textContent).toMatch(
      /Model default \(bf16\)/,
    );
    const input = screen.getByRole('combobox', { name: 'Hugging Face model ID' });
    fireEvent.change(input, {
      target: { value: 'bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw' },
    });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).toMatch(
        /Model default \(exl3 2\.52bpw\)/,
      ),
    );
    fireEvent.click(screen.getByRole('combobox', { name: 'Weights' }));
    expect(await screen.findByRole('option', { name: 'EXL3 2.52bpw (model default)' })).toBeTruthy();
  });
});

describe('estimator inputs', () => {
  it('rejects TP and world-size errors without computing a result', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    mount();
    await screen.findByText(/qwen:/);
    await screen.findByTestId('prefill-tps');
    fireEvent.change(screen.getByLabelText(/^TP/), { target: { value: '3' } });
    await waitFor(() => expect(screen.queryByTestId('prefill-tps')).toBeNull());
    expect(screen.getByLabelText(/^TP/).getAttribute('aria-invalid')).toBe('true');
    expect(screen.getByText(/Valid TP:/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText(/^TP/), { target: { value: '1' } });
    expect(screen.queryByLabelText(/^DP/)).toBeNull();
    expect(screen.queryByLabelText(/^EP/)).toBeNull();
    fireEvent.change(screen.getByLabelText(/^TP/), { target: { value: '2' } });
    expect(screen.getByRole('spinbutton', { name: /^(GPUs|Devices) in one machine$/ }).getAttribute('aria-invalid')).toBe('true');
    expect(screen.getAllByText(/TP × PP = 2, but the rig has 1 device/).length).toBeGreaterThan(0);
  });
  it('restores the URL and writes canonical state after a change', async () => {
    history.replaceState(
      null,
      '',
      '/?m=Qwen/Qwen3.8-27B&d=rtx-4090&u=vram-mod.48gb&n=2&tp=2&kv=fp8_e4m3&b=4',
    );
    mount();
    await screen.findByText(/qwen:/);
    expect(screen.getByRole('combobox', { name: 'Device' }).textContent).toMatch(/RTX 4090/);
    expect(screen.getByRole('combobox', { name: 'KV cache' }).textContent).toMatch(/fp8/i);
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
    let gated = true;
    const loader = vi.fn<ModelLoader>().mockImplementation(async (id, options) => {
      if (id === 'zai-org/GLM-5.3' && gated) {
        gated = false;
        return { kind: 'unauthorized', message: 'Gated model. Add a token.', status: 401 };
      }
      return fixtureLoader(id, options);
    });
    mount(loader);
    fireEvent.change(screen.getByRole('combobox', { name: 'Hugging Face model ID' }), {
      target: { value: 'zai-org/GLM-5.3' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    fireEvent.change(await screen.findByLabelText('Hugging Face token'), {
      target: { value: 'hf_test_secret' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save token and retry' }));
    await waitFor(() => expect(loader.mock.calls.at(-1)?.[1]?.token).toBe('hf_test_secret'));
    expect(localStorage.getItem(HF_TOKEN_KEY)).toBe('hf_test_secret');
    expect(location.search).not.toContain('secret');
    fireEvent.click(screen.getByRole('button', { name: 'Forget token' }));
    expect(localStorage.getItem(HF_TOKEN_KEY)).toBeNull();
  });
  it('resets quantization overrides to the model defaults', async () => {
    mount();
    fireEvent.click(screen.getByRole('combobox', { name: 'Weights' }));
    fireEvent.click(await screen.findByRole('option', { name: /int4/i }));
    expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).toMatch(/int4/i);
    fireEvent.click(screen.getByRole('button', { name: 'Reset to model default' }));
    expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).not.toMatch(/int4/i);
    expect(screen.queryByRole('button', { name: 'Reset to model default' })).toBeNull();
    await screen.findByTestId('decode-tps');
  });
  it('EXL3 weights at 2.5 bpw change the result and survive a URL round-trip', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    const first = mount();
    await screen.findByText(/qwen:/);
    const decodeBf16 = (await screen.findByTestId('decode-tps')).textContent;
    expect(screen.queryByLabelText('Bits per weight')).toBeNull();
    fireEvent.click(screen.getByRole('combobox', { name: 'Weights' }));
    fireEvent.click(await screen.findByRole('option', { name: 'EXL3 2.5bpw' }));
    await waitFor(() => expect(new URLSearchParams(location.search).get('bpw')).toBe('2.5'));
    expect(new URLSearchParams(location.search).get('w')).toBe('exl3');
    expect(screen.queryByLabelText('Bits per weight')).toBeNull();
    expect(screen.getByLabelText('Head bits per weight')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'About Head bits per weight' })).toBeTruthy(); // the "?" help icon
    const decode25 = screen.getByTestId('decode-tps').textContent;
    expect(decode25).not.toBe(decodeBf16);

    fireEvent.click(screen.getByRole('combobox', { name: 'Weights' }));
    fireEvent.click(await screen.findByRole('option', { name: 'EXL3 custom…' }));
    const bpw = screen.getByLabelText('Bits per weight') as HTMLInputElement;
    expect(bpw.value).toBe('2.5');
    expect(bpw.step).toBe('0.05');
    fireEvent.change(bpw, { target: { value: '4' } });
    await waitFor(() => expect(screen.getByTestId('decode-tps').textContent).not.toBe(decode25));
    fireEvent.change(bpw, { target: { value: '2.5' } });
    await waitFor(() => expect(screen.getByTestId('decode-tps').textContent).toBe(decode25));

    // Round trip: a fresh app from the written URL selects the 2.5 preset and shows the same result.
    const url = location.search;
    first.unmount();
    history.replaceState(null, '', `/${url}`);
    mount();
    await screen.findByText(/qwen:/);
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).toMatch(/EXL3 2\.5bpw/),
    );
    expect(screen.queryByLabelText('Bits per weight')).toBeNull();
    await waitFor(() => expect(screen.getByTestId('decode-tps').textContent).toBe(decode25));
  });
  it('ignores stale HF responses', async () => {
    const pending = new Map<string, (value: Awaited<ReturnType<ModelLoader>>) => void>();
    const loader: ModelLoader = (id) =>
      id === 'Qwen/Qwen3.8-27B'
        ? fixtureLoader(id)
        : new Promise((r) => {
            pending.set(id, r);
          });
    mount(loader);
    fireEvent.change(screen.getByRole('combobox', { name: 'Hugging Face model ID' }), {
      target: { value: 'zai-org/GLM-5.3' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    fireEvent.change(screen.getByRole('combobox', { name: 'Hugging Face model ID' }), {
      target: { value: 'Qwen/Qwen3.8-Flash-Next' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    pending.get('Qwen/Qwen3.8-Flash-Next')?.(mapHfModel(fixtureRepoFiles('Qwen/Qwen3.8-Flash-Next')));
    await screen.findByTestId('decode-tps');
    pending.get('zai-org/GLM-5.3')?.(mapHfModel(fixtureRepoFiles('zai-org/GLM-5.3')));
    await new Promise((r) => setTimeout(r, 300));
    expect(location.search).toContain('Flash-Next');
    expect(screen.queryByText(/glm/i, { selector: '[data-testid="model-facts"] *' })).toBeNull();
  });
  it('keeps revision and quant overrides when reloading the same model', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&rev=test-revision&w=int4');
    const loader = vi.fn<ModelLoader>().mockImplementation(fixtureLoader);
    mount(loader);
    await screen.findByText(/qwen:/);
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    await waitFor(() => expect(loader).toHaveBeenCalledTimes(2));
    expect(loader.mock.calls.at(-1)?.[1]?.revision).toBe('test-revision');
    expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).toMatch(/int4/i);
    await waitFor(() =>
      expect(new URLSearchParams(location.search).get('rev')).toBe('test-revision'),
    );
  });
  it('explains field errors in the results pane without assuming the input layout', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&n=2&tp=3');
    const { container } = mount();
    await screen.findByText(/qwen:/);
    const output = container.querySelector('.output')!;
    expect(output.textContent).toMatch(/TP × PP = 3, but the rig has 2 devices/);
    expect(output.textContent).not.toMatch(/inputs below/);
  });
  it('opens on the default state: Qwen3.8-27B, 8k / 8k, numbers without any loader call', async () => {
    const loader = vi.fn<ModelLoader>().mockRejectedValue(new Error('offline'));
    mount(loader);
    expect((screen.getByLabelText(/^Prefill length/) as HTMLInputElement).value).toBe('8192');
    expect((screen.getByLabelText(/^Decode context/) as HTMLInputElement).value).toBe('8192');
    expect(screen.getByTestId('model-facts').textContent).toMatch(/\d+(\.\d+)?B total, \d+(\.\d+)?B active, \d+ layers/);
    expect((await screen.findByTestId('decode-tps')).textContent).toMatch(/[1-9]/);
    expect(screen.queryByRole('alert')).toBeNull();
    await waitFor(() => expect(document.title).toMatch(/@ 8k/));
  });
  it('picking a multi-device setup sets the counts and links in the URL', async () => {
    mount();
    fireEvent.click(screen.getByRole('combobox', { name: 'Device' }));
    fireEvent.click(await screen.findByRole('option', { name: /RTX 5090 ×4/ }));
    expect(screen.getByRole('combobox', { name: 'Device' }).textContent).toMatch(/RTX 5090 ×4/);
    await waitFor(() => expect(new URLSearchParams(location.search).get('n')).toBe('4'));
    expect(new URLSearchParams(location.search).get('il')).toBeTruthy();
  });
  it.each([['DGX Spark ×4', ''], ['R9700 ×4', '?n=2&tp=2'], ['DGX Spark ×4', '?n=3&pp=3']])(
    'picking %s from %j chooses TP and PP that use all 4 devices',
    async (name, query) => {
      history.replaceState(null, '', `/${query}${query ? '&' : '?'}m=Qwen/Qwen3.8-27B`);
      mount();
      await screen.findByTestId('prefill-tps');
      fireEvent.click(screen.getByRole('combobox', { name: 'Device' }));
      fireEvent.click(await screen.findByRole('option', { name: new RegExp(name) }));
      await waitFor(() => expect(screen.getByTestId('device-total').textContent).toMatch(/= 4 /));
      const tp = Number((screen.getByLabelText(/^TP/) as HTMLInputElement).value);
      const pp = Number((screen.getByLabelText(/^PP/) as HTMLInputElement).value);
      expect(tp * pp).toBe(4);
      expect(await screen.findByTestId('prefill-tps')).toBeTruthy();
      expect(screen.queryByText(/unused|must be equal/)).toBeNull();
    },
  );
  it('a huge device count shows validation errors instead of hanging', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    const { container } = mount();
    await screen.findByTestId('prefill-tps');
    (container.querySelector('details.advanced') as HTMLDetailsElement).open = true;
    fireEvent.change(screen.getByRole('spinbutton', { name: /^(GPUs|Devices) in one machine$/ }), { target: { value: '18014398509481984' } });
    await waitFor(() => expect(screen.queryByTestId('prefill-tps')).toBeNull());
    expect(screen.getAllByText(/must be equal|TP × PP/).length).toBeGreaterThan(0);
  });
  it('TP × PP below the device count blocks the simulation', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&n=4&tp=2');
    mount();
    await screen.findByText(/qwen:/);
    expect(await screen.findAllByText(/TP × PP = 2, but the rig has 4 devices/)).not.toHaveLength(0);
    expect(screen.queryByTestId('prefill-tps')).toBeNull();
  });
  it('filters the device list by search and selects with the keyboard', async () => {
    mount();
    fireEvent.click(screen.getByRole('combobox', { name: 'Device' }));
    const search = await screen.findByRole('combobox', { name: /search/i });
    fireEvent.change(search, { target: { value: '4090' } });
    const options = screen.getAllByRole('option');
    expect(options.length).toBeGreaterThan(0);
    expect(options.every((o) => /4090/.test(o.textContent ?? ''))).toBe(true);
    fireEvent.keyDown(search, { key: 'ArrowDown' });
    fireEvent.keyDown(search, { key: 'Enter' });
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Device' }).textContent).toMatch(/4090/));
  });
  it('keeps Advanced closed until opened, then its subsections are open', async () => {
    const { container } = mount();
    await screen.findByTestId('decode-tps');
    const advanced = container.querySelector('details.advanced') as HTMLDetailsElement;
    expect(advanced.open).toBe(false);
    advanced.open = true;
    expect((container.querySelector('details.quick-picks') as HTMLDetailsElement).open).toBe(true);
    expect((container.querySelector('details.spec-editor') as HTMLDetailsElement).open).toBe(true);
  });
  it('labels the counts in plain language with a total', async () => {
    history.replaceState(null, '', '/?n=4');
    mount();
    expect(screen.getByRole('spinbutton', { name: 'GPUs in one machine' })).toBeTruthy();
    expect(screen.getByRole('spinbutton', { name: 'Machines' })).toBeTruthy();
    expect(screen.getByTestId('device-total').textContent).toMatch(/= 4 GPUs total/);
  });
  it('typing a model ID and pressing Enter loads it', async () => {
    const loader = vi.fn<ModelLoader>().mockImplementation(fixtureLoader);
    mount(loader);
    const input = screen.getByRole('combobox', { name: 'Hugging Face model ID' });
    fireEvent.change(input, { target: { value: 'zai-org/GLM-5.3-Flash' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(loader.mock.calls.some((c) => c[0] === 'zai-org/GLM-5.3-Flash')).toBe(true));
  });
});
