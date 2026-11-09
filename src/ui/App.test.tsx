// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { App, type ModelLoader } from './App';
import { inlineRunner } from './sim/runner';
import { FAMILIES, mapHfModel, type FamilyEntry } from '../sim/models/hf';
import { REF_DENSE_8B } from '../sim/models/reference';
import { fixtureRepoFiles, fixtureStem, readFixtureJson } from '../sim/models/__fixtures__/fixtures.testutil';
import { createCache } from '../hf/cache';
import { isHfError } from '../hf/errors';
import { fetchHfRepoFiles } from '../hf/fetch';
import { memoryStorage } from '../hf/storage';
import { HF_TOKEN_KEY } from '../hf/token';
import { apiUrl, resolveUrl } from '../hf/urls';
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

describe('lookup-table picker', () => {
  it('shows only for models with n-gram tables and labels the resolved default', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    mount();
    await screen.findByText(/qwen:/);
    await screen.findByTestId('prefill-tps');
    expect(screen.queryAllByRole('combobox', { name: 'Lookup tables' })).toHaveLength(0);
    cleanup();
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-Flash-Next&lt=nvfp4');
    mount();
    await screen.findByTestId('prefill-tps');
    // The Quantization panel's picker comes before the offload panel's placement picker of the same name.
    const picker = (await screen.findAllByRole('combobox', { name: 'Lookup tables' }))[0];
    expect(picker.textContent).toMatch(/nvfp4/);
    fireEvent.click(picker);
    expect(await screen.findByRole('option', { name: /Model default \(bf16\)/ })).toBeTruthy();
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
    expect(screen.getByRole('radiogroup', { name: 'Machines' }).getAttribute('aria-invalid')).toBe('true');
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
  it('picking a popular setup sets the counts and links in the URL', async () => {
    mount();
    fireEvent.click(screen.getByRole('combobox', { name: 'Device' }));
    expect(await screen.findByRole('group', { name: 'Popular setups' })).toBeTruthy();
    fireEvent.click(await screen.findByRole('option', { name: /RTX 5090 32GB ×2/ }));
    expect(screen.getByRole('combobox', { name: 'Device' }).textContent).toMatch(/RTX 5090 32GB ×2/);
    await waitFor(() => expect(new URLSearchParams(location.search).get('n')).toBe('2'));
    expect(new URLSearchParams(location.search).get('il')).toBeTruthy();
  });
  it('picking the DGX Spark ×4 ring writes fab=ring and names the cabling in the hero', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    mount();
    await screen.findByTestId('prefill-tps');
    fireEvent.click(screen.getByRole('combobox', { name: 'Device' }));
    fireEvent.click(await screen.findByRole('option', { name: /DGX Spark.*×4 · ConnectX-7 ring/ }));
    await waitFor(() => expect(new URLSearchParams(location.search).get('fab')).toBe('ring'));
    expect(screen.getByRole('combobox', { name: 'Cabling' }).textContent).toBe('Ring');
    await waitFor(() => expect(document.querySelector('.rig')?.textContent).toMatch(/on 4 × NVIDIA DGX Spark \(GB10, 128GB memory\) over ConnectX-7 \(ring\), with/));
    // Changing the cabling leaves the setup: the trigger shows the device again.
    fireEvent.click(screen.getByRole('combobox', { name: 'Cabling' }));
    fireEvent.click(await screen.findByRole('option', { name: 'Switch' }));
    await waitFor(() => expect(new URLSearchParams(location.search).get('fab')).toBeNull());
    expect(screen.getByRole('combobox', { name: 'Device' }).textContent).toMatch(/×4 · ConnectX-7 switch/);
    expect(document.querySelector('.rig')?.textContent).not.toMatch(/\(ring\)/);
  });
  it('offers only the cablings the ports allow, and one as plain text', async () => {
    history.replaceState(null, '', '/?d=ryzen-ai-max-plus-395&nodes=3');
    mount();
    await screen.findByTestId('setup-row');
    fireEvent.click(screen.getByRole('combobox', { name: 'Cabling' }));
    const names = (await screen.findAllByRole('option')).map((o) => o.textContent?.replace('✓', ''));
    expect(names).toEqual(['Full mesh', 'Daisy chain']);
    fireEvent.keyDown(screen.getByRole('combobox', { name: 'Cabling' }), { key: 'Escape' });
    cleanup();
    history.replaceState(null, '', '/?d=dgx-spark&nodes=2');
    mount();
    await screen.findByTestId('setup-row');
    expect(screen.queryByRole('combobox', { name: 'Cabling' })).toBeTruthy();
    cleanup();
    history.replaceState(null, '', '/?n=2&nodes=2');
    mount();
    await screen.findByTestId('setup-row');
    expect(screen.queryByRole('combobox', { name: 'Cabling' })).toBeNull();
    expect(screen.getByTestId('cabling').textContent).toBe('Switch');
  });
  it('offers the 2-card NVLink bridge only at 2 GPUs, and more GPUs fall back to PCIe', async () => {
    history.replaceState(null, '', '/?d=rtx-3090&n=2&il=rtx-3090-nvlink-bridge');
    mount();
    await screen.findByTestId('setup-row');
    fireEvent.click(screen.getByRole('combobox', { name: 'Local link' }));
    expect((await screen.findAllByRole('option')).some((o) => /NVLink bridge/.test(o.textContent ?? ''))).toBe(true);
    fireEvent.keyDown(screen.getByRole('combobox', { name: 'Local link' }), { key: 'Escape' });
    const perMachine = screen.getByRole('radiogroup', { name: 'GPUs in one machine' });
    fireEvent.click(within(perMachine).getByRole('radio', { name: '4' }));
    await waitFor(() => expect(new URLSearchParams(location.search).get('il')).toBeNull());
    expect(new URLSearchParams(location.search).get('n')).toBe('4');
    fireEvent.click(screen.getByRole('combobox', { name: 'Local link' }));
    const options = await screen.findAllByRole('option');
    expect(options.length).toBeGreaterThan(0);
    expect(options.some((o) => /NVLink bridge/.test(o.textContent ?? ''))).toBe(false);
  });
  it.each([['×4 · ConnectX-7 switch', ''], ['×4 · USB4 ring', '?n=2&tp=2'], ['×4 · ConnectX-7 ring', '?n=3&pp=3']])(
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
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&n=1024');
    mount();
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
  it('shows the prefetch accuracy only with offload, 80 by default, and writes opf', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-Flash-Next');
    mount();
    await screen.findByTestId('decode-tps');
    expect(screen.queryByLabelText('Prefetch accuracy %')).toBeNull();
    cleanup();
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-Flash-Next&ox=ram');
    mount();
    await screen.findByTestId('decode-tps');
    const field = screen.getByLabelText('Prefetch accuracy %') as HTMLInputElement;
    expect(field.value).toBe('80');
    fireEvent.change(field, { target: { value: '90' } });
    await waitFor(() => expect(location.search).toContain('opf=0.9'));
    fireEvent.change(screen.getByLabelText('Prefetch accuracy %'), { target: { value: '80' } });
    await waitFor(() => expect(location.search).not.toContain('opf='));
  });
  it('keeps Advanced closed until opened, then its subsections are open', async () => {
    const { container } = mount();
    await screen.findByTestId('decode-tps');
    const advanced = container.querySelector('details.advanced') as HTMLDetailsElement;
    expect(advanced.open).toBe(false);
    advanced.open = true;
    expect((container.querySelector('details.spec-editor') as HTMLDetailsElement).open).toBe(true);
  });
  it.each([
    { name: 'offload off', query: '?m=Qwen/Qwen3.8-27B', offload: [] },
    { name: 'dense offload on', query: '?m=Qwen/Qwen3.8-27B&odl=32', offload: ['Queue depth', 'Prefetch accuracy %'] },
    { name: 'a lookup-table model', query: '?m=Qwen/Qwen3.8-Flash-Next', offload: ['Lookup tables'] },
    {
      name: 'expert offload on a lookup-table model',
      query: '?m=Qwen/Qwen3.8-Flash-Next&ox=ram',
      offload: ['Lookup tables', 'Queue depth', 'Prefetch accuracy %'],
    },
  ])('shows one Advanced, with an Offload subsection only when it applies ($name)', async ({ query, offload }) => {
    history.replaceState(null, '', `/${query}`);
    const { container } = mount();
    await screen.findByTestId('decode-tps');
    const summaries = [...container.querySelectorAll('summary')].filter((s) => s.textContent === 'Advanced');
    expect(summaries).toHaveLength(1);
    const advanced = summaries[0].parentElement as HTMLDetailsElement;
    expect(advanced.open).toBe(false);
    expect(within(screen.getByTestId('offload-section')).queryByText('Advanced')).toBeNull();
    const section = screen.queryByTestId('offload-advanced');
    if (offload.length === 0) {
      expect(section).toBeNull();
      return;
    }
    expect(section).not.toBeNull();
    expect(advanced.contains(section)).toBe(true);
    // The subsection comes first, before the device unlocks and the spec editor.
    expect(section!.compareDocumentPosition(advanced.querySelector('details.spec-editor')!)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
    expect(section!.querySelector('legend')!.textContent).toBe('Offload');
    const labels = [...section!.querySelectorAll('.field-label')].map((l) => l.textContent?.trim());
    expect(labels).toEqual(offload);
  });
  it('labels the counts in plain language with a total', async () => {
    history.replaceState(null, '', '/?n=4');
    mount();
    expect(screen.getByRole('radiogroup', { name: 'GPUs in one machine' })).toBeTruthy();
    expect((within(screen.getByRole('radiogroup', { name: 'GPUs in one machine' })).getByRole('radio', { name: '4' }) as HTMLInputElement).checked).toBe(true);
    expect(screen.getByRole('radiogroup', { name: 'Machines' })).toBeTruthy();
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

describe('config read from another repo', () => {
  /**
   * Mentions of where a config was read from; the user wants none of them shown. `unsloth/` is the mirror's org as
   * a repo path (a hardware note names "Unsloth" the trainer, which is unrelated).
   */
  const FROM_TEXT = /read from|unsloth\//i;
  const official = 'meta-models/Muse-Spark-Open-816B-A42B';
  const mirror = 'unsloth/Muse-Spark-Open-816B-A42B';

  /** The Hub without a token: the official repo answers 401, the mirror serves its checked-in fixtures. */
  const gatedHub = (async (url: string) => {
    if (url.includes(official)) return new Response('', { status: 401 });
    const kind = url === apiUrl(mirror) ? 'api' : url === resolveUrl(mirror, 'config.json') ? 'config' : null;
    const body = kind && readFixtureJson(`${fixtureStem(mirror)}.${kind}.json`);
    return body ? new Response(JSON.stringify(body)) : new Response('', { status: 404 });
  }) as unknown as typeof fetch;
  /** A stand-in muse_spark family (a dense 8B shape) used only until the real family is registered. */
  const stub: FamilyEntry = {
    family: 'muse_spark',
    modelTypes: ['muse_spark', 'muse_spark_text'],
    architectures: ['MuseSparkForConditionalGeneration'],
    build: (_text, _top, files) => ({
      kind: 'ok',
      model: { ...REF_DENSE_8B, name: files.id, source: files.id },
      notes: [],
      mtpParamsEstimate: 0,
      visionParamsEstimate: 0,
      excludedParts: [],
    }),
  };
  const families = FAMILIES.some((f) => f.family === 'muse_spark') ? FAMILIES : [...FAMILIES, stub];
  /** `loadHfModel` with the fake Hub (and the stand-in family while the real one is missing). */
  const gatedLoader: ModelLoader = async (id, options) => {
    const files = await fetchHfRepoFiles(id, {
      ...options,
      fetchImpl: gatedHub,
      cache: createCache(memoryStorage(), new Map()),
      tokenStorage: memoryStorage(),
    });
    return isHfError(files) ? files : mapHfModel(files, families);
  };

  it('loads a gated model from its mirror without a token prompt or any mention of the mirror', async () => {
    history.replaceState(null, '', `/?m=${official}`);
    mount(gatedLoader);
    await waitFor(() => expect(screen.getByTestId('model-facts').textContent).toMatch(/muse_spark/));
    expect((await screen.findByTestId('decode-tps')).textContent).toMatch(/[1-9]/);
    expect(screen.queryByLabelText('Hugging Face token')).toBeNull();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(document.body.textContent).not.toMatch(FROM_TEXT);
  });

  it('does not show the base repo a Mistral NVFP4 config was read from', async () => {
    const id = 'mistralai/Mistral-Large-4-1T-A52B-NVFP4';
    history.replaceState(null, '', `/?m=${id}`);
    mount(async (requested) =>
      requested === id
        ? mapHfModel(fixtureRepoFiles(id, { dir: 'quant', configFrom: 'mistralai/Mistral-Large-4-1T-A52B' }))
        : fixtureLoader(requested),
    );
    await waitFor(() => expect(screen.getByTestId('model-facts').textContent).toMatch(/mistral/i));
    await screen.findByTestId('decode-tps');
    expect(document.body.textContent).not.toMatch(/read from/i);
  });
});

describe('speculative-decoding acceptance hints', () => {
  it('flags a preset α used at a k it was not fitted at, and clears the flag at a fitted k or a user α', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&sd=dflash2&sdr=hyp');
    mount();
    await screen.findByText(/qwen:/);
    const section = await screen.findByTestId('specdec-section');
    const k = within(section).getByLabelText('Draft length k');
    // DFlash 2 prose 0.79 is fitted at k 7 (Qwen3.8 card) only; the k 15 reports are cited, not fitted. Default k 7.
    expect((k as HTMLInputElement).value).toBe('7');
    expect(within(section).queryByText(/fitted at k/)).toBeNull();
    fireEvent.change(k, { target: { value: '5' } });
    expect(await within(section).findByText('⚠ fitted at k = 7')).toBeTruthy();
    fireEvent.change(k, { target: { value: '7' } });
    await waitFor(() => expect(within(section).queryByText(/fitted at k/)).toBeNull());
    // A user α is not a preset: no warning at any k.
    fireEvent.change(k, { target: { value: '5' } });
    fireEvent.change(within(section).getByLabelText('Acceptance α'), { target: { value: '0.8' } });
    await waitFor(() => expect(within(section).queryByText(/fitted at k/)).toBeNull());
  });
  it('labels DFlash 2 agentic as DFlash v1 data fitted at k 15', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&sd=dflash2&sdr=hyp&sdu=agentic');
    mount();
    await screen.findByText(/qwen:/);
    const section = await screen.findByTestId('specdec-section');
    expect(await within(section).findByText('from DFlash v1 · ⚠ fitted at k = 15')).toBeTruthy();
  });
  it('labels a pooled preset with the k of its runs, with ⚠ at any other k', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&sd=dspark&sdr=hyp&sdu=code');
    mount();
    await screen.findByText(/qwen:/);
    const section = await screen.findByTestId('specdec-section');
    // default k 7 is one of the pooled runs' k (paper k 7, Inco k 15)
    expect(await within(section).findByText('pooled from k = 7, 15')).toBeTruthy();
    fireEvent.change(within(section).getByLabelText('Draft length k'), { target: { value: '5' } });
    expect(await within(section).findByText('⚠ pooled from k = 7, 15')).toBeTruthy();
  });
});
