/**
 * The URL query ↔ `AppState`. Short keys, defaults omitted, a fixed key order, and `/ : , .` left unescaped,
 * so links stay readable (`?m=Qwen/Qwen3.8-27B&n=2&tp=2`). Decoding never throws: unknown keys and invalid
 * values are dropped and listed in `dropped` (the UI shows a notice). The HF token is NEVER in the URL.
 *
 * | key | field | example |
 * |---|---|---|
 * | `m`, `rev` | HF repo id, revision (absent: the default model `Qwen/Qwen3.8-27B`; `rev` default `main`) | `m=zai-org/GLM-5.3` |
 * | `w`, `a`, `kv` | weight / activation / KV dtype overrides | `kv=fp8_e4m3` |
 * | `d` | device preset id (default `rtx-5090`) | `d=rtx-4090` |
 * | `u` | unlocks, `axis.option` comma list | `u=pcie-p2p.direct,vram-mod.48gb` |
 * | `eng`, `cap` | engine id, capacity option index | `cap=1` |
 * | `n`, `nodes` | devices per node, nodes | `n=2` |
 * | `il`, `xl` | intra / inter link preset ids (`linkPresetId`) | `il=pcie-5-0-x16-direct-p2p` |
 * | `x`, `xi`, `xe` | spec edits on device / intra link / inter link, `path:value` comma list, SI units | `x=memory.bandwidthBytesPerSec:2e12` |
 * | `tp`, `pp`, `dp`, `ep` | parallel degrees | `tp=2` |
 * | `b`, `p`, `c` | batch, prefill length (default 8192), decode context (default 8192) | `b=8` |
 * | `ov`, `fu`, `ar` | overlap, elementwise fusion, all-reduce algorithm | `ov=none` |
 */

import type { SimOptions } from '../../sim/engine/types';
import type { DeviceSpec } from '../../sim/hw/types';
import type { UnlockSelection } from '../../sim/hw/unlocks';
import { numericLeafPaths } from '../../sim/hw/validate';
import { deviceUnlocks } from '../../sim/hw/presets/index';
import { isValidModelId } from '../../hf/urls';
import { DEFAULT_HW, DEFAULT_STATE, normalizeOptions, type AppState, type HardwareChoice, type ModelChoice, type QuantOverrides, type SpecEdits } from './appState';
import { findDevice, findLink, resolveDevice, resolveLinks } from './hardware';
import { isQuantDtype, isValidBpw, isValidHeadBpw, isWeightFormat } from './quant';

/** Largest accepted devices-per-node and node count (keeps derived loops bounded). Count. */
export const MAX_COUNT = 1024;

/** Largest accepted parallel degree or workload value (tokens, sequences). */
export const MAX_VALUE = 1e9;

/** Allowed values of the option keys, in URL key → [SimOptions field, values]. */
const OPTION_KEYS: [string, keyof SimOptions, readonly string[]][] = [
  ['ov', 'overlap', ['ideal', 'none']],
  ['fu', 'elementwiseFusion', ['none', 'ideal']],
  ['ar', 'collectiveAlgorithm', ['best', 'ring', 'tree']],
];

/** Every key the codec knows, in encoding order. */
export const URL_KEYS = [
  'm', 'rev', 'w', 'bpw', 'hbpw', 'a', 'kv', 'd', 'u', 'eng', 'cap', 'n', 'nodes', 'il', 'xl', 'x', 'xi', 'xe',
  'tp', 'pp', 'dp', 'ep', 'b', 'p', 'c', 'ov', 'fu', 'ar',
] as const;

/** Percent-encodes a value but keeps `/ : ,` readable (`.` is never escaped). */
function encodeValue(v: string): string {
  return encodeURIComponent(v).replace(/%2F/gi, '/').replace(/%3A/gi, ':').replace(/%2C/gi, ',');
}

/** The shorter of plain and exponent notation, both exact (`2e12`, `1.792e12`, `1024`, `0.5`). */
export function encodeNumber(v: number): string {
  const plain = String(v);
  const exp = v.toExponential().replace('e+', 'e');
  return exp.length < plain.length ? exp : plain;
}

/** `path:value` comma list of edits, paths sorted. */
function encodeEdits(edits: SpecEdits): string {
  return Object.keys(edits)
    .sort()
    .map((p) => `${p}:${encodeNumber(edits[p])}`)
    .join(',');
}

/**
 * The query string of a state, without the leading `?` (`''` for the default state). Defaults are omitted;
 * keys follow {@link URL_KEYS}; unlocks follow the device's axis order and edits are sorted by path.
 */
export function encodeState(s: AppState): string {
  const out: [string, string][] = [];
  const put = (key: string, value: string | number | undefined) => {
    if (value !== undefined) out.push([key, typeof value === 'number' ? encodeNumber(value) : value]);
  };
  const dm = DEFAULT_STATE.model;
  const isDefaultModel = s.model.id === dm.id && (s.model.revision ?? 'main') === (dm.revision ?? 'main');
  if (!isDefaultModel) {
    put('m', s.model.id);
    if (s.model.revision !== undefined && s.model.revision !== 'main') put('rev', s.model.revision);
  }
  put('w', s.quant.weights);
  if (s.quant.weights === 'exl3') {
    put('bpw', s.quant.bpw);
    put('hbpw', s.quant.headBpw);
  }
  put('a', s.quant.activations);
  put('kv', s.quant.kvCache);
  const hw = s.hw;
  if (hw.deviceId !== DEFAULT_HW.deviceId) put('d', hw.deviceId);
  const axisOrder = deviceUnlocks({ id: hw.deviceId }).map((a) => a.id);
  const axes = Object.keys(hw.unlocks).sort((a, b) => axisOrder.indexOf(a) - axisOrder.indexOf(b));
  if (axes.length > 0) put('u', axes.map((a) => `${a}.${hw.unlocks[a]}`).join(','));
  put('eng', hw.engineId);
  if (hw.capacityIndex !== 0) put('cap', hw.capacityIndex);
  if (hw.devicesPerNode !== 1) put('n', hw.devicesPerNode);
  if (hw.nodes !== 1) put('nodes', hw.nodes);
  put('il', hw.intraLinkId);
  put('xl', hw.interLinkId);
  if (Object.keys(hw.deviceEdits).length > 0) put('x', encodeEdits(hw.deviceEdits));
  if (Object.keys(hw.intraEdits).length > 0) put('xi', encodeEdits(hw.intraEdits));
  if (Object.keys(hw.interEdits).length > 0) put('xe', encodeEdits(hw.interEdits));
  for (const key of ['tp', 'pp', 'dp', 'ep'] as const) if (s.parallel[key] !== 1) put(key, s.parallel[key]);
  const w = s.workload;
  const dw = DEFAULT_STATE.workload;
  if (w.batch !== dw.batch) put('b', w.batch);
  if (w.promptLen !== dw.promptLen) put('p', w.promptLen);
  if (w.decodeContextLen !== dw.decodeContextLen) put('c', w.decodeContextLen);
  const options = normalizeOptions(s.options);
  for (const [key, field] of OPTION_KEYS) put(key, options[field]);
  return out.map(([k, v]) => `${k}=${encodeValue(v)}`).join('&');
}

/** Result of {@link decodeState}. */
export interface DecodedState {
  /** The state; every value in it is valid (known ids, integers ≥ 1, known paths). */
  state: AppState;
  /** `key=value` of each dropped parameter (unknown key or invalid value), for a notice. */
  dropped: string[];
}

/** Parses an integer in [min, max], or undefined. */
function parseInteger(raw: string, min: number, max: number): number | undefined {
  if (!/^\d+$/.test(raw)) return undefined;
  const v = Number(raw);
  return Number.isSafeInteger(v) && v >= min && v <= max ? v : undefined;
}

/** Parses a `path:value` edit list against the numeric leaves of `spec`; invalid entries go to `bad`. */
function parseEdits(raw: string, spec: object, bad: string[]): SpecEdits {
  const leaves = new Set(numericLeafPaths(spec).map((l) => l.path));
  const edits: SpecEdits = {};
  for (const entry of raw.split(',')) {
    const i = entry.lastIndexOf(':');
    const path = entry.slice(0, i);
    const value = i < 0 ? NaN : Number(entry.slice(i + 1));
    if (i > 0 && leaves.has(path) && entry.slice(i + 1) !== '' && Number.isFinite(value) && value >= 0) edits[path] = value;
    else bad.push(entry);
  }
  return edits;
}

/**
 * The state of a query string (with or without the leading `?`). Starts from the default state; each known
 * key with a valid value overrides its field. Device-dependent values (unlocks, engine, capacity, edits,
 * links) are checked against the decoded device; link edits against the link in use at that level.
 */
export function decodeState(search: string): DecodedState {
  const params = new URLSearchParams(search);
  const dropped: string[] = [];
  const drop = (key: string, value: string) => dropped.push(`${key}=${value}`);
  const get = (key: string): string | undefined => params.get(key) ?? undefined;
  for (const [key, value] of params) if (!(URL_KEYS as readonly string[]).includes(key)) drop(key, value);

  // Model.
  let model: ModelChoice = DEFAULT_STATE.model;
  const m = get('m');
  const rev = get('rev');
  if (m !== undefined) {
    if (isValidModelId(m)) {
      model = rev !== undefined && rev !== '' && rev !== 'main' ? { kind: 'hf', id: m, revision: rev } : { kind: 'hf', id: m };
    } else drop('m', m);
  }
  if (rev !== undefined && m === undefined) drop('rev', rev);

  // Quant overrides.
  const quant: QuantOverrides = {};
  for (const [key, field] of [['w', 'weights'], ['a', 'activations'], ['kv', 'kvCache']] as const) {
    const v = get(key);
    if (v === undefined) continue;
    if (field === 'weights' ? isWeightFormat(v) : isQuantDtype(v)) Object.assign(quant, { [field]: v });
    else drop(key, v);
  }
  // EXL3 bits per weight; meaningful only with `w=exl3`.
  for (const [key, field, valid] of [['bpw', 'bpw', isValidBpw], ['hbpw', 'headBpw', isValidHeadBpw]] as const) {
    const v = get(key);
    if (v === undefined) continue;
    const n = Number(v);
    if (quant.weights === 'exl3' && v.trim() !== '' && valid(n)) quant[field] = n;
    else drop(key, v);
  }

  // Device and its unlocks.
  const hw: HardwareChoice = { ...DEFAULT_HW, unlocks: {}, deviceEdits: {}, intraEdits: {}, interEdits: {} };
  const d = get('d');
  if (d !== undefined) {
    if (findDevice(d) !== undefined) hw.deviceId = d;
    else drop('d', d);
  }
  const base = findDevice(hw.deviceId) as DeviceSpec;
  const axes = deviceUnlocks(base);
  const u = get('u');
  if (u !== undefined) {
    const unlocks: UnlockSelection = {};
    const bad: string[] = [];
    for (const entry of u.split(',')) {
      const [axisId, optionId] = entry.split('.');
      const axis = axes.find((a) => a.id === axisId);
      if (axis !== undefined && axis.options.some((o) => o.id === optionId) && unlocks[axisId] === undefined) unlocks[axisId] = optionId;
      else bad.push(entry);
    }
    hw.unlocks = unlocks;
    if (bad.length > 0) drop('u', bad.join(','));
  }
  const unlocked = resolveDevice({ ...hw, deviceEdits: {} }).unlocked;
  const eng = get('eng');
  if (eng !== undefined) {
    if (unlocked.engines.some((e) => e.id === eng)) hw.engineId = eng;
    else drop('eng', eng);
  }
  const cap = get('cap');
  if (cap !== undefined) {
    const v = parseInteger(cap, 0, unlocked.memory.capacityOptionsBytes.length - 1);
    if (v !== undefined) hw.capacityIndex = v;
    else drop('cap', cap);
  }
  const x = get('x');
  if (x !== undefined) {
    const bad: string[] = [];
    hw.deviceEdits = parseEdits(x, unlocked, bad);
    if (bad.length > 0) drop('x', bad.join(','));
  }

  // Counts and links.
  for (const [key, field] of [['n', 'devicesPerNode'], ['nodes', 'nodes']] as const) {
    const v = get(key);
    if (v === undefined) continue;
    const parsed = parseInteger(v, 1, MAX_COUNT);
    if (parsed !== undefined) hw[field] = parsed;
    else drop(key, v);
  }
  for (const [key, field] of [['il', 'intraLinkId'], ['xl', 'interLinkId']] as const) {
    const id = get(key);
    if (id === undefined) continue;
    if (findLink(id) !== undefined) hw[field] = id;
    else drop(key, id);
  }
  // Link edits are checked against the link in use at their level (preset + unlock modifiers).
  const links = resolveLinks(hw, resolveDevice({ ...hw, deviceEdits: {} }));
  for (const [key, field, link] of [['xi', 'intraEdits', links.intra], ['xe', 'interEdits', links.inter]] as const) {
    const edits = get(key);
    if (edits === undefined) continue;
    if (link === undefined) {
      drop(key, edits);
      continue;
    }
    const bad: string[] = [];
    hw[field] = parseEdits(edits, link.unlocked, bad);
    if (bad.length > 0) drop(key, bad.join(','));
  }

  // Parallelism and workload.
  const parallel = { ...DEFAULT_STATE.parallel };
  for (const key of ['tp', 'pp', 'dp', 'ep'] as const) {
    const v = get(key);
    if (v === undefined) continue;
    const parsed = parseInteger(v, 1, MAX_VALUE);
    if (parsed !== undefined) parallel[key] = parsed;
    else drop(key, v);
  }
  const workload = { ...DEFAULT_STATE.workload };
  for (const [key, field] of [['b', 'batch'], ['p', 'promptLen'], ['c', 'decodeContextLen']] as const) {
    const v = get(key);
    if (v === undefined) continue;
    const parsed = parseInteger(v, 1, MAX_VALUE);
    if (parsed !== undefined) workload[field] = parsed;
    else drop(key, v);
  }

  // Options.
  const options: Partial<SimOptions> = {};
  for (const [key, field, values] of OPTION_KEYS) {
    const v = get(key);
    if (v === undefined) continue;
    if (values.includes(v)) Object.assign(options, { [field]: v });
    else drop(key, v);
  }

  return { state: { model, quant, hw, parallel, workload, options: normalizeOptions(options) }, dropped };
}
