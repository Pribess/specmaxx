/**
 * The URL query ↔ `AppState`. Short keys, defaults omitted, a fixed key order, and `/ : , .` left unescaped,
 * so links stay readable (`?m=Qwen/Qwen3.8-27B&n=2&tp=2`). Decoding never throws: unknown keys and invalid
 * values are dropped and listed in `dropped` (the UI shows a notice). The HF token is NEVER in the URL.
 *
 * | key | field | example |
 * |---|---|---|
 * | `m`, `rev` | HF repo id, revision (absent: the default model `Qwen/Qwen3.8-27B`; `rev` default `main`) | `m=zai-org/GLM-5.3` |
 * | `w`, `a`, `kv` | weight / activation / KV dtype overrides | `kv=fp8_e4m3` |
 * | `lt` | lookup-table (n-gram / Engram) dtype override; absent = the model default | `lt=nvfp4` |
 * | `d` | device preset id (default `rtx-5090`) | `d=rtx-4090` |
 * | `u` | unlocks, `axis.option` comma list | `u=pcie-p2p.direct,vram-mod.48gb` |
 * | `eng`, `cap` | engine id, capacity option index | `cap=1` |
 * | `n`, `nodes` | devices per node, nodes | `n=2` |
 * | `il`, `xl` | intra / inter link preset ids (`linkPresetId`; renamed ids load through `LEGACY_LINK_IDS`) | `il=pcie-5-0-x16-direct-p2p` |
 * | `fab` | cabling between machines: `switched`, `mesh`, `mesh2` (2 cables per pair), `ring`, `chain` (absent: the default for the link and counts) | `fab=ring` |
 * | `x`, `xi`, `xe` | spec edits on device / intra link / inter link, `path:value` comma list, SI units | `x=memory.bandwidthBytesPerSec:2e12` |
 * | `tp`, `pp` | parallel degrees (old `dp` / `ep` are ignored: both fixed at 1) | `tp=2` |
 * | `b`, `p`, `c` | batch, prefill length (default 8192), decode context (default 8192) | `b=8` |
 * | `ov`, `fu`, `ar`, `oo` | overlap, elementwise fusion, all-reduce algorithm, offload overlap (`ideal` default, `none`) | `ov=none` |
 * | `hr`, `hrc` | host-RAM preset id (default `ddr5-6000-2ch`), its capacity in GB (1e9 B) | `hr=ddr5-6400-8ch&hrc=512` |
 * | `ssd` | SSD preset id (default: the device's own SSD on unified memory, else `nvme-990-pro-2tb`) | `ssd=nvme-t705-2tb` |
 * | `hl` | host-link override, `<PCIe gen>x<lanes>` (default: the device's host interface) | `hl=4x16` |
 * | `ox` | routed experts tier: `ram`, `ssd` (absent: on the device) | `ox=ram` |
 * | `oxl`, `oxf`, `oxh` | (with `ox`) first offloaded layer (default 0), resident fraction 0–1 (default 0), hit-ratio model `classic`, `fine` or a custom h 0–1 (default uniform) | `oxf=0.25&oxh=fine` |
 * | `odl`, `odt` | dense layers kept on the device (the rest stream; absent: none stream), their tier `ram` (default) or `ssd` | `odl=40` |
 * | `olt`, `ols` | lookup-tables tier `ram`, `ssd` (absent: on the device); (with `olt`) scale layout `separate`, `pinned`, `interleaved` (default: the tier's) | `olt=ram` |
 * | `oqd` | (with offload) queue depth, integer ≥ 1 (default: the tier's rated maximum) | `oqd=32` |
 * | `sd` | speculative decoding method: `mtp`, `dflash2`, `dspark` (absent: off) | `sd=mtp` |
 * | `sdr` | (block methods) drafter repo id, or `hyp` for the hypothetical drafter (default: the known one for the model) | `sdr=hyp` |
 * | `sdk`, `sdu`, `sda` | draft length k 1–16 (default per method), use case `code`, `math`, `reasoning`, `agentic` (default `prose`), acceptance α 0–0.99 (custom; overrides the use case) | `sdk=3&sda=0.8` |
 */

import type { SimOptions } from '../../sim/engine/types';
import type { DeviceSpec } from '../../sim/hw/types';
import type { UnlockSelection } from '../../sim/hw/unlocks';
import { numericLeafPaths } from '../../sim/hw/validate';
import { deviceUnlocks } from '../../sim/hw/presets/index';
import { isValidModelId } from '../../hf/urls';
import { DEFAULT_HW, DEFAULT_STATE, normalizeOptions, type AppState, type HardwareChoice, type ModelChoice, type QuantOverrides, type SpecEdits } from './appState';
import {
  SPEC_DEC_USE_CASES,
  HYPOTHETICAL_DRAFTER,
  normalizeDeployment,
  normalizeHost,
  type DeploymentChoice,
  type HitRatioChoice,
  type HostChoice,
  type OffloadChoice,
  type ScaleLayoutChoice,
  type SpecDecChoice,
  type SpecDecMethod,
  type TierChoice,
} from './deployment';
import { currentLinkId, findDevice, findHostRam, findLink, findSsd, cablingOptions, isEditableDevicePath, normalizeCabling, normalizeLocalLink, resolveDevice, resolveLinks } from './hardware';
import { CABLING_IDS } from '../../sim/hw/presets/catalog';
import { isLookupTableDtype, isQuantDtype, isValidBpw, isValidHeadBpw, isWeightFormat } from './quant';

/** Largest accepted devices-per-node and node count (keeps derived loops bounded). Count. */
export const MAX_COUNT = 1024;

/** Largest accepted parallel degree or workload value (tokens, sequences). */
export const MAX_VALUE = 1e9;

/** Allowed values of the option keys, in URL key → [SimOptions field, values]. */
const OPTION_KEYS: [string, keyof SimOptions, readonly string[]][] = [
  ['ov', 'overlap', ['ideal', 'none']],
  ['fu', 'elementwiseFusion', ['none', 'ideal']],
  ['ar', 'collectiveAlgorithm', ['best', 'ring', 'tree']],
  ['oo', 'offloadOverlap', ['ideal', 'none']],
];

/** Every key the codec knows, in encoding order. */
export const URL_KEYS = [
  'm', 'rev', 'w', 'bpw', 'hbpw', 'a', 'kv', 'lt', 'd', 'u', 'eng', 'cap', 'n', 'nodes', 'il', 'xl', 'fab', 'x', 'xi', 'xe',
  'hr', 'hrc', 'ssd', 'hl', 'tp', 'pp', 'dp', 'ep', 'b', 'p', 'c', 'ov', 'fu', 'ar', 'oo',
  'ox', 'oxl', 'oxf', 'oxh', 'odl', 'odt', 'olt', 'ols', 'oqd', 'sd', 'sdr', 'sdk', 'sdu', 'sda',
] as const;

/** URL value of a tier. */
const TIER_KEYS: Record<TierChoice, string> = { host_ram: 'ram', nvme: 'ssd' };

/** URL value of a named hit-ratio model. */
const HIT_RATIO_KEYS: Record<Exclude<HitRatioChoice, number>, string> = {
  uniform: 'uniform',
  skewed_classic: 'classic',
  skewed_fine_grained: 'fine',
};

/** URL value of a scale layout. */
const SCALE_LAYOUT_KEYS: Record<ScaleLayoutChoice, string> = { separate: 'separate', pinned_host: 'pinned', interleaved: 'interleaved' };

/** Every spec-dec method, as stored and in the URL. */
const SPEC_DEC_METHODS: readonly SpecDecMethod[] = ['mtp', 'dflash2', 'dspark'];

/** Largest draft length k. Count. */
export const MAX_DRAFT_LENGTH = 16;

/** Largest acceptance α the UI accepts (the i.i.d. closed form divides by 1 − α). Dimensionless. */
export const MAX_ACCEPTANCE = 0.99;

/** The key of `value` in `table`, or undefined. */
function keyOf<K extends string>(table: Record<K, string>, value: string): K | undefined {
  return (Object.keys(table) as K[]).find((k) => table[k] === value);
}

/** Parses a finite number in [min, max], or undefined. */
function parseNumberIn(raw: string, min: number, max: number): number | undefined {
  if (raw.trim() === '') return undefined;
  const v = Number(raw);
  return Number.isFinite(v) && v >= min && v <= max ? v : undefined;
}

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
  put('lt', s.quant.lookupTables);
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
  // Absent when it is the default (the state keeps only a non-default cabling, `normalizeCabling`).
  put('fab', normalizeCabling(hw).cabling);
  if (Object.keys(hw.deviceEdits).length > 0) put('x', encodeEdits(hw.deviceEdits));
  if (Object.keys(hw.intraEdits).length > 0) put('xi', encodeEdits(hw.intraEdits));
  if (Object.keys(hw.interEdits).length > 0) put('xe', encodeEdits(hw.interEdits));
  const host = normalizeHost(hw.host ?? {});
  put('hr', host.ramId);
  if (host.ramCapacityBytes !== undefined) put('hrc', host.ramCapacityBytes / 1e9);
  put('ssd', host.ssdId);
  if (host.link !== undefined) put('hl', `${host.link.generation}x${host.link.lanes}`);
  for (const key of ['tp', 'pp'] as const) if (s.parallel[key] !== 1) put(key, s.parallel[key]);
  const w = s.workload;
  const dw = DEFAULT_STATE.workload;
  if (w.batch !== dw.batch) put('b', w.batch);
  if (w.promptLen !== dw.promptLen) put('p', w.promptLen);
  if (w.decodeContextLen !== dw.decodeContextLen) put('c', w.decodeContextLen);
  const options = normalizeOptions(s.options);
  for (const [key, field] of OPTION_KEYS) put(key, options[field]);
  const deployment = normalizeDeployment(s.deployment);
  const o = deployment.offload ?? {};
  if (o.experts !== undefined) {
    put('ox', TIER_KEYS[o.experts.tier]);
    put('oxl', o.experts.fromLayer);
    put('oxf', o.experts.residentFraction);
    const h = o.experts.hitRatio;
    put('oxh', typeof h === 'number' ? h : h === undefined ? undefined : HIT_RATIO_KEYS[h]);
  }
  if (o.dense !== undefined) {
    put('odl', o.dense.layersOnDevice);
    if (o.dense.tier !== 'host_ram') put('odt', TIER_KEYS[o.dense.tier]);
  }
  if (o.tables !== undefined) {
    put('olt', TIER_KEYS[o.tables.tier]);
    if (o.tables.scaleLayout !== undefined) put('ols', SCALE_LAYOUT_KEYS[o.tables.scaleLayout]);
  }
  put('oqd', o.queueDepth);
  const sd = deployment.specDec;
  if (sd !== undefined) {
    put('sd', sd.method);
    put('sdr', sd.drafter);
    put('sdk', sd.k);
    put('sdu', sd.useCase);
    put('sda', sd.alpha);
  }
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

/**
 * Parses a `path:value` edit list against the numeric leaves of `spec` that `editable` allows; invalid entries go
 * to `bad`.
 */
function parseEdits(raw: string, spec: object, bad: string[], editable: (path: string) => boolean = () => true): SpecEdits {
  const leaves = new Set(numericLeafPaths(spec).map((l) => l.path).filter(editable));
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

/** The host choice of the `hr`, `hrc`, `ssd`, `hl` parameters (see the module table). */
function decodeHost(get: (key: string) => string | undefined, drop: (key: string, value: string) => void): HostChoice {
  const host: HostChoice = {};
  const hr = get('hr');
  if (hr !== undefined) {
    if (findHostRam(hr) !== undefined) host.ramId = hr;
    else drop('hr', hr);
  }
  const hrc = get('hrc');
  if (hrc !== undefined) {
    const gb = parseNumberIn(hrc, Number.MIN_VALUE, 1e6);
    if (gb !== undefined) host.ramCapacityBytes = Math.round(gb * 1e9);
    else drop('hrc', hrc);
  }
  const ssd = get('ssd');
  if (ssd !== undefined) {
    if (findSsd(ssd) !== undefined) host.ssdId = ssd;
    else drop('ssd', ssd);
  }
  const hl = get('hl');
  if (hl !== undefined) {
    const match = /^([345])x(4|8|16)$/.exec(hl);
    if (match !== null) host.link = { generation: Number(match[1]) as 3 | 4 | 5, lanes: Number(match[2]) as 4 | 8 | 16 };
    else drop('hl', hl);
  }
  return host;
}

/**
 * The deployment choice of the offload (`ox`…`oqd`) and spec-dec (`sd`…`sda`) parameters. A dependent key without
 * its parent (`oxf` without `ox`, `ols` without `olt`, `sdk` without `sd`, `oqd` with nothing offloaded) is dropped.
 */
function decodeDeployment(get: (key: string) => string | undefined, drop: (key: string, value: string) => void): DeploymentChoice {
  /** Reads `key` with `parse`; drops it when invalid or when its parent is missing (`allowed` false). */
  const read = <T>(key: string, allowed: boolean, parse: (raw: string) => T | undefined): T | undefined => {
    const raw = get(key);
    if (raw === undefined) return undefined;
    const v = allowed ? parse(raw) : undefined;
    if (v === undefined) drop(key, raw);
    return v;
  };
  const tier = (raw: string) => keyOf(TIER_KEYS, raw);
  const offload: OffloadChoice = {};
  const expertsTier = read('ox', true, tier);
  const fromLayer = read('oxl', expertsTier !== undefined, (r) => parseInteger(r, 0, MAX_VALUE));
  const residentFraction = read('oxf', expertsTier !== undefined, (r) => parseNumberIn(r, 0, 1));
  const hitRatio = read<HitRatioChoice>('oxh', expertsTier !== undefined, (r) => keyOf(HIT_RATIO_KEYS, r) ?? parseNumberIn(r, 0, 1));
  if (expertsTier !== undefined) offload.experts = { tier: expertsTier, fromLayer, residentFraction, hitRatio };
  const layersOnDevice = read('odl', true, (r) => parseInteger(r, 0, MAX_VALUE));
  const denseTier = read('odt', layersOnDevice !== undefined, tier);
  if (layersOnDevice !== undefined) offload.dense = { tier: denseTier ?? 'host_ram', layersOnDevice };
  const tablesTier = read('olt', true, tier);
  const scaleLayout = read('ols', tablesTier !== undefined, (r) => keyOf(SCALE_LAYOUT_KEYS, r));
  if (tablesTier !== undefined) offload.tables = { tier: tablesTier, scaleLayout };
  const any = offload.experts !== undefined || offload.dense !== undefined || offload.tables !== undefined;
  offload.queueDepth = read('oqd', any, (r) => parseInteger(r, 1, MAX_VALUE));

  const method = read('sd', true, (r) => SPEC_DEC_METHODS.find((m) => m === r));
  const on = method !== undefined;
  const drafter = read('sdr', on && method !== 'mtp', (r) => (r === HYPOTHETICAL_DRAFTER || isValidModelId(r) ? r : undefined));
  const k = read('sdk', on, (r) => parseInteger(r, 1, MAX_DRAFT_LENGTH));
  const useCase = read('sdu', on, (r) => SPEC_DEC_USE_CASES.find((u) => u === r));
  const alpha = read('sda', on, (r) => parseNumberIn(r, 0, MAX_ACCEPTANCE));
  const specDec: SpecDecChoice | undefined = method !== undefined ? { method, drafter, k, useCase, alpha } : undefined;
  return normalizeDeployment({ offload, specDec });
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
  for (const [key, field] of [['w', 'weights'], ['a', 'activations'], ['kv', 'kvCache'], ['lt', 'lookupTables']] as const) {
    const v = get(key);
    if (v === undefined) continue;
    if (field === 'weights' ? isWeightFormat(v) : field === 'lookupTables' ? isLookupTableDtype(v) : isQuantDtype(v)) Object.assign(quant, { [field]: v });
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
    hw.deviceEdits = parseEdits(x, unlocked, bad, isEditableDevicePath);
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
    if (findLink(id) !== undefined) hw[field] = currentLinkId(id);
    else drop(key, id);
  }
  // A catalog local link that cannot join the decoded devices per machine (the 2-card NVLink bridge at n=4) is dropped.
  if (hw.intraLinkId !== undefined && normalizeLocalLink(hw).intraLinkId === undefined) {
    drop('il', get('il') ?? hw.intraLinkId);
    delete hw.intraLinkId;
  }
  // Cabling: only one the decoded device, link and counts allow (`allowedCablings`); a default one is stored as absent.
  const fab = get('fab');
  if (fab !== undefined) {
    const cabling = CABLING_IDS.find((c) => c === fab);
    if (cabling !== undefined && cablingOptions(hw).includes(cabling)) {
      const normalized = normalizeCabling({ ...hw, cabling });
      if (normalized.cabling !== undefined) hw.cabling = normalized.cabling;
    } else drop('fab', fab);
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
  // Old URLs may carry dp / ep; they are accepted and forced to 1.
  get('dp');
  get('ep');
  for (const key of ['tp', 'pp'] as const) {
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

  const host = normalizeHost(decodeHost(get, drop));
  if (Object.keys(host).length > 0) hw.host = host;
  const deployment = decodeDeployment(get, drop);

  return { state: { model, quant, hw, parallel, workload, options: normalizeOptions(options), deployment }, dropped };
}
