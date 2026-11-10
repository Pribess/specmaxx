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
 * | `tp`, `pp`, `dp`, `ep` | parallel degrees (`ep` applies to MoE models only; old URLs with `dp` load it as written) | `tp=2&dp=2` |
 * | `b`, `p`, `c` | batch, prefill length (default 8192), decode context (default 8192) | `b=8` |
 * | `pc` | prefill chunk of the memory fit, tokens (default 8192) | `pc=2048` |
 * | `ov`, `fu`, `ar` | overlap, elementwise fusion, all-reduce algorithm | `ov=none` |
 * | `lf` | per-op latency floor in µs, 0–1000 (default 0: off) | `lf=1` |
 * | `ed` | engine dispatch overlay on: `1` (absent: off) | `ed=1` |
 * | `hr`, `hrc` | host-RAM preset id (default `ddr5-6000-2ch`), its capacity in GB (1e9 B) | `hr=ddr5-6400-8ch&hrc=512` |
 * | `ssd` | SSD preset id (default: the device's own SSD on unified memory, else `nvme-990-pro-2tb`) | `ssd=nvme-t705-2tb` |
 * | `hl` | host-link override, `<PCIe gen>x<lanes>` (default: the device's host interface) | `hl=4x16` |
 * | `gpu` | (unified memory) GPU-usable share of memory in percent, (0, 100] (default: the device's) | `gpu=90` |
 * | `rr` | runtime reserve per device in GB (1e9 B), ≥ 0 (default 0) | `rr=1.5` |
 * | `ox` | routed experts tier: `ram`, `ssd`, `split` (host RAM + SSD) (absent: on the device) | `ox=ram` |
 * | `oxl`, `oxf`, `oxh` | (with `ox`) first offloaded layer (default 0), resident fraction 0–1 (default 0), hit-ratio model `classic`, `fine` or a custom h 0–1 (default uniform) | `oxf=0.25&oxh=fine` |
 * | `oxr`, `oxe` | (with `ox=split`) offloaded layers in host RAM, integer ≥ 0 (default: half, rounded down), which end is in host RAM: `last` (default `first`, never written) | `oxr=24&oxe=last` |
 * | `odl`, `odt` | legacy whole-layer offload (removed), read only: accepted and ignored, never written | `odl=40` |
 * | `olt`, `ols` | lookup-tables tier `ram`, `ssd`, `split` (host RAM + SSD) (absent: on the device); (with `olt`) scale layout `separate`, `pinned`, `interleaved` (default: the tier's; with `split`, each side's tier's) | `olt=ram` |
 * | `olr`, `ole` | (with `olt=split`) layers with a table kept in host RAM, integer ≥ 0 (default: half, rounded down), which end is in host RAM: `last` (default `first`, never written) | `olr=24` |
 * | `oet` | token-embedding table tier `ram`, `ssd` (absent: on the device; `split` is not allowed) | `oet=ssd` |
 * | `oqd` | (with offload) queue depth, integer ≥ 1 (default: the tier's rated maximum) | `oqd=32` |
 * | `osr` | (with SSD offload) SSD read rate: `datasheet` (default), `sustained` | `osr=sustained` |
 * | `opf` | (with offload) prefetch accuracy p 0–1 (default 0.8) | `opf=0.9` |
 * | `oo` | legacy offload overlap, read only: with offload and no `opf`, `ideal` → `opf=1`, `none` → `opf=0`; otherwise ignored | `oo=none` |
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
  offloadsToSsd,
  normalizeDeployment,
  normalizeHost,
  type DeploymentChoice,
  type HitRatioChoice,
  type HostChoice,
  type OffloadChoice,
  type ScaleLayoutChoice,
  type SpecDecChoice,
  type SpecDecMethod,
  type PlacementChoice,
  type TierChoice,
  type TierSplitChoice,
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
];

/** Largest per-op latency floor the URL and the UI accept. Unit: µs. */
export const MAX_LATENCY_FLOOR_US = 1000;

/**
 * A per-op latency floor typed in µs as the state stores it: seconds rounded to whole ns, so the URL's 3-decimal µs
 * (`lf`) round-trips exactly. Unit: µs in, seconds out.
 */
export function latencyFloorSecFromMicros(us: number): number {
  return Math.round(us * 1e3) / 1e9;
}

/**
 * Every key the codec knows, in encoding order (`oo`, `odl`, `odt` are legacy keys: read, never written), except the
 * split keys of the last two lines: `oxr`, `oxe` are written right after `oxh`, and `olr`, `ole`
 * right after `ols`.
 */
export const URL_KEYS = [
  'm', 'rev', 'w', 'bpw', 'hbpw', 'a', 'kv', 'lt', 'd', 'u', 'eng', 'cap', 'n', 'nodes', 'il', 'xl', 'fab', 'x', 'xi', 'xe',
  'hr', 'hrc', 'ssd', 'hl', 'gpu', 'rr', 'tp', 'pp', 'dp', 'ep', 'b', 'p', 'c', 'pc', 'ph', 'ov', 'fu', 'ar', 'lf', 'ed', 'oo',
  'ox', 'oxl', 'oxf', 'oxh', 'odl', 'odt', 'olt', 'ols', 'oet', 'oqd', 'osr', 'opf', 'sd', 'sdr', 'sdk', 'sdu', 'sda',
  'oxr', 'oxe',
  'olr', 'ole',
] as const;

/** URL value of a tier. */
const TIER_KEYS: Record<TierChoice, string> = { host_ram: 'ram', nvme: 'ssd' };

/** URL value of a placement of experts or lookup tables: a tier, or `split` (host RAM + SSD). */
const PLACEMENT_KEYS: Record<PlacementChoice, string> = { ...TIER_KEYS, split: 'split' };

/** Puts a split's `<ram>` (layers in host RAM) and `<end>` (`last` only) keys; nothing without a split. */
function putSplit(put: (key: string, value: string | number | undefined) => void, ramKey: string, endKey: string, split: TierSplitChoice | undefined): void {
  put(ramKey, split?.ramLayers);
  if (split?.ramEnd === 'last') put(endKey, 'last');
}

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
  // Percent with at most 4 decimals, so 0.9 encodes as 90 (not 90.00000000000001).
  if (hw.gpuUsableFraction !== undefined) put('gpu', Math.round(hw.gpuUsableFraction * 1e6) / 1e4);
  if (hw.runtimeReserveBytes !== undefined) put('rr', hw.runtimeReserveBytes / 1e9);
  for (const key of ['tp', 'pp', 'dp', 'ep'] as const) if (s.parallel[key] !== 1) put(key, s.parallel[key]);
  const w = s.workload;
  const dw = DEFAULT_STATE.workload;
  if (w.batch !== dw.batch) put('b', w.batch);
  if (w.promptLen !== dw.promptLen) put('p', w.promptLen);
  if (w.decodeContextLen !== dw.decodeContextLen) put('c', w.decodeContextLen);
  if (w.prefillChunkTokens !== undefined) put('pc', w.prefillChunkTokens);
  // Percent with at most 4 decimals (0.9 encodes as 90).
  if (w.prefixCacheHitRatio !== undefined) put('ph', Math.round(w.prefixCacheHitRatio * 1e6) / 1e4);
  const options = normalizeOptions(s.options);
  for (const [key, field] of OPTION_KEYS) put(key, options[field]);
  // µs with at most 3 decimals (ns), so 0.6 µs encodes as 0.6 (not 0.6000000000000001).
  if (options.perOpLatencyFloorSec !== undefined) put('lf', Math.round(options.perOpLatencyFloorSec * 1e9) / 1e3);
  if (s.engineDispatchOverlay) put('ed', '1');
  const deployment = normalizeDeployment(s.deployment);
  const o = deployment.offload ?? {};
  if (o.experts !== undefined) {
    put('ox', PLACEMENT_KEYS[o.experts.tier]);
    put('oxl', o.experts.fromLayer);
    put('oxf', o.experts.residentFraction);
    const h = o.experts.hitRatio;
    put('oxh', typeof h === 'number' ? h : h === undefined ? undefined : HIT_RATIO_KEYS[h]);
    putSplit(put, 'oxr', 'oxe', o.experts.split);
  }
  if (o.tables !== undefined) {
    put('olt', PLACEMENT_KEYS[o.tables.tier]);
    if (o.tables.scaleLayout !== undefined) put('ols', SCALE_LAYOUT_KEYS[o.tables.scaleLayout]);
    putSplit(put, 'olr', 'ole', o.tables.split);
  }
  if (o.embeddings !== undefined) put('oet', TIER_KEYS[o.embeddings.tier]);
  put('oqd', o.queueDepth);
  put('osr', o.ssdReadRate);
  put('opf', o.prefetchAccuracy);
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
 * The deployment choice of the offload (`ox`…`opf`, legacy `oo`) and spec-dec (`sd`…`sda`) parameters. A dependent key without
 * its parent (`oxf` without `ox`, `ols` without `olt`, `sdk` without `sd`, `oqd` or `opf` with nothing offloaded) is
 * dropped. The legacy `oo` (offload overlap) maps to the equivalent prefetch accuracy when something is offloaded and
 * `opf` is absent (`ideal` → 1, `none` → 0); otherwise it is ignored without a notice (it was valid on its own).
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
  const placement = (raw: string) => keyOf(PLACEMENT_KEYS, raw);
  /** The split of the `<ram>` / `<end>` keys; both are dropped unless the group's placement is `split`. */
  const readSplit = (ramKey: string, endKey: string, isSplit: boolean): TierSplitChoice | undefined => {
    const ramLayers = read(ramKey, isSplit, (r) => parseInteger(r, 0, MAX_VALUE));
    const ramEnd = read(endKey, isSplit, (r) => (r === 'first' || r === 'last' ? r : undefined));
    return isSplit ? { ramLayers, ramEnd } : undefined;
  };
  const offload: OffloadChoice = {};
  const expertsTier = read('ox', true, placement);
  const fromLayer = read('oxl', expertsTier !== undefined, (r) => parseInteger(r, 0, MAX_VALUE));
  const residentFraction = read('oxf', expertsTier !== undefined, (r) => parseNumberIn(r, 0, 1));
  const hitRatio = read<HitRatioChoice>('oxh', expertsTier !== undefined, (r) => keyOf(HIT_RATIO_KEYS, r) ?? parseNumberIn(r, 0, 1));
  const expertsSplit = readSplit('oxr', 'oxe', expertsTier === 'split');
  if (expertsTier !== undefined) offload.experts = { tier: expertsTier, split: expertsSplit, fromLayer, residentFraction, hitRatio };
  // Whole-layer offload was removed: old URLs may carry odl / odt; they are accepted and ignored.
  get('odl');
  get('odt');
  const tablesTier = read('olt', true, placement);
  const scaleLayout = read('ols', tablesTier !== undefined, (r) => keyOf(SCALE_LAYOUT_KEYS, r));
  const tablesSplit = readSplit('olr', 'ole', tablesTier === 'split');
  if (tablesTier !== undefined) offload.tables = { tier: tablesTier, split: tablesSplit, scaleLayout };
  const embeddingsTier = read('oet', true, (r) => keyOf(TIER_KEYS, r));
  if (embeddingsTier !== undefined) offload.embeddings = { tier: embeddingsTier };
  const any = offload.experts !== undefined || offload.tables !== undefined || embeddingsTier !== undefined;
  offload.queueDepth = read('oqd', any, (r) => parseInteger(r, 1, MAX_VALUE));
  offload.ssdReadRate = read('osr', offloadsToSsd(offload), (r) =>
    r === 'datasheet' || r === 'sustained' ? r : undefined,
  );
  offload.prefetchAccuracy = read('opf', any, (r) => parseNumberIn(r, 0, 1));
  if (any && offload.prefetchAccuracy === undefined && get('oo') !== undefined) {
    // Legacy offload overlap: 'ideal' was perfect prefetch (p = 1), 'none' no prefetching (p = 0).
    offload.prefetchAccuracy = read('oo', true, (r) => (r === 'ideal' ? 1 : r === 'none' ? 0 : undefined));
  }

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
  // A catalog local link that cannot join the decoded devices per machine (the 2-card NVLink bridge at n=4), or that the
  // decoded unlocks' PCIe host interface cannot run (a CMP 170HX PCIe 2.0 link without the Gen2 unlock), is dropped.
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
  for (const key of ['tp', 'pp', 'dp', 'ep'] as const) {
    const v = get(key);
    if (v === undefined) continue;
    const parsed = parseInteger(v, 1, MAX_VALUE);
    if (parsed !== undefined) parallel[key] = parsed;
    else drop(key, v);
  }
  const workload = { ...DEFAULT_STATE.workload };
  for (const [key, field] of [['b', 'batch'], ['p', 'promptLen'], ['c', 'decodeContextLen'], ['pc', 'prefillChunkTokens']] as const) {
    const v = get(key);
    if (v === undefined) continue;
    const parsed = parseInteger(v, 1, MAX_VALUE);
    if (parsed !== undefined) workload[field] = parsed;
    else drop(key, v);
  }
  const ph = get('ph');
  if (ph !== undefined) {
    const percent = parseNumberIn(ph, Number.MIN_VALUE, 99.9999);
    if (percent !== undefined) workload.prefixCacheHitRatio = percent / 100;
    else drop('ph', ph);
  }

  // Options.
  const options: Partial<SimOptions> = {};
  for (const [key, field, values] of OPTION_KEYS) {
    const v = get(key);
    if (v === undefined) continue;
    if (values.includes(v)) Object.assign(options, { [field]: v });
    else drop(key, v);
  }
  // Per-op latency floor in µs (0 = off, the default, so it stays absent from the state).
  const lf = get('lf');
  if (lf !== undefined) {
    const us = parseNumberIn(lf, 0, MAX_LATENCY_FLOOR_US);
    if (us === undefined) drop('lf', lf);
    else if (us > 0) options.perOpLatencyFloorSec = latencyFloorSecFromMicros(us);
  }
  const ed = get('ed');
  const engineDispatchOverlay = ed === '1';
  if (ed !== undefined && !engineDispatchOverlay) drop('ed', ed);

  const host = normalizeHost(decodeHost(get, drop));
  if (Object.keys(host).length > 0) hw.host = host;
  // Memory-fit knobs: the GPU limit only on unified memory (it means nothing on a dedicated device).
  const gpu = get('gpu');
  if (gpu !== undefined) {
    const percent = parseNumberIn(gpu, Number.MIN_VALUE, 100);
    if (percent !== undefined && unlocked.memory.kind === 'unified') hw.gpuUsableFraction = percent / 100;
    else drop('gpu', gpu);
  }
  const rr = get('rr');
  if (rr !== undefined) {
    const gb = parseNumberIn(rr, 0, 1e6);
    if (gb !== undefined) hw.runtimeReserveBytes = Math.round(gb * 1e9);
    else drop('rr', rr);
  }
  const deployment = decodeDeployment(get, drop);

  const state: AppState = { model, quant, hw, parallel, workload, options: normalizeOptions(options), deployment };
  if (engineDispatchOverlay) state.engineDispatchOverlay = true;
  return { state, dropped };
}
