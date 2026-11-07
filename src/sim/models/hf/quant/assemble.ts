/**
 * From per-probe formats to a {@link QuantConfig}: majority per class, splits reported as notes.
 *
 * Per class: inside each layer, the probes (projections) vote with their size weight; across layers, each
 * layer votes once. The winner is the class format; layers or projections with another format are listed in a
 * note (e.g. "routed_experts: layer 2 stays bf16"). `weights` is the bulk class's format (routed experts in a
 * MoE model, else the dense MLP); `modules` lists every class whose format differs from its default.
 */

import type { Dtype, TensorFormat } from '../../../ir/dtype';
import type { ModuleQuant, QuantConfig, WeightModule } from '../../../ir/model';
import { WEIGHT_MODULE_FALLBACK } from '../../../ir/model';
import type { ModuleClass, Probe, SchemeContext, SchemeReading } from './common';
import { describeFormat, formatKey, formatRanges, moduleFormat } from './common';
import type { ProbeSet } from './moduleClass';
import { compilePatterns, firstMatch } from './patterns';
import type { QuantDefaults } from '../types';

/** Weight classes in the order `modules` lists them (parents before their fallback sub-classes). */
const MODULE_ORDER: WeightModule[] = [
  'attention',
  'attention_kv_down',
  'attention_kv_up',
  'linear_attention',
  'indexer',
  'indexer_key',
  'indexer_head_weights',
  'dense_mlp',
  'shared_experts',
  'routed_experts',
  'residual_mixer',
  'ngram',
];

/** The decided format of one class, with the minority layers / projections. */
interface ClassDecision {
  /** Winning format. */
  format: ModuleQuant;
  /** Layers (index, or -1 for top-level) whose format differs, by format key. */
  minorityLayers: Map<string, { format: ModuleQuant; layers: number[] }>;
  /** Projections that differ from the winner in most of their layers. */
  minorityLeaves: Map<string, { format: ModuleQuant; leaves: string[] }>;
  /** Layers voting. Count. */
  layerCount: number;
}

/** Weighted vote: the key with the largest total; ties go to the first key seen. */
function vote(entries: { key: string; weight: number }[]): string {
  const totals = new Map<string, number>();
  for (const e of entries) totals.set(e.key, (totals.get(e.key) ?? 0) + e.weight);
  let best = '';
  let bestTotal = -1;
  for (const [k, t] of totals) {
    if (t > bestTotal + 1e-9) {
      best = k;
      bestTotal = t;
    }
  }
  return best;
}

/** Decide the format of one class from its probes. */
function decideClass(probes: Probe[], resolved: Map<Probe, ModuleQuant>): ClassDecision {
  const byKey = new Map<string, ModuleQuant>();
  const byLayer = new Map<number, Probe[]>();
  for (const p of probes) {
    const layer = p.layer ?? -1;
    const list = byLayer.get(layer) ?? [];
    list.push(p);
    byLayer.set(layer, list);
  }
  const layerKey = new Map<number, string>();
  for (const [layer, ps] of byLayer) {
    const entries = ps.map((p) => {
      const f = resolved.get(p) as ModuleQuant;
      const key = formatKey(f);
      byKey.set(key, f);
      return { key, weight: p.weight };
    });
    layerKey.set(layer, vote(entries));
  }
  const winner = vote([...layerKey.values()].map((key) => ({ key, weight: 1 })));
  const minorityLayers = new Map<string, { format: ModuleQuant; layers: number[] }>();
  for (const [layer, key] of layerKey) {
    if (key === winner) continue;
    const m = minorityLayers.get(key) ?? { format: byKey.get(key) as ModuleQuant, layers: [] };
    m.layers.push(layer);
    minorityLayers.set(key, m);
  }

  // Projections that differ from the winner in most of the winner's layers.
  const leafStats = new Map<string, { other: Map<string, number>; total: number }>();
  for (const [layer, ps] of byLayer) {
    if (layerKey.get(layer) !== winner) continue;
    for (const p of ps) {
      const key = formatKey(resolved.get(p) as ModuleQuant);
      const s = leafStats.get(p.leaf) ?? { other: new Map<string, number>(), total: 0 };
      s.total++;
      if (key !== winner) s.other.set(key, (s.other.get(key) ?? 0) + 1);
      leafStats.set(p.leaf, s);
    }
  }
  const minorityLeaves = new Map<string, { format: ModuleQuant; leaves: string[] }>();
  for (const [leaf, s] of leafStats) {
    for (const [key, n] of s.other) {
      if (n * 2 <= s.total) continue;
      const m = minorityLeaves.get(key) ?? { format: byKey.get(key) as ModuleQuant, leaves: [] };
      m.leaves.push(leaf);
      minorityLeaves.set(key, m);
    }
  }
  return { format: byKey.get(winner) as ModuleQuant, minorityLayers, minorityLeaves, layerCount: byLayer.size };
}

/** Drop `gemmInput` when it equals the activations dtype (that is the default). */
function normalize(mq: ModuleQuant, activations: Dtype): ModuleQuant {
  return mq.gemmInput === activations ? { weights: mq.weights } : mq;
}

/** Same weights and GEMM input. */
function sameFormat(a: ModuleQuant, b: ModuleQuant): boolean {
  return formatKey(a) === formatKey(b);
}

/** Same storage format. */
function sameTensor(a: TensorFormat, b: TensorFormat): boolean {
  return formatKey({ weights: a }) === formatKey({ weights: b });
}

/** Notes for the split classes. */
function splitNotes(cls: ModuleClass, d: ClassDecision): string[] {
  const notes: string[] = [];
  for (const m of d.minorityLayers.values()) {
    const layers = m.layers.filter((l) => l >= 0);
    if (layers.length === 0) continue;
    const what = layers.length === 1 ? `layer ${layers[0]}` : `layers ${formatRanges(layers)}`;
    notes.push(
      `${cls}: ${describeFormat(d.format)} in ${d.layerCount - layers.length} of ${d.layerCount} layers; ` +
        `${what} ${layers.length === 1 ? 'is' : 'are'} ${describeFormat(m.format)} (majority used)`,
    );
  }
  for (const m of d.minorityLeaves.values()) {
    notes.push(`${cls}: ${m.leaves.join(', ')} ${describeFormat(m.format)} (class uses ${describeFormat(d.format)})`);
  }
  return notes;
}

/** Notes for ignore-list entries that match no probe and do not name a module outside the IR. */
function unknownIgnoreNotes(reading: SchemeReading, probes: Probe[], numLayers: number): string[] {
  const known =
    /(visual|vision|audio|mtp|norm|bias|conv|sconv|embed|projector|patch|aligner|merger|sink|gate|a_log|scale|lm_head|head|output|router|eh_proj|_hc|hc_|hyper_connection|mapping_proj|attn_mha|attn_mqa|position|rotary|lerope|ssmax)/i;
  const notes: string[] = [];
  for (const { label, patterns } of reading.ignoreLists) {
    const unknown: string[] = [];
    for (const raw of patterns) {
      if (known.test(raw)) continue;
      // MTP layers sit at index num_hidden_layers and up: `layers.45.x`, `layers.45*`, `re:.*layers\.45\..*`.
      const layer = /layers\\?\.(\d+)(?!\d)/.exec(raw);
      if (layer && Number(layer[1]) >= numLayers) continue;
      const list = compilePatterns([raw], { regexKeys: label === 'extra_config' });
      if (list.invalid.length > 0 || !probes.some((p) => firstMatch(list, p) >= 0)) unknown.push(raw);
    }
    if (unknown.length > 0) {
      const examples = unknown.slice(0, 3).map((u) => `'${u}'`).join(', ');
      notes.push(`${label}: ${unknown.length} entr${unknown.length === 1 ? 'y matches' : 'ies match'} no known module (${examples}); ignored`);
    }
  }
  return notes;
}

/**
 * Build {@link QuantDefaults} from a reading: resolve every probe, decide per class, and fill `QuantConfig`.
 */
export function assemble(reading: SchemeReading, set: ProbeSet, ctx: SchemeContext): QuantDefaults {
  const activations = reading.activations ?? ctx.modelDtype;
  const highPrecision = ctx.modelDtype;
  const resolved = new Map<Probe, ModuleQuant>();
  for (const p of set.probes) {
    const f = reading.resolve(p);
    // An unconverted module keeps its own checkpoint dtype when that differs from the model's (fp32 hc_*).
    const own = f === ctx.native && p.nativeDtype !== undefined ? moduleFormat(p.nativeDtype) : f;
    resolved.set(p, normalize(own, activations));
  }

  const byClass = new Map<ModuleClass, Probe[]>();
  for (const p of set.probes) {
    const list = byClass.get(p.cls) ?? [];
    list.push(p);
    byClass.set(p.cls, list);
  }
  const decisions = new Map<ModuleClass, ClassDecision>();
  for (const [cls, ps] of byClass) decisions.set(cls, decideClass(ps, resolved));

  const notes = [...reading.notes];
  for (const [cls, d] of decisions) notes.push(...splitNotes(cls, d));
  notes.push(...unknownIgnoreNotes(reading, set.probes, set.layout.numLayers));

  const hasMoe = set.layout.mlp.includes('moe') && decisions.has('routed_experts');
  const bulkCls: WeightModule = hasMoe ? 'routed_experts' : decisions.has('dense_mlp') ? 'dense_mlp' : 'attention';
  const bulk = decisions.get(bulkCls)?.format ?? ctx.native;
  const quant: QuantConfig = {
    weights: bulk.weights,
    activations,
    kvCache: reading.kvCache ?? { dtype: activations },
  };
  if (highPrecision !== 'bf16') quant.highPrecision = highPrecision;

  const modules: Partial<Record<WeightModule, ModuleQuant>> = {};
  for (const cls of MODULE_ORDER) {
    const d = decisions.get(cls);
    if (!d) continue;
    const parent = WEIGHT_MODULE_FALLBACK[cls];
    const parentEntry = parent === undefined ? undefined : (modules[parent] ?? decisions.get(parent)?.format);
    const fallback: ModuleQuant =
      parentEntry ?? (cls === 'residual_mixer' ? moduleFormat(highPrecision) : { weights: quant.weights });
    if (!sameFormat(d.format, fallback)) modules[cls] = d.format;
  }
  if (Object.keys(modules).length > 0) quant.modules = modules;

  const hp: TensorFormat = { dtype: highPrecision };
  const embed = decisions.get('embeddings')?.format.weights;
  if (embed && !sameTensor(embed, hp)) quant.embeddings = embed;
  const embedEffective = quant.embeddings ?? hp;
  const head = decisions.get('lm_head')?.format;
  if (head && !sameTensor(head.weights, embedEffective)) quant.lmHead = head.weights;
  if (head?.gemmInput !== undefined) quant.lmHeadGemmInput = head.gemmInput;
  const tables = decisions.get('lookup_tables')?.format.weights;
  if (tables && !sameTensor(tables, embedEffective)) quant.lookupTables = tables;
  if (reading.indexerKeyCache) quant.indexerKeyCache = reading.indexerKeyCache;

  // Fields the repo's metadata states; family quantHints never override them (see `mergeQuantHints`).
  const explicit: (keyof QuantConfig)[] = [];
  if (reading.kvCache) explicit.push('kvCache');
  if (reading.indexerKeyCache) explicit.push('indexerKeyCache');
  if (quant.modules) explicit.push('modules');
  if (quant.lookupTables) explicit.push('lookupTables');

  return { quant, scheme: reading.label, source: reading.source, notes, explicit };
}
