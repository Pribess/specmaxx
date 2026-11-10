/**
 * Module-name patterns as quant configs write them, compiled once and matched against probes.
 *
 * Syntaxes (one entry of an ignore / exclude / targets list, or a key of a per-layer map):
 * - plain name (`'lm_head'`, `'kv_b_proj'`, `'model.layers.0.self_attn'`): matches a module whose dotted name
 *   contains it at component boundaries (equal, a dotted prefix, a dotted suffix, or a dotted infix). This
 *   covers exact names, parent modules, and the short names of `modules_to_not_convert`. With `exact`
 *   (compressed-tensors, which matches each module by its own name), only the whole name matches: ignoring
 *   `layers.0.linear_attn` does not ignore `layers.0.linear_attn.out_proj`.
 *   Either way a name written with another stack prefix matches from `layers.<n>.` on
 *   (`language_model.layers.3.x` = `layers.3.x`), except MTP / draft / vision stacks.
 * - glob (`'*.attn.*'`, `'model.layers.0*'`): fnmatch, the whole name (ModelOpt, Quark).
 * - `re:` regex (`'re:.*mlp\\.gate$'`): anchored at the start, as Python's `re.match` (compressed-tensors).
 * - torch class name (`'Linear'`, `'Embedding'`, `'RoutedExperts'`): matches by probe kind / class.
 * With `regexKeys` (AutoRound `extra_config`), a key with regex syntax is a regex searched anywhere.
 */

import type { Probe } from './common';

/** One compiled pattern. */
export type NamePattern =
  | { kind: 'name'; raw: string; exact?: boolean }
  | { kind: 'regex'; raw: string; re: RegExp }
  | { kind: 'class'; raw: string };

/** A compiled pattern list: plain names indexed for O(name depth) lookups, the rest scanned in order. */
export interface PatternList {
  /** Entries in their original order. */
  entries: NamePattern[];
  /** Plain-name entries → their first index in `entries`. */
  names: Map<string, number>;
  /** Plain names match whole module names only. */
  exact: boolean;
  /** Entries that failed to compile (invalid regex); never match. */
  invalid: string[];
}

/** Options of {@link compilePatterns}. */
export interface CompileOptions {
  /** Treat entries with regex syntax as unanchored regexes (AutoRound `extra_config` keys). */
  regexKeys?: boolean;
  /** Plain names match whole module names only (compressed-tensors). */
  exact?: boolean;
}

/** Characters that make an AutoRound key a regex. */
const REGEX_SYNTAX = /[*\\^$|()[\]+?{}]/;

/** Translate an fnmatch glob into an anchored RegExp (`*` any run, `?` one char, `[...]` a class). */
function globToRegExp(glob: string): RegExp {
  let out = '';
  for (const ch of glob) {
    if (ch === '*') out += '.*';
    else if (ch === '?') out += '.';
    else if (ch === '[' || ch === ']') out += ch;
    else out += ch.replace(/[.+^${}()|\\/]/g, '\\$&');
  }
  return new RegExp(`^${out}$`);
}

/** Compile one entry; `null` for an invalid regex. */
export function compilePattern(raw: string, opts: CompileOptions = {}): NamePattern | null {
  try {
    if (raw.startsWith('re:'))
      return { kind: 'regex', raw, re: new RegExp(`^(?:${raw.slice(3)})`) };
    if (opts.regexKeys && REGEX_SYNTAX.test(raw))
      return { kind: 'regex', raw, re: new RegExp(raw) };
    if (/[*?[]/.test(raw)) return { kind: 'regex', raw, re: globToRegExp(raw) };
  } catch {
    return null;
  }
  if (/^[A-Z][A-Za-z0-9]*$/.test(raw)) return { kind: 'class', raw };
  return opts.exact ? { kind: 'name', raw, exact: true } : { kind: 'name', raw };
}

/** Stack prefixes that are not the decoder (their `layers.<n>` are other layers). */
const OTHER_STACK = /mtp|dspark|draft|vision|visual|audio|encoder|decoder|merger/i;

/** Compile a list of entries (order kept, invalid ones collected). */
export function compilePatterns(raws: readonly string[], opts: CompileOptions = {}): PatternList {
  const entries: NamePattern[] = [];
  const names = new Map<string, number>();
  const invalid: string[] = [];
  for (const raw of raws) {
    const p = compilePattern(raw, opts);
    if (p === null) {
      invalid.push(raw);
      continue;
    }
    if (p.kind === 'name') {
      if (!names.has(p.raw)) names.set(p.raw, entries.length);
      const tail = layerTail(p.raw);
      if (tail !== null && !names.has(tail)) names.set(tail, entries.length);
    }
    entries.push(p);
  }
  return { entries, names, exact: opts.exact === true, invalid };
}

/**
 * The part of a module name from `layers.<n>.` on, when it has a stack prefix that is the decoder
 * (`language_model.layers.3.x` → `layers.3.x`); null otherwise (no prefix, or an MTP / draft / vision stack).
 */
function layerTail(name: string): string | null {
  const m = /^(.+?\.)(layers\.\d+\..+)$/.exec(name);
  return m && !OTHER_STACK.test(m[1]) ? m[2] : null;
}

/** True when the plain name `raw` names the whole module `name` (directly or through {@link layerTail}). */
function exactNameMatches(name: string, raw: string): boolean {
  if (name === raw) return true;
  const a = layerTail(name) ?? name;
  const b = layerTail(raw) ?? raw;
  return a === b;
}

/** True when a class-name target selects the probe: `Linear`, `Embedding`, or an experts module class. */
function classMatches(cls: string, probe: Probe): boolean {
  if (cls === 'Linear') return probe.kind === 'linear';
  if (cls === 'Embedding') return probe.kind === 'embedding';
  if (/expert/i.test(cls)) return probe.cls === 'routed_experts';
  return false;
}

/** True when one compiled pattern matches the probe. */
export function patternMatches(p: NamePattern, probe: Probe): boolean {
  switch (p.kind) {
    case 'name':
      return p.exact ? exactNameMatches(probe.name, p.raw) : dottedContains(probe.name, p.raw);
    case 'regex':
      return p.re.test(probe.name);
    case 'class':
      return classMatches(p.raw, probe);
  }
}

/** True when `part` occurs in the dotted name `name` at component boundaries. */
export function dottedContains(name: string, part: string): boolean {
  if (name === part) return true;
  if (!name.includes(part)) return false;
  return name.startsWith(`${part}.`) || name.endsWith(`.${part}`) || name.includes(`.${part}.`);
}

/** Every contiguous run of dotted components of `name` (the plain-name keys that could match it). */
function dottedRuns(name: string): string[] {
  const parts = name.split('.');
  const runs: string[] = [];
  for (let i = 0; i < parts.length; i++) {
    for (let j = i + 1; j <= parts.length; j++) runs.push(parts.slice(i, j).join('.'));
  }
  return runs;
}

/**
 * Index of the first entry of `list` that matches the probe, or -1. Plain names are looked up by every
 * dotted run of the probe name (with `exact`, by the whole name and its layer tail); the earliest matching
 * entry wins.
 */
export function firstMatch(list: PatternList, probe: Probe): number {
  let best = -1;
  if (list.names.size > 0) {
    const keys = list.exact
      ? [probe.name, layerTail(probe.name) ?? probe.name]
      : dottedRuns(probe.name);
    for (const run of keys) {
      const i = list.names.get(run);
      if (i !== undefined && (best < 0 || i < best)) best = i;
    }
  }
  for (let i = 0; i < list.entries.length && (best < 0 || i < best); i++) {
    const p = list.entries[i];
    if (p.kind !== 'name' && patternMatches(p, probe)) return i;
  }
  return best;
}

/** True when any entry of `list` matches the probe. */
export function anyMatch(list: PatternList, probe: Probe): boolean {
  return firstMatch(list, probe) >= 0;
}
