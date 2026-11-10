/**
 * A tiny typed reader over parsed config JSON.
 *
 * Every getter takes a key, checks the value's type, and on failure records a path-qualified error such as
 * `text_config.hidden_size: missing number` in a shared list and returns a placeholder (0, '', false, [], an
 * empty reader). It never throws, so a family builder reads every field it needs, then checks
 * `errors.length` once and returns all problems together.
 *
 * Optional getters (`opt*`) return `undefined` for a missing or `null` key and record an error only for a
 * present value of the wrong type.
 */

import type { JsonObject } from './types';

/** Typed accessors over one JSON object, sharing an error list with the reader that created them. */
export interface ConfigReader {
  /** Path of this object from the root, e.g. `'text_config'` (empty at the root). */
  path: string;
  /** The object read (an empty object when it was missing or not an object). */
  raw: JsonObject;
  /** Errors collected by this reader and every reader derived from it (shared list). */
  errors: string[];
  /** True if `key` is present and not `null`. */
  has(key: string): boolean;
  /** A required finite number. */
  num(key: string): number;
  /** An optional finite number. */
  optNum(key: string): number | undefined;
  /** A required integer ≥ 0 (counts, dims, ids). */
  int(key: string): number;
  /** An optional integer ≥ 0. */
  optInt(key: string): number | undefined;
  /** A boolean; `fallback` when missing (error when missing and no fallback is given). */
  bool(key: string, fallback?: boolean): boolean;
  /** A required string. */
  str(key: string): string;
  /** An optional string. */
  optStr(key: string): string | undefined;
  /** A required array of finite numbers. */
  numArray(key: string): number[];
  /** An optional array of finite numbers. */
  optNumArray(key: string): number[] | undefined;
  /** A required array of strings. */
  strArray(key: string): string[];
  /** An optional array of strings. */
  optStrArray(key: string): string[] | undefined;
  /** A required sub-object, as a reader with the same error list. */
  obj(key: string): ConfigReader;
  /** An optional sub-object, as a reader with the same error list. */
  optObj(key: string): ConfigReader | undefined;
}

/** True for a plain JSON object (not null, not an array). */
export function isJsonObject(v: unknown): v is JsonObject {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/**
 * Create a reader over `value`. When `value` is not an object, records `<path>: missing object` and reads
 * from an empty object. `errors` defaults to a new list; pass one to share it.
 */
export function createReader(value: unknown, path = '', errors: string[] = []): ConfigReader {
  if (!isJsonObject(value)) errors.push(`${path || '(root)'}: missing object`);
  const raw: JsonObject = isJsonObject(value) ? value : {};
  const at = (key: string) => (path ? `${path}.${key}` : key);
  const present = (key: string) => raw[key] !== undefined && raw[key] !== null;
  const fail = <T>(key: string, what: string, placeholder: T): T => {
    errors.push(`${at(key)}: ${present(key) ? 'expected' : 'missing'} ${what}`);
    return placeholder;
  };

  const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
  const isInt = (v: unknown): v is number => isNum(v) && Number.isInteger(v) && v >= 0;
  const isNumArray = (v: unknown): v is number[] => Array.isArray(v) && v.every(isNum);
  const isStrArray = (v: unknown): v is string[] =>
    Array.isArray(v) && v.every((x) => typeof x === 'string');

  /** Optional getter: undefined when absent, error + undefined when of the wrong type. */
  const opt = <T>(key: string, ok: (v: unknown) => v is T, what: string): T | undefined => {
    if (!present(key)) return undefined;
    const v = raw[key];
    return ok(v) ? v : fail<T | undefined>(key, what, undefined);
  };
  /** Required getter: error + placeholder when absent or of the wrong type. */
  const req = <T>(key: string, ok: (v: unknown) => v is T, what: string, placeholder: T): T => {
    const v = raw[key];
    return present(key) && ok(v) ? v : fail(key, what, placeholder);
  };

  return {
    path,
    raw,
    errors,
    has: present,
    num: (key) => req(key, isNum, 'number', 0),
    optNum: (key) => opt(key, isNum, 'number'),
    int: (key) => req(key, isInt, 'non-negative integer', 0),
    optInt: (key) => opt(key, isInt, 'non-negative integer'),
    bool: (key, fallback) => {
      if (!present(key) && fallback !== undefined) return fallback;
      return req(key, (v): v is boolean => typeof v === 'boolean', 'boolean', false);
    },
    str: (key) => req(key, (v): v is string => typeof v === 'string', 'string', ''),
    optStr: (key) => opt(key, (v): v is string => typeof v === 'string', 'string'),
    numArray: (key) => req(key, isNumArray, 'number array', []),
    optNumArray: (key) => opt(key, isNumArray, 'number array'),
    strArray: (key) => req(key, isStrArray, 'string array', []),
    optStrArray: (key) => opt(key, isStrArray, 'string array'),
    obj: (key) => {
      if (!present(key) || !isJsonObject(raw[key])) {
        fail(key, 'object', undefined);
        return createReader({}, at(key), errors);
      }
      return createReader(raw[key], at(key), errors);
    },
    optObj: (key) => {
      if (!present(key)) return undefined;
      if (!isJsonObject(raw[key])) return fail(key, 'object', undefined);
      return createReader(raw[key], at(key), errors);
    },
  };
}
