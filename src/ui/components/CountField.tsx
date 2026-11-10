import type { ReactNode } from 'react';
import type { FieldIssue } from '../state';
import { Picker } from './Picker';
import { Segmented } from './Segmented';

/** Most options a count shows as a {@link Segmented} control; more become a {@link Picker}. */
export const MAX_SEGMENTED_COUNTS = 6;

/** The offered counts, plus the current count when it is not offered (an older link), ascending. */
export function countChoices(offered: number[], current: number): number[] {
  return offered.includes(current) ? offered : [...offered, current].sort((a, b) => a - b);
}

/**
 * A count field (devices per machine, machines): a segmented control while it has at most
 * {@link MAX_SEGMENTED_COUNTS} choices, else a picker (which adds a search box above 8 choices).
 */
export function CountField({
  label,
  value,
  counts,
  onChange,
  help,
  aside,
  issues,
}: {
  /** Visible label and accessible name. */
  label: string;
  /** The current count. */
  value: number;
  /** The counts on offer (a count outside it is added, so the current value always shows). */
  counts: number[];
  /** Called with the chosen count. */
  onChange(count: number): void;
  /** Rendered as a "?" help icon next to the label. */
  help?: ReactNode;
  /** Text at the right end of the label row. */
  aside?: ReactNode;
  /** Validation messages for this field. */
  issues?: FieldIssue[];
}) {
  const choices = countChoices(counts, value);
  if (choices.length <= MAX_SEGMENTED_COUNTS) {
    return (
      <Segmented
        label={label}
        value={String(value)}
        segments={choices.map((n) => ({ value: String(n), label: String(n) }))}
        onChange={(v) => onChange(Number(v))}
        help={help}
        aside={aside}
        issues={issues}
      />
    );
  }
  return (
    <Picker
      label={label}
      value={String(value)}
      options={choices.map((n) => ({ value: String(n), label: String(n) }))}
      onChange={(v) => onChange(Number(v))}
      help={help}
      aside={aside}
      issues={issues}
    />
  );
}
