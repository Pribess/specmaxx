import { useId, type ReactNode } from 'react';
import type { FieldIssue } from '../state';
import { HelpIcon } from './HelpIcon';

/** One segment of a {@link Segmented} control. */
export interface Segment {
  /** The stored value. */
  value: string;
  /** The visible text. */
  label: string;
}

/**
 * A labelled segmented control: a radiogroup of styled native radio inputs (not a chooser element, so the
 * no-native-choosers rule allows it, as for the unlock axes). Arrow keys move the choice natively; the group's
 * accessible name is `label`. `aside` sits at the right of the label row (e.g. a device total); `issues` show
 * under the control and mark the group invalid when one is an error.
 */
export function Segmented({
  label,
  value,
  segments,
  onChange,
  help,
  aside,
  issues = [],
}: {
  /** Visible label; also the radiogroup's accessible name. */
  label: string;
  /** The checked segment's value. */
  value: string;
  /** The segments, in display order. */
  segments: Segment[];
  /** Called with the newly checked value. */
  onChange(value: string): void;
  /** Rendered as a "?" help icon next to the label. */
  help?: ReactNode;
  /** Text at the right end of the label row. */
  aside?: ReactNode;
  /** Validation messages for this field. */
  issues?: FieldIssue[];
}) {
  const id = useId();
  const labelId = `${id}-label`;
  const hintId = `${id}-hint`;
  return (
    <div className="field segmented-field">
      <div className="field-head">
        <span id={labelId} className="field-label">
          {label}
        </span>
        {help && <HelpIcon term={label}>{help}</HelpIcon>}
        {aside && <span className="field-aside">{aside}</span>}
      </div>
      <div
        className="segmented"
        role="radiogroup"
        aria-labelledby={labelId}
        aria-invalid={issues.some((i) => i.level === 'error')}
        aria-describedby={issues.length > 0 ? hintId : undefined}
      >
        {segments.map((s) => (
          <label className="segment" key={s.value}>
            <input
              type="radio"
              name={id}
              value={s.value}
              checked={s.value === value}
              onChange={() => onChange(s.value)}
            />
            <span>{s.label}</span>
          </label>
        ))}
      </div>
      {issues.length > 0 && (
        <span className="hint" id={hintId}>
          {issues.map((i, n) => (
            <span className={i.level === 'error' ? 'error' : ''} key={n}>
              {i.message}
            </span>
          ))}
        </span>
      )}
    </div>
  );
}
