import { useId, type ReactNode } from 'react';
import type { FieldIssue } from '../state';
import { HelpIcon } from './HelpIcon';

/**
 * A labelled numeric input. The input's accessible name is exactly `label`: the "?" help icon sits beside
 * the label, not inside it, and the help popup is referenced by aria-describedby together with the hint and
 * the validation messages.
 */
export function NumberField({
  label,
  value,
  onChange,
  issues = [],
  hint,
  step = '1',
  min,
  max,
  help,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  issues?: FieldIssue[];
  hint?: string;
  step?: string;
  /** Smallest value the input's arrows and form validation allow. */
  min?: number;
  /** Largest value the input's arrows and form validation allow. */
  max?: number;
  /** Definition shown by a "?" help icon next to the label. */
  help?: ReactNode;
}) {
  const id = useId();
  const helpId = `${id}-help`;
  const hintId = `${id}-hint`;
  const describedBy = [help ? helpId : undefined, hintId].filter(Boolean).join(' ');
  return (
    <div className="field">
      <div className="field-head">
        <label className="field-label" htmlFor={id}>
          {label}
        </label>
        {help && (
          <HelpIcon term={label} id={helpId}>
            {help}
          </HelpIcon>
        )}
      </div>
      <input
        id={id}
        type="number"
        step={step}
        min={min}
        max={max}
        value={Number.isFinite(value) ? value : ''}
        onChange={(e) => onChange(e.target.value === '' ? 0 : Number(e.target.value))}
        aria-invalid={issues.some((i) => i.level === 'error')}
        aria-describedby={describedBy}
      />
      <span className="hint" id={hintId}>
        {hint}
        {issues.map((i, n) => (
          <span className={i.level === 'error' ? 'error' : ''} key={n}>
            {i.message}
          </span>
        ))}
      </span>
    </div>
  );
}
