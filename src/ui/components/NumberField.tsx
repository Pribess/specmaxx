import { useId } from 'react';
import type { FieldIssue } from '../state';

export function NumberField({
  label,
  value,
  onChange,
  issues = [],
  hint,
  step = '1',
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  issues?: FieldIssue[];
  hint?: string;
  step?: string;
}) {
  const id = useId();
  return (
    <label className="field" htmlFor={id}>
      <span>{label}</span>
      <input
        id={id}
        type="number"
        step={step}
        value={Number.isFinite(value) ? value : ''}
        onChange={(e) => onChange(e.target.value === '' ? 0 : Number(e.target.value))}
        aria-invalid={issues.some((i) => i.level === 'error')}
        aria-describedby={`${id}-help`}
      />
      <span className="hint" id={`${id}-help`}>
        {hint}
        {issues.map((i, n) => (
          <span className={i.level === 'error' ? 'error' : ''} key={n}>
            {i.message}
          </span>
        ))}
      </span>
    </label>
  );
}
