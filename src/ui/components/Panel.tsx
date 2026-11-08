import { useState, type ReactNode } from 'react';
import { isStacked } from '../layout';

/**
 * One input section (Model, Quantization, …) as a disclosure. `open` sections start expanded on the split
 * (desktop) layout; on the stacked (phone) layout every section starts collapsed, showing `value` as a summary.
 */
export function Panel({
  title,
  value,
  children,
  open = false,
}: {
  title: string;
  value?: string;
  children: ReactNode;
  open?: boolean;
}) {
  const [expanded] = useState(() => open && !isStacked());
  return (
    <details className="panel" open={expanded}>
      <summary>
        <strong>{title}</strong>
        <span className="panel-value">{value}</span>
      </summary>
      <div className="panel-body">{children}</div>
    </details>
  );
}
