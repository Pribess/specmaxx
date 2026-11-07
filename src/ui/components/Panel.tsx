import { useState, type ReactNode } from 'react';

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
  const [expanded] = useState(
    () => open && !(typeof matchMedia === 'function' && matchMedia('(max-width: 760px)').matches),
  );
  return (
    <details className="panel" open={expanded}>
      <summary>
        <strong>{title}</strong>
        <span>{value}</span>
      </summary>
      <div className="panel-body">{children}</div>
    </details>
  );
}
