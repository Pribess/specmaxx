import * as ScrollAreaPrimitive from '@radix-ui/react-scroll-area';
import type { ComponentPropsWithoutRef, ReactNode, Ref } from 'react';

/** Which way a {@link ScrollArea} scrolls. */
export type ScrollAxis = 'vertical' | 'horizontal' | 'both';

/**
 * A scroll container with a thin themed scrollbar (shadcn/ui's ScrollArea on the Radix primitive), used for
 * every scrolling region inside the page; the document's own scrollbar is styled in tokens.css instead.
 *
 * The viewport scrolls natively (wheel and trackpad momentum, touch, and keyboard once it or a child has
 * focus); Radix only hides the OS scrollbar and draws its own, which shows while the pointer is over the
 * area. Thumb and track colours are the `--scrollbar-*` tokens.
 *
 * - `axis`: vertical areas keep their content at the viewport's width (text wraps and ellipses work);
 *   horizontal areas let content grow wider and scroll sideways (tables).
 * - `viewportProps`: attributes for the scrolling element itself, e.g. `tabIndex`, `role`, `aria-label`,
 *   or a `style.maxHeight` that bounds it.
 */
export function ScrollArea({
  axis = 'vertical',
  className,
  viewportRef,
  viewportProps,
  children,
  ...rootProps
}: {
  axis?: ScrollAxis;
  className?: string;
  /** Ref to the scrolling element (the viewport). */
  viewportRef?: Ref<HTMLDivElement>;
  viewportProps?: ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.Viewport>;
  children: ReactNode;
} & Omit<ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.Root>, 'children' | 'className'>) {
  const vertical = axis !== 'horizontal';
  const horizontal = axis !== 'vertical';
  return (
    <ScrollAreaPrimitive.Root
      className={`scroll-area${className ? ` ${className}` : ''}`}
      data-axis={axis}
      {...rootProps}
    >
      <ScrollAreaPrimitive.Viewport
        {...viewportProps}
        ref={viewportRef}
        className={`scroll-area-viewport${viewportProps?.className ? ` ${viewportProps.className}` : ''}`}
        disableImplicitContentElement
      >
        {/* Radix sizes its content as a table so it can grow sideways; vertical areas stay block-level. */}
        <ScrollAreaPrimitive.Content
          className="scroll-area-content"
          style={horizontal ? undefined : { display: 'block', minWidth: 0 }}
        >
          {children}
        </ScrollAreaPrimitive.Content>
      </ScrollAreaPrimitive.Viewport>
      {vertical && <ScrollBar orientation="vertical" />}
      {horizontal && <ScrollBar orientation="horizontal" />}
      {vertical && horizontal && <ScrollAreaPrimitive.Corner />}
    </ScrollAreaPrimitive.Root>
  );
}

/** One themed scrollbar of a {@link ScrollArea}: a track with a rounded thumb. */
function ScrollBar({ orientation }: { orientation: 'vertical' | 'horizontal' }) {
  return (
    <ScrollAreaPrimitive.Scrollbar className="scroll-area-bar" orientation={orientation}>
      <ScrollAreaPrimitive.Thumb className="scroll-area-thumb" />
    </ScrollAreaPrimitive.Scrollbar>
  );
}
