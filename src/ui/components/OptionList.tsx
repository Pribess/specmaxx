import type { MouseEvent, RefObject } from 'react';
import { groupOptions, type PickerOption } from './fuzzy';

/** Props of {@link OptionList}. The popup container is owned by the caller (it positions and scrolls it). */
export interface OptionListProps {
  /** Id prefix: options get `${idPrefix}-opt-<index>`, group headers `${idPrefix}-grp-<n>`. */
  idPrefix: string;
  /** Id of the element naming the listbox (the field label). */
  labelId: string;
  /** The listbox's own id, referenced by the trigger's aria-controls. */
  listId: string;
  /** Options to show, already filtered and ranked. Grouped by `group` here. */
  options: PickerOption[];
  /** Value of the selected option (aria-selected), if any. */
  selectedValue?: string;
  /** Value of the keyboard-active option (highlighted), if any. */
  activeValue?: string;
  /** A click or tap picked this option. */
  onPick(option: PickerOption): void;
  /** The pointer moved over this option (it becomes active). */
  onHover(option: PickerOption): void;
  /** Ref to the listbox element, for the caller's focus handling. */
  listRef?: RefObject<HTMLDivElement | null>;
}

/** The DOM id of the option at `index` in the flat (group-ordered) list. */
export function optionDomId(idPrefix: string, index: number): string {
  return `${idPrefix}-opt-${index}`;
}

/**
 * The listbox body: options grouped under headers, with ARIA `listbox` / `group` / `option` roles.
 * Pointer-down is cancelled so a click does not move focus out of the trigger or search box.
 * Renders nothing for an empty list; the caller shows its own "no matches" text.
 */
export function OptionList({
  idPrefix,
  labelId,
  listId,
  options,
  selectedValue,
  activeValue,
  onPick,
  onHover,
  listRef,
}: OptionListProps) {
  const groups = groupOptions(options);
  const flat = groups.flatMap((g) => g.options);
  const indexOf = new Map(flat.map((o, i) => [o.value, i] as const));
  const stop = (e: MouseEvent) => e.preventDefault();
  let groupNo = 0;
  return (
    <div
      ref={listRef}
      id={listId}
      role="listbox"
      aria-labelledby={labelId}
      className="listbox"
      tabIndex={-1}
      onMouseDown={stop}
    >
      {groups.map((g) => {
        const headerId = g.group !== undefined ? `${idPrefix}-grp-${groupNo++}` : undefined;
        return (
          <div role="group" aria-labelledby={headerId} key={g.group ?? '\u0000'}>
            {g.group !== undefined && (
              <div id={headerId} role="presentation" className="listbox-group">
                {g.group}
              </div>
            )}
            {g.options.map((o) => {
              const index = indexOf.get(o.value) ?? -1;
              const isSelected = o.value === selectedValue;
              return (
                <div
                  id={optionDomId(idPrefix, index)}
                  role="option"
                  aria-selected={isSelected}
                  key={o.value}
                  className={
                    'listbox-option' +
                    (o.value === activeValue ? ' is-active' : '') +
                    (isSelected ? ' is-selected' : '')
                  }
                  onClick={() => onPick(o)}
                  onMouseEnter={() => {
                    if (o.value !== activeValue) onHover(o);
                  }}
                >
                  <span className="listbox-check" aria-hidden="true">
                    {isSelected ? '✓' : ''}
                  </span>
                  <span className="listbox-text">
                    <span className="listbox-label">{o.label}</span>
                    {o.description && <span className="listbox-desc">{o.description}</span>}
                  </span>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
