import { Command } from 'cmdk';
import type { Ref } from 'react';
import { groupOptions, type PickerOption } from './fuzzy';
import { ScrollArea } from './ScrollArea';

/**
 * Value given to cmdk while no option is active. cmdk highlights its first item whenever its value is empty;
 * a value that matches no item keeps every option unhighlighted (the combobox before the arrow keys).
 */
export const NO_ACTIVE_OPTION = '\u0000none';

/** Props of {@link OptionList}. It must render inside a cmdk `Command` with `shouldFilter={false}`. */
export interface OptionListProps {
  /** Accessible name of the listbox (the field label). */
  label: string;
  /** Options to show, already filtered and ranked by fuzzyFilter. Grouped by `group` here. */
  options: PickerOption[];
  /** Value of the chosen option (check mark, `data-checked`), if any. */
  selectedValue?: string;
  /** A click, tap, or Enter picked this option. */
  onPick(option: PickerOption): void;
  /** The listbox element (cmdk's `Command.List`), for its DOM id and option lookups. */
  listRef?: Ref<HTMLDivElement>;
}

/**
 * The popup body of a picker: cmdk's listbox with options grouped under headings, inside a themed
 * {@link ScrollArea} whose viewport is the only scrolling element (the search header sits outside it).
 *
 * cmdk gives the ARIA roles (`listbox`, `group` named by its heading, `option`), pointer highlight and click
 * picks; the keyboard-active option is the cmdk value the caller controls (`aria-selected` / `data-selected`).
 * Filtering and ranking stay in fuzzy.ts: options arrive filtered, so cmdk's own filter is off.
 */
export function OptionList({ label, options, selectedValue, onPick, listRef }: OptionListProps) {
  const groups = groupOptions(options);
  return (
    <ScrollArea className="picker-scroll">
      <Command.List ref={listRef} label={label} className="listbox">
        {groups.map((g) => {
          const items = g.options.map((o) => {
            const isSelected = o.value === selectedValue;
            return (
              <Command.Item
                key={o.value}
                value={o.value}
                className="listbox-option"
                data-checked={isSelected || undefined}
                onSelect={() => onPick(o)}
              >
                <span className="listbox-check" aria-hidden="true">
                  {isSelected ? '✓' : ''}
                </span>
                <span className="listbox-text">
                  <span className="listbox-label">{o.label}</span>
                  {o.description && <span className="listbox-desc">{o.description}</span>}
                </span>
              </Command.Item>
            );
          });
          return (
            <Command.Group key={g.group ?? '\u0000'} value={g.group ?? '\u0000'} heading={g.group}>
              {items}
            </Command.Group>
          );
        })}
      </Command.List>
    </ScrollArea>
  );
}

/** The DOM element of the option whose value is `value` inside `list`, if it is rendered. */
export function findOptionElement(
  list: HTMLElement | null,
  value: string | undefined,
): HTMLElement | undefined {
  if (!list || value === undefined) return undefined;
  for (const el of list.querySelectorAll<HTMLElement>('[cmdk-item]')) {
    if (el.getAttribute('data-value') === value.trim()) return el;
  }
  return undefined;
}

/**
 * Scrolls the option element into the list's viewport (nearest edge). For the first option of a group the
 * group heading comes into view too, so a heading is never left hidden above its first row.
 */
export function revealOption(el: HTMLElement | undefined): void {
  if (!el) return;
  if (el.parentElement?.firstElementChild === el) {
    el.closest('[cmdk-group]')
      ?.querySelector<HTMLElement>('[cmdk-group-heading]')
      ?.scrollIntoView?.({ block: 'nearest' });
  }
  el.scrollIntoView?.({ block: 'nearest' });
}

/**
 * Points `owner`'s `aria-activedescendant` at the active option `el` (or removes it). cmdk generates the
 * option ids, so they are only known once rendered; the attribute is set on the DOM after each render
 * instead of through React state, and the owner must not render the attribute itself.
 */
export function setActiveDescendant(owner: HTMLElement | null, el: HTMLElement | undefined): void {
  if (!owner) return;
  if (el?.id) owner.setAttribute('aria-activedescendant', el.id);
  else owner.removeAttribute('aria-activedescendant');
}
