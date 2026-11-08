import * as Popover from '@radix-ui/react-popover';
import { Command } from 'cmdk';
import {
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { fuzzyFilter, groupOptions, stepIndex, type NavStep, type PickerOption } from './fuzzy';
import { HelpIcon } from './HelpIcon';
import { OptionList, findOptionElement, revealOption, setActiveDescendant } from './OptionList';

/** Keys that move the active option, and the step each makes. */
const NAV_KEYS: Record<string, NavStep> = {
  ArrowDown: 'next',
  ArrowUp: 'prev',
  Home: 'first',
  End: 'last',
  PageDown: 'pageDown',
  PageUp: 'pageUp',
};

/** The value of the first option a list shows (its first group's first option), if any. */
function firstValue(list: PickerOption[]): string | undefined {
  return groupOptions(list)[0]?.options[0]?.value;
}

/**
 * Select-only custom picker (shadcn/ui's combobox pattern): a Radix Popover anchored to the trigger, with a
 * cmdk Command listbox in a themed ScrollArea. It replaces the native select element.
 *
 * - Trigger: a button with `role="combobox"` named by `label`; Enter, Space, ArrowDown or ArrowUp open the
 *   list. Without a search box, focus stays on the trigger and `aria-activedescendant` names the active
 *   option.
 * - Keys (list open): ArrowUp/ArrowDown move, Home/End jump, PageUp/PageDown move 10 rows, Enter or Space
 *   pick the active option, Escape closes and returns focus to the trigger, Tab closes and moves on from
 *   the trigger.
 * - Search: with more than `searchThreshold` options a search box sits in a header above the scrolling list
 *   (the rows scroll beneath it, never through it) and takes focus on open. Typing filters and ranks with
 *   {@link fuzzyFilter} (cmdk's own filter is off); the best match is active, so Enter picks it. Home and
 *   End stay in the search box.
 * - Popup: as wide as the trigger, below it (above when there is more room there), its height capped at
 *   360px and at the room the viewport has.
 * - Pointer: hovering highlights, a click picks; an outside click closes without changing the value.
 */
export function Picker({
  label,
  value,
  options,
  onChange,
  placeholder,
  searchThreshold = 8,
  help,
  renderValue,
}: {
  /** Visible label; also the accessible name of the trigger. */
  label: string;
  /** The selected option's value, or undefined for none. */
  value: string | undefined;
  /** All choices; `group` makes listbox groups in first-appearance order. */
  options: PickerOption[];
  /** Called with the picked option's value. */
  onChange(value: string): void;
  /** Trigger text when nothing is selected. */
  placeholder?: string;
  /** Show a search box at the top of the open list when options.length > this. Default 8. */
  searchThreshold?: number;
  /** Rendered as a "?" help icon next to the label. */
  help?: ReactNode;
  /** Trigger text for the selected option (default: its label). */
  renderValue?(option: PickerOption | undefined): ReactNode;
}) {
  const id = useId();
  const labelId = `${id}-label`;
  const triggerRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const [listEl, setListEl] = useState<HTMLDivElement | null>(null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [activeValue, setActiveValue] = useState<string | undefined>(undefined);

  const searchable = options.length > searchThreshold;
  const selected = options.find((o) => o.value === value);
  const shown = useMemo(
    () => fuzzyFilter(options, searchable ? query : ''),
    [options, query, searchable],
  );
  const flat = useMemo(() => groupOptions(shown).flatMap((g) => g.options), [shown]);
  const activeIdx = flat.findIndex((o) => o.value === activeValue);
  const activeOpt = activeIdx >= 0 ? flat[activeIdx] : undefined;

  const openPopup = () => {
    setQuery('');
    setActiveValue(selected?.value ?? firstValue(options));
    setOpen(true);
  };
  const close = (restoreFocus: boolean) => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  };
  const pick = (o: PickerOption) => {
    onChange(o.value);
    close(true);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLElement>, inSearch: boolean) => {
    const step = NAV_KEYS[e.key];
    if (step) {
      if (inSearch && (e.key === 'Home' || e.key === 'End')) {
        e.stopPropagation(); // the caret, not the list (cmdk would jump)
        return;
      }
      e.preventDefault();
      if (!open) openPopup();
      else {
        const i = stepIndex(activeIdx, flat.length, step);
        if (i >= 0) setActiveValue(flat[i].value);
      }
      return;
    }
    switch (e.key) {
      case 'Enter':
        e.preventDefault();
        if (!open) openPopup();
        else if (activeOpt) pick(activeOpt);
        else close(true);
        return;
      case ' ':
        if (inSearch) return;
        e.preventDefault();
        if (!open) openPopup();
        else if (activeOpt) pick(activeOpt);
        return;
      case 'Tab':
        // Back on the trigger before the browser moves focus, so Tab continues from the field.
        if (open) close(inSearch);
        return;
    }
  };

  // aria-activedescendant on the focused element (the search box, else the trigger) names the active option,
  // whose id cmdk generates; keep that option in view.
  useLayoutEffect(() => {
    const el = open ? findOptionElement(listEl, activeValue) : undefined;
    setActiveDescendant(searchRef.current, el);
    setActiveDescendant(triggerRef.current, searchable ? undefined : el);
    revealOption(el);
  }, [open, listEl, activeValue, shown, searchable]);

  return (
    <div className="field picker">
      <div className="field-head">
        <span id={labelId} className="field-label">
          {label}
        </span>
        {help && <HelpIcon term={label}>{help}</HelpIcon>}
      </div>
      <Popover.Root open={open} onOpenChange={(next) => (next ? openPopup() : close(false))}>
        <Popover.Trigger asChild>
          <button
            ref={triggerRef}
            type="button"
            role="combobox"
            aria-haspopup="listbox"
            aria-expanded={open}
            aria-controls={open ? listEl?.id : undefined}
            aria-labelledby={labelId}
            className="picker-trigger"
            onKeyDown={(e) => onKeyDown(e, false)}
          >
            <span className="picker-value">
              {renderValue ? renderValue(selected) : (selected?.label ?? placeholder ?? '')}
            </span>
            <span className="picker-chevron" aria-hidden="true" />
          </button>
        </Popover.Trigger>
        <Popover.Portal>
          <Popover.Content
            className="picker-popup"
            role="presentation"
            side="bottom"
            align="start"
            sideOffset={4}
            collisionPadding={8}
            onOpenAutoFocus={(e) => {
              e.preventDefault();
              searchRef.current?.focus({ preventScroll: true });
            }}
            onCloseAutoFocus={(e) => e.preventDefault()}
            onEscapeKeyDown={() => triggerRef.current?.focus()}
            onMouseDown={(e) => {
              if (e.target !== searchRef.current) e.preventDefault();
            }}
          >
            <Command
              className="picker-command"
              shouldFilter={false}
              vimBindings={false}
              value={activeValue ?? ''}
              onValueChange={setActiveValue}
            >
              {searchable && (
                <div className="picker-search-head">
                  <input
                    ref={searchRef}
                    type="text"
                    role="combobox"
                    aria-label={`Search ${label}`}
                    aria-expanded={true}
                    aria-controls={listEl?.id}
                    aria-autocomplete="list"
                    className="picker-search"
                    value={query}
                    autoComplete="off"
                    spellCheck={false}
                    placeholder="Search"
                    onChange={(e) => {
                      const q = e.target.value;
                      setQuery(q);
                      setActiveValue(firstValue(fuzzyFilter(options, q)));
                    }}
                    onKeyDown={(e) => onKeyDown(e, true)}
                  />
                </div>
              )}
              <OptionList
                label={label}
                listRef={setListEl}
                options={shown}
                selectedValue={value}
                onPick={pick}
              />
              {shown.length === 0 && (
                <div className="listbox-empty" role="status">
                  No matches
                </div>
              )}
            </Command>
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
    </div>
  );
}
