import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { fuzzyFilter, groupOptions, stepIndex, type NavStep, type PickerOption } from './fuzzy';
import { HelpIcon } from './HelpIcon';
import { useFloating } from './floating';
import { OptionList, optionDomId } from './OptionList';

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
 * Select-only custom picker: an ARIA combobox (`role="combobox"`, popup `listbox`) that replaces the native
 * select element.
 *
 * - Trigger: a button named by `label`; Enter, Space, ArrowDown or ArrowUp open the list. While the list is
 *   open, focus stays on the trigger and `aria-activedescendant` names the active option.
 * - Keys (list open): ArrowUp/ArrowDown move, Home/End jump, PageUp/PageDown move 10 rows, Enter or Space
 *   pick the active option, Escape closes and returns focus to the trigger, Tab closes.
 * - Search: with more than `searchThreshold` options the list has a search box, focused on open. Typing
 *   filters with {@link fuzzyFilter}; the best match is active, so Enter picks it. Home and End stay in the
 *   search box.
 * - Pointer: a click picks; an outside click closes without changing the value.
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
  const triggerId = `${id}-trigger`;
  const listId = `${id}-list`;
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
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
  const popupStyle = useFloating(triggerRef, popupRef, open, {
    preferred: 'below',
    maxHeight: 360,
    width: 'anchor',
  });

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
  const move = (step: NavStep) => {
    const i = stepIndex(activeIdx, flat.length, step);
    if (i >= 0) setActiveValue(flat[i].value);
  };
  /** Enter / Space: open the list when closed, otherwise pick the active option (or just close). */
  const commit = () => {
    if (!open) openPopup();
    else if (activeOpt) pick(activeOpt);
    else close(true);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLElement>, inSearch: boolean) => {
    const step = NAV_KEYS[e.key];
    if (step) {
      if (inSearch && (e.key === 'Home' || e.key === 'End')) return; // the caret, not the list
      e.preventDefault();
      if (open) move(step);
      else openPopup();
      return;
    }
    switch (e.key) {
      case 'Enter':
        e.preventDefault();
        commit();
        return;
      case ' ':
        if (inSearch) return;
        e.preventDefault();
        commit();
        return;
      case 'Escape':
        if (!open) return;
        e.preventDefault();
        e.stopPropagation();
        close(true);
        return;
      case 'Tab':
        if (open) close(false);
        return;
    }
  };

  // Focus the search box when the list opens (it is the keyboard target while searching).
  useEffect(() => {
    if (open && searchable) searchRef.current?.focus();
  }, [open, searchable]);

  // An outside pointer closes the list without changing the value.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!triggerRef.current?.contains(t) && !popupRef.current?.contains(t)) close(false);
    };
    document.addEventListener('pointerdown', onDown);
    return () => document.removeEventListener('pointerdown', onDown);
  }, [open]);

  // Keep the active option visible while moving with the keyboard.
  useEffect(() => {
    if (open && activeIdx >= 0) {
      document.getElementById(optionDomId(id, activeIdx))?.scrollIntoView?.({ block: 'nearest' });
    }
  }, [open, activeIdx, id]);

  return (
    <div className="field picker">
      <div className="field-head">
        <span id={labelId} className="field-label">
          {label}
        </span>
        {help && <HelpIcon term={label}>{help}</HelpIcon>}
      </div>
      <button
        ref={triggerRef}
        id={triggerId}
        type="button"
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        aria-labelledby={labelId}
        aria-activedescendant={
          !searchable && open && activeOpt ? optionDomId(id, activeIdx) : undefined
        }
        className="picker-trigger"
        onClick={() => (open ? close(true) : openPopup())}
        onKeyDown={(e) => onKeyDown(e, false)}
      >
        <span className="picker-value">
          {renderValue ? renderValue(selected) : (selected?.label ?? placeholder ?? '')}
        </span>
        <span className="picker-chevron" aria-hidden="true" />
      </button>
      {open && (
        <div
          ref={popupRef}
          className="picker-popup"
          style={popupStyle}
          onMouseDown={(e) => {
            if (e.target !== searchRef.current) e.preventDefault();
          }}
        >
          {searchable && (
            <input
              ref={searchRef}
              type="text"
              role="combobox"
              aria-label={`Search ${label}`}
              aria-expanded={true}
              aria-controls={listId}
              aria-autocomplete="list"
              aria-activedescendant={activeOpt ? optionDomId(id, activeIdx) : undefined}
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
          )}
          <OptionList
            idPrefix={id}
            labelId={labelId}
            listId={listId}
            options={shown}
            selectedValue={value}
            activeValue={activeValue}
            onPick={pick}
            onHover={(o) => setActiveValue(o.value)}
          />
          {shown.length === 0 && (
            <div className="listbox-empty" role="status">
              No matches
            </div>
          )}
        </div>
      )}
    </div>
  );
}
