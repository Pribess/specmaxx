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

/** Keys that move the active suggestion, and the step each makes. */
const NAV_KEYS: Record<string, NavStep> = {
  ArrowDown: 'next',
  ArrowUp: 'prev',
  PageDown: 'pageDown',
  PageUp: 'pageUp',
};

/**
 * Free-text input with a filtered suggestion list (ARIA combobox with `aria-autocomplete="list"`).
 * The user can type or paste any value; suggestions are a shortcut, not a restriction.
 *
 * - Typing filters the suggestions with {@link fuzzyFilter} and opens the list; focus opens it too.
 * - ArrowUp/ArrowDown move the active suggestion (none is active until a key moves into the list).
 * - Enter with an active suggestion calls `onCommit(option.value)`; Enter without one calls
 *   `onCommit(value)` with the typed text. Either closes the list.
 * - Escape closes the list (a second Escape does nothing); Tab closes it and keeps the typed text.
 * - A click or tap on a suggestion calls `onCommit(option.value)`.
 */
export function Combobox({
  label,
  value,
  onValueChange,
  options,
  onCommit,
  placeholder,
  help,
}: {
  /** Visible label; the accessible name of the input. */
  label: string;
  /** The text in the input. */
  value: string;
  /** Called on every keystroke with the new text. */
  onValueChange(text: string): void;
  /** Suggestions, filtered by the typed text with fuzzyFilter. */
  options: PickerOption[];
  /** Enter with no active suggestion commits the typed text; picking a suggestion commits its value. */
  onCommit(value: string): void;
  placeholder?: string;
  /** Rendered as a "?" help icon next to the label. */
  help?: ReactNode;
}) {
  const id = useId();
  const labelId = `${id}-label`;
  const listId = `${id}-list`;
  const boxRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [activeValue, setActiveValue] = useState<string | undefined>(undefined);

  const shown = useMemo(() => fuzzyFilter(options, value), [options, value]);
  const flat = useMemo(() => groupOptions(shown).flatMap((g) => g.options), [shown]);
  const activeIdx = flat.findIndex((o) => o.value === activeValue);
  const activeOpt = activeIdx >= 0 ? flat[activeIdx] : undefined;
  const isOpen = open && flat.length > 0;
  const popupStyle = useFloating(boxRef, popupRef, isOpen, {
    preferred: 'below',
    maxHeight: 360,
    width: 'anchor',
  });

  const close = () => {
    setOpen(false);
    setActiveValue(undefined);
  };
  const commit = (text: string) => {
    onCommit(text);
    close();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    const step = NAV_KEYS[e.key];
    if (step) {
      e.preventDefault();
      if (!open) setOpen(true);
      const i = stepIndex(activeIdx, flat.length, step);
      if (i >= 0) setActiveValue(flat[i].value);
      return;
    }
    switch (e.key) {
      case 'Enter':
        e.preventDefault();
        commit(activeOpt ? activeOpt.value : value);
        return;
      case 'Escape':
        if (!isOpen) return;
        e.preventDefault();
        e.stopPropagation();
        close();
        return;
      case 'Tab':
        close();
        return;
    }
  };

  // An outside pointer closes the list; the typed text stays.
  useEffect(() => {
    if (!isOpen) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!boxRef.current?.contains(t) && !popupRef.current?.contains(t)) {
        setOpen(false);
        setActiveValue(undefined);
      }
    };
    document.addEventListener('pointerdown', onDown);
    return () => document.removeEventListener('pointerdown', onDown);
  }, [isOpen]);

  // Keep the active suggestion visible while moving with the keyboard.
  useEffect(() => {
    if (isOpen && activeIdx >= 0) {
      document.getElementById(optionDomId(id, activeIdx))?.scrollIntoView?.({ block: 'nearest' });
    }
  }, [isOpen, activeIdx, id]);

  return (
    <div className="field combobox">
      <div className="field-head">
        <span id={labelId} className="field-label">
          {label}
        </span>
        {help && <HelpIcon term={label}>{help}</HelpIcon>}
      </div>
      <div className="combobox-box" ref={boxRef}>
        <input
          ref={inputRef}
          type="text"
          role="combobox"
          aria-labelledby={labelId}
          aria-expanded={isOpen}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={isOpen && activeOpt ? optionDomId(id, activeIdx) : undefined}
          value={value}
          placeholder={placeholder}
          autoComplete="off"
          spellCheck={false}
          onFocus={() => setOpen(true)}
          onChange={(e) => {
            onValueChange(e.target.value);
            setOpen(true);
            setActiveValue(undefined);
          }}
          onKeyDown={onKeyDown}
        />
      </div>
      {isOpen && (
        <div
          ref={popupRef}
          className="picker-popup"
          style={popupStyle}
          onMouseDown={(e) => {
            if (e.target !== inputRef.current) e.preventDefault();
          }}
        >
          <OptionList
            idPrefix={id}
            labelId={labelId}
            listId={listId}
            options={shown}
            selectedValue={value}
            activeValue={activeValue}
            onPick={(o) => commit(o.value)}
            onHover={(o) => setActiveValue(o.value)}
          />
        </div>
      )}
    </div>
  );
}
