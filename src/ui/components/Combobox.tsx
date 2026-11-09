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
import {
  NO_ACTIVE_OPTION,
  OptionList,
  findOptionElement,
  revealOption,
  setActiveDescendant,
} from './OptionList';

/** Keys that move the active suggestion, and the step each makes. */
const NAV_KEYS: Record<string, NavStep> = {
  ArrowDown: 'next',
  ArrowUp: 'prev',
  PageDown: 'pageDown',
  PageUp: 'pageUp',
};

/**
 * Free-text input with a filtered suggestion list (ARIA combobox with `aria-autocomplete="list"`), built
 * like {@link Picker}: a Radix Popover anchored to the input, with a cmdk listbox in a themed ScrollArea.
 * The user can type or paste any value; suggestions are a shortcut, not a restriction.
 *
 * - Typing filters the suggestions with {@link fuzzyFilter} and opens the list; focus opens it too.
 * - ArrowUp/ArrowDown move the active suggestion (none is active until a key or the pointer moves into
 *   the list, so Enter after typing commits the typed text).
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
  selected,
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
  /**
   * Value of the option the text stands for, when the text is its label rather than a search: it gets the check
   * mark, and the list shows every option until the text is edited.
   */
  selected?: string;
  /** Rendered as a "?" help icon next to the label. */
  help?: ReactNode;
}) {
  const id = useId();
  const labelId = `${id}-label`;
  const boxRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [listEl, setListEl] = useState<HTMLDivElement | null>(null);
  const [open, setOpen] = useState(false);
  const [activeValue, setActiveValue] = useState<string | undefined>(undefined);

  // While the text is still what the selected option shows (its label, or the repo id of a choice that is no
  // option), the whole list is shown: nothing has been searched for.
  const selectedOption =
    selected === undefined ? undefined : options.find((o) => o.value === selected);
  const unedited = selected !== undefined && value === (selectedOption?.label ?? selected);
  const shown = useMemo(
    () => (unedited ? options : fuzzyFilter(options, value)),
    [options, value, unedited],
  );
  const flat = useMemo(() => groupOptions(shown).flatMap((g) => g.options), [shown]);
  const activeIdx = flat.findIndex((o) => o.value === activeValue);
  const activeOpt = activeIdx >= 0 ? flat[activeIdx] : undefined;
  const isOpen = open && flat.length > 0;

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
      case 'Home':
      case 'End':
        e.stopPropagation(); // the caret, not the list (cmdk would jump)
        return;
      case 'Tab':
        close();
        return;
    }
  };

  // aria-activedescendant on the input names the active suggestion, whose id cmdk generates; keep it in view.
  useLayoutEffect(() => {
    const el = isOpen ? findOptionElement(listEl, activeValue) : undefined;
    setActiveDescendant(inputRef.current, el);
    revealOption(el);
  }, [isOpen, listEl, activeValue, shown]);

  return (
    <div className="field combobox">
      <div className="field-head">
        <span id={labelId} className="field-label">
          {label}
        </span>
        {help && <HelpIcon term={label}>{help}</HelpIcon>}
      </div>
      {/* The input sits inside cmdk's root so the portalled list shares its state. */}
      <Command
        className="combobox-command"
        shouldFilter={false}
        vimBindings={false}
        value={activeValue ?? NO_ACTIVE_OPTION}
        onValueChange={setActiveValue}
      >
        <Popover.Root open={isOpen} onOpenChange={(next) => (next ? setOpen(true) : close())}>
          <Popover.Anchor asChild>
            <div className="combobox-box" ref={boxRef}>
              <input
                ref={inputRef}
                type="text"
                role="combobox"
                aria-labelledby={labelId}
                aria-expanded={isOpen}
                aria-controls={isOpen ? listEl?.id : undefined}
                aria-autocomplete="list"
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
          </Popover.Anchor>
          <Popover.Portal>
            <Popover.Content
              className="picker-popup"
              role="presentation"
              side="bottom"
              align="start"
              sideOffset={4}
              collisionPadding={8}
              onOpenAutoFocus={(e) => e.preventDefault()}
              onCloseAutoFocus={(e) => e.preventDefault()}
              onInteractOutside={(e) => {
                // Pressing in the input itself keeps the list open.
                if (boxRef.current?.contains(e.target as Node)) e.preventDefault();
              }}
              onMouseDown={(e) => e.preventDefault()}
            >
              <OptionList
                label={label}
                listRef={setListEl}
                options={shown}
                selectedValue={selected ?? value}
                onPick={(o) => commit(o.value)}
              />
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
      </Command>
    </div>
  );
}
