import { knownDrafters } from '../../sim/models/hf/drafter';
import type { ModelSpec } from '../../sim/ir/model';
import { HYPOTHETICAL_DRAFTER } from '../state';
import type { PickerOption } from './fuzzy';

/** The option value of "Model default" (not a repo id, so a typed repo cannot collide with it): committing it clears the override (`drafter: undefined`, no `sdr` in the URL). */
export const DEFAULT_DRAFTER_OPTION = 'default';

/** Display name of a block method, for the default option's label. */
const METHOD_NAMES = { dflash2: 'DFlash 2', dspark: 'DSpark' } as const;

/** What the drafter combobox shows and offers; pure, so the tests cover it without React. */
export interface DrafterChoices {
  /** The list: the model default first (when there is one), then the other released drafters, then the stand-in. */
  options: PickerOption[];
  /** `value` of the option the field currently stands for (check mark; the full list is shown while it is unedited). */
  selected: string;
  /** The text the input shows when unedited: the selected option's label, or the typed repo id. */
  text: string;
  /** The known released drafter that is the default (no checkpoint drafter), whose repo id equals "Model default". */
  defaultRepo: string | undefined;
}

/**
 * The drafter options of a block method for `model` (repo id `targetId`), given the user's `chosen` drafter
 * (`SpecDecChoice.drafter`; undefined = the default).
 *
 * The default is the checkpoint's own drafter of that method, else the known released drafter for the target, else
 * the hypothetical stand-in (which then is the default, and is not listed twice). The default is always an option, so
 * a user who picked something else can go back; a released drafter that is the default is not also listed as itself,
 * and choosing its repo id (typed, or from the URL) is the same as the default.
 */
export function drafterChoices(
  method: 'dflash2' | 'dspark',
  targetId: string,
  model: ModelSpec,
  chosen: string | undefined,
): DrafterChoices {
  const released = knownDrafters(targetId).filter((d) => d.method === method);
  const own =
    model.draft?.kind === 'block_drafter' && model.draft.method === method
      ? model.draft
      : undefined;
  const defaultRepo = own === undefined ? released[0]?.repoId : undefined;
  const options: PickerOption[] = [];
  if (own !== undefined) {
    options.push({
      value: DEFAULT_DRAFTER_OPTION,
      label: `Model default (checkpoint’s ${METHOD_NAMES[method]} drafter)`,
      description: `${own.layers} layers`,
    });
  } else if (defaultRepo !== undefined) {
    options.push({
      value: DEFAULT_DRAFTER_OPTION,
      label: `Model default (${defaultRepo})`,
      description: 'Released drafter',
    });
  }
  for (const d of released) {
    if (d.repoId !== defaultRepo)
      options.push({ value: d.repoId, label: d.repoId, group: 'Released drafters' });
  }
  options.push({
    value: HYPOTHETICAL_DRAFTER,
    label: 'Hypothetical drafter',
    description: '5 layers, 32/8 heads, window 2048',
    group: 'Stand-in',
  });
  const hasDefaultOption = options[0].value === DEFAULT_DRAFTER_OPTION;
  // With no default drafter, the hypothetical one is the default, so an absent choice selects it.
  const effective = chosen === undefined || chosen === defaultRepo ? undefined : chosen;
  const selected =
    effective === undefined
      ? hasDefaultOption
        ? DEFAULT_DRAFTER_OPTION
        : HYPOTHETICAL_DRAFTER
      : effective;
  const text = options.find((o) => o.value === selected)?.label ?? selected;
  return { options, selected, text, defaultRepo };
}

/**
 * What committing `value` (a picked option or typed text) means: the new `SpecDecChoice.drafter`, or `'invalid'` when
 * the text is no repo id. The default option, empty text, and the default released drafter's repo id all return to the
 * default (undefined); the label of an option is its value.
 */
export function drafterFromCommit(
  value: string,
  choices: DrafterChoices,
  isValid: (id: string) => boolean,
): string | undefined | 'invalid' {
  const v = value.trim();
  if (v === '' || v === DEFAULT_DRAFTER_OPTION) return undefined;
  const byLabel = choices.options.find((o) => o.label === v);
  if (byLabel !== undefined)
    return byLabel.value === DEFAULT_DRAFTER_OPTION ? undefined : byLabel.value;
  if (v === choices.defaultRepo) return undefined;
  if (v === HYPOTHETICAL_DRAFTER || isValid(v)) return v;
  return 'invalid';
}
