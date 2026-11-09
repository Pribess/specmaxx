import { useState } from 'react';
import {
  ACCEPTANCE_PRESETS,
  pooledDraftLengths,
  presetDraftLengthExtrapolation,
  presetMethod,
  type AcceptancePreset,
} from '../../sim/engine/specdec';
import { knownDrafters } from '../../sim/models/hf/drafter';
import {
  HYPOTHETICAL_DRAFTER,
  MAX_ACCEPTANCE,
  MAX_DRAFT_LENGTH,
  SPEC_DEC_USE_CASES,
  draftSpecOf,
  drafterSource,
  effectiveAcceptance,
  effectiveDraftLength,
  type DrafterStatus,
  type SpecDecChoice,
  type SpecDecMethod,
  type SpecDecUseCase,
} from '../state';
import { isValidModelId } from '../../hf/urls';
import { Combobox } from './Combobox';
import { NumberField } from './NumberField';
import { Picker } from './Picker';
import { helpLines } from './helpLines';
import type { PickerOption } from './fuzzy';
import type { PanelProps } from './types';

/** Method labels, in picker order. */
const METHOD_LABELS: Record<SpecDecMethod, string> = {
  mtp: 'MTP (model head)',
  dflash2: 'DFlash 2',
  dspark: 'DSpark',
};

/** Use-case labels. */
const USE_CASE_LABELS: Record<SpecDecUseCase, string> = {
  prose: 'Prose / chat',
  code: 'Code',
  math: 'Math',
  reasoning: 'Reasoning',
  agentic: 'Agentic code',
};

/** The picker value of "custom" acceptance (an α the user typed). */
const CUSTOM = 'custom';

const METHOD_HELP = [
  'A cheap draft guesses several tokens, then the model checks them all in one pass.',
  'Accepted guesses give more tokens per step, so decode is faster.',
];
const PP_HELP = ['Needs PP 1: pipeline parallelism is not supported yet.'];

/**
 * Workload → "Speculative decoding" (design doc §6.1): method, drafter (block methods), draft length k, use case and
 * acceptance α. MTP is offered only when the model has an MTP head; every method is unavailable at PP > 1.
 */
export function SpecDecodeSection({
  state,
  dispatch,
  loaded,
  built,
  drafter,
}: PanelProps & { drafter?: DrafterStatus }) {
  const spec = state.deployment.specDec;
  const model = loaded?.model;
  const ppBlocked = state.parallel.pp > 1;
  const methods: SpecDecMethod[] = ppBlocked
    ? []
    : model?.draft?.kind === 'mtp'
      ? ['mtp', 'dflash2', 'dspark']
      : ['dflash2', 'dspark'];
  const methodOptions: PickerOption[] = [
    { value: 'off', label: 'Off' },
    ...methods.map((m) => ({ value: m, label: METHOD_LABELS[m] })),
    // A method from the URL that this model or PP does not allow stays visible, so it can be turned off.
    ...(spec !== undefined && !methods.includes(spec.method)
      ? [{ value: spec.method, label: METHOD_LABELS[spec.method] }]
      : []),
  ];
  const set = (next: Partial<SpecDecChoice>) =>
    dispatch({ type: 'setSpecDec', specDec: { ...(spec ?? { method: 'dflash2' }), ...next } });
  const issues = built.issues.filter((i) => i.field === 'specDec');
  const source = spec && model ? drafterSource(spec, model, state.model.id) : undefined;
  const draft = spec && model ? draftSpecOf(spec.method, source, model, drafter) : undefined;
  const k = spec ? effectiveDraftLength(spec, draft) : 1;
  const acceptance = spec ? effectiveAcceptance(spec, k, draft) : undefined;
  const alpha = acceptance?.kind === 'iid' ? acceptance.alpha : acceptance?.alphas[0];
  const maxDepth = draft?.kind === 'mtp' ? draft.maxReliableDepth : undefined;
  const maxK =
    draft?.kind === 'block_drafter'
      ? Math.min(MAX_DRAFT_LENGTH, draft.blockSize - 1)
      : MAX_DRAFT_LENGTH;
  const useCase = spec?.alpha !== undefined ? CUSTOM : (spec?.useCase ?? 'prose');
  const preset =
    spec && spec.alpha === undefined
      ? ACCEPTANCE_PRESETS[presetMethod(spec.method, draft)][spec.useCase ?? 'prose']
      : undefined;
  const extrapolatedFrom = spec
    ? presetDraftLengthExtrapolation(spec.method, spec.useCase ?? 'prose', k, draft, spec.alpha)
    : undefined;
  // A pooled preset names the k of its runs once ("pooled from k = 7, 15"), with ⚠ when k is none of them.
  const alphaHint =
    preset?.basis === 'pooled'
      ? `${extrapolatedFrom ? '⚠ ' : ''}${basisHint(preset)}`
      : [
          preset ? basisHint(preset) : undefined,
          extrapolatedFrom ? `⚠ fitted at k = ${extrapolatedFrom.join(', ')}` : undefined,
        ]
          .filter(Boolean)
          .join(' · ');
  return (
    <div className="deployment-section" data-testid="specdec-section">
      <Picker
        label="Speculative decoding"
        value={spec?.method ?? 'off'}
        options={methodOptions}
        onChange={(v) =>
          dispatch({
            type: 'setSpecDec',
            specDec: v === 'off' ? undefined : { method: v as SpecDecMethod },
          })
        }
        help={helpLines(ppBlocked ? [...METHOD_HELP, ...PP_HELP] : METHOD_HELP)}
      />
      {spec !== undefined && spec.method !== 'mtp' && (
        <DrafterField
          method={spec.method}
          targetId={state.model.id}
          chosen={spec.drafter}
          sourceKind={source?.kind}
          sourceRepo={source?.kind === 'repo' ? source.repoId : undefined}
          onChange={(drafterId) => set({ drafter: drafterId })}
        />
      )}
      {spec !== undefined && (
        <>
          <NumberField
            label="Draft length k"
            value={k}
            min={1}
            max={maxK}
            onChange={(v) =>
              set({ k: Number.isInteger(v) && v >= 1 && v <= MAX_DRAFT_LENGTH ? v : spec.k })
            }
            hint={
              maxDepth !== undefined && k > maxDepth
                ? `⚠ head trained for k ≤ ${maxDepth}`
                : undefined
            }
            help={helpLines([
              'How many tokens the draft guesses per step.',
              'Longer drafts can accept more tokens, but cost more to check.',
            ])}
          />
          <Picker
            label="Use case"
            value={useCase}
            options={[
              ...SPEC_DEC_USE_CASES.map((u) => ({ value: u, label: USE_CASE_LABELS[u] })),
              ...(useCase === CUSTOM ? [{ value: CUSTOM, label: 'Custom' }] : []),
            ]}
            onChange={(v) => {
              if (v !== CUSTOM) set({ useCase: v as SpecDecUseCase, alpha: undefined });
            }}
            help={helpLines([
              'Fills in the acceptance rate from published measurements.',
              'Code means code generation.',
            ])}
          />
          <NumberField
            label="Acceptance α"
            value={alpha ?? 0}
            step="0.01"
            min={0}
            max={MAX_ACCEPTANCE}
            onChange={(v) =>
              set({
                alpha: Number.isFinite(v) ? Math.min(MAX_ACCEPTANCE, Math.max(0, v)) : undefined,
              })
            }
            hint={alphaHint || undefined}
            help={helpLines([
              'Chance that each drafted token is accepted.',
              'Higher means more tokens per step and faster decode.',
              'Editing it sets the use case to Custom.',
            ])}
          />
        </>
      )}
      {issues.length > 0 && (
        <span className="hint" role="status">
          {issues.map((i, n) => (
            <span className={i.level === 'error' ? 'error' : ''} key={n}>
              {i.message}
            </span>
          ))}
        </span>
      )}
    </div>
  );
}

/** Short label for a preset whose α reproduces no run of this method × use case; undefined for a measured one. */
function basisHint(preset: AcceptancePreset): string | undefined {
  switch (preset.basis) {
    case 'measured':
      return undefined;
    case 'pooled':
      return `pooled from k = ${pooledDraftLengths(preset).join(', ')}`;
    case 'carried_over':
      return 'interpolated';
    case 'borrowed':
      return `from ${preset.measurements[0].methodVersion}`;
    default: {
      const _never: never = preset.basis;
      throw new Error(`unknown acceptance basis ${String(_never)}`);
    }
  }
}
/**
 * The drafter of a block method: a Hub repo id typed or picked from the known drafters of the target, or the
 * hypothetical stand-in. An empty commit returns to the default (`chosen` undefined).
 */
function DrafterField({
  method,
  targetId,
  chosen,
  sourceKind,
  sourceRepo,
  onChange,
}: {
  method: 'dflash2' | 'dspark';
  targetId: string;
  chosen: string | undefined;
  sourceKind: 'repo' | 'checkpoint' | 'hypothetical' | undefined;
  sourceRepo: string | undefined;
  onChange(drafter: string | undefined): void;
}) {
  const shown =
    sourceKind === 'hypothetical' ? '' : sourceKind === 'checkpoint' ? '' : (sourceRepo ?? '');
  const [text, setText] = useState(shown);
  // A new effective drafter (another method or model) resets the text, adjusted during render.
  const [prevShown, setPrevShown] = useState(shown);
  if (prevShown !== shown) {
    setPrevShown(shown);
    setText(shown);
  }
  const options: PickerOption[] = [
    ...knownDrafters(targetId)
      .filter((d) => d.method === method)
      .map((d) => ({ value: d.repoId, label: d.repoId, group: 'Released drafters' })),
    {
      value: HYPOTHETICAL_DRAFTER,
      label: 'Hypothetical drafter',
      description: '5 layers, 32/8 heads, window 2048',
      group: 'Stand-in',
    },
  ];
  const placeholder =
    sourceKind === 'checkpoint' ? 'The checkpoint’s own drafter' : 'Hypothetical drafter';
  return (
    <div className="drafter-field">
      <Combobox
        label="Drafter"
        value={text}
        onValueChange={setText}
        options={options}
        placeholder={placeholder}
        onCommit={(value) => {
          const v = value.trim();
          if (v === '') onChange(undefined);
          else if (v === HYPOTHETICAL_DRAFTER || isValidModelId(v))
            onChange(v === chosen ? chosen : v);
          else setText(shown);
        }}
        help={helpLines([
          'The small model that drafts the tokens, from its Hugging Face config.',
          'Hypothetical is a configurable stand-in, not a real model.',
        ])}
      />
      {sourceKind === 'hypothetical' && (
        <span className="badge" data-testid="hypothetical-badge">
          hypothetical
        </span>
      )}
    </div>
  );
}
