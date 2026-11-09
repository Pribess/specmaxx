import { validateDevice, validateLink } from '../../sim/hw/validate';
import {
  editableFields,
  editableLinkFields,
  editsKey,
  fromDisplay,
  toDisplay,
  type EditTarget,
  type EditableField,
} from '../state';
import { NumberField } from './NumberField';
import { HelpIcon } from './HelpIcon';
import { helpLines } from './helpLines';
import type { PanelProps as Props } from './types';

export function CustomSpecEditor({ state, dispatch, built }: Props) {
  const hardware = built.hardware;
  if (!hardware) return null;
  const specs: { target: EditTarget; name: string; fields: EditableField[]; errors: string[] }[] = [
    {
      target: 'device',
      name: 'Device',
      fields: editableFields(hardware.device.device),
      errors: validateDevice(hardware.device.device),
    },
  ];
  for (const target of ['intra', 'inter'] as const) {
    const spec = hardware.links[target]?.link;
    if (spec)
      specs.push({
        target,
        name: `${target === 'intra' ? 'Intra' : 'Inter'}-node link`,
        fields: editableLinkFields(spec),
        errors: validateLink(spec),
      });
  }
  return (
    <details className="spec-editor" open>
      <summary>
        Edit spec numbers{hardware.edited ? ' (edited)' : ''}
        <HelpIcon term="Edit spec numbers">
          {helpLines([
            'Starts as a copy of this preset.',
            'Rates and capacity are per device.',
            'Link bandwidth is per direction.',
          ])}
        </HelpIcon>
      </summary>
      {specs.map((s) => (
        <details key={s.target}>
          <summary>{s.name}</summary>
          {s.errors.map((e) => (
            <p className="error" key={e}>
              {e}
            </p>
          ))}
          {(['memory', 'matmul', 'vector', 'tiling', 'link'] as const).map((group) => {
            const fields = s.fields.filter((f) => f.group === group);
            return (
              fields.length > 0 && (
                <details key={group}>
                  <summary>{group === 'tiling' ? 'Shape efficiency / tiling' : group}</summary>
                  {fields.map((f) => (
                    <div
                      className={
                        state.hw[editsKey(s.target)][f.path] !== undefined
                          ? 'spec-field edited'
                          : 'spec-field'
                      }
                      key={f.path}
                    >
                      <NumberField
                        label={`${f.label} ${f.unit.label}`}
                        step="any"
                        value={toDisplay(f.value, f.unit)}
                        {...(f.help ? { help: helpLines(f.help) } : {})}
                        onChange={(v) =>
                          dispatch({
                            type: 'setEdit',
                            target: s.target,
                            path: f.path,
                            value: fromDisplay(v, f.unit),
                          })
                        }
                      />
                      {state.hw[editsKey(s.target)][f.path] !== undefined && (
                        <button
                          onClick={() =>
                            dispatch({
                              type: 'setEdit',
                              target: s.target,
                              path: f.path,
                              value: undefined,
                            })
                          }
                        >
                          Reset
                        </button>
                      )}
                    </div>
                  ))}
                </details>
              )
            );
          })}
        </details>
      ))}
      {hardware.edited && (
        <button onClick={() => dispatch({ type: 'resetEdits' })}>Reset all edits</button>
      )}
    </details>
  );
}
