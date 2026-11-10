import type { InNetworkReduction } from '../../sim/hw/types';
import { inNetworkReductionsOf, nicLabel, nicOptions } from '../state';
import { Picker } from './Picker';
import { helpLines } from './helpLines';
import type { PanelProps } from './types';

const NICS_HELP = helpLines([
  "Network cards in each machine, each at the network link's speed.",
  'Default is one per GPU.',
  "The machine's network speed is shared evenly by its GPUs.",
]);

const IN_NETWORK_HELP = helpLines([
  "The network switch adds up the GPUs' all-reduce data itself (NVLS on NVLink Switch, SHARP on InfiniBand).",
  'Each GPU then sends its data about once instead of twice.',
  'Used when the collective algorithm is best.',
]);

/** Display name of each in-network reduction. */
const IN_NETWORK_NAME: Record<InNetworkReduction, string> = { nvls: 'NVLS', sharp: 'SHARP' };

/**
 * Hardware → Advanced → "Network": the inter-node NICs per machine (`HardwareChoice.nicsPerNode`, shown for several
 * machines of more than one GPU each on a switch, `nicOptions`) and the in-network reduction switch
 * (`SimOptions.inNetworkReduction`, shown when a link in use offers NVLS or SHARP, `inNetworkReductionsOf`;
 * docs/simulation-model.md §6 "Multi-NIC nodes", "In-network reduction (NVLS, SHARP)"). Hidden when neither shows.
 */
export function NetworkFields({ state, dispatch, built }: PanelProps) {
  const hw = state.hw;
  const nics = nicOptions(hw);
  const reductions = built.hardware ? inNetworkReductionsOf(built.hardware) : [];
  if (nics.length === 0 && reductions.length === 0) return null;
  const inNetworkOn = (state.options.inNetworkReduction ?? 'off') === 'on';
  return (
    <fieldset className="unlock advanced-fields" data-testid="network-advanced">
      <legend>Network</legend>
      {nics.length > 0 && (
        <Picker
          label="NICs per machine"
          value={String(hw.nicsPerNode ?? hw.devicesPerNode)}
          options={nics.map((n) => ({ value: String(n), label: nicLabel(n, hw.devicesPerNode) }))}
          help={NICS_HELP}
          onChange={(v) => dispatch({ type: 'setNics', nicsPerNode: Number(v) })}
        />
      )}
      {reductions.length > 0 && (
        <Picker
          label="In-network reduction"
          value={inNetworkOn ? 'on' : 'off'}
          options={[
            { value: 'off', label: 'Off' },
            {
              value: 'on',
              label: `On (${reductions.map((r) => IN_NETWORK_NAME[r]).join(', ')})`,
            },
          ]}
          help={IN_NETWORK_HELP}
          onChange={(v) =>
            dispatch({
              type: 'setOptions',
              options: { inNetworkReduction: v === 'on' ? 'on' : 'off' },
            })
          }
        />
      )}
    </fieldset>
  );
}
