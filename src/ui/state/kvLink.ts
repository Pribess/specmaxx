/**
 * The P→D KV transfer link of prefill/decode disaggregation (docs/design/prefill-decode-disaggregation.md §4): the
 * presets the user picks from (the sourced RDMA scale-out links already in the catalog; no new numbers), the
 * default, and the link the engine reads after the user's edits. Pure; no React, no DOM.
 */

import type { LinkSpec } from '../../sim/hw/types';
import { LINK_PRESETS } from '../../sim/hw/presets/links';
import { linkPresetId } from '../../sim/hw/presets/topologies';
import { DEFAULT_KV_TRANSFER_LINK, type KvTransferConfig } from '../../sim/engine/kvTransfer';
import type { DisaggregationChoice, KvEditPath } from './appState';

/**
 * The KV link presets: the RDMA scale-out NICs of the catalog (InfiniBand and RoCE), one NIC per device, each with
 * its cited bandwidth per direction and latency.
 */
export const KV_LINK_PRESETS: LinkSpec[] = LINK_PRESETS.filter(
  (l) => l.transport === 'rdma' && (l.kind === 'infiniband' || l.kind === 'roce'),
);

/** `linkPresetId` of the default KV link (InfiniBand NDR400, `DEFAULT_KV_TRANSFER_LINK`). */
export const DEFAULT_KV_LINK_ID = linkPresetId(DEFAULT_KV_TRANSFER_LINK);

/** The editable KV link fields, in display and URL order. */
export const KV_EDIT_PATHS: readonly KvEditPath[] = [
  'bandwidthPerDirectionBytesPerSec',
  'latencySec',
];

/** The id of a KV link preset (its `linkPresetId`), as stored in `kvLinkId` and the URL's `kvl`. */
export function kvLinkIdOf(link: Pick<LinkSpec, 'name'>): string {
  return linkPresetId(link);
}

/** The KV link preset with this id, or undefined. */
export function findKvLink(id: string): LinkSpec | undefined {
  return KV_LINK_PRESETS.find((l) => linkPresetId(l) === id);
}

/** The preset the choice uses: its `kvLinkId`, else the default. */
export function kvLinkPreset(choice: DisaggregationChoice): LinkSpec {
  return (
    findKvLink(choice.kvLinkId ?? DEFAULT_KV_LINK_ID) ??
    (findKvLink(DEFAULT_KV_LINK_ID) as LinkSpec)
  );
}

/**
 * The engine's KV transfer config of a choice: the preset with the user's edits (bandwidth B/s per direction,
 * latency s) and the overlap mode. Always set while disaggregation is on.
 */
export function kvTransferConfig(choice: DisaggregationChoice): KvTransferConfig {
  const preset = kvLinkPreset(choice);
  const edits = choice.kvEdits ?? {};
  return {
    link: {
      name: preset.name,
      bandwidthPerDirectionBytesPerSec:
        edits.bandwidthPerDirectionBytesPerSec ?? preset.bandwidthPerDirectionBytesPerSec,
      latencySec: edits.latencySec ?? preset.latencySec,
    },
    overlap: choice.kvOverlap ?? 'layerwise',
  };
}
