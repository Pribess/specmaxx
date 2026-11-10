import type { Dispatch } from 'react';
import type { AppState, AppAction, LoadedModel, BuildResult, PoolId } from '../state';

/** Shared estimator state and derived validation supplied to the input panels. */
export interface PanelProps {
  state: AppState;
  dispatch: Dispatch<AppAction>;
  loaded: LoadedModel | null;
  built: BuildResult;
  /**
   * With disaggregation on: the pool the panel edits (`state`, `dispatch` and `built` are then that pool's view),
   * named in the per-pool panels' titles. Absent while off.
   */
  pool?: PoolId;
}

/** The title of a per-pool panel: `Hardware`, or `Hardware (decode pool)` while it edits a pool. */
export function poolTitle(title: string, pool: PoolId | undefined): string {
  return pool === undefined ? title : `${title} (${pool} pool)`;
}
