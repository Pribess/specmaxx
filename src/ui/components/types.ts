import type { Dispatch } from 'react';
import type { AppState, AppAction, LoadedModel, BuildResult } from '../state';

/** Shared estimator state and derived validation supplied to the input panels. */
export interface PanelProps {
  state: AppState;
  dispatch: Dispatch<AppAction>;
  loaded: LoadedModel | null;
  built: BuildResult;
}
