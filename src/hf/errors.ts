/** Why a Hub request failed; one variant per `kind`, each with a user-facing `message`. */
export type HfError =
  | { kind: 'invalid_id'; message: string }
  | { kind: 'unauthorized'; message: string; status: 401 }
  | { kind: 'forbidden'; message: string; status: 403 }
  | { kind: 'not_found'; message: string; status: 404 }
  | { kind: 'rate_limited'; message: string; status: 429 }
  | { kind: 'http'; message: string; status: number }
  | { kind: 'network'; message: string }
  | { kind: 'bad_json'; message: string };

/**
 * Narrow a fetch result to an error.
 * @param x value returned by a fetch function.
 */
export function isHfError(x: unknown): x is HfError {
  return typeof x === 'object' && x !== null && 'kind' in x && 'message' in x;
}

/**
 * Map an HTTP status of a failed response to an {@link HfError}.
 * @param status HTTP status (not 2xx).
 * @param what what was requested, for the message (e.g. `meta-llama/X`).
 * @param hasToken whether a token was sent, to word 401/403 advice.
 */
export function errorFromStatus(status: number, what: string, hasToken: boolean): HfError {
  switch (status) {
    case 401:
      return {
        kind: 'unauthorized',
        status: 401,
        message: hasToken
          ? `Hugging Face rejected the token for ${what} (401). Check that the token is valid.`
          : `${what} is gated or private (401). Add a Hugging Face access token to load it.`,
      };
    case 403:
      return {
        kind: 'forbidden',
        status: 403,
        message: `Access to ${what} is forbidden (403). Your token may lack access, or you have not accepted the model's terms on its Hugging Face page.`,
      };
    case 404:
      return {
        kind: 'not_found',
        status: 404,
        message: `${what} was not found on Hugging Face (404). Check the model id and revision.`,
      };
    case 429:
      return {
        kind: 'rate_limited',
        status: 429,
        message: 'Hugging Face rate limit hit (429). Wait a bit, or add an access token.',
      };
    default:
      return { kind: 'http', status, message: `Hugging Face returned HTTP ${status} for ${what}.` };
  }
}
