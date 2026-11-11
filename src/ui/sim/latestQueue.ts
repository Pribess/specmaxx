/**
 * One running job and one replaceable queued job. A task boundary before each run lets worker message
 * events drain first, even when the simulation itself is synchronous. Superseded requests are settled
 * explicitly so the caller never retains an unresolved promise.
 */
export function latestQueue<T>(run: (job: T) => Promise<void>, supersede: (job: T) => void) {
  let queued: T | undefined;
  let scheduled = false;
  const drain = async () => {
    const job = queued!;
    queued = undefined;
    try {
      await run(job);
    } finally {
      if (queued !== undefined) setTimeout(() => void drain(), 0);
      else scheduled = false;
    }
  };
  return (job: T) => {
    if (queued !== undefined) supersede(queued);
    queued = job;
    if (!scheduled) {
      scheduled = true;
      setTimeout(() => void drain(), 0);
    }
  };
}
