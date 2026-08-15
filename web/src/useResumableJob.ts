import { useCallback, useEffect, useState } from 'react';
import { ApiError, getJob } from '@/api';
import type { JobSnapshot } from '@/types';

function clearJobParam() {
  window.history.replaceState(null, '', window.location.pathname);
}

/**
 * The job this tab is following, and the `?job=` parameter that survives a reload.
 *
 * A run outlives any single request, so the id goes in the URL as soon as there
 * is one: reloading mid-run picks the job back up and the stream replays its
 * whole log.
 *
 * `onResume` must be stable — it is an effect dependency, and this fetch is
 * meant to happen once.
 */
export function useResumableJob(onResume: (snapshot: JobSnapshot) => void) {
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [expired, setExpired] = useState(false);

  useEffect(() => {
    const existing = new URLSearchParams(window.location.search).get('job');
    if (!existing) return;

    getJob(existing)
      .then((snapshot) => {
        setJob(snapshot);
        onResume(snapshot);
      })
      .catch((error: unknown) => {
        // Only a 404 means the job is really gone. A network blip should not
        // tell the reader their translation expired.
        if (error instanceof ApiError && error.status === 404) setExpired(true);
        clearJobParam();
      });
  }, [onResume]);

  /** A job that did not exist a moment ago; its id belongs in the URL. */
  const adopt = useCallback((snapshot: JobSnapshot) => {
    setJob(snapshot);
    window.history.replaceState(null, '', `?job=${snapshot.id}`);
  }, []);

  const clear = useCallback(() => {
    setJob(null);
    clearJobParam();
  }, []);

  return { job, setJob, adopt, expired, clear };
}
