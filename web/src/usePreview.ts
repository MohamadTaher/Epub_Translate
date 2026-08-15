import { useCallback, useEffect, useRef, useState } from 'react';
import { previewJob } from '@/api';
import { PREVIEW_DEBOUNCE_MS } from '@/constants';
import type { JobSnapshot, JobStats } from '@/types';

/**
 * Re-cost a chapter selection on the server rather than guessing at it here.
 *
 * Debounced, because ticking a run of chapters shouldn't fire a request per
 * click, and guarded twice over: the in-flight request is aborted, and a
 * monotonic token means a reply that arrives anyway can't overwrite a newer one.
 *
 * Only a job still in `ready` is worth costing, and only once the reader has
 * changed something — until then the plan the server sent already stands.
 */
export function usePreview(job: JobSnapshot | null, selectionList: string[], touched: boolean) {
  const [preview, setPreview] = useState<JobStats | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const latest = useRef(0);

  useEffect(() => {
    if (!job || job.status !== 'ready' || !touched) return;

    const token = ++latest.current;
    const controller = new AbortController();
    setPreviewing(true);

    const timer = window.setTimeout(() => {
      previewJob(job.id, selectionList, controller.signal)
        .then((stats) => {
          if (token === latest.current) setPreview(stats);
        })
        .catch(() => undefined)
        .finally(() => {
          if (token === latest.current) setPreviewing(false);
        });
    }, PREVIEW_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
      // The `finally` above only runs if the timer fired at all. Clearing here
      // as well is what keeps this from sticking on when the job leaves `ready`
      // mid-debounce and the effect stops re-running.
      setPreviewing(false);
    };
  }, [job, touched, selectionList]);

  const reset = useCallback(() => setPreview(null), []);

  return { preview, previewing, reset };
}
