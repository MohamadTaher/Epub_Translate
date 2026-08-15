import { useCallback, useEffect, useState } from 'react';
import { getStatus } from '@/api';
import { ASSUMED_REQUESTS_PER_MINUTE } from '@/constants';
import type { Status } from '@/types';

/**
 * The server's limits, model and remaining budget.
 *
 * Re-read after anything that spends budget, since the figures shown to a
 * reader are the server's and not worth guessing at locally. A failure is
 * ignored on purpose: a stale status is better than an error over the whole app.
 */
export function useServerStatus() {
  const [status, setStatus] = useState<Status | null>(null);

  const refresh = useCallback(() => {
    getStatus()
      .then(setStatus)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    status,
    refresh,
    /** The pace every estimate is built on, defaulted until the server answers. */
    requestsPerMinute: status?.requests_per_minute ?? ASSUMED_REQUESTS_PER_MINUTE,
  };
}
