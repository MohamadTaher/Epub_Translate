import { useEffect, useReducer } from 'react';
import { eventsUrl } from '@/api';
import type { JobEvent, JobSnapshot, RequestProgress } from '@/types';

export interface RateLimitWait {
  seconds: number;
  reason: string;
  /** When the wait began, so the countdown can be worked out on each tick. */
  startedAt: number;
}

export interface StreamState {
  events: JobEvent[];
  /** Keyed by 1-based request number, as carried on the events. */
  requests: Record<number, RequestProgress>;
  /** Seconds each finished request took, for estimating what is left. */
  durations: number[];
  rateLimit: RateLimitWait | null;
  /** True once anything translated has been saved, so downloading is worthwhile. */
  hasOutput: boolean;
  /** The snapshot carried by the terminal `end` event. */
  final: JobSnapshot | null;
  lastEventAt: number;
}

export const empty: StreamState = {
  events: [],
  requests: {},
  durations: [],
  rateLimit: null,
  hasOutput: false,
  final: null,
  lastEventAt: Date.now(),
};

export type Action =
  | { type: 'reset' }
  | { type: 'event'; event: JobEvent }
  | { type: 'end'; snapshot: JobSnapshot };

function withRequest(
  state: StreamState,
  patch: number | undefined,
  next: RequestProgress,
): Record<number, RequestProgress> {
  if (patch === undefined) return state.requests;
  return { ...state.requests, [patch]: next };
}

/** Exported for its tests; the hook below is the only production caller. */
export function reduce(state: StreamState, action: Action): StreamState {
  switch (action.type) {
    case 'reset':
      return { ...empty, lastEventAt: Date.now() };

    case 'end':
      return { ...state, final: action.snapshot, rateLimit: null };

    case 'event': {
      const event = action.event;
      const next: StreamState = {
        ...state,
        events: [...state.events, event],
        lastEventAt: Date.now(),
      };

      switch (event.event) {
        case 'patch_start':
          next.requests = withRequest(state, event.patch, {
            state: 'active',
            attempt: event.attempt ?? 1,
          });
          // A request going out means whatever was blocking has cleared.
          next.rateLimit = null;
          break;

        case 'attempt_failed':
          next.requests = withRequest(state, event.patch, {
            state: 'active',
            attempt: (event.attempt ?? 0) + 1,
          });
          break;

        case 'patch_done':
          next.requests = withRequest(state, event.patch, { state: 'done', attempt: 0 });
          if (typeof event.seconds === 'number') {
            next.durations = [...state.durations, event.seconds];
          }
          break;

        case 'patch_failed':
          next.requests = withRequest(state, event.patch, { state: 'failed', attempt: 0 });
          break;

        case 'saved':
          next.hasOutput = true;
          break;

        case 'rate_limit_wait':
          next.rateLimit = {
            seconds: event.seconds ?? 0,
            reason: event.reason ?? 'rate limit',
            startedAt: Date.now(),
          };
          break;

        case 'rate_limit_done':
          next.rateLimit = null;
          break;

        default:
          break;
      }

      return next;
    }

    default:
      return state;
  }
}

/**
 * Follow a job's progress.
 *
 * Every connection replays the job's whole event log before going live, so the
 * state is thrown away and rebuilt on open rather than appended to — that is
 * also what makes reconnecting after a dropped stream lossless.
 */
export function useJobStream(jobId: string | null, active: boolean): StreamState {
  const [state, dispatch] = useReducer(reduce, empty);

  useEffect(() => {
    if (!jobId || !active) return;

    dispatch({ type: 'reset' });
    const source = new EventSource(eventsUrl(jobId));

    source.addEventListener('message', (message) => {
      try {
        dispatch({ type: 'event', event: JSON.parse(message.data) as JobEvent });
      } catch {
        // A malformed frame is not worth tearing the stream down for.
      }
    });

    source.addEventListener('end', (message) => {
      try {
        dispatch({ type: 'end', snapshot: JSON.parse((message as MessageEvent).data) as JobSnapshot });
      } catch {
        // Fall through: App re-fetches the snapshot when the stream ends anyway.
      }
      // Required. The server closes the stream after `end`, and EventSource
      // reconnects on close — the job is terminal by then, so it would replay
      // the log and send `end` again, forever.
      source.close();
    });

    return () => source.close();
  }, [jobId, active]);

  return state;
}
