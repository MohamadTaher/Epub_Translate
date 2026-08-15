import { describe, expect, it } from 'vitest';
import type { JobEvent, JobSnapshot } from '@/types';
import { empty, reduce, type StreamState } from '@/useJobStream';

const event = (fields: Partial<JobEvent>): JobEvent => ({
  level: 'info',
  message: '',
  ...fields,
});

const apply = (state: StreamState, ...events: JobEvent[]): StreamState =>
  events.reduce((current, e) => reduce(current, { type: 'event', event: e }), state);

const waiting: StreamState = {
  ...empty,
  rateLimit: { seconds: 30, reason: 'requests per minute', startedAt: 0 },
};

describe('reduce', () => {
  it('keeps every event, in order', () => {
    const state = apply(empty, event({ message: 'first' }), event({ message: 'second' }));
    expect(state.events.map((e) => e.message)).toEqual(['first', 'second']);
  });

  it('treats a request going out as proof the rate limit has cleared', () => {
    const state = apply(waiting, event({ event: 'patch_start', patch: 1 }));
    expect(state.rateLimit).toBeNull();
    expect(state.requests[1]).toEqual({ state: 'active', attempt: 1 });
  });

  it('counts a failed attempt as the next attempt, still active', () => {
    const state = apply(empty, event({ event: 'attempt_failed', patch: 2, attempt: 3 }));
    expect(state.requests[2]).toEqual({ state: 'active', attempt: 4 });
  });

  it('records how long each finished request took', () => {
    const state = apply(
      empty,
      event({ event: 'patch_done', patch: 1, seconds: 42 }),
      event({ event: 'patch_done', patch: 2, seconds: 38 }),
      // The server omits `seconds` when it has none; that must not become a NaN.
      event({ event: 'patch_done', patch: 3 }),
    );
    expect(state.durations).toEqual([42, 38]);
    expect(state.requests[3]).toEqual({ state: 'done', attempt: 0 });
  });

  it('marks a request failed once its retries run out', () => {
    const state = apply(empty, event({ event: 'patch_failed', patch: 1 }));
    expect(state.requests[1]).toEqual({ state: 'failed', attempt: 0 });
  });

  it('notes that something has been written, so downloading is worthwhile', () => {
    expect(empty.hasOutput).toBe(false);
    expect(apply(empty, event({ event: 'saved' })).hasOutput).toBe(true);
  });

  it('starts and ends a rate-limit wait', () => {
    const waited = apply(
      empty,
      event({ event: 'rate_limit_wait', seconds: 12, reason: 'tokens per minute' }),
    );
    expect(waited.rateLimit).toMatchObject({ seconds: 12, reason: 'tokens per minute' });
    expect(apply(waited, event({ event: 'rate_limit_done' })).rateLimit).toBeNull();
  });

  it('leaves the requests alone for an event that names none', () => {
    const started = apply(empty, event({ event: 'patch_start', patch: 1 }));
    const after = apply(started, event({ event: 'started' }), event({ level: 'error' }));
    expect(after.requests).toBe(started.requests);
  });

  it('takes the final snapshot from `end`, and stops any countdown', () => {
    const snapshot = { id: 'j', status: 'done', completed: 3, total: 3 } as JobSnapshot;
    const state = reduce(waiting, { type: 'end', snapshot });
    expect(state.final).toBe(snapshot);
    expect(state.rateLimit).toBeNull();
  });

  it('throws the whole log away on reset, because reconnecting replays it', () => {
    const state = reduce(apply(waiting, event({ message: 'stale' })), { type: 'reset' });
    expect(state.events).toEqual([]);
    expect(state.rateLimit).toBeNull();
    expect(state.final).toBeNull();
  });
});
