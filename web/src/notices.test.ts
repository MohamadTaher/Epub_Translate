import { describe, expect, it } from 'vitest';
import { describeBlock, describeOutcome } from '@/notices';
import type { JobSnapshot, Status } from '@/types';

const status = (fields: Partial<Status>): Status =>
  ({
    configured: true,
    cooldown_seconds: 0,
    busy: false,
    remaining_requests: 100,
    ...fields,
  }) as Status;

const job = (fields: Partial<JobSnapshot>): JobSnapshot =>
  ({ id: 'j', error: null, completed: 0, total: 0, stats: null, ...fields }) as JobSnapshot;

describe('describeBlock', () => {
  it('says nothing until the status has arrived, or when nothing is wrong', () => {
    expect(describeBlock(null)).toBeNull();
    expect(describeBlock(status({}))).toBeNull();
  });

  it('reports a missing key ahead of every other reason', () => {
    const notice = describeBlock(
      status({ configured: false, busy: true, cooldown_seconds: 120, remaining_requests: 0 }),
    );
    expect(notice?.title).toBe('This server has no API key');
    expect(notice?.tone).toBe('error');
  });

  it('rounds the cooldown up to whole minutes', () => {
    expect(describeBlock(status({ cooldown_seconds: 61 }))?.body).toContain('about 2 minutes');
    expect(describeBlock(status({ cooldown_seconds: 30 }))?.body).toContain('about 1 minute.');
  });

  it('distinguishes a busy server from a spent budget', () => {
    expect(describeBlock(status({ busy: true }))?.title).toBe('The server is busy');
    expect(describeBlock(status({ remaining_requests: 0 }))?.title).toBe('Today’s budget is spent');
  });
});

describe('describeOutcome', () => {
  it('says nothing about a run still going', () => {
    expect(describeOutcome(job({ status: 'running', total: 4 }))).toBeNull();
    expect(describeOutcome(job({ status: 'ready' }))).toBeNull();
  });

  it('celebrates only a run that actually finished everything', () => {
    const notice = describeOutcome(job({ status: 'done', completed: 4, total: 4 }));
    expect(notice?.tone).toBe('success');
    expect(notice?.title).toBe('Translated');
  });

  it('does not call an empty run a success', () => {
    const notice = describeOutcome(job({ status: 'done', completed: 0, total: 0 }));
    expect(notice?.tone).toBe('info');
    expect(notice?.title).toBe('Nothing to translate');
  });

  it('does not call a wholly failed run a success either', () => {
    const notice = describeOutcome(job({ status: 'done', completed: 0, total: 5 }));
    expect(notice?.tone).toBe('warning');
    expect(notice?.body).toContain('0 of 5');
  });

  it('reports a partial run with its counts', () => {
    expect(describeOutcome(job({ status: 'done', completed: 3, total: 5 }))?.body).toContain(
      '3 of 5',
    );
  });

  it('passes the server’s own words through when a run fails', () => {
    expect(describeOutcome(job({ status: 'failed', error: 'Quota exhausted.' }))?.body).toBe(
      'Quota exhausted.',
    );
    expect(describeOutcome(job({ status: 'failed' }))?.body).toBe(
      'The translation could not be completed.',
    );
  });
});
