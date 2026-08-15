import { useEffect, useState } from 'react';
import { downloadUrl } from '../api';
import { duration, estimateSeconds } from '../format';
import type { JobSnapshot } from '../types';
import type { StreamState } from '../useJobStream';
import { ActivityLog } from './ActivityLog';
import { BookHeader } from './BookHeader';
import { ChapterList } from './ChapterList';
import { GlossaryEditor } from './GlossaryEditor';
import { Button, Callout, Disclosure, Eyebrow, LinkButton, Meter } from './ui';
import styles from './RunPanel.module.css';

/** Ticks once a second, but only while something is actually counting down. */
function useSecondTicker(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [active]);
  return now;
}

interface Props {
  job: JobSnapshot;
  stream: StreamState;
  onCancel: () => void;
  onStartOver: () => void;
  cancelling: boolean;
  requestsPerMinute: number;
}

export function RunPanel({
  job,
  stream,
  onCancel,
  onStartOver,
  cancelling,
  requestsPerMinute,
}: Props) {
  const running = job.status === 'running';
  const now = useSecondTicker(running);

  const stats = job.stats;
  const completed = job.completed;
  const total = job.total;
  const remaining = Math.max(0, total - completed);

  const canDownload = stream.hasOutput || completed > 0;
  const outcome = describeOutcome(job);

  // The countdown from the last rate-limit wait, if it hasn't run out.
  const waitLeft = stream.rateLimit
    ? Math.max(0, Math.round(stream.rateLimit.seconds - (now - stream.rateLimit.startedAt) / 1000))
    : 0;

  // Nothing is emitted while a worker sleeps on the rate limiter, so a long
  // silence is itself worth reporting rather than leaving the bar looking stuck.
  const quietFor = running ? (now - stream.lastEventAt) / 1000 : 0;
  const stalled = running && !stream.rateLimit && quietFor > 25;

  return (
    <section className={styles.panel}>
      <div className={styles.sticky}>
        <div className={styles.stickyInner}>
          {stats && (
            <BookHeader
              jobId={job.id}
              book={stats.book}
              sourceLanguage={stats.source_language}
              targetLanguage={stats.target_language}
              size="compact"
            />
          )}

          <div className={styles.progress}>
            <div className={styles.progressLine}>
              <span className="tabular">
                <strong>{completed}</strong> of {total} requests
              </span>
              {running && (
                <span className={styles.eta}>
                  {remaining > 0
                    ? `${duration(
                        estimateSeconds(remaining, requestsPerMinute, stream.durations),
                      )} left`
                    : 'finishing up'}
                </span>
              )}
            </div>
            <Meter value={completed} max={Math.max(1, total)} />
          </div>

          <div className={styles.controls}>
            {canDownload && (
              <LinkButton href={downloadUrl(job.id)} download variant={running ? 'secondary' : 'primary'}>
                Download{running ? ' what’s done' : ''}
              </LinkButton>
            )}
            {running && (
              <Button variant="danger" onClick={onCancel} disabled={cancelling}>
                {cancelling ? 'Stopping…' : 'Stop'}
              </Button>
            )}
            {!running && (
              <Button variant="secondary" onClick={onStartOver}>
                Translate another
              </Button>
            )}
          </div>
        </div>
      </div>

      {waitLeft > 0 && (
        <p className={styles.waiting} role="status">
          <span aria-hidden="true" className={styles.waitingGlyph}>
            ◔
          </span>
          Waiting on the {stream.rateLimit?.reason ?? 'rate limit'} — <span className="tabular">{waitLeft}s</span>
        </p>
      )}

      {stalled && (
        <p className={styles.waiting} role="status">
          <span aria-hidden="true" className={styles.waitingGlyph}>
            ◔
          </span>
          Still working. Long gaps are normal while requests are spaced out.
        </p>
      )}

      {outcome && (
        <Callout tone={outcome.tone} title={outcome.title}>
          {outcome.body}
        </Callout>
      )}

      <div className={styles.columns}>
        <div className={styles.column}>
          <div className={styles.columnHead}>
            <Eyebrow>Chapters</Eyebrow>
            <span className={styles.columnMeta}>grouped by request</span>
          </div>
          {stats && <ChapterList chapters={stats.chapters} requests={stream.requests} />}
        </div>

        <div className={styles.column}>
          <div className={styles.columnHead}>
            <Eyebrow>Activity</Eyebrow>
          </div>
          <div className={styles.logWrap}>
            <ActivityLog events={stream.events} />
          </div>
          <details className={styles.logCollapsed}>
            <summary>Activity ({stream.events.length} lines)</summary>
            <ActivityLog events={stream.events} />
          </details>
        </div>
      </div>

      {running && (
        <Disclosure summary="Glossary">
          <GlossaryEditor jobId={job.id} running />
        </Disclosure>
      )}
    </section>
  );
}

function describeOutcome(job: JobSnapshot) {
  const partial = job.completed > 0 && job.completed < job.total;

  switch (job.status) {
    case 'done':
      return partial
        ? {
            tone: 'warning' as const,
            title: 'Finished, but not all of it',
            body: `${job.completed} of ${job.total} requests were translated. The rest ran out of retries. The download holds everything that succeeded, with the untranslated chapters left in their original language.`,
          }
        : {
            tone: 'success' as const,
            title: 'Translated',
            body: 'Every request completed. The book is ready to download.',
          };

    case 'cancelled':
      return {
        tone: 'info' as const,
        title: 'Stopped',
        body: `You stopped this after ${job.completed} of ${job.total} requests. Everything finished up to that point is saved and downloadable.`,
      };

    case 'failed':
      return {
        tone: 'error' as const,
        title: 'This run failed',
        body: job.error ?? 'The translation could not be completed.',
      };

    default:
      return null;
  }
}
