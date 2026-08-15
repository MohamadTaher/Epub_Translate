import { useEffect, useState } from 'react';
import { downloadUrl } from '@/api';
import { STALL_SECONDS } from '@/constants';
import { duration, estimateSeconds } from '@/format';
import { describeOutcome } from '@/notices';
import type { JobSnapshot } from '@/types';
import type { GlossaryState } from '@/useGlossary';
import type { StreamState } from '@/useJobStream';
import { ActivityLog } from './ActivityLog';
import { BookHeader } from './BookHeader';
import { ChapterList } from './ChapterList';
import { GlossaryEditor } from './GlossaryEditor';
import { Button, Disclosure, Eyebrow, LinkButton, Meter, NoticeCallout } from './ui';
import styles from './RunPanel.module.css';

/** The same breakpoint the two-column layout uses in RunPanel.module.css. */
const NARROW = '(max-width: 900px)';

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

/**
 * Whether a media query matches, so the log can be one element that collapses
 * on a narrow screen rather than two copies of itself.
 */
function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const update = () => setMatches(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, [query]);
  return matches;
}

interface Props {
  job: JobSnapshot;
  stream: StreamState;
  glossary: GlossaryState;
  onCancel: () => void;
  onStartOver: () => void;
  cancelling: boolean;
  requestsPerMinute: number;
}

export function RunPanel({
  job,
  stream,
  glossary,
  onCancel,
  onStartOver,
  cancelling,
  requestsPerMinute,
}: Props) {
  const running = job.status === 'running';
  const now = useSecondTicker(running);

  // The log is one <details>: open and un-collapsible on a wide screen, closed
  // and openable on a narrow one. Held in state rather than left to the DOM
  // because this panel re-renders on every event, and an uncontrolled <details>
  // would be argued with a few times a second.
  const narrow = useMediaQuery(NARROW);
  const [logOpen, setLogOpen] = useState(!narrow);
  useEffect(() => setLogOpen(!narrow), [narrow]);

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
  const stalled = running && !stream.rateLimit && quietFor > STALL_SECONDS;

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

      <NoticeCallout notice={outcome} />

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
          {/* One log, not two. Below the breakpoint its summary is the only way
              in; above it, the summary is hidden and this stays open. Rendering
              it twice put every line in the DOM twice and left two auto-scroll
              effects fighting over the same events. */}
          <details open={logOpen} onToggle={(e) => setLogOpen(e.currentTarget.open)}>
            <summary className={styles.logSummary}>Activity ({stream.events.length} lines)</summary>
            <ActivityLog events={stream.events} />
          </details>
        </div>
      </div>

      {/* Shown after the run as well as during it: by then the glossary holds
          every name the book settled on, which is the thing worth exporting and
          carrying into the next one. */}
      <Disclosure summary="Glossary">
        <GlossaryEditor glossary={glossary} running={running} />
      </Disclosure>
    </section>
  );
}

