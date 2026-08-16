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
import { PatchTrack } from './PatchTrack';
import {
  Button,
  Disclosure,
  DownloadIcon,
  Eyebrow,
  LinkButton,
  NoticeCallout,
  RefreshIcon,
  StopIcon,
} from './ui';
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

  const narrow = useMediaQuery(NARROW);
  const [logOpen, setLogOpen] = useState(!narrow);
  useEffect(() => setLogOpen(!narrow), [narrow]);

  const stats = job.stats;
  // The snapshot's counts only move when the run ends — mid-run they are the
  // ones `/start` returned. The stream carries them as they change, and replays
  // them in full on reconnect, so the higher of the two is the current one.
  const completed = Math.max(job.completed, stream.completed);
  const total = Math.max(job.total, stream.total);
  const remaining = Math.max(0, total - completed);
  const percent = total > 0 ? Math.round((completed / total) * 100) : 0;

  const canDownload = stream.hasOutput || completed > 0;
  const outcome = describeOutcome(job);
  const termCount = Object.keys(glossary.terms).length;

  // The countdown from the last rate-limit wait, if it hasn't run out.
  const waitLeft = stream.rateLimit
    ? Math.max(0, Math.round(stream.rateLimit.seconds - (now - stream.rateLimit.startedAt) / 1000))
    : 0;

  // Nothing is emitted while a worker sleeps on the rate limiter.
  const quietFor = running ? (now - stream.lastEventAt) / 1000 : 0;
  const stalled = running && !stream.rateLimit && quietFor > STALL_SECONDS;

  return (
    <section className={styles.panel}>
      <div className={styles.sticky}>
        <div className={styles.stickyInner}>
          {stats && (
            <div className={styles.bookSummary}>
              <BookHeader
                jobId={job.id}
                book={stats.book}
                sourceLanguage={stats.source_language}
                targetLanguage={stats.target_language}
                size="compact"
              />
            </div>
          )}

          <div className={styles.progress}>
            <div className={styles.progressLine}>
              <span className="tabular">
                <strong>{completed}</strong> of {total} patches ({percent}%)
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
            <PatchTrack total={total} patches={stream.patches} />
          </div>

          <div className={styles.controls}>
            {canDownload && (
              <LinkButton
                href={downloadUrl(job.id)}
                download
                variant={running ? 'secondary' : 'primary'}
                icon={<DownloadIcon />}
              >
                Download{running ? ' partial EPUB' : ' EPUB'}
              </LinkButton>
            )}
            {running && (
              <Button
                variant="danger"
                onClick={onCancel}
                disabled={cancelling}
                icon={<StopIcon />}
              >
                {cancelling ? 'Stopping…' : 'Stop'}
              </Button>
            )}
            {!running && (
              <Button
                variant="secondary"
                onClick={onStartOver}
                icon={<RefreshIcon />}
              >
                Translate another
              </Button>
            )}
          </div>
        </div>
      </div>

      {waitLeft > 0 && (
        <div className={styles.waiting} role="status">
          <span aria-hidden="true" className={styles.waitingGlyph}>
            ⏳
          </span>
          <span>
            Pacing delay for {stream.rateLimit?.reason ?? 'rate limit'} — resuming in{' '}
            <strong className="tabular">{waitLeft}s</strong>
          </span>
        </div>
      )}

      {stalled && (
        <div className={styles.waiting} role="status">
          <span aria-hidden="true" className={styles.waitingGlyph}>
            ⏳
          </span>
          <span>Still working. Spacing requests out according to API limits.</span>
        </div>
      )}

      <NoticeCallout notice={outcome} />

      <div className={styles.columns}>
        <div className={styles.column}>
          <div className={styles.columnHead}>
            <Eyebrow>Chapters Progress</Eyebrow>
            <span className={styles.columnMeta}>grouped by patch</span>
          </div>
          {stats && <ChapterList chapters={stats.chapters} patches={stream.patches} />}
        </div>

        <div className={styles.column}>
          <div className={styles.columnHead}>
            <Eyebrow>Translation Console</Eyebrow>
            <span className={styles.columnMeta}>every patch, as it happens</span>
          </div>
          <details open={logOpen} onToggle={(e) => setLogOpen(e.currentTarget.open)}>
            <summary className={styles.logSummary}>Translation Console</summary>
            <ActivityLog events={stream.events} live={running} />
          </details>
        </div>
      </div>

      <Disclosure
        summary="Translation Glossary & Terms"
        badge={termCount > 0 ? `${termCount} term${termCount === 1 ? '' : 's'}` : undefined}
      >
        <GlossaryEditor glossary={glossary} running={running} />
      </Disclosure>
    </section>
  );
}

