import { useEffect, useRef } from 'react';
import { LOG_NEAR_BOTTOM_PX } from '@/constants';
import { clockTime } from '@/format';
import type { JobEvent } from '@/types';
import styles from './ActivityLog.module.css';

/**
 * The run, line by line in a developer-grade terminal window.
 */
export function ActivityLog({ events, live }: { events: JobEvent[]; live: boolean }) {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const lines = events.filter(worthShowing);

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;

    const nearBottom =
      scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < LOG_NEAR_BOTTOM_PX;
    // Not `scrollIntoView`: that scrolls every scrollable ancestor, so each new
    // line dragged the whole page down to keep the log's end on screen.
    if (nearBottom) scroller.scrollTop = scroller.scrollHeight;
  }, [lines.length]);

  return (
    <div className={styles.terminal}>
      <div className={styles.terminalHeader}>
        <div className={styles.windowDots} aria-hidden="true">
          <span className={styles.dotRed} />
          <span className={styles.dotYellow} />
          <span className={styles.dotGreen} />
        </div>
        <span className={styles.lineCount}>
          {lines.length} line{lines.length === 1 ? '' : 's'}
        </span>
        {live ? (
          <span className={styles.liveIndicator}>
            <span className={styles.livePulse} aria-hidden="true" />
            STREAM
          </span>
        ) : (
          <span className={styles.idleIndicator}>ENDED</span>
        )}
      </div>

      <div className={styles.scroller} ref={scrollerRef}>
        <ol className={styles.log} aria-label="Translation activity">
          {lines.length === 0 ? (
            <li className={styles.emptyState}>Waiting for translation activity…</li>
          ) : (
            lines.map((event, index) => {
              const eventTone = tone(event);
              return (
                <li key={index} className={`${styles.line} ${styles[`level-${eventTone}`]}`}>
                  <span className={styles.time}>{clockTime(event.at)}</span>
                  <span className={`${styles.levelBadge} ${styles[`badge-${eventTone}`]}`}>
                    {label(event)}
                  </span>
                  <span className={styles.message}>{message(event)}</span>
                </li>
              );
            })
          )}
        </ol>
      </div>
    </div>
  );
}

/**
 * The book is re-saved after every patch, so a `saved` event lands beside every
 * `patch_done` saying the same thing in other words — and the meter above says
 * it a third time. Dropping it here doesn't affect the download button, which
 * reads `saved` off the stream state rather than off what is drawn.
 */
function worthShowing(event: JobEvent): boolean {
  return event.event !== 'saved';
}

function tone(event: JobEvent): 'error' | 'warning' | 'success' | 'muted' | 'normal' {
  // A resumed run is reported at the level of the wait that caused it; amber
  // for the news that the waiting is over reads as a fresh problem.
  if (event.event === 'rate_limit_done') return 'muted';

  switch (event.level) {
    case 'error':
      return 'error';
    case 'warning':
    case 'rate_limit':
    case 'interrupt':
      return 'warning';
    case 'success':
      return 'success';
    case 'system':
    case 'concurrency':
      return 'muted';
    default:
      return 'normal';
  }
}

function label(event: JobEvent): string {
  switch (event.event) {
    case 'patch_start':
      return 'SENDING';
    case 'patch_done':
      return 'DONE';
    case 'attempt_failed':
      return 'RETRY';
    case 'patch_misaligned':
      return 'MISMATCH';
    case 'patch_incomplete':
      return 'EMPTY';
    case 'patch_failed':
      return 'FAILED';
    case 'glossary_learned':
      return 'GLOSSARY';
    case 'finished':
      return 'FINISH';
    case 'rate_limit_wait':
      return 'WAIT';
    case 'rate_limit_done':
      return 'RESUME';
    case 'started':
      return 'START';
    case 'stopping':
      return 'STOPPING';
    default:
      return (event.level === 'error' ? 'ERROR' : event.level === 'warning' ? 'WARN' : 'NOTE').toUpperCase();
  }
}

/**
 * The server's own wording, except for `finished`, which is one line summing up
 * the whole run and reads better said in full.
 *
 * There used to be a rewriting pass here, turning the server's "patch" into
 * "request" with four regexes — it silently missed any wording the server
 * changed. Both sides say "patch" now, so a message arrives ready to show.
 */
function message(event: JobEvent): string {
  if (event.event !== 'finished' || typeof event.successful !== 'number') return event.message;

  if (event.successful === 0) return 'Nothing could be translated.';
  if (event.successful === event.total) return `All ${event.total} patches completed successfully.`;
  return `Finished with ${event.successful} of ${event.total} patches translated.`;
}
