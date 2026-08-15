import { useEffect, useRef } from 'react';
import { LOG_NEAR_BOTTOM_PX } from '@/constants';
import type { JobEvent } from '@/types';
import styles from './ActivityLog.module.css';

/**
 * The run, line by line.
 *
 * Two messages are rewritten rather than shown: the server puts its own
 * filesystem paths in the "saved" and "finished" text, which means nothing to a
 * reader and shouldn't be on screen.
 */
export function ActivityLog({ events }: { events: JobEvent[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;

    // Only follow along if the reader hasn't scrolled up to look at something.
    const nearBottom =
      scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < LOG_NEAR_BOTTOM_PX;
    if (nearBottom) endRef.current?.scrollIntoView({ block: 'end' });
  }, [events.length]);

  return (
    <div className={styles.scroller} ref={scrollerRef}>
      <ol className={styles.log} aria-live="polite" aria-label="Translation activity">
        {events.map((event, index) => (
          <li key={index} className={`${styles.line} ${styles[`level-${tone(event.level)}`]}`}>
            <span className={styles.level}>{label(event)}</span>
            <span className={styles.message}>{message(event)}</span>
          </li>
        ))}
      </ol>
      <div ref={endRef} />
    </div>
  );
}

function tone(level: string): 'error' | 'warning' | 'success' | 'muted' | 'normal' {
  switch (level) {
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
      return 'sending';
    case 'patch_done':
      return 'done';
    case 'attempt_failed':
      return 'retrying';
    case 'patch_failed':
      return 'failed';
    case 'saved':
      return 'saved';
    case 'finished':
      return 'finished';
    case 'rate_limit_wait':
    case 'rate_limit_done':
      return 'waiting';
    case 'started':
      return 'started';
    case 'stopping':
      return 'stopping';
    default:
      return event.level === 'error' ? 'error' : event.level === 'warning' ? 'warning' : 'note';
  }
}

function message(event: JobEvent): string {
  // These two carry server-side paths, so they get their own wording.
  if (event.event === 'saved') {
    return `Progress saved — ${event.completed ?? 0} of ${event.total ?? 0} requests done`;
  }
  if (event.event === 'finished') {
    if (typeof event.successful !== 'number') return 'The run ended.';
    if (event.successful === 0) return 'Nothing could be translated.';
    if (event.successful === event.total) return `All ${event.total} requests completed.`;
    return `Finished with ${event.successful} of ${event.total} requests translated.`;
  }
  return asRequests(event.message);
}

/**
 * The server calls a batch of chapters a "patch"; everywhere a reader can see,
 * this calls it a request — that is the unit the limits and budget are counted
 * in. The log would otherwise be the one place using the other word.
 */
function asRequests(text: string): string {
  return text
    .replace(/\bPatches\b/g, 'Requests')
    .replace(/\bPatch\b/g, 'Request')
    .replace(/\bpatches\b/g, 'requests')
    .replace(/\bpatch\b/g, 'request');
}
