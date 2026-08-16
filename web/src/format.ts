/** Small display helpers, kept together so wording stays consistent. */

export function tokens(count: number): string {
  if (count < 1000) return String(count);
  if (count < 10_000) return `${(count / 1000).toFixed(1)}k`;
  return `${Math.round(count / 1000)}k`;
}

export function fileSize(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return mb < 0.1 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${mb.toFixed(1)} MB`;
}

/**
 * "about 4 minutes", "under a minute" — never a false precision like "3m 42s".
 *
 * The hedge is part of the phrase rather than something callers prepend, so
 * "about under a minute" can't happen.
 */
export function duration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'a moment';
  if (seconds < 60) return 'under a minute';

  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `about ${minutes} minute${minutes === 1 ? '' : 's'}`;

  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `about ${hours}h ${rest}m` : `about ${hours} hour${hours === 1 ? '' : 's'}`;
}

/**
 * How long a run should take.
 *
 * Throughput is bounded by the per-minute request limit rather than by how fast
 * the model answers, so before anything has run the rate limit is the estimate.
 * Once requests start landing, their measured pace is better.
 *
 * The server runs as many requests at a time as it allows per minute, so
 * `requestsPerMinute` stands in for concurrency too.
 */
export function estimateSeconds(
  remainingPatches: number,
  requestsPerMinute: number,
  observedDurations: number[],
): number {
  if (remainingPatches <= 0) return 0;

  const pace = Math.max(1, requestsPerMinute);
  const rateLimited = (remainingPatches / pace) * 60;

  if (observedDurations.length < 2) return rateLimited;

  const mean = observedDurations.reduce((a, b) => a + b, 0) / observedDurations.length;
  const byPace = (remainingPatches / pace) * mean;

  // Whichever constraint binds harder is the one that will actually be felt.
  return Math.max(rateLimited, byPace);
}

/**
 * A log line's wall-clock time, "14:03:22".
 *
 * Pinned to en-GB rather than the reader's locale: a log column wants a fixed
 * width and a 24-hour clock, not "2:03:22 PM".
 */
export function clockTime(epochSeconds: number | undefined): string {
  if (typeof epochSeconds !== 'number') return '';
  return new Date(epochSeconds * 1000).toLocaleTimeString('en-GB', { hour12: false });
}

/** Language names arrive lowercase from the server; titles read better. */
export function languageLabel(name: string): string {
  if (!name || name === 'auto') return 'Not detected';
  return name.charAt(0).toUpperCase() + name.slice(1);
}
