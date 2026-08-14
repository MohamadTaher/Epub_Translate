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
 */
export function estimateSeconds(
  remainingRequests: number,
  requestsPerMinute: number,
  observedDurations: number[],
  concurrency: number,
): number {
  if (remainingRequests <= 0) return 0;

  const rateLimited = (remainingRequests / Math.max(1, requestsPerMinute)) * 60;

  if (observedDurations.length < 2) return rateLimited;

  const mean = observedDurations.reduce((a, b) => a + b, 0) / observedDurations.length;
  const byPace = (remainingRequests / Math.max(1, concurrency)) * mean;

  // Whichever constraint binds harder is the one that will actually be felt.
  return Math.max(rateLimited, byPace);
}

/** Language names arrive lowercase from the server; titles read better. */
export function languageLabel(name: string): string {
  if (!name || name === 'auto') return 'Not detected';
  return name.charAt(0).toUpperCase() + name.slice(1);
}

export function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}
