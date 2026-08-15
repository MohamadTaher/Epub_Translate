/** Numbers that were spelled out in more than one place, or that mirror the server. */

import type { JobStatus } from '@/types';

/** A job in one of these is finished with; the stream will send nothing more. */
export const TERMINAL: JobStatus[] = ['done', 'failed', 'cancelled'];

/** Until /api/status answers, the server's own default is the best guess. */
export const ASSUMED_REQUESTS_PER_MINUTE = 4;

/** Likewise for the upload ceiling, which the dropzone has to state up front. */
export const DEFAULT_MAX_UPLOAD_MB = 25;

/** Mirrors `max_retries` in epub_translate/translator.py — shown in retry labels. */
export const MAX_ATTEMPTS = 10;

/** How long a silence has to run before it is worth saying the run is not stuck. */
export const STALL_SECONDS = 25;

/** Long enough that ticking a run of chapters costs one request, not one per click. */
export const PREVIEW_DEBOUNCE_MS = 300;

/** Within this much of the bottom, the activity log keeps following along. */
export const LOG_NEAR_BOTTOM_PX = 80;
