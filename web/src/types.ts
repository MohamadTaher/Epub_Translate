/** Wire types. These mirror server/app.py and server/jobs.py exactly. */

export interface Defaults {
  max_tokens_per_patch: number;
  max_concurrent: number;
  max_requests_per_minute: number;
}

export interface Status {
  configured: boolean;
  model: string;
  remaining_requests: number;
  daily_budget: number;
  max_patches_per_job: number;
  max_upload_mb: number;
  /** Seconds until this visitor may start another translation; 0 when free. */
  cooldown_seconds: number;
  /** The server is already running as many translations as it allows. */
  busy: boolean;
  defaults: Defaults;
  languages: string[];
  detectable_languages: string[];
}

export type JobStatus = 'preparing' | 'ready' | 'running' | 'done' | 'failed' | 'cancelled';

export interface Chapter {
  id: string;
  title: string;
  file_name: string;
  tokens: number;
  /** 1-based request number, matching the `patch` field on progress events. Null when skipped. */
  patch: number | null;
  /** Why this chapter is not being translated, or null when it is. */
  skip_reason: string | null;
}

export interface BookInfo {
  title: string | null;
  author: string | null;
  has_cover: boolean;
  filename: string;
}

export interface JobStats {
  book: BookInfo;
  chapter_count: number;
  patch_count: number;
  total_tokens: number;
  /** A language name, or the literal "auto" when detection found no non-Latin script. */
  source_language: string;
  target_language: string;
  chapters: Chapter[];
}

export interface JobSnapshot {
  id: string;
  status: JobStatus;
  error: string | null;
  /** Requests finished and planned — not chapters. */
  completed: number;
  total: number;
  stats: JobStats | null;
}

/**
 * A progress event.
 *
 * Only `level` and `message` are guaranteed; everything else depends on which
 * event it is, and `finished` has two shapes — the one the server emits when a
 * run crashes carries no counts. Treat every optional field as possibly absent.
 */
export interface JobEvent {
  level: string;
  message: string;
  event?: string;
  total?: number;
  completed?: number;
  patch?: number;
  attempt?: number;
  successful?: number;
  seconds?: number;
  reason?: string;
}

export type Glossary = Record<string, string>;
