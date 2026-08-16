/**
 * Wire types, mirroring server/app.py and server/jobs.py exactly, and the few
 * shared shapes derived from them.
 *
 * The derived ones live here rather than in whichever module first needed them,
 * so a presentational component never has to import from the transport layer.
 */

export interface Status {
  configured: boolean;
  model: string;
  remaining_requests: number;
  daily_budget: number;
  max_upload_mb: number;
  /** Seconds until this visitor may start another translation; 0 when free. */
  cooldown_seconds: number;
  /** The server is already running as many translations as it allows. */
  busy: boolean;
  /** The server's pace, which is also its concurrency. Set in .env, not here. */
  requests_per_minute: number;
  languages: string[];
  detectable_languages: string[];
}

export type JobStatus = 'preparing' | 'ready' | 'running' | 'done' | 'failed' | 'cancelled';

export interface Chapter {
  id: string;
  title: string;
  file_name: string;
  tokens: number;
  /**
   * 1-based number of the patch that translates this chapter, matching the
   * `patch` field on progress events. Null when the chapter is skipped.
   *
   * A patch is a group of chapters sent together. It is called that everywhere,
   * including on screen; "request" is kept for the API call it becomes, which
   * is what the rate limit and the daily budget count. See CLAUDE.md.
   */
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
  /** Patches finished and planned — not chapters. */
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
  /** Unix seconds, stamped by the server as the event was published. */
  at?: number;
  total?: number;
  completed?: number;
  patch?: number;
  attempt?: number;
  successful?: number;
  seconds?: number;
  reason?: string;
  /**
   * On `patch_misaligned` and `patch_incomplete`: how many chapters went out,
   * and how many usable ones came back.
   */
  expected?: number;
  received?: number;
  /** On `glossary_learned`: terms this reply taught, and the size of the glossary now. */
  added?: number;
}

export type Glossary = Record<string, string>;

/* Derived ------------------------------------------------------------------ */

/**
 * Language is the only thing a visitor chooses. Pace and patch size are the
 * server owner's to set in .env, since it is their quota being spent.
 */
export interface UploadSettings {
  source_lang: string;
  target_lang: string;
}

export type PatchState = 'queued' | 'active' | 'done' | 'failed';

/** Where one patch has got to, as worked out from the event stream. */
export interface PatchProgress {
  state: PatchState;
  attempt: number;
}
