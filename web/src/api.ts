import type { Glossary, JobSnapshot, JobStats, Status } from './types';

/**
 * A failed request.
 *
 * The server answers every failure with `{"detail": "<prose>"}` and no error
 * code, so the status is the only thing worth branching on — the message is
 * already written for a reader and can be shown as it is.
 */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) return response.json() as Promise<T>;

  let detail = `Something went wrong (${response.status}).`;
  try {
    const body = await response.json();
    if (typeof body?.detail === 'string') {
      detail = body.detail;
    } else if (Array.isArray(body?.detail)) {
      detail = body.detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join('; ') || detail;
    }
  } catch {
    // A non-JSON error body tells us nothing useful; the default stands.
  }
  throw new ApiError(response.status, detail);
}

const json = (method: 'POST' | 'PUT', body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export function getStatus(): Promise<Status> {
  return fetch('/api/status').then(unwrap<Status>);
}

interface UploadOptions {
  file: File;
  source_lang: string;
  target_lang: string;
}

export function createJob(options: UploadOptions): Promise<JobSnapshot> {
  const form = new FormData();
  form.append('file', options.file);
  form.append('source_lang', options.source_lang);
  form.append('target_lang', options.target_lang);

  return fetch('/api/jobs', { method: 'POST', body: form }).then(unwrap<JobSnapshot>);
}

export function getJob(jobId: string): Promise<JobSnapshot> {
  return fetch(`/api/jobs/${jobId}`).then(unwrap<JobSnapshot>);
}

/**
 * Re-cost a selection without committing to it.
 *
 * `chapterIds` of null means "the server's own default", which leaves out
 * chapters that already look translated. An explicit list means exactly those.
 */
export function previewJob(
  jobId: string,
  chapterIds: string[] | null,
  signal?: AbortSignal,
): Promise<JobStats> {
  return fetch(`/api/jobs/${jobId}/preview`, {
    ...json('POST', { chapter_ids: chapterIds }),
    signal,
  }).then(unwrap<JobStats>);
}

export function startJob(jobId: string, chapterIds: string[] | null): Promise<JobSnapshot> {
  return fetch(`/api/jobs/${jobId}/start`, json('POST', { chapter_ids: chapterIds })).then(
    unwrap<JobSnapshot>,
  );
}

export function cancelJob(jobId: string): Promise<JobSnapshot> {
  return fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' }).then(unwrap<JobSnapshot>);
}

export function getGlossary(jobId: string): Promise<{ terms: Glossary }> {
  return fetch(`/api/jobs/${jobId}/glossary`).then(unwrap<{ terms: Glossary }>);
}

/** Replaces the whole glossary — the server has no way to change one term. */
export function putGlossary(jobId: string, terms: Glossary): Promise<{ terms: Glossary }> {
  return fetch(`/api/jobs/${jobId}/glossary`, json('PUT', { terms })).then(
    unwrap<{ terms: Glossary }>,
  );
}

export const downloadUrl = (jobId: string) => `/api/jobs/${jobId}/download`;
export const coverUrl = (jobId: string) => `/api/jobs/${jobId}/cover`;
/** Opened by EventSource rather than fetch, but it is still this module's URL. */
export const eventsUrl = (jobId: string) => `/api/jobs/${jobId}/events`;
