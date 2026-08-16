/**
 * Everything the app says when it has to explain itself.
 *
 * Both of these answer the same question in different phases — "is there
 * something you need to know before going on?" — and both feed a Callout, so
 * they share a shape and live together. The copy is the interface here; keeping
 * it in one file is what stops two panels drifting apart in wording.
 */

import type { JobSnapshot, Status } from '@/types';

export type Tone = 'info' | 'warning' | 'error' | 'success';

export interface Notice {
  tone: Tone;
  title: string;
  body: string;
}

/** Why a book can't be uploaded right now, or null when it can. */
export function describeBlock(status: Status | null): Notice | null {
  if (!status) return null;

  if (!status.configured) {
    return {
      tone: 'error',
      title: 'This server has no API key',
      body: 'Translation needs a Gemini API key in the server’s .env file. Nothing can be uploaded until one is set.',
    };
  }

  if (status.cooldown_seconds > 0) {
    const minutes = Math.ceil(status.cooldown_seconds / 60);
    return {
      tone: 'warning',
      title: 'You started a translation recently',
      body: `You can start another in about ${minutes} minute${minutes === 1 ? '' : 's'}. This limit keeps one visitor from spending the whole day’s budget.`,
    };
  }

  if (status.busy) {
    return {
      tone: 'warning',
      title: 'The server is busy',
      body: 'Other translations are running right now. Try again in a few minutes.',
    };
  }

  if (status.remaining_requests <= 0) {
    return {
      tone: 'warning',
      title: 'Today’s budget is spent',
      body: 'The daily request budget resets at midnight UTC.',
    };
  }

  return null;
}

/** How a run ended, or null while it is still going. */
export function describeOutcome(job: JobSnapshot): Notice | null {
  switch (job.status) {
    case 'done': {
      // A run with nothing in it is not a success story; say what happened.
      if (job.total === 0) {
        return {
          tone: 'info',
          title: 'Nothing to translate',
          body: 'This run had no patches to send. Nothing in the book was left to translate.',
        };
      }

      return job.completed < job.total
        ? {
            tone: 'warning',
            title: 'Finished, but not all of it',
            body: `${job.completed} of ${job.total} patches were translated. The rest ran out of retries. The download holds everything that succeeded, with the untranslated chapters left in their original language.`,
          }
        : {
            tone: 'success',
            title: 'Translated',
            body: 'Every patch completed. The book is ready to download.',
          };
    }

    case 'cancelled':
      return {
        tone: 'info',
        title: 'Stopped',
        body: `You stopped this after ${job.completed} of ${job.total} patches. Everything finished up to that point is saved and downloadable.`,
      };

    case 'failed':
      return {
        tone: 'error',
        title: 'This run failed',
        body: job.error ?? 'The translation could not be completed.',
      };

    default:
      return null;
  }
}
