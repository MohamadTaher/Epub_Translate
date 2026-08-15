import { useMemo } from 'react';
import type { Chapter } from '@/types';

export interface ChapterGroup {
  patch: number | null;
  chapters: Chapter[];
  /**
   * False for a run of chapters belonging to a request already introduced
   * further up the book, so the header appears once and later runs say so.
   */
  firstRun: boolean;
}

export interface ChapterPlan {
  /** Runs of consecutive chapters that share a request, in book order. */
  groups: ChapterGroup[];
  /** Each request's real token cost. */
  totals: Map<number, number>;
  /** Each request's chapter ids, for selecting all of them at once. */
  members: Map<number, string[]>;
}

/**
 * The chapter list as it is drawn: grouped by request, plus each request's
 * totals.
 *
 * The totals are gathered across the whole book rather than per run, because a
 * request's chapters are often split up by skipped ones in between — showing a
 * run's own tokens would understate what the request actually costs. `firstRun`
 * is worked out here for the same reason: it depends on the whole book so far,
 * not on the group in hand.
 */
export function plan(chapters: Chapter[]): ChapterPlan {
  const groups: ChapterGroup[] = [];
  const totals = new Map<number, number>();
  const members = new Map<number, string[]>();
  const introduced = new Set<number>();

  for (const chapter of chapters) {
    const last = groups[groups.length - 1];
    if (last && last.patch === chapter.patch) {
      last.chapters.push(chapter);
    } else {
      groups.push({
        patch: chapter.patch,
        chapters: [chapter],
        firstRun: chapter.patch === null || !introduced.has(chapter.patch),
      });
    }

    if (chapter.patch !== null) {
      introduced.add(chapter.patch);
      totals.set(chapter.patch, (totals.get(chapter.patch) ?? 0) + chapter.tokens);
      members.set(chapter.patch, [...(members.get(chapter.patch) ?? []), chapter.id]);
    }
  }

  return { groups, totals, members };
}

export function useChapterGroups(chapters: Chapter[]): ChapterPlan {
  return useMemo(() => plan(chapters), [chapters]);
}
