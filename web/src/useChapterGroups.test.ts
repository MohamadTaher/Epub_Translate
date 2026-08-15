import { describe, expect, it } from 'vitest';
import type { Chapter } from '@/types';
import { plan } from '@/useChapterGroups';

const chapter = (id: string, patch: number | null, tokens: number): Chapter => ({
  id,
  title: id,
  file_name: `${id}.xhtml`,
  tokens,
  patch,
  skip_reason: patch === null ? 'Not selected' : null,
});

describe('plan', () => {
  it('groups consecutive chapters that share a request, in book order', () => {
    const { groups } = plan([chapter('a', 1, 100), chapter('b', 1, 100), chapter('c', 2, 100)]);
    expect(groups.map((g) => [g.patch, g.chapters.map((c) => c.id)])).toEqual([
      [1, ['a', 'b']],
      [2, ['c']],
    ]);
  });

  it('totals a request across the whole book, not per contiguous run', () => {
    // A skipped chapter in the middle splits request 1 into two runs. Either run
    // alone would understate what the request costs.
    const { groups, totals, members } = plan([
      chapter('a', 1, 100),
      chapter('skipped', null, 900),
      chapter('c', 1, 150),
    ]);

    expect(groups).toHaveLength(3);
    expect(totals.get(1)).toBe(250);
    expect(members.get(1)).toEqual(['a', 'c']);
  });

  it('marks only the first run of a request, so the header appears once', () => {
    const { groups } = plan([
      chapter('a', 1, 100),
      chapter('skipped', null, 900),
      chapter('c', 1, 150),
      chapter('d', 2, 100),
    ]);
    expect(groups.map((g) => g.firstRun)).toEqual([true, true, false, true]);
  });

  it('leaves skipped chapters out of the totals entirely', () => {
    const { totals, members } = plan([chapter('skipped', null, 900)]);
    expect(totals.size).toBe(0);
    expect(members.size).toBe(0);
  });

  it('keeps consecutive skipped chapters together as one group', () => {
    const { groups } = plan([chapter('x', null, 10), chapter('y', null, 10)]);
    expect(groups).toHaveLength(1);
    expect(groups[0].chapters).toHaveLength(2);
  });
});
