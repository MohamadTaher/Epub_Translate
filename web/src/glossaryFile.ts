import type { Glossary } from '@/types';

/**
 * Bringing in a glossary from somewhere else — the file the CLI writes, or one
 * saved from an earlier book in the same series.
 *
 * The reading and the merging live here rather than in the editor because they
 * are the part worth being sure about: a book of names is tedious to retype,
 * and a bad merge silently renames a character.
 */

/**
 * Read a glossary file's text.
 *
 * Every failure gets a sentence a reader can act on. The server refuses the
 * same shapes for the same reasons, but it never sees the file — this is where
 * someone finds out they picked the wrong one.
 */
export function parseGlossaryFile(text: string): Glossary {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error('That file isn’t JSON — a glossary looks like {"道玄": "Dao Xuan"}.');
  }

  if (data === null || typeof data !== 'object' || Array.isArray(data)) {
    throw new Error('A glossary is a JSON object of "term": "translation" pairs.');
  }

  const terms: Glossary = {};
  for (const [from, to] of Object.entries(data)) {
    if (typeof to !== 'string') {
      throw new Error(`The translation for “${from}” isn’t text.`);
    }
    const term = from.trim();
    if (term) terms[term] = to.trim();
  }

  return terms;
}

export interface MergeReport {
  terms: Glossary;
  /** Terms the file brought that weren't here. */
  added: number;
  /** Terms this book had already decided, left as they were. */
  kept: number;
  /**
   * Of those kept, the ones the file translated differently.
   *
   * Counted separately because it is the only part of an import that is worth
   * arguing with: the reader handed over a translation and did not get it, and
   * a silent count of "already here" would not tell them so.
   */
  differed: number;
}

/**
 * Merge an imported glossary into the one this book already has.
 *
 * What is already here wins, which is the same rule the server applies to terms
 * the model reports: a book that has settled on a name keeps it. Compared
 * case-insensitively, so an imported "IVAN" can't sit beside "Ivan" giving the
 * model two answers. A term whose translation is blank counts as undecided and
 * is filled in by the import.
 */
export function mergeGlossary(current: Glossary, incoming: Glossary): MergeReport {
  const terms = { ...current };
  const settled = new Map(
    Object.entries(current)
      .filter(([, to]) => to.trim())
      .map(([from, to]) => [from.toLowerCase(), to] as const),
  );

  let added = 0;
  let kept = 0;
  let differed = 0;

  for (const [from, to] of Object.entries(incoming)) {
    const settledAs = settled.get(from.toLowerCase());
    if (settledAs !== undefined) {
      kept += 1;
      // A blank in the file is no translation offered, not a disagreement.
      if (to.trim() && to !== settledAs) differed += 1;
      continue;
    }
    terms[from] = to;
    if (to.trim()) settled.set(from.toLowerCase(), to);
    added += 1;
  }

  return { terms, added, kept, differed };
}

/**
 * Everything this book knows right now: the list on screen, plus whatever the
 * run has learned since that list was last in step with the server.
 *
 * A term the reader deleted stays deleted — it was in `saved`, so it is known
 * to have been seen and dropped on purpose, rather than being something new the
 * screen has never heard of.
 */
export function withLearned(shown: Glossary, saved: Glossary, onServer: Glossary): Glossary {
  const learned = Object.fromEntries(
    Object.entries(onServer).filter(([term]) => !(term in saved)),
  );
  return { ...learned, ...shown };
}

/**
 * What importing a file would do, worked out in one place.
 *
 * The merge happens against what the book knows *including* terms the run has
 * learned, not just the rows on screen. Otherwise an import would quietly
 * overwrite a name the book settled on mid-run, and report it as new.
 */
export function planImport(
  fileText: string,
  shown: Glossary,
  saved: Glossary,
  onServer: Glossary,
): MergeReport {
  return mergeGlossary(withLearned(shown, saved, onServer), parseGlossaryFile(fileText));
}

/** Named so it can't shadow the `terms` maps this module is full of. */
const termCount = (n: number) => `${n} term${n === 1 ? '' : 's'}`;

/**
 * What the reader is told once the file has been merged in.
 *
 * Three things can happen to a term in the file and all three get said: it came
 * in, it was already here, or it was already here and the file disagreed. The
 * counts are what explain a file of fifty terms importing three.
 */
export function describeImport({ added, kept, differed }: MergeReport): string {
  if (added === 0 && kept === 0) return 'That file had no terms in it.';

  const opening =
    added > 0
      ? `Imported ${termCount(added)}.`
      : `Nothing new — ${kept === 1 ? 'that term was' : `all ${termCount(kept)} were`} already here.`;

  const sentences = [opening];

  if (added > 0 && kept > 0) {
    sentences.push(`${kept} ${kept === 1 ? 'was' : 'were'} already here.`);
  }

  if (differed > 0) {
    const subject = differed === 1 ? 'One of those was' : `${differed} of those were`;
    sentences.push(`${subject} translated differently in the file, and this book’s own was kept.`);
  }

  return sentences.join(' ');
}
