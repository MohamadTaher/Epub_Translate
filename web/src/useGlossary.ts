import { useCallback, useEffect, useRef, useState } from 'react';
import { getGlossary, putGlossary } from '@/api';
import { describeImport, planImport, withLearned } from '@/glossaryFile';
import type { Glossary } from '@/types';

/**
 * A row being edited.
 *
 * The id exists only to key the inputs: keying by array index re-associates
 * every row below one that is removed, so the value of row 4 would appear to
 * jump into row 3's box.
 */
export interface GlossaryRow {
  id: number;
  from: string;
  to: string;
}

let nextRowId = 0;
const makeRow = (from: string, to: string): GlossaryRow => ({ id: ++nextRowId, from, to });
const toRows = (terms: Glossary): GlossaryRow[] =>
  Object.entries(terms).map(([from, to]) => makeRow(from, to));

/**
 * The job's glossary, held above the phase machine.
 *
 * The editor is rendered from both the plan panel and the run panel, in
 * different subtrees, so starting a run unmounts one and mounts the other. Held
 * here instead, unsaved edits survive that — they used to be discarded silently.
 *
 * The glossary is no longer only what the reader typed: a run adds the terms
 * each response reports, so the server's copy moves while this one sits still.
 * Every write therefore takes in what has been learned since (see `persist`),
 * and `revision` pulls the list down again each time the run has learned more.
 */
export function useGlossary(jobId: string | null, revision = 0) {
  const [rows, setRows] = useState<GlossaryRow[]>([]);
  const [saved, setSaved] = useState<Glossary>({});
  const [saving, setSaving] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    setRows([]);
    setSaved({});
    setError(null);
    setNotice(null);
    if (!jobId) return;

    let cancelled = false;
    getGlossary(jobId)
      .then(({ terms }) => {
        if (cancelled) return;
        setSaved(terms);
        setRows(toRows(terms));
      })
      .catch(() => {
        // An unreadable glossary is an empty one; nothing here is load-bearing.
      });

    return () => {
      cancelled = true;
    };
  }, [jobId]);

  const terms = toTerms(rows);
  const dirty = JSON.stringify(terms) !== JSON.stringify(saved);

  // Read by the effect below without making it depend on every keystroke. A row
  // with a blank source is one being typed rather than a term, so `toTerms`
  // drops it and `dirty` never sees it — but a refresh would still take it away.
  const editing = useRef(false);
  const busy = dirty || rows.some((row) => !row.from.trim());
  useEffect(() => {
    editing.current = busy;
  }, [busy]);

  // A run learns terms as it goes and writes them to the server's copy, so this
  // list is stale the moment a patch lands. `revision` moves with each one, and
  // once more when the run ends. Fetch what the book has taught itself, unless
  // the reader is midway through editing — their work outranks a tidier list.
  useEffect(() => {
    if (!jobId || !revision || editing.current) return;

    let cancelled = false;
    getGlossary(jobId)
      .then(({ terms: stored }) => {
        if (cancelled || editing.current) return;
        setSaved(stored);
        setRows(toRows(stored));
      })
      .catch(() => undefined);

    return () => {
      cancelled = true;
    };
  }, [jobId, revision]);

  const add = useCallback(() => setRows((current) => [...current, makeRow('', '')]), []);

  const update = useCallback((id: number, fields: Partial<GlossaryRow>) => {
    setRows((current) => current.map((row) => (row.id === id ? { ...row, ...fields } : row)));
  }, []);

  const remove = useCallback((id: number) => {
    setRows((current) => current.filter((row) => row.id !== id));
  }, []);

  /**
   * The server's copy, which a run moves while this list sits still.
   *
   * Read before every write, because saving replaces the whole glossary: a list
   * fetched before the run would otherwise delete every term the book taught
   * itself. If it can't be read, assume it still matches what was last seen —
   * saving what the reader has beats refusing to save.
   */
  async function serverTerms(id: string): Promise<Glossary> {
    try {
      return (await getGlossary(id)).terms;
    } catch {
      return saved;
    }
  }

  async function save() {
    if (!jobId) return;
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      const next = withLearned(terms, saved, await serverTerms(jobId));
      const { terms: stored } = await putGlossary(jobId, next);
      setSaved(stored);
      setRows(toRows(stored));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save the glossary.');
    } finally {
      setSaving(false);
    }
  }

  /**
   * Merge a glossary file from elsewhere into this book's, and save it.
   *
   * Saving is part of importing rather than a second step: the file is a
   * decision already made, and a list of imported terms left unsaved is one
   * stray refresh away from being lost. Unsaved edits go up with it, since they
   * are what the reader is looking at.
   *
   * What the merge does is `planImport`'s to decide; all that happens here is
   * reading the file, sending the result, and saying what became of it.
   */
  async function importFile(file: File) {
    if (!jobId) return;
    setImporting(true);
    setError(null);
    setNotice(null);
    try {
      const text = await file.text();
      const report = planImport(text, terms, saved, await serverTerms(jobId));
      const { terms: stored } = await putGlossary(jobId, report.terms);
      setSaved(stored);
      setRows(toRows(stored));
      setNotice(describeImport(report));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not read that glossary.');
    } finally {
      setImporting(false);
    }
  }

  return {
    rows, add, update, remove, save, importFile,
    terms, dirty, saving, importing, error, notice,
  };
}

export type GlossaryState = ReturnType<typeof useGlossary>;

/** Blank keys are rows in progress, not terms; the server never sees them. */
function toTerms(rows: GlossaryRow[]): Glossary {
  const terms: Glossary = {};
  for (const { from, to } of rows) {
    const key = from.trim();
    if (key) terms[key] = to.trim();
  }
  return terms;
}
