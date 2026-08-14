import { useEffect, useState } from 'react';
import { getGlossary, putGlossary } from '../api';
import type { Glossary } from '../types';
import { Button } from './ui';
import styles from './GlossaryEditor.module.css';

interface Row {
  from: string;
  to: string;
}

/**
 * Names and terms that should be translated the same way every time.
 *
 * The server replaces the whole glossary on every save — there is no way to
 * change one term — so this edits a local list and sends all of it at once.
 */
export function GlossaryEditor({ jobId, running }: { jobId: string; running: boolean }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [saved, setSaved] = useState<Glossary>({});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getGlossary(jobId)
      .then(({ terms }) => {
        if (cancelled) return;
        setSaved(terms);
        setRows(Object.entries(terms).map(([from, to]) => ({ from, to })));
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

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const { terms: stored } = await putGlossary(jobId, terms);
      setSaved(stored);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save the glossary.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className={styles.editor}>
      <p className={styles.intro}>
        Terms to translate the same way every time — character names especially.
        {running && ' Changes apply to requests that haven’t been sent yet.'}
      </p>

      {rows.length > 0 && (
        <ol className={styles.rows}>
          <li className={styles.headings} aria-hidden="true">
            <span>Original</span>
            <span>Translation</span>
            <span />
          </li>
          {rows.map((row, index) => (
            <li key={index} className={styles.row}>
              <input
                aria-label={`Original term ${index + 1}`}
                value={row.from}
                placeholder="道玄"
                onChange={(e) => setRows(replace(rows, index, { ...row, from: e.target.value }))}
              />
              <input
                aria-label={`Translation ${index + 1}`}
                value={row.to}
                placeholder="Dao Xuan"
                onChange={(e) => setRows(replace(rows, index, { ...row, to: e.target.value }))}
              />
              <button
                type="button"
                className={styles.remove}
                aria-label={`Remove ${row.from || `term ${index + 1}`}`}
                onClick={() => setRows(rows.filter((_, i) => i !== index))}
              >
                ✕
              </button>
            </li>
          ))}
        </ol>
      )}

      {error && <p className={styles.error}>{error}</p>}

      <div className={styles.actions}>
        <Button variant="secondary" size="sm" onClick={() => setRows([...rows, { from: '', to: '' }])}>
          Add a term
        </Button>
        <Button variant="primary" size="sm" disabled={!dirty || saving} onClick={save}>
          {saving ? 'Saving…' : 'Save glossary'}
        </Button>
        {!dirty && Object.keys(terms).length > 0 && <span className={styles.savedNote}>Saved</span>}
      </div>
    </div>
  );
}

function replace(rows: Row[], index: number, row: Row): Row[] {
  return rows.map((existing, i) => (i === index ? row : existing));
}

function toTerms(rows: Row[]): Glossary {
  const terms: Glossary = {};
  for (const { from, to } of rows) {
    const key = from.trim();
    if (key) terms[key] = to.trim();
  }
  return terms;
}
