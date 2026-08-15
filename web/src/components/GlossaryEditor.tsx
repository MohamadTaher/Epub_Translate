import { useRef } from 'react';
import type { GlossaryState } from '@/useGlossary';
import { Button } from './ui';
import styles from './GlossaryEditor.module.css';

/**
 * Names and terms that should be translated the same way every time.
 *
 * The server replaces the whole glossary on every save — there is no way to
 * change one term — so this edits a local list and sends all of it at once. The
 * list itself lives in `useGlossary`, above the phase machine, so edits made
 * while reviewing the plan are still here once the run starts.
 *
 * A glossary can also arrive as a file: the one the CLI writes, or one saved
 * from an earlier book in the same series. Importing merges rather than
 * replaces, so a list already being edited here is never lost to it.
 */
export function GlossaryEditor({ glossary, running }: { glossary: GlossaryState; running: boolean }) {
  const { rows, add, update, remove, save, importFile, terms, dirty, saving, importing, error, notice } =
    glossary;
  const fileRef = useRef<HTMLInputElement>(null);

  const count = Object.keys(terms).length;

  /**
   * Hand the list back as the same kind of file it can be imported from, so a
   * series can carry its names from one book to the next.
   */
  function exportTerms() {
    const blob = new Blob([JSON.stringify(terms, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'glossary.json';
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={styles.editor}>
      <p className={styles.intro}>
        Terms to translate the same way every time — character names especially. The translation
        adds the names it meets as it goes, so this fills itself in.
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
            <li key={row.id} className={styles.row}>
              <input
                aria-label={`Original term ${index + 1}`}
                value={row.from}
                placeholder="道玄"
                onChange={(e) => update(row.id, { from: e.target.value })}
              />
              <input
                aria-label={`Translation ${index + 1}`}
                value={row.to}
                placeholder="Dao Xuan"
                onChange={(e) => update(row.id, { to: e.target.value })}
              />
              <button
                type="button"
                className={styles.remove}
                aria-label={`Remove ${row.from || `term ${index + 1}`}`}
                onClick={() => remove(row.id)}
              >
                ✕
              </button>
            </li>
          ))}
        </ol>
      )}

      {notice && <p className={styles.notice}>{notice}</p>}
      {error && <p className={styles.error}>{error}</p>}

      <div className={styles.actions}>
        <Button variant="secondary" size="sm" onClick={add}>
          Add a term
        </Button>

        <Button
          variant="quiet"
          size="sm"
          disabled={importing}
          onClick={() => fileRef.current?.click()}
        >
          {importing ? 'Importing…' : 'Import a file'}
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="visually-hidden"
          onChange={(e) => {
            const chosen = e.target.files?.[0];
            if (chosen) importFile(chosen);
            // Cleared so choosing the same file twice still counts as a change.
            e.target.value = '';
          }}
        />

        <Button variant="quiet" size="sm" disabled={count === 0} onClick={exportTerms}>
          Export
        </Button>

        <span className={styles.spacer} />

        <Button variant="primary" size="sm" disabled={!dirty || saving} onClick={save}>
          {saving ? 'Saving…' : 'Save glossary'}
        </Button>
        {!dirty && count > 0 && <span className={styles.savedNote}>Saved</span>}
      </div>
    </div>
  );
}
