import { useRef } from 'react';
import type { GlossaryState } from '@/useGlossary';
import { Button, CheckIcon, DownloadIcon, FolderIcon, PlusIcon } from './ui';
import styles from './GlossaryEditor.module.css';

/**
 * Names and terms that should be translated the same way every time.
 */
export function GlossaryEditor({ glossary, running }: { glossary: GlossaryState; running: boolean }) {
  const { rows, add, update, remove, save, importFile, terms, dirty, saving, importing, error, notice } =
    glossary;
  const fileRef = useRef<HTMLInputElement>(null);

  const count = Object.keys(terms).length;

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
        Custom glossary for names, techniques, and specific terms to keep translations identical across chapters.
        The AI automatically learns and records names as it reads through the book.
        {running && ' Any saved edits will apply to upcoming patches.'}
      </p>

      {rows.length > 0 ? (
        <ol className={styles.rows}>
          <li className={styles.headings} aria-hidden="true">
            <span>Original Term (Source)</span>
            <span>Canonical Translation (Target)</span>
            <span />
          </li>
          {rows.map((row, index) => (
            <li key={row.id} className={styles.row}>
              <input
                aria-label={`Original term ${index + 1}`}
                value={row.from}
                placeholder="e.g. 李慕生"
                onChange={(e) => update(row.id, { from: e.target.value })}
              />
              <input
                aria-label={`Translation ${index + 1}`}
                value={row.to}
                placeholder="e.g. Li Musheng"
                onChange={(e) => update(row.id, { to: e.target.value })}
              />
              <button
                type="button"
                className={styles.remove}
                aria-label={`Remove ${row.from || `term ${index + 1}`}`}
                onClick={() => remove(row.id)}
                title="Remove this term"
              >
                ✕
              </button>
            </li>
          ))}
        </ol>
      ) : (
        <div className={styles.emptyPrompt}>
          <p>No custom terms defined yet. Terms identified during translation will automatically appear here.</p>
        </div>
      )}

      {notice && <p className={styles.notice}>✓ {notice}</p>}
      {error && <p className={styles.error}>⚠️ {error}</p>}

      <div className={styles.actions}>
        <Button variant="secondary" size="sm" onClick={add} icon={<PlusIcon />}>
          Add term
        </Button>

        <Button
          variant="quiet"
          size="sm"
          disabled={importing}
          onClick={() => fileRef.current?.click()}
          icon={<FolderIcon />}
        >
          {importing ? 'Importing…' : 'Import JSON'}
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="visually-hidden"
          onChange={(e) => {
            const chosen = e.target.files?.[0];
            if (chosen) importFile(chosen);
            e.target.value = '';
          }}
        />

        <Button variant="quiet" size="sm" disabled={count === 0} onClick={exportTerms} icon={<DownloadIcon />}>
          Export JSON
        </Button>

        <span className={styles.spacer} />

        {!dirty && count > 0 && (
          <span className={styles.savedNote}>
            <CheckIcon /> Saved
          </span>
        )}
        <Button variant="primary" size="sm" disabled={!dirty || saving} onClick={save}>
          {saving ? 'Saving…' : 'Save glossary'}
        </Button>
      </div>
    </div>
  );
}
