import { useId, useRef, useState } from 'react';
import { DEFAULT_MAX_UPLOAD_MB } from '@/constants';
import { fileSize, languageLabel } from '@/format';
import { describeBlock } from '@/notices';
import type { Status, UploadSettings } from '@/types';
import { Button, Callout, Field, NoticeCallout } from './ui';
import styles from './UploadPanel.module.css';

interface Props {
  status: Status | null;
  settings: UploadSettings;
  onSettingsChange: (settings: UploadSettings) => void;
  file: File | null;
  onFileChange: (file: File | null) => void;
  onAnalyse: () => void;
  analysing: boolean;
  error: string | null;
}

export function UploadPanel({
  status,
  settings,
  onSettingsChange,
  file,
  onFileChange,
  onAnalyse,
  analysing,
  error,
}: Props) {
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const sourceId = useId();
  const targetId = useId();
  const listId = useId();

  const maxUploadMb = status?.max_upload_mb ?? DEFAULT_MAX_UPLOAD_MB;
  const maxBytes = maxUploadMb * 1024 * 1024;

  function accept(candidate: File | undefined) {
    if (!candidate) return;

    // Checked here as well as on the server, so the answer is instant rather
    // than arriving after a pointless upload.
    if (!candidate.name.toLowerCase().endsWith('.epub')) {
      setLocalError('That is not an EPUB. Books usually end in .epub.');
      return;
    }
    if (candidate.size > maxBytes) {
      setLocalError(`That file is ${fileSize(candidate.size)}, over the ${maxUploadMb} MB limit.`);
      return;
    }
    setLocalError(null);
    onFileChange(candidate);
  }

  const blocked = describeBlock(status);
  const shownError = error ?? localError;

  return (
    <section className={styles.panel}>
      <div className={styles.hero}>
        <h1 className={styles.title}>Translate a book, keeping its voice.</h1>
        <p className={styles.lede}>
          A whole EPUB, chapter by chapter, with a glossary that keeps names consistent all the way
          through.
        </p>
      </div>

      {blocked ? (
        <NoticeCallout notice={blocked} />
      ) : (
        <>
          <div
            className={`${styles.dropzone} ${dragging ? styles.dragging : ''} ${
              analysing ? styles.busy : ''
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              if (!analysing) setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              if (!analysing) accept(e.dataTransfer.files[0]);
            }}
          >
            <div className={styles.plate}>
              {analysing && file ? (
                <div className={styles.analysing}>
                  <p className={styles.analysingName}>{file.name}</p>
                  <p className={styles.analysingNote}>Reading the book…</p>
                  <div className={styles.shimmer} aria-hidden="true" />
                  <p className={styles.analysingHint}>
                    Finding chapters and measuring how much there is to translate.
                  </p>
                </div>
              ) : file ? (
                <div className={styles.chosen}>
                  <p className={styles.chosenName}>{file.name}</p>
                  <p className={styles.chosenSize}>{fileSize(file.size)}</p>
                  <Button variant="quiet" size="sm" onClick={() => inputRef.current?.click()}>
                    Choose a different book
                  </Button>
                </div>
              ) : (
                <button className={styles.prompt} onClick={() => inputRef.current?.click()} type="button">
                  <span className={styles.promptTitle}>Drop an EPUB here, or browse</span>
                  <span className={styles.promptHint}>up to {maxUploadMb} MB</span>
                </button>
              )}
            </div>

            <input
              ref={inputRef}
              type="file"
              accept=".epub,application/epub+zip"
              className="visually-hidden"
              onChange={(e) => {
                accept(e.target.files?.[0]);
                e.target.value = '';
              }}
            />
          </div>

          {shownError && (
            <Callout tone="error" title="That didn't work">
              {shownError}
            </Callout>
          )}

          <div className={styles.languages}>
            <Field label="From" htmlFor={sourceId}>
              <select
                id={sourceId}
                value={settings.source_lang}
                onChange={(e) => onSettingsChange({ ...settings, source_lang: e.target.value })}
              >
                <option value="auto">Auto-detect</option>
                {(status?.languages ?? []).map((language) => (
                  <option key={language} value={language}>
                    {languageLabel(language)}
                  </option>
                ))}
              </select>
            </Field>

            <span aria-hidden="true" className={styles.arrow}>
              →
            </span>

            <Field label="Into" htmlFor={targetId}>
              <input
                id={targetId}
                list={listId}
                value={settings.target_lang}
                placeholder="English"
                onChange={(e) => onSettingsChange({ ...settings, target_lang: e.target.value })}
              />
              <datalist id={listId}>
                {(status?.languages ?? []).map((language) => (
                  <option key={language} value={languageLabel(language)} />
                ))}
              </datalist>
            </Field>
          </div>

          <div className={styles.actions}>
            <Button variant="primary" disabled={!file || analysing} onClick={onAnalyse}>
              {analysing ? 'Reading…' : 'Analyse this book'}
            </Button>
            <p className={styles.actionsNote}>
              Nothing is translated yet — you'll see the plan and what it costs first.
            </p>
          </div>
        </>
      )}
    </section>
  );
}

