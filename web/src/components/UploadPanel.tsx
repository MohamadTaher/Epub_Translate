import { useId, useRef, useState } from 'react';
import { DEFAULT_MAX_UPLOAD_MB } from '@/constants';
import { fileSize, languageLabel } from '@/format';
import { describeBlock } from '@/notices';
import type { Status, UploadSettings } from '@/types';
import { Button, Callout, NoticeCallout, SparklesIcon } from './ui';
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

    if (!candidate.name.toLowerCase().endsWith('.epub')) {
      setLocalError('Please upload an EPUB file (usually ending in .epub).');
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
        <h1 className={styles.title}>
          Translate EPUB books, <span className={styles.titleHighlight}>preserving voice & style.</span>
        </h1>
        <p className={styles.lede}>
          Chapter-by-chapter AI translation with consistent terminology, formatting, and a live plan.
        </p>
      </div>

      {blocked ? (
        <NoticeCallout notice={blocked} />
      ) : (
        <div className={styles.card}>
          <div
            className={`${styles.dropzone} ${dragging ? styles.dragging : ''} ${
              analysing ? styles.busy : ''
            } ${file ? styles.hasFile : ''}`}
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
            {analysing && file ? (
              <div className={styles.analysing}>
                <div className={styles.analysingIconWrapper}>
                  <svg className={styles.spinIcon} width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                  </svg>
                </div>
                <div className={styles.analysingInfo}>
                  <p className={styles.analysingName}>{file.name}</p>
                  <p className={styles.analysingStatus}>Analyzing book structure & chapters…</p>
                </div>
                <div className={styles.shimmer} aria-hidden="true" />
              </div>
            ) : file ? (
              <div className={styles.chosen}>
                <div className={styles.chosenLeft}>
                  <div className={styles.chosenBookBadge} aria-hidden="true">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
                      <path d="M6 6h10" />
                      <path d="M6 10h10" />
                    </svg>
                  </div>
                  <div className={styles.chosenDetails}>
                    <div className={styles.chosenHeader}>
                      <p className={styles.chosenName} title={file.name}>{file.name}</p>
                      <span className={styles.chosenBadge}>{fileSize(file.size)}</span>
                    </div>
                    <span className={styles.chosenStatus}>✓ Ready to analyze</span>
                  </div>
                </div>
                <div className={styles.chosenActions}>
                  <button
                    type="button"
                    className={styles.changeFileButton}
                    onClick={() => inputRef.current?.click()}
                  >
                    Change book
                  </button>
                </div>
              </div>
            ) : (
              <div
                className={styles.prompt}
                onClick={() => inputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    inputRef.current?.click();
                  }
                }}
              >
                <div className={styles.promptIconWrapper} aria-hidden="true">
                  <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
                    <path d="M12 11v6" />
                    <path d="m9 14 3-3 3 3" />
                  </svg>
                </div>
                <div className={styles.promptText}>
                  <span className={styles.promptTitle}>
                    Drop your <strong>.epub</strong> book here, or <span className={styles.promptAction}>browse files</span>
                  </span>
                  <span className={styles.promptHint}>Standard EPUB format · Up to {maxUploadMb} MB</span>
                </div>
              </div>
            )}

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
            <Callout tone="error" title="Could not load file">
              {shownError}
            </Callout>
          )}

          <div className={styles.languagesRow}>
            <div className={styles.langField}>
              <label htmlFor={sourceId} className={styles.langLabel}>
                Original Language
              </label>
              <div className={styles.selectWrapper}>
                <select
                  id={sourceId}
                  className={styles.langSelect}
                  value={settings.source_lang}
                  onChange={(e) => onSettingsChange({ ...settings, source_lang: e.target.value })}
                >
                  <option value="auto">Auto-detect from book</option>
                  {(status?.languages ?? []).map((language) => (
                    <option key={language} value={language}>
                      {languageLabel(language)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className={styles.arrowBox} aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14" />
                <path d="m12 5 7 7-7 7" />
              </svg>
            </div>

            <div className={styles.langField}>
              <label htmlFor={targetId} className={styles.langLabel}>
                Translate Into
              </label>
              <input
                id={targetId}
                list={listId}
                className={styles.langInput}
                value={settings.target_lang}
                placeholder="English"
                onChange={(e) => onSettingsChange({ ...settings, target_lang: e.target.value })}
              />
              <datalist id={listId}>
                {(status?.languages ?? []).map((language) => (
                  <option key={language} value={languageLabel(language)} />
                ))}
              </datalist>
            </div>
          </div>

          <div className={styles.actions}>
            <Button
              variant="primary"
              size="lg"
              icon={<SparklesIcon />}
              disabled={!file || analysing}
              onClick={onAnalyse}
              className={styles.analyseButton}
            >
              {analysing ? 'Reading book…' : 'Analyse book & plan translation'}
            </Button>
            <p className={styles.actionsNote}>
              🔒 Nothing is spent yet — you’ll review chapter breakdown and cost before translating.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
