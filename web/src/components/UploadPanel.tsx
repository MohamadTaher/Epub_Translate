import { useId, useRef, useState } from 'react';
import { clamp, fileSize, languageLabel } from '../format';
import type { Status } from '../types';
import { Button, Callout, Disclosure, Field } from './ui';
import styles from './UploadPanel.module.css';

export interface UploadSettings {
  source_lang: string;
  target_lang: string;
  max_tokens_per_patch: number;
  max_concurrent: number;
  max_requests_per_minute: number;
}

/**
 * The server accepts these without checking them, so the bounds live here. The
 * lower bound on tokens-per-patch matters most: it is the lever a reader is told
 * to pull when a book needs more requests than a translation is allowed.
 */
const LIMITS = {
  max_tokens_per_patch: [2000, 60000],
  max_concurrent: [1, 8],
  max_requests_per_minute: [1, 20],
} as const;

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

  const maxBytes = (status?.max_upload_mb ?? 25) * 1024 * 1024;

  function accept(candidate: File | undefined) {
    if (!candidate) return;

    // Checked here as well as on the server, so the answer is instant rather
    // than arriving after a pointless upload.
    if (!candidate.name.toLowerCase().endsWith('.epub')) {
      setLocalError('That is not an EPUB. Books usually end in .epub.');
      return;
    }
    if (candidate.size > maxBytes) {
      setLocalError(
        `That file is ${fileSize(candidate.size)}, over the ${status?.max_upload_mb ?? 25} MB limit.`,
      );
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
          Translate a book,
          <br />
          keeping its voice.
        </h1>
        <p className={styles.lede}>
          A whole EPUB, chapter by chapter, with a glossary that keeps names consistent all the way
          through.
        </p>
      </div>

      {blocked ? (
        <Callout tone={blocked.tone} title={blocked.title}>
          {blocked.body}
        </Callout>
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
                  <span className={styles.promptHint}>up to {status?.max_upload_mb ?? 25} MB</span>
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

          <Disclosure summary="Translation settings">
            <div className={styles.settings}>
              <NumberField
                label="Tokens per request"
                hint="Bigger requests mean fewer of them. Raise this if a book needs more requests than one translation allows."
                value={settings.max_tokens_per_patch}
                bounds={LIMITS.max_tokens_per_patch}
                step={1000}
                onChange={(v) => onSettingsChange({ ...settings, max_tokens_per_patch: v })}
              />
              <NumberField
                label="Requests at once"
                hint="How many chapters are translated in parallel."
                value={settings.max_concurrent}
                bounds={LIMITS.max_concurrent}
                step={1}
                onChange={(v) => onSettingsChange({ ...settings, max_concurrent: v })}
              />
              <NumberField
                label="Requests per minute"
                hint="Keep this at or below your Gemini quota, or requests will be rejected and retried."
                value={settings.max_requests_per_minute}
                bounds={LIMITS.max_requests_per_minute}
                step={1}
                onChange={(v) => onSettingsChange({ ...settings, max_requests_per_minute: v })}
              />
            </div>
          </Disclosure>

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

function NumberField({
  label,
  hint,
  value,
  bounds,
  step,
  onChange,
}: {
  label: string;
  hint: string;
  value: number;
  bounds: readonly [number, number];
  step: number;
  onChange: (value: number) => void;
}) {
  const id = useId();
  const [draft, setDraft] = useState(String(value));

  return (
    <Field label={label} hint={hint} htmlFor={id}>
      <input
        id={id}
        type="number"
        inputMode="numeric"
        min={bounds[0]}
        max={bounds[1]}
        step={step}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          const clamped = clamp(Number(draft), bounds[0], bounds[1]);
          setDraft(String(clamped));
          onChange(clamped);
        }}
      />
    </Field>
  );
}

function describeBlock(status: Status | null) {
  if (!status) return null;

  if (!status.configured) {
    return {
      tone: 'error' as const,
      title: 'This server has no API key',
      body: 'Translation needs a Gemini API key in the server’s .env file. Nothing can be uploaded until one is set.',
    };
  }

  if (status.cooldown_seconds > 0) {
    const minutes = Math.ceil(status.cooldown_seconds / 60);
    return {
      tone: 'warning' as const,
      title: 'You started a translation recently',
      body: `You can start another in about ${minutes} minute${minutes === 1 ? '' : 's'}. This limit keeps one visitor from spending the whole day’s budget.`,
    };
  }

  if (status.busy) {
    return {
      tone: 'warning' as const,
      title: 'The server is busy',
      body: 'Other translations are running right now. Try again in a few minutes.',
    };
  }

  if (status.remaining_requests <= 0) {
    return {
      tone: 'warning' as const,
      title: 'Today’s budget is spent',
      body: 'The daily request budget resets at midnight UTC.',
    };
  }

  return null;
}
