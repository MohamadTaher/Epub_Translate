import { useState } from 'react';
import { coverUrl } from '../api';
import { languageLabel } from '../format';
import type { BookInfo } from '../types';
import styles from './BookHeader.module.css';

/**
 * The book itself, so a reader recognises what they uploaded.
 *
 * Plenty of EPUBs carry no cover and some carry no title, so every part of this
 * has a fallback — the typographic plate is used often enough to be a real state,
 * not an edge case.
 */
export function BookHeader({
  jobId,
  book,
  sourceLanguage,
  targetLanguage,
  size = 'large',
  children,
}: {
  jobId: string;
  book: BookInfo;
  sourceLanguage: string;
  targetLanguage: string;
  size?: 'large' | 'compact';
  children?: React.ReactNode;
}) {
  const [coverFailed, setCoverFailed] = useState(false);
  const showCover = book.has_cover && !coverFailed;
  const title = book.title ?? book.filename.replace(/\.epub$/i, '');

  return (
    <div className={`${styles.header} ${size === 'compact' ? styles.compact : ''}`}>
      <div className={styles.coverFrame}>
        {showCover ? (
          <img
            className={styles.cover}
            src={coverUrl(jobId)}
            alt={`Cover of ${title}`}
            onError={() => setCoverFailed(true)}
          />
        ) : (
          <div className={styles.plate} aria-hidden="true">
            <span className={styles.plateText}>{initials(title)}</span>
          </div>
        )}
      </div>

      <div className={styles.details}>
        <h2 className={styles.title}>{title}</h2>
        {book.author && <p className={styles.author}>{book.author}</p>}

        <p className={styles.languages}>
          <span>{languageLabel(sourceLanguage)}</span>
          <span aria-hidden="true" className={styles.arrow}>
            →
          </span>
          <span>{targetLanguage || 'English'}</span>
        </p>

        {children}
      </div>
    </div>
  );
}

function initials(title: string): string {
  const trimmed = title.trim();
  if (!trimmed) return '📖';
  // Works for scripts without spaces too, where the first glyphs carry the sense.
  const words = trimmed.split(/\s+/).filter(Boolean);
  return words.length > 1 ? words.slice(0, 2).map((w) => [...w][0]).join('') : [...trimmed].slice(0, 2).join('');
}
