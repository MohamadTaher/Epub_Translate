import { useState } from 'react';
import { coverUrl } from '@/api';
import { languageLabel } from '@/format';
import type { BookInfo } from '@/types';
import styles from './BookHeader.module.css';

/**
 * The book itself, so a reader recognises what they uploaded.
 */
export function BookHeader({
  jobId,
  book,
  sourceLanguage,
  targetLanguage,
  size = 'large',
  action,
  children,
}: {
  jobId: string;
  book: BookInfo;
  sourceLanguage: string;
  targetLanguage: string;
  size?: 'large' | 'compact';
  action?: React.ReactNode;
  children?: React.ReactNode;
}) {
  const [coverFailed, setCoverFailed] = useState(false);
  const showCover = book.has_cover && !coverFailed;
  const title = book.title ?? book.filename.replace(/\.epub$/i, '');

  return (
    <div className={`${styles.header} ${size === 'compact' ? styles.compact : ''}`}>
      <div className={styles.coverWrapper}>
        <div className={styles.coverFrame}>
          <div className={styles.spineHighlight} aria-hidden="true" />
          {showCover ? (
            <img
              className={styles.cover}
              src={coverUrl(jobId)}
              alt={`Cover of ${title}`}
              onError={() => setCoverFailed(true)}
            />
          ) : (
            <div className={styles.plate} aria-hidden="true">
              <div className={styles.plateInner}>
                <span className={styles.plateIcon}>📖</span>
                <span className={styles.plateText}>{initials(title)}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className={styles.details}>
        <h2 className={styles.title}>{title}</h2>
        {book.author && (
          <p className={styles.author}>
            <span className={styles.authorBy}>by</span> {book.author}
          </p>
        )}

        <div className={styles.languages}>
          <span className={styles.langPill}>{languageLabel(sourceLanguage)}</span>
          <span aria-hidden="true" className={styles.arrow}>
            →
          </span>
          <span className={`${styles.langPill} ${styles.targetLangPill}`}>
            {targetLanguage || 'English'}
          </span>
        </div>

        {children}
      </div>

      {action && <div className={styles.action}>{action}</div>}
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
