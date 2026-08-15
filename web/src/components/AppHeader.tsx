import type { Status } from '@/types';
import styles from './AppHeader.module.css';

export function AppHeader({ status }: { status: Status | null }) {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <span className={styles.wordmark}>
          <span aria-hidden="true" className={styles.mark}>
            ❦
          </span>
          EPUB Translate
        </span>

        {status && (
          <p className={styles.meta}>
            <span className={styles.model}>{status.model}</span>
            <span aria-hidden="true" className={styles.dot} />
            <span className="tabular">
              {status.remaining_requests} of {status.daily_budget} requests left today
            </span>
          </p>
        )}
      </div>
    </header>
  );
}
