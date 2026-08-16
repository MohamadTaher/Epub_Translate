import type { Status } from '@/types';
import styles from './AppHeader.module.css';

export function AppHeader({ status }: { status: Status | null }) {
  const budgetRatio = status && status.daily_budget > 0 ? status.remaining_requests / status.daily_budget : 1;
  const isBudgetLow = budgetRatio < 0.2;

  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <div className={styles.markBadge}>
            <span aria-hidden="true" className={styles.mark}>
              ❦
            </span>
          </div>
          <span className={styles.wordmark}>
            <strong>EPUB</strong> Translate
          </span>
        </div>

        {status && (
          <div className={styles.meta}>
            <div className={styles.modelPill}>
              <span className={styles.liveDot} aria-hidden="true" />
              <span className={styles.model}>{status.model}</span>
            </div>

            <div className={`${styles.budgetPill} ${isBudgetLow ? styles.budgetLow : ''}`}>
              <div className={styles.miniGauge} aria-hidden="true">
                <div
                  className={styles.miniGaugeFill}
                  style={{ width: `${Math.max(0, Math.min(100, budgetRatio * 100))}%` }}
                />
              </div>
              <span className="tabular">
                <strong>{status.remaining_requests}</strong> of {status.daily_budget} left
              </span>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
