import type { PatchProgress, PatchState } from '@/types';
import styles from './PatchTrack.module.css';

/**
 * The run's progress as one block per patch, in patch order.
 *
 * A plain bar can only say how much is left; a book that lost a patch to the
 * safety filter looks the same as one that is simply slow. One block each says
 * which patch, and colour says what became of it.
 */

const LABEL: Record<PatchState, string> = {
  queued: 'Queued',
  active: 'Translating',
  done: 'Completed',
  failed: 'Failed',
};

/**
 * Below the blocks, only the states the headline count doesn't already give.
 * `done` is `3 of 12 patches`, and `queued` is everything else, so naming
 * either would be repeating the line above it.
 */
const LEGEND: PatchState[] = ['active', 'failed'];

/** Past this many, the blocks are thinner than their own borders. */
const DENSE_FROM = 32;

interface Props {
  total: number;
  /** Keyed by 1-based patch number; anything absent has not started. */
  patches: Record<number, PatchProgress>;
}

export function PatchTrack({ total, patches }: Props) {
  if (total <= 0) return null;

  const states = Array.from(
    { length: total },
    (_, index): PatchState => patches[index + 1]?.state ?? 'queued',
  );

  const done = states.filter((state) => state === 'done').length;
  const keys = LEGEND.map((state) => ({
    state,
    count: states.filter((s) => s === state).length,
  })).filter((key) => key.count > 0);

  return (
    <div
      className={styles.track}
      role="progressbar"
      aria-label="Patches translated"
      aria-valuenow={done}
      aria-valuemin={0}
      aria-valuemax={total}
    >
      <div className={`${styles.blocks} ${total > DENSE_FROM ? styles.dense : ''}`}>
        {states.map((state, index) => (
          <span
            key={index}
            className={`${styles.block} ${styles[`is-${state}`]}`}
            title={`Patch ${index + 1} — ${LABEL[state]}`}
          />
        ))}
      </div>

      {keys.length > 0 && (
        <p className={styles.legend}>
          {keys.map(({ state, count }) => (
            <span
              key={state}
              className={`${styles.key} ${state === 'failed' ? styles.keyFailed : ''}`}
            >
              <span className={`${styles.dot} ${styles[`is-${state}`]}`} aria-hidden="true" />
              <span className="tabular">{count}</span> {LABEL[state].toLowerCase()}
            </span>
          ))}
        </p>
      )}
    </div>
  );
}
