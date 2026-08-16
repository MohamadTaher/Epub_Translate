import { Fragment } from 'react';
import { MAX_ATTEMPTS } from '@/constants';
import { tokens as formatTokens } from '@/format';
import type { Chapter, PatchProgress, PatchState } from '@/types';
import { useChapterGroups } from '@/useChapterGroups';
import styles from './ChapterList.module.css';

/**
 * The chapters, in book order, with the skipped ones left where they belong.
 */
interface Props {
  chapters: Chapter[];
  /** Present while reviewing a plan; absent during a run, when nothing is selectable. */
  selection?: {
    selected: ReadonlySet<string>;
    onToggle: (id: string) => void;
    onToggleMany: (ids: string[], next: boolean) => void;
  };
  /** Present during a run: patch number to its progress. */
  patches?: Record<number, PatchProgress>;
}

const STATE_LABEL: Record<PatchState, string> = {
  queued: 'Queued',
  active: 'Translating',
  done: 'Completed',
  failed: 'Failed',
};

export function ChapterList({ chapters, selection, patches }: Props) {
  const { groups, totals, members } = useChapterGroups(chapters);

  return (
    <ol className={styles.list}>
      {groups.map((group, index) => {
        if (group.patch === null) {
          return group.chapters.map((chapter) => (
            <ChapterRow key={chapter.file_name || chapter.id} chapter={chapter} selection={selection} />
          ));
        }

        const ids = members.get(group.patch) ?? [];
        const allSelected = selection ? ids.every((id) => selection.selected.has(id)) : false;
        const progress = patches?.[group.patch];
        const state: PatchState = progress?.state ?? 'queued';

        const rows = group.chapters.map((chapter) => (
          <ChapterRow
            key={chapter.file_name || chapter.id}
            chapter={chapter}
            selection={selection}
            state={patches ? state : undefined}
          />
        ));

        // A patch split up by skipped chapters keeps the header it was
        // introduced with; the later chapters just carry on beneath it.
        if (!group.firstRun) {
          return <Fragment key={`${group.patch}-${index}`}>{rows}</Fragment>;
        }

        return (
          <Fragment key={`${group.patch}-${index}`}>
            <li className={styles.groupHeader}>
              <div className={styles.groupHeaderLeft}>
                <span className={styles.groupBadge}>Patch {group.patch}</span>

                {patches ? (
                  <span className={`${styles.groupState} ${styles[`state-${state}`]}`}>
                    {state === 'active' && progress && progress.attempt > 1
                      ? `Retrying (${progress.attempt}/${MAX_ATTEMPTS})`
                      : STATE_LABEL[state]}
                  </span>
                ) : (
                  <span className={styles.groupTokens}>
                    <span className="tabular">{formatTokens(totals.get(group.patch) ?? 0)}</span> tokens
                  </span>
                )}
              </div>

              {selection && ids.length > 1 && (
                <button
                  type="button"
                  className={styles.groupToggle}
                  onClick={() => selection.onToggleMany(ids, !allSelected)}
                >
                  {allSelected ? 'Deselect all' : 'Select all'}
                </button>
              )}
            </li>

            {rows}
          </Fragment>
        );
      })}
    </ol>
  );
}

function ChapterRow({
  chapter,
  selection,
  state,
}: {
  chapter: Chapter;
  selection: Props['selection'];
  state?: PatchState;
}) {
  const checked = selection?.selected.has(chapter.id) ?? false;
  const skipped = chapter.patch === null;

  const statusIndicator = state && (
    <span className={`${styles.statusBadge} ${styles[`status-${state}`]}`} aria-hidden="true">
      {state === 'active' && <span className={styles.spinner} />}
      {state === 'done' && '✓'}
      {state === 'failed' && '✕'}
      {state === 'queued' && '·'}
    </span>
  );

  const body = (
    <div className={styles.rowContent}>
      {statusIndicator}
      <span className={styles.title}>{chapter.title}</span>
      {skipped && chapter.skip_reason && (
        <span className={styles.reason}>{humaniseReason(chapter.skip_reason)}</span>
      )}
      <span className={`${styles.tokens} tabular`}>{formatTokens(chapter.tokens)} tokens</span>
      {state && <span className="visually-hidden">{STATE_LABEL[state]}</span>}
    </div>
  );

  if (!selection) {
    return <li className={`${styles.row} ${skipped ? styles.skipped : ''}`}>{body}</li>;
  }

  return (
    <li className={`${styles.row} ${styles.selectable} ${skipped && !checked ? styles.skipped : ''} ${checked ? styles.selected : ''}`}>
      <label className={styles.label}>
        <input
          type="checkbox"
          className={styles.checkbox}
          checked={checked}
          onChange={() => selection.onToggle(chapter.id)}
        />
        {body}
      </label>
    </li>
  );
}

/** The server's reasons are written for a log; these read better in a list. */
function humaniseReason(reason: string): string {
  if (reason === 'Not selected') return 'not selected';
  const match = /^No (.+) characters detected$/.exec(reason);
  return match ? `already translated` : reason.toLowerCase();
}
