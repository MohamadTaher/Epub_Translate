import { Fragment } from 'react';
import { MAX_ATTEMPTS } from '@/constants';
import { tokens as formatTokens } from '@/format';
import type { Chapter, RequestProgress, RequestState } from '@/types';
import { useChapterGroups } from '@/useChapterGroups';
import styles from './ChapterList.module.css';

/**
 * The chapters, in book order, with the skipped ones left where they belong so a
 * reader can recognise their book. Rows are grouped under the request that will
 * translate them, which is the same numbering the progress events use.
 */
interface Props {
  chapters: Chapter[];
  /** Present while reviewing a plan; absent during a run, when nothing is selectable. */
  selection?: {
    selected: ReadonlySet<string>;
    onToggle: (id: string) => void;
    onToggleMany: (ids: string[], next: boolean) => void;
  };
  /** Present during a run: request number to its progress. */
  requests?: Record<number, RequestProgress>;
}

const STATE_LABEL: Record<RequestState, string> = {
  queued: 'Waiting',
  active: 'Translating',
  done: 'Translated',
  failed: 'Failed',
};

const STATE_GLYPH: Record<RequestState, string> = {
  queued: '·',
  active: '◐',
  done: '✓',
  failed: '✕',
};

export function ChapterList({ chapters, selection, requests }: Props) {
  const { groups, totals, members } = useChapterGroups(chapters);

  return (
    <ol className={styles.list}>
      {groups.map((group, index) => {
        // Skipped chapters interleave with translated ones all through a book, so
        // they get no band of their own — a repeated "not being translated"
        // header every few rows is noise. The dimmed row and its reason say it.
        if (group.patch === null) {
          return group.chapters.map((chapter) => (
            <ChapterRow key={chapter.file_name || chapter.id} chapter={chapter} selection={selection} />
          ));
        }

        const ids = members.get(group.patch) ?? [];
        const allSelected = selection ? ids.every((id) => selection.selected.has(id)) : false;
        const progress = requests?.[group.patch];
        const state: RequestState = progress?.state ?? 'queued';

        return (
          <Fragment key={`${group.patch}-${index}`}>
            <li className={styles.groupHeader}>
              <span className={styles.groupTitle}>Request {group.patch}</span>

              {!group.firstRun ? (
                <span className={styles.groupMeta}>continued</span>
              ) : requests ? (
                <span className={`${styles.groupState} ${styles[`state-${state}`]}`}>
                  {state === 'active' && progress && progress.attempt > 1
                    ? `Retrying, attempt ${progress.attempt} of ${MAX_ATTEMPTS}`
                    : STATE_LABEL[state]}
                </span>
              ) : (
                <span className={styles.groupMeta}>
                  <span className="tabular">{formatTokens(totals.get(group.patch) ?? 0)}</span> tokens
                </span>
              )}

              {group.firstRun && selection && ids.length > 1 && (
                <button
                  type="button"
                  className={styles.groupToggle}
                  onClick={() => selection.onToggleMany(ids, !allSelected)}
                >
                  {allSelected ? 'Deselect all' : 'Select all'}
                </button>
              )}
            </li>

            {group.chapters.map((chapter) => (
              <ChapterRow
                key={chapter.file_name || chapter.id}
                chapter={chapter}
                selection={selection}
                state={requests ? state : undefined}
              />
            ))}
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
  state?: RequestState;
}) {
  const checked = selection?.selected.has(chapter.id) ?? false;
  const skipped = chapter.patch === null;

  const body = (
    <>
      {state && (
        <span className={`${styles.glyph} ${styles[`state-${state}`]}`} aria-hidden="true">
          {STATE_GLYPH[state]}
        </span>
      )}
      <span className={styles.title}>{chapter.title}</span>
      {skipped && chapter.skip_reason && (
        <span className={styles.reason}>{humaniseReason(chapter.skip_reason)}</span>
      )}
      <span className={`${styles.tokens} tabular`}>{formatTokens(chapter.tokens)}</span>
      {state && <span className="visually-hidden">{STATE_LABEL[state]}</span>}
    </>
  );

  if (!selection) {
    return <li className={`${styles.row} ${skipped ? styles.skipped : ''}`}>{body}</li>;
  }

  return (
    <li className={`${styles.row} ${styles.selectable} ${skipped && !checked ? styles.skipped : ''}`}>
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
