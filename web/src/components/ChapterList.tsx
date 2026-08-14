import { Fragment, useMemo } from 'react';
import { tokens as formatTokens } from '../format';
import type { Chapter } from '../types';
import type { RequestProgress, RequestState } from '../useJobStream';
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
  const { groups, totals, members } = useMemo(() => plan(chapters), [chapters]);
  const introduced = new Set<number>();

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

        // A request's chapters need not be contiguous, so the header appears once
        // and later runs of the same request are marked as continuing it.
        const firstTime = !introduced.has(group.patch);
        introduced.add(group.patch);

        const ids = members.get(group.patch) ?? [];
        const allSelected = selection ? ids.every((id) => selection.selected.has(id)) : false;
        const progress = requests?.[group.patch];
        const state: RequestState = progress?.state ?? 'queued';

        return (
          <Fragment key={`${group.patch}-${index}`}>
            <li className={styles.groupHeader}>
              <span className={styles.groupTitle}>Request {group.patch}</span>

              {!firstTime ? (
                <span className={styles.groupMeta}>continued</span>
              ) : requests ? (
                <span className={`${styles.groupState} ${styles[`state-${state}`]}`}>
                  {state === 'active' && progress && progress.attempt > 1
                    ? `Retrying, attempt ${progress.attempt} of 10`
                    : STATE_LABEL[state]}
                </span>
              ) : (
                <span className={styles.groupMeta}>
                  <span className="tabular">{formatTokens(totals.get(group.patch) ?? 0)}</span> tokens
                </span>
              )}

              {firstTime && selection && ids.length > 1 && (
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

interface Group {
  patch: number | null;
  chapters: Chapter[];
}

/**
 * Runs of consecutive chapters that share a request, in book order, plus each
 * request's real totals.
 *
 * The totals are gathered across the whole book rather than per run, because a
 * request's chapters are often split up by skipped ones in between — showing a
 * run's own tokens would understate what the request actually costs.
 */
function plan(chapters: Chapter[]) {
  const groups: Group[] = [];
  const totals = new Map<number, number>();
  const members = new Map<number, string[]>();

  for (const chapter of chapters) {
    const last = groups[groups.length - 1];
    if (last && last.patch === chapter.patch) last.chapters.push(chapter);
    else groups.push({ patch: chapter.patch, chapters: [chapter] });

    if (chapter.patch !== null) {
      totals.set(chapter.patch, (totals.get(chapter.patch) ?? 0) + chapter.tokens);
      members.set(chapter.patch, [...(members.get(chapter.patch) ?? []), chapter.id]);
    }
  }

  return { groups, totals, members };
}

/** The server's reasons are written for a log; these read better in a list. */
function humaniseReason(reason: string): string {
  if (reason === 'Not selected') return 'not selected';
  const match = /^No (.+) characters detected$/.exec(reason);
  return match ? `already translated` : reason.toLowerCase();
}
