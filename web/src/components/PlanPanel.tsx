import { duration, estimateSeconds, tokens as formatTokens } from '@/format';
import type { JobStats, Status } from '@/types';
import type { GlossaryState } from '@/useGlossary';
import { BookHeader } from './BookHeader';
import { ChapterList } from './ChapterList';
import { GlossaryEditor } from './GlossaryEditor';
import { Button, Callout, Disclosure, Eyebrow, Meter, Stat } from './ui';
import styles from './PlanPanel.module.css';

interface Props {
  jobId: string;
  stats: JobStats;
  status: Status | null;
  glossary: GlossaryState;
  selected: ReadonlySet<string>;
  onToggle: (id: string) => void;
  onToggleMany: (ids: string[], next: boolean) => void;
  onStart: () => void;
  onStartOver: () => void;
  starting: boolean;
  previewing: boolean;
  requestsPerMinute: number;
  error: string | null;
}

export function PlanPanel({
  jobId,
  stats,
  status,
  glossary,
  selected,
  onToggle,
  onToggleMany,
  onStart,
  onStartOver,
  starting,
  previewing,
  requestsPerMinute,
  error,
}: Props) {
  const requests = stats.patch_count;
  const perJobLimit = status?.max_requests_per_book ?? Infinity;
  const budgetLeft = status?.remaining_requests ?? Infinity;
  const allowed = Math.min(perJobLimit, budgetLeft);

  const overJobLimit = requests > perJobLimit;
  const overBudget = requests > budgetLeft;
  const over = overJobLimit || overBudget;
  const nothingSelected = requests === 0;

  const estimate = estimateSeconds(requests, requestsPerMinute, []);
  const selectedCount = stats.chapters.filter((c) => c.patch !== null).length;

  /**
   * The line under the button.
   *
   * The button sits above everything it depends on, so when it is disabled this
   * has to say why — the meter and its warnings are further down the page, past
   * the fold on a small screen.
   */
  function startNote(): string {
    if (nothingSelected) return 'Select at least one chapter below.';
    if (overJobLimit) return `Over the ${perJobLimit}-request limit — deselect some chapters below.`;
    if (overBudget) return `Only ${budgetLeft} requests left today — deselect some chapters below.`;
    return `${requests} request${requests === 1 ? '' : 's'} · ${duration(estimate)}`;
  }

  return (
    <section className={styles.panel}>
      <BookHeader
        jobId={jobId}
        book={stats.book}
        sourceLanguage={stats.source_language}
        targetLanguage={stats.target_language}
        action={
          <>
            <Button
              variant="primary"
              onClick={onStart}
              disabled={starting || over || nothingSelected}
            >
              {starting ? 'Starting…' : 'Start translating'}
            </Button>
            <p className={styles.actionsNote}>{startNote()}</p>
          </>
        }
      >
        <button type="button" className={styles.startOver} onClick={onStartOver}>
          Choose a different book or language
        </button>
      </BookHeader>

      {stats.source_language === 'auto' && (
        <Callout tone="info" title="Source language not detected">
          This book uses the Latin alphabet, so chapters that are already translated can’t be told
          apart from ones that aren’t. Everything is included below — deselect anything that
          shouldn’t be translated.
        </Callout>
      )}

      {error && (
        <Callout tone="error" title="Couldn’t start">
          {error}
        </Callout>
      )}

      <div className={styles.stats}>
        <Stat value={`${selectedCount} of ${stats.chapter_count}`} label="Chapters" />
        <Stat
          value={
            <span className={previewing ? styles.recalculating : undefined}>{requests}</span>
          }
          label="Requests"
        />
        <Stat value={formatTokens(stats.total_tokens)} label="Tokens" />
        <Stat value={requests > 0 ? duration(estimate) : '—'} label="Estimated" />
      </div>

      <div className={styles.budget}>
        <div className={styles.budgetLine}>
          <span>
            <strong className="tabular">{requests}</strong> of {perJobLimit} requests per translation
          </span>
          <span className={styles.budgetRight}>
            <span className="tabular">{budgetLeft}</span> left in today’s budget
          </span>
        </div>
        <Meter value={requests} max={Number.isFinite(allowed) ? allowed : requests} over={over} />

        {overJobLimit && (
          <p className={styles.budgetWarning}>
            Over the {perJobLimit}-request limit for one translation. Deselect some chapters and
            translate the rest afterwards.
          </p>
        )}
        {!overJobLimit && overBudget && (
          <p className={styles.budgetWarning}>
            Only {budgetLeft} requests are left today. Deselect some chapters, or come back after
            the budget resets at midnight UTC.
          </p>
        )}
      </div>

      <div className={styles.section}>
        <div className={styles.sectionHead}>
          <Eyebrow>Chapters</Eyebrow>
          <span className={styles.sectionMeta}>
            in book order · {selectedCount} selected
          </span>
        </div>
        <ChapterList
          chapters={stats.chapters}
          selection={{ selected, onToggle, onToggleMany }}
        />
      </div>

      <Disclosure summary="Glossary">
        <GlossaryEditor glossary={glossary} running={false} />
      </Disclosure>
    </section>
  );
}
