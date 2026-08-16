import { duration, estimateSeconds, tokens as formatTokens } from '@/format';
import type { JobStats, Status } from '@/types';
import type { GlossaryState } from '@/useGlossary';
import { BookHeader } from './BookHeader';
import { ChapterList } from './ChapterList';
import { GlossaryEditor } from './GlossaryEditor';
import { ArrowRightIcon, Button, Callout, Disclosure, Eyebrow, Stat } from './ui';
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
  const patches = stats.patch_count ?? 0;
  const budgetLeft = status?.remaining_requests ?? Infinity;

  const overBudget = patches > budgetLeft;
  const over = overBudget;
  const nothingSelected = patches === 0;

  const estimate = estimateSeconds(patches, requestsPerMinute, []);
  const selectedCount = stats.chapters.filter((c) => c.patch !== null).length;
  const termCount = Object.keys(glossary.terms).length;

  /**
   * The line under the button.
   */
  function startNote(): string {
    if (nothingSelected) return 'Select at least one chapter below.';
    if (overBudget) return `Only ${budgetLeft} requests left today — deselect some chapters.`;
    return `${patches} patch${patches === 1 ? '' : 'es'} · ${duration(estimate)}`;
  }

  return (
    <section className={styles.panel}>
      <BookHeader
        jobId={jobId}
        book={stats.book}
        sourceLanguage={stats.source_language}
        targetLanguage={stats.target_language}
        action={
          <div className={styles.actionBlock}>
            <Button
              variant="primary"
              size="lg"
              icon={<ArrowRightIcon />}
              iconPosition="right"
              onClick={onStart}
              disabled={starting || over || nothingSelected}
              className={styles.startButton}
            >
              {starting ? 'Starting translation…' : 'Start translating'}
            </Button>
            <p className={styles.actionsNote}>{startNote()}</p>
          </div>
        }
      >
        <button type="button" className={styles.startOver} onClick={onStartOver}>
          ← Choose a different book or language
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

      <div className={styles.statsGrid}>
        <Stat
          label="Chapters"
          value={`${selectedCount} of ${stats.chapter_count}`}
          subtext="in queue"
          icon="📖"
        />
        <Stat
          label="Patches"
          value={
            <span className={previewing ? styles.recalculating : undefined}>{patches}</span>
          }
          subtext="API calls"
          icon="⚡"
        />
        <Stat
          label="Tokens"
          value={formatTokens(stats.total_tokens)}
          subtext="approx volume"
          icon="🪙"
        />
        <Stat
          label="Estimated Time"
          value={patches > 0 ? duration(estimate) : '—'}
          subtext={patches > 0 ? `paced @ ${requestsPerMinute}/min` : 'select chapters'}
          icon="⏱️"
        />
      </div>

      <div className={styles.section}>
        <div className={styles.sectionHead}>
          <Eyebrow>Chapters to Translate</Eyebrow>
          <span className={styles.sectionMeta}>
            in book order · <strong>{selectedCount}</strong> selected
          </span>
        </div>
        <ChapterList
          chapters={stats.chapters}
          selection={{ selected, onToggle, onToggleMany }}
        />
      </div>

      <Disclosure
        summary="Translation Glossary & Names"
        badge={termCount > 0 ? `${termCount} term${termCount === 1 ? '' : 's'}` : undefined}
      >
        <GlossaryEditor glossary={glossary} running={false} />
      </Disclosure>
    </section>
  );
}
