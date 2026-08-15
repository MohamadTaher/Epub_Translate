import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from 'react';
import type { Notice, Tone } from '@/notices';
import styles from './ui.module.css';

type Variant = 'primary' | 'secondary' | 'quiet' | 'danger';
type Size = 'md' | 'sm';

const buttonClass = (variant: Variant, size: Size, className?: string) =>
  [styles.button, styles[variant], styles[size], className].filter(Boolean).join(' ');

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
};

export function Button({ variant = 'secondary', size = 'md', className, ...rest }: ButtonProps) {
  return <button className={buttonClass(variant, size, className)} {...rest} />;
}

type LinkButtonProps = AnchorHTMLAttributes<HTMLAnchorElement> & {
  /** A link is never destructive, and there is nothing quiet to link to. */
  variant?: Extract<Variant, 'primary' | 'secondary'>;
  size?: Size;
};

export function LinkButton({ variant = 'secondary', size = 'md', className, ...rest }: LinkButtonProps) {
  return <a className={buttonClass(variant, size, className)} {...rest} />;
}

export function Callout({
  tone = 'info',
  title,
  children,
}: {
  tone?: Tone;
  title?: string;
  children: ReactNode;
}) {
  return (
    <div className={`${styles.callout} ${styles[`tone-${tone}`]}`} role={tone === 'error' ? 'alert' : undefined}>
      {title && <p className={styles.calloutTitle}>{title}</p>}
      <div className={styles.calloutBody}>{children}</div>
    </div>
  );
}

/** A Callout for one of the notices in `notices.ts`; nothing at all when there is none. */
export function NoticeCallout({ notice }: { notice: Notice | null }) {
  if (!notice) return null;
  return (
    <Callout tone={notice.tone} title={notice.title}>
      {notice.body}
    </Callout>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return <p className={styles.eyebrow}>{children}</p>;
}

export function Stat({ value, label }: { value: ReactNode; label: string }) {
  return (
    <div className={styles.stat}>
      <span className={styles.statValue}>{value}</span>
      <span className={styles.statLabel}>{label}</span>
    </div>
  );
}

/**
 * A labelled progress bar.
 *
 * `over` is styled distinctly rather than just clipped, because going over a
 * limit is the one thing the reader has to act on.
 */
export function Meter({ value, max, over = false }: { value: number; max: number; over?: boolean }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div
      className={`${styles.meter} ${over ? styles.meterOver : ''}`}
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={max}
    >
      <div className={styles.meterFill} style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Field({
  label,
  hint,
  htmlFor,
  children,
}: {
  label: string;
  hint?: string;
  htmlFor?: string;
  children: ReactNode;
}) {
  return (
    <div className={styles.field}>
      <label className={styles.fieldLabel} htmlFor={htmlFor}>
        {label}
      </label>
      {children}
      {hint && <p className={styles.fieldHint}>{hint}</p>}
    </div>
  );
}

export function Disclosure({
  summary,
  children,
  defaultOpen = false,
}: {
  summary: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className={styles.disclosure} open={defaultOpen}>
      <summary className={styles.disclosureSummary}>{summary}</summary>
      <div className={styles.disclosureBody}>{children}</div>
    </details>
  );
}
