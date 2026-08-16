import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from 'react';
import type { Notice, Tone } from '@/notices';
import styles from './ui.module.css';

type Variant = 'primary' | 'secondary' | 'quiet' | 'danger';
type Size = 'lg' | 'md' | 'sm';

const buttonClass = (variant: Variant, size: Size, className?: string) =>
  [styles.button, styles[variant], styles[size], className].filter(Boolean).join(' ');

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
  icon?: ReactNode;
  iconPosition?: 'left' | 'right';
};

export function Button({
  variant = 'secondary',
  size = 'md',
  icon,
  iconPosition = 'left',
  className,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button className={buttonClass(variant, size, className)} {...rest}>
      {icon && iconPosition === 'left' && (
        <span className={styles.buttonIcon} aria-hidden="true">
          {icon}
        </span>
      )}
      <span className={styles.buttonText}>{children}</span>
      {icon && iconPosition === 'right' && (
        <span className={styles.buttonIcon} aria-hidden="true">
          {icon}
        </span>
      )}
    </button>
  );
}

type LinkButtonProps = AnchorHTMLAttributes<HTMLAnchorElement> & {
  variant?: Extract<Variant, 'primary' | 'secondary'>;
  size?: Size;
  icon?: ReactNode;
  iconPosition?: 'left' | 'right';
};

export function LinkButton({
  variant = 'secondary',
  size = 'md',
  icon,
  iconPosition = 'left',
  className,
  children,
  ...rest
}: LinkButtonProps) {
  return (
    <a className={buttonClass(variant, size, className)} {...rest}>
      {icon && iconPosition === 'left' && (
        <span className={styles.buttonIcon} aria-hidden="true">
          {icon}
        </span>
      )}
      <span className={styles.buttonText}>{children}</span>
      {icon && iconPosition === 'right' && (
        <span className={styles.buttonIcon} aria-hidden="true">
          {icon}
        </span>
      )}
    </a>
  );
}

/* Modern Minimalist SVG Icons */

export function SparklesIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3L12 3z" />
    </svg>
  );
}

export function DownloadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

export function RefreshIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
    </svg>
  );
}

export function StopIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <rect x="5" y="5" width="14" height="14" rx="2" />
    </svg>
  );
}

export function ArrowRightIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  );
}

export function PlusIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

export function FolderIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z" />
    </svg>
  );
}

export function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
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
      <div className={styles.calloutIcon} aria-hidden="true">
        {tone === 'error' && '✕'}
        {tone === 'warning' && '⚠'}
        {tone === 'success' && '✓'}
        {tone === 'info' && 'ℹ'}
      </div>
      <div className={styles.calloutContent}>
        {title && <p className={styles.calloutTitle}>{title}</p>}
        <div className={styles.calloutBody}>{children}</div>
      </div>
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

export function Stat({
  value,
  label,
  subtext,
  icon,
}: {
  value: ReactNode;
  label: string;
  subtext?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className={styles.stat}>
      <div className={styles.statHeader}>
        <span className={styles.statLabel}>{label}</span>
        {icon && <span className={styles.statIcon} aria-hidden="true">{icon}</span>}
      </div>
      <span className={styles.statValue}>{value}</span>
      {subtext && <span className={styles.statSubtext}>{subtext}</span>}
    </div>
  );
}

/**
 * A labelled progress bar with optional shimmer/active state.
 */
export function Meter({
  value,
  max,
  over = false,
  animated = false,
}: {
  value: number;
  max: number;
  over?: boolean;
  animated?: boolean;
}) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div
      className={`${styles.meter} ${over ? styles.meterOver : ''} ${animated ? styles.meterAnimated : ''}`}
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
  badge,
  defaultOpen = false,
}: {
  summary: ReactNode;
  children: ReactNode;
  badge?: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className={styles.disclosure} open={defaultOpen}>
      <summary className={styles.disclosureSummary}>
        <span className={styles.summaryTitle}>{summary}</span>
        {badge && <span className={styles.summaryBadge}>{badge}</span>}
      </summary>
      <div className={styles.disclosureBody}>{children}</div>
    </details>
  );
}
