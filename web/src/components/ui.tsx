import type { ButtonHTMLAttributes, ReactNode } from 'react';
import styles from './ui.module.css';

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'quiet' | 'danger';
  size?: 'md' | 'sm';
};

export function Button({ variant = 'secondary', size = 'md', className, ...rest }: ButtonProps) {
  return (
    <button
      className={[styles.button, styles[variant], styles[size], className].filter(Boolean).join(' ')}
      {...rest}
    />
  );
}

export function LinkButton({
  variant = 'secondary',
  size = 'md',
  className,
  ...rest
}: React.AnchorHTMLAttributes<HTMLAnchorElement> & { variant?: 'primary' | 'secondary'; size?: 'md' | 'sm' }) {
  return (
    <a
      className={[styles.button, styles[variant], styles[size], className].filter(Boolean).join(' ')}
      {...rest}
    />
  );
}

export function Callout({
  tone = 'info',
  title,
  children,
}: {
  tone?: 'info' | 'warning' | 'error' | 'success';
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
