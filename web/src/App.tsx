import { useCallback, useEffect, useMemo, useState } from 'react';
import { cancelJob, createJob, getJob, startJob } from '@/api';
import { AppHeader } from '@/components/AppHeader';
import { PlanPanel } from '@/components/PlanPanel';
import { RunPanel } from '@/components/RunPanel';
import { UploadPanel } from '@/components/UploadPanel';
import { Callout } from '@/components/ui';
import { TERMINAL } from '@/constants';
import type { JobSnapshot, UploadSettings } from '@/types';
import { useGlossary } from '@/useGlossary';
import { useJobStream } from '@/useJobStream';
import { usePreview } from '@/usePreview';
import { useResumableJob } from '@/useResumableJob';
import { useServerStatus } from '@/useServerStatus';
import styles from './App.module.css';

const DEFAULT_SETTINGS: UploadSettings = {
  source_lang: 'auto',
  target_lang: 'English',
};

export default function App() {
  const { status, refresh: refreshStatus, requestsPerMinute } = useServerStatus();

  const [settings, setSettings] = useState<UploadSettings>(DEFAULT_SETTINGS);
  const [file, setFile] = useState<File | null>(null);

  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [touched, setTouched] = useState(false);

  const [analysing, setAnalysing] = useState(false);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  const [uploadError, setUploadError] = useState<string | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);

  // A job arrives with a plan already made; the selection starts as whatever the
  // server chose. Stable, so resuming from the URL happens once.
  const takePlannedSelection = useCallback((snapshot: JobSnapshot) => {
    const planned = snapshot.stats?.chapters.filter((c) => c.patch !== null).map((c) => c.id) ?? [];
    setSelected(new Set(planned));
    setTouched(false);
  }, []);

  const { job, setJob, adopt, expired, clear } = useResumableJob(takePlannedSelection);

  const streaming = job !== null && (job.status === 'running' || TERMINAL.includes(job.status));
  const stream = useJobStream(job?.id ?? null, streaming);

  const selectionList = useMemo(() => [...selected].sort(), [selected]);
  const { preview, previewing, reset: resetPreview } = usePreview(job, selectionList, touched);

  // Held here rather than in either panel: both render the editor, in different
  // subtrees, so starting a run would otherwise throw away unsaved edits.
  //
  // Every patch may have taught the book a name, and each one is written to the
  // server's glossary as it lands — so the count moves with the run, and the
  // editor re-reads it each time rather than only at the end.
  const finished = job !== null && TERMINAL.includes(job.status);
  const glossary = useGlossary(job?.id ?? null, stream.completed + (finished ? 1 : 0));

  // The terminal `end` event carries the final snapshot; take it, and refresh the
  // budget, which the run has been spending the whole time.
  useEffect(() => {
    if (!stream.final) return;
    setJob(stream.final);
    refreshStatus();
  }, [stream.final, setJob, refreshStatus]);

  /* Actions --------------------------------------------------------------- */

  async function analyse() {
    if (!file) return;
    setAnalysing(true);
    setUploadError(null);
    try {
      const snapshot = await createJob({ file, ...settings });
      adopt(snapshot);
      takePlannedSelection(snapshot);
      resetPreview();
      setPlanError(null);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Could not read that book.');
      refreshStatus();
    } finally {
      setAnalysing(false);
    }
  }

  async function start() {
    if (!job) return;
    setStarting(true);
    setPlanError(null);
    try {
      // Null means "use the server's own default", which leaves out chapters that
      // already look translated. Only send a list once the reader has changed it.
      const snapshot = await startJob(job.id, touched ? selectionList : null);
      setJob(snapshot);
      resetPreview();
    } catch (error) {
      setPlanError(error instanceof Error ? error.message : 'Could not start.');

      // Starting re-plans the job before it checks the limits, so a rejected start
      // leaves the stored plan describing the selection that was refused. Re-read it.
      getJob(job.id).then(setJob).catch(() => undefined);
      refreshStatus();
    } finally {
      setStarting(false);
    }
  }

  async function stop() {
    if (!job) return;
    setCancelling(true);
    try {
      setJob(await cancelJob(job.id));
    } catch {
      // The run ends on its own terms; the stream will report what happened.
    } finally {
      setCancelling(false);
    }
  }

  function startOver() {
    clear();
    resetPreview();
    setSelected(new Set());
    setTouched(false);
    setPlanError(null);
    setUploadError(null);
    refreshStatus();
  }

  const toggle = useCallback((id: string) => {
    setTouched(true);
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleMany = useCallback((ids: string[], on: boolean) => {
    setTouched(true);
    setSelected((current) => {
      const next = new Set(current);
      for (const id of ids) {
        if (on) next.add(id);
        else next.delete(id);
      }
      return next;
    });
  }, []);

  /* Render ---------------------------------------------------------------- */

  const phase = !job ? 'upload' : job.status === 'ready' ? 'plan' : 'run';
  const stats = preview ?? job?.stats ?? null;

  return (
    <div className={styles.app}>
      <AppHeader status={status} />

      <main className={styles.main}>
        {expired && phase === 'upload' && (
          <div className={styles.notice}>
            <Callout tone="info" title="That translation is no longer available">
              Jobs are kept for an hour and are lost if the server restarts. Upload the book again to
              start over.
            </Callout>
          </div>
        )}

        {phase === 'upload' && (
          <UploadPanel
            status={status}
            settings={settings}
            onSettingsChange={setSettings}
            file={file}
            onFileChange={setFile}
            onAnalyse={analyse}
            analysing={analysing}
            error={uploadError}
          />
        )}

        {phase === 'plan' && job && stats && (
          <PlanPanel
            jobId={job.id}
            stats={stats}
            status={status}
            glossary={glossary}
            selected={selected}
            onToggle={toggle}
            onToggleMany={toggleMany}
            onStart={start}
            onStartOver={startOver}
            starting={starting}
            previewing={previewing}
            requestsPerMinute={requestsPerMinute}
            error={planError}
          />
        )}

        {phase === 'run' && job && (
          <RunPanel
            job={job}
            stream={stream}
            glossary={glossary}
            onCancel={stop}
            onStartOver={startOver}
            cancelling={cancelling}
            requestsPerMinute={requestsPerMinute}
          />
        )}
      </main>

      <footer className={styles.footer}>
        <div className={styles.footerInner}>
          <p className={styles.footerText}>
            🔒 The API key stays safely on the server — books are translated with Gemini, not in your browser.
            Uploads are kept temporarily and automatically deleted after one hour.
          </p>
          <span className={styles.footerBrand}>EPUB Translate · Literary Edition</span>
        </div>
      </footer>
    </div>
  );
}
