import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { cancelJob, createJob, getJob, getStatus, previewJob, startJob } from './api';
import { AppHeader } from './components/AppHeader';
import { PlanPanel } from './components/PlanPanel';
import { RunPanel } from './components/RunPanel';
import type { UploadSettings } from './components/UploadPanel';
import { UploadPanel } from './components/UploadPanel';
import { Callout } from './components/ui';
import type { JobSnapshot, JobStats, Status } from './types';
import { useJobStream } from './useJobStream';
import styles from './App.module.css';

const DEFAULT_SETTINGS: UploadSettings = {
  source_lang: 'auto',
  target_lang: 'English',
};

const TERMINAL = ['done', 'failed', 'cancelled'];

/** Until /api/status answers, the server's own default is the best guess. */
const ASSUMED_REQUESTS_PER_MINUTE = 4;

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [settings, setSettings] = useState<UploadSettings>(DEFAULT_SETTINGS);
  const [file, setFile] = useState<File | null>(null);

  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [preview, setPreview] = useState<JobStats | null>(null);
  const [previewing, setPreviewing] = useState(false);

  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [touched, setTouched] = useState(false);

  const [analysing, setAnalysing] = useState(false);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  const [uploadError, setUploadError] = useState<string | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [expired, setExpired] = useState(false);

  const streaming = job !== null && (job.status === 'running' || TERMINAL.includes(job.status));
  const stream = useJobStream(job?.id ?? null, streaming);

  const refreshStatus = useCallback(() => {
    getStatus().then(setStatus).catch(() => undefined);
  }, []);

  // Startup: the server's limits, and any job this tab was already following.
  useEffect(() => {
    refreshStatus();

    const existing = new URLSearchParams(window.location.search).get('job');
    if (!existing) return;

    getJob(existing)
      .then((snapshot) => {
        setJob(snapshot);
        setSelectionFrom(snapshot);
      })
      .catch(() => {
        setExpired(true);
        clearJobParam();
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function setSelectionFrom(snapshot: JobSnapshot) {
    const planned = snapshot.stats?.chapters.filter((c) => c.patch !== null).map((c) => c.id) ?? [];
    setSelected(new Set(planned));
    setTouched(false);
  }

  function clearJobParam() {
    window.history.replaceState(null, '', window.location.pathname);
  }

  // The terminal `end` event carries the final snapshot; take it, and refresh the
  // budget, which the run has been spending the whole time.
  useEffect(() => {
    if (!stream.final) return;
    setJob(stream.final);
    refreshStatus();
  }, [stream.final, refreshStatus]);

  /* Upload ---------------------------------------------------------------- */

  async function analyse() {
    if (!file) return;
    setAnalysing(true);
    setUploadError(null);
    try {
      const snapshot = await createJob({ file, ...settings });
      setJob(snapshot);
      setSelectionFrom(snapshot);
      setPreview(null);
      setPlanError(null);
      window.history.replaceState(null, '', `?job=${snapshot.id}`);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Could not read that book.');
      refreshStatus();
    } finally {
      setAnalysing(false);
    }
  }

  /* Plan ------------------------------------------------------------------ */

  const selectionList = useMemo(() => [...selected].sort(), [selected]);
  const previewRequest = useRef(0);

  // Re-cost the selection on the server rather than guessing at it here. Debounced,
  // because ticking a run of chapters shouldn't fire a request per click.
  useEffect(() => {
    if (!job || job.status !== 'ready' || !touched) return;

    const token = ++previewRequest.current;
    const controller = new AbortController();
    setPreviewing(true);

    const timer = window.setTimeout(() => {
      previewJob(job.id, selectionList, controller.signal)
        .then((stats) => {
          if (token === previewRequest.current) setPreview(stats);
        })
        .catch(() => undefined)
        .finally(() => {
          if (token === previewRequest.current) setPreviewing(false);
        });
    }, 300);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [job, touched, selectionList]);

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

  async function start() {
    if (!job) return;
    setStarting(true);
    setPlanError(null);
    try {
      // Null means "use the server's own default", which leaves out chapters that
      // already look translated. Only send a list once the reader has changed it.
      const snapshot = await startJob(job.id, touched ? selectionList : null);
      setJob(snapshot);
      setPreview(null);
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
    setJob(null);
    setPreview(null);
    setSelected(new Set());
    setTouched(false);
    setPlanError(null);
    setUploadError(null);
    clearJobParam();
    refreshStatus();
  }

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
            selected={selected}
            onToggle={toggle}
            onToggleMany={toggleMany}
            onStart={start}
            onStartOver={startOver}
            starting={starting}
            previewing={previewing}
            requestsPerMinute={status?.requests_per_minute ?? ASSUMED_REQUESTS_PER_MINUTE}
            error={planError}
          />
        )}

        {phase === 'run' && job && (
          <RunPanel
            job={job}
            stream={stream}
            onCancel={stop}
            onStartOver={startOver}
            cancelling={cancelling}
            requestsPerMinute={status?.requests_per_minute ?? ASSUMED_REQUESTS_PER_MINUTE}
          />
        )}
      </main>

      <footer className={styles.footer}>
        <div>
          <p>
            The API key stays on the server — books are translated here, not in your browser.
            Uploads are deleted an hour after they finish.
          </p>
        </div>
      </footer>
    </div>
  );
}
