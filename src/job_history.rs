use std::{
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::process::Command;

use anyhow::{Context, Result};

use crate::{
    script_store::{command_for_script, resolve_script_in},
    state_store::{
        self, JobHistoryLoad as PersistedLoad, JobRecord as PersistedJob, JOBS_MAX,
        JOB_INTERRUPTED_EXIT, JOB_RUNNING_EXIT,
    },
};

#[derive(Debug, Clone)]
pub struct JobRecord {
    pub id: u64,
    pub name: String,
    pub kind: String,
    pub status: String,
    pub command: Vec<String>,
    pub started_label: String,
    pub elapsed_label: String,
    pub exit_label: String,
    pub mode: String,
    pub script_path: String,
    pub exit_code: Option<i32>,
    started_at: Option<Instant>,
}

impl JobRecord {
    pub fn elapsed_seconds(&self) -> Option<u64> {
        self.started_at.map(|started| started.elapsed().as_secs())
    }

    pub fn is_running(&self) -> bool {
        self.exit_label == JOB_RUNNING_EXIT
    }

    fn persisted(&self) -> PersistedJob {
        PersistedJob {
            name: self.name.clone(),
            started: self.started_label.clone(),
            elapsed: self.elapsed_label.clone(),
            exit: self.exit_label.clone(),
            mode: self.mode.clone(),
            script_path: self.script_path.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct JobHistory {
    jobs: Vec<JobRecord>,
    next_id: u64,
    persistence_enabled: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadNotice {
    pub reconciled_running: usize,
    pub truncated_rows: usize,
    pub malformed_rows: usize,
    pub normalization_error: Option<String>,
}

impl LoadNotice {
    pub fn is_empty(&self) -> bool {
        self.reconciled_running == 0
            && self.truncated_rows == 0
            && self.malformed_rows == 0
            && self.normalization_error.is_none()
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.reconciled_running > 0 {
            parts.push(format!(
                "reconciled {} interrupted job(s)",
                self.reconciled_running
            ));
        }
        if self.truncated_rows > 0 {
            parts.push(format!("trimmed {} old job(s)", self.truncated_rows));
        }
        if self.malformed_rows > 0 {
            parts.push(format!("ignored {} malformed job(s)", self.malformed_rows));
        }
        if let Some(error) = &self.normalization_error {
            parts.push(format!("normalization save failed: {error}"));
        }
        parts.join("; ")
    }
}

impl Default for JobHistory {
    fn default() -> Self {
        Self {
            jobs: Vec::new(),
            next_id: 1,
            persistence_enabled: true,
        }
    }
}

impl JobHistory {
    pub fn load(repository_root: &Path) -> Result<(Self, LoadNotice)> {
        let data_root = state_store::data_root()?;
        Self::load_in(&data_root, repository_root)
    }

    pub fn load_in(data_root: &Path, repository_root: &Path) -> Result<(Self, LoadNotice)> {
        let loaded = state_store::load_jobs_in(data_root)?;
        let mut notice = notice_for(&loaded);
        if !notice.is_empty() {
            if let Err(error) = state_store::save_jobs_in(data_root, &loaded.jobs) {
                notice.normalization_error = Some(format!("{error:#}"));
            }
        }
        let jobs = loaded
            .jobs
            .into_iter()
            .enumerate()
            .rev()
            .map(|(index, job)| hydrate_job(index as u64 + 1, job, repository_root))
            .collect::<Vec<_>>();
        let next_id = jobs.iter().map(|job| job.id).max().unwrap_or(0) + 1;
        Ok((
            Self {
                jobs,
                next_id,
                persistence_enabled: true,
            },
            notice,
        ))
    }

    pub fn records(&self) -> &[JobRecord] {
        &self.jobs
    }

    pub fn get(&self, index: usize) -> Option<&JobRecord> {
        self.jobs.get(index)
    }

    pub fn unavailable() -> Self {
        Self {
            persistence_enabled: false,
            ..Self::default()
        }
    }

    pub fn persistence_enabled(&self) -> bool {
        self.persistence_enabled
    }

    pub fn begin(
        &mut self,
        name: String,
        kind: String,
        command: Vec<String>,
        mode: String,
        script_path: String,
    ) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.jobs.insert(
            0,
            JobRecord {
                id,
                name,
                kind,
                status: "running".into(),
                command,
                started_label: current_clock_label(),
                elapsed_label: "-".into(),
                exit_label: JOB_RUNNING_EXIT.into(),
                mode,
                script_path,
                exit_code: None,
                started_at: Some(Instant::now()),
            },
        );
        self.jobs.truncate(JOBS_MAX);
        id
    }

    pub fn finish(&mut self, id: u64, exit_code: i32) -> Option<&JobRecord> {
        let job = self.jobs.iter_mut().find(|job| job.id == id)?;
        let elapsed = job
            .started_at
            .map(|started| started.elapsed().as_secs())
            .unwrap_or_default();
        job.elapsed_label = format_elapsed(elapsed);
        job.exit_label = exit_code.to_string();
        job.exit_code = Some(exit_code);
        job.status = if exit_code == 0 {
            "done".into()
        } else {
            "failed".into()
        };
        job.started_at = None;
        Some(job)
    }

    pub fn clear_for_recovery(&mut self) {
        self.jobs.clear();
        self.persistence_enabled = true;
    }

    pub fn persist(&self) -> Result<()> {
        let data_root = state_store::data_root()?;
        self.persist_in(&data_root)
    }

    pub fn persist_in(&self, data_root: &Path) -> Result<()> {
        anyhow::ensure!(
            self.persistence_enabled,
            "job history persistence is disabled until the unreadable history is explicitly cleared"
        );
        let oldest_first = self
            .jobs
            .iter()
            .rev()
            .map(JobRecord::persisted)
            .collect::<Vec<_>>();
        state_store::save_jobs_in(data_root, &oldest_first)
    }
}

fn notice_for(loaded: &PersistedLoad) -> LoadNotice {
    LoadNotice {
        reconciled_running: loaded.reconciled_running,
        truncated_rows: loaded.truncated_rows,
        malformed_rows: loaded.issues.len(),
        normalization_error: None,
    }
}

fn hydrate_job(id: u64, persisted: PersistedJob, repository_root: &Path) -> JobRecord {
    let (kind, command) = retry_command(&persisted, repository_root);
    let exit_code = persisted.exit.parse::<i32>().ok();
    let status = match persisted.exit.as_str() {
        JOB_INTERRUPTED_EXIT => "interrupted",
        "0" => "done",
        _ if exit_code.is_some() => "failed",
        _ => "finished",
    }
    .to_owned();
    JobRecord {
        id,
        name: persisted.name,
        kind,
        status,
        command,
        started_label: persisted.started,
        elapsed_label: persisted.elapsed,
        exit_label: persisted.exit,
        mode: persisted.mode,
        script_path: persisted.script_path,
        exit_code,
        started_at: None,
    }
}

fn retry_command(persisted: &PersistedJob, repository_root: &Path) -> (String, Vec<String>) {
    match persisted.mode.as_str() {
        "run" if !persisted.script_path.trim().is_empty() => (
            "model-load".into(),
            vec![
                "llama-swap".into(),
                "load".into(),
                persisted.script_path.clone(),
            ],
        ),
        "bench" | "maintenance" if !persisted.script_path.trim().is_empty() => {
            let mode = persisted.mode.clone();
            let path = PathBuf::from(&persisted.script_path);
            let expected_directory = repository_root.join(if mode == "bench" {
                "bench-models"
            } else {
                "maintenance"
            });
            let expected_directory =
                fs::canonicalize(&expected_directory).unwrap_or(expected_directory);
            let safe_path = resolve_script_in(&path, repository_root)
                .ok()
                .filter(|resolved| resolved.is_file() && resolved.starts_with(expected_directory));
            match safe_path {
                Some(path) => (mode, command_for_script(path, &[])),
                None => (mode, Vec::new()),
            }
        }
        mode if !mode.is_empty() => (mode.to_owned(), Vec::new()),
        _ => ("legacy".into(), Vec::new()),
    }
}

fn clock_label(now: SystemTime) -> String {
    let seconds = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() % 86_400;
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3_600,
        (seconds % 3_600) / 60,
        seconds % 60
    )
}

fn current_clock_label() -> String {
    #[cfg(unix)]
    {
        if let Ok(output) = Command::new("date").arg("+%H:%M:%S").output() {
            let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if output.status.success()
                && value.len() == 8
                && value.chars().enumerate().all(|(index, character)| {
                    matches!(index, 2 | 5) && character == ':'
                        || !matches!(index, 2 | 5) && character.is_ascii_digit()
                })
            {
                return value;
            }
        }
    }
    clock_label(SystemTime::now())
}

fn format_elapsed(seconds: u64) -> String {
    if seconds < 120 {
        format!("{seconds}s")
    } else {
        format!("{:.1}m", seconds as f64 / 60.0)
    }
}

pub fn persist_with_context(history: &JobHistory, context: &str) -> Result<()> {
    history
        .persist()
        .with_context(|| format!("persist job history after {context}"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn loads_newest_first_and_reconciles_running_rows() {
        let data = TempDir::new().unwrap();
        let repository = repository_fixture();
        state_store::save_jobs_in(
            data.path(),
            &[
                persisted("old", "0", "", ""),
                persisted("new", JOB_RUNNING_EXIT, "run", "model-a"),
            ],
        )
        .unwrap();

        let (history, notice) = JobHistory::load_in(data.path(), repository.path()).unwrap();
        assert_eq!(history.records()[0].name, "new");
        assert_eq!(history.records()[0].status, "interrupted");
        assert_eq!(history.records()[0].command[2], "model-a");
        assert_eq!(notice.reconciled_running, 1);
        let normalized = state_store::load_jobs_in(data.path()).unwrap();
        assert_eq!(normalized.reconciled_running, 0);
    }

    #[test]
    fn begin_finish_and_persist_keep_legacy_order_and_shape() {
        let data = TempDir::new().unwrap();
        let mut history = JobHistory::default();
        let first = history.begin(
            "first".into(),
            "bench".into(),
            vec!["bash".into(), "/repo/bench-models/bench-a.sh".into()],
            "bench".into(),
            "/repo/bench-models/bench-a.sh".into(),
        );
        history.finish(first, 0);
        history.begin(
            "second".into(),
            "download".into(),
            vec!["download".into()],
            "download".into(),
            String::new(),
        );
        history.persist_in(data.path()).unwrap();

        let persisted = state_store::load_jobs_in(data.path()).unwrap();
        assert_eq!(persisted.jobs[0].name, "first");
        assert_eq!(persisted.jobs[0].exit, "0");
        assert_eq!(persisted.jobs[1].name, "second");
        assert_eq!(persisted.jobs[1].exit, JOB_INTERRUPTED_EXIT);
    }

    #[test]
    fn only_reconstructs_contained_scripts_in_the_expected_directory() {
        let repository = repository_fixture();
        let bench = repository.path().join("bench-models/bench-a.sh");
        fs::write(&bench, "#!/bin/sh\n").unwrap();
        let good = persisted("good", "0", "bench", &bench.to_string_lossy());
        let (kind, command) = retry_command(&good, repository.path());
        assert_eq!(kind, "bench");
        let canonical_bench = fs::canonicalize(&bench).unwrap();
        assert_eq!(command, ["bash", canonical_bench.to_str().unwrap()]);

        let outside = TempDir::new().unwrap();
        let outside_script = outside.path().join("bench-evil.sh");
        fs::write(&outside_script, "#!/bin/sh\n").unwrap();
        let bad = persisted("bad", "0", "bench", &outside_script.to_string_lossy());
        assert!(retry_command(&bad, repository.path()).1.is_empty());
    }

    #[test]
    fn formatting_matches_legacy_display_shapes() {
        assert_eq!(clock_label(UNIX_EPOCH), "00:00:00");
        assert_eq!(format_elapsed(42), "42s");
        assert_eq!(format_elapsed(150), "2.5m");
    }

    #[test]
    fn unavailable_history_cannot_overwrite_state_until_explicit_recovery() {
        let data = TempDir::new().unwrap();
        let path = state_store::jobs_path_in(data.path());
        fs::write(&path, "not json").unwrap();
        assert!(JobHistory::load_in(data.path(), repository_fixture().path()).is_err());

        let mut history = JobHistory::unavailable();
        history.begin(
            "new".into(),
            "download".into(),
            vec!["download".into()],
            "download".into(),
            String::new(),
        );
        assert!(history.persist_in(data.path()).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "not json");

        history.clear_for_recovery();
        history.persist_in(data.path()).unwrap();
        assert_eq!(
            state_store::load_jobs_in(data.path()).unwrap().jobs.len(),
            0
        );
    }

    fn persisted(name: &str, exit: &str, mode: &str, script_path: &str) -> PersistedJob {
        PersistedJob {
            name: name.into(),
            started: "12:34:56".into(),
            elapsed: "1s".into(),
            exit: exit.into(),
            mode: mode.into(),
            script_path: script_path.into(),
        }
    }

    fn repository_fixture() -> TempDir {
        let repository = TempDir::new().unwrap();
        fs::create_dir(repository.path().join("bench-models")).unwrap();
        fs::create_dir(repository.path().join("maintenance")).unwrap();
        repository
    }
}
