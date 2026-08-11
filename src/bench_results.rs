//! Bounded discovery and comparison of benchmark result records.
//!
//! Benchmark runners write JSONL under `bench-models/logs/results/`. Older
//! hand-written benchmark notes may still be Markdown, so the reader accepts
//! both formats without allowing an untrusted result file to consume
//! unbounded memory.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const MAX_RESULT_FILE_BYTES: usize = 8 * 1024 * 1024;
const MAX_RESULT_LINE_BYTES: usize = 512 * 1024;
const MAX_RESULTS: usize = 10_000;

/// A normalized benchmark result record. Unknown JSON fields are intentionally
/// ignored so newer runner metadata remains readable by older binaries.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BenchResult {
    pub ts: String,
    pub model_key: String,
    pub model: String,
    pub backend: String,
    pub strategy: String,
    pub ngl: Option<u64>,
    pub n_cpu_moe: Option<u64>,
    pub override_tensor: String,
    pub fit_ctx: Option<u64>,
    pub fit_target: Option<u64>,
    pub ctx: String,
    pub ctk: String,
    pub ctv: String,
    pub threads: Option<u64>,
    pub repetitions: Option<u64>,
    pub pp_tokens: Option<u64>,
    pub pp_ts: Option<f64>,
    pub pp_std: Option<f64>,
    pub tg_tokens: Option<u64>,
    pub tg_ts: Option<f64>,
    pub tg_std: Option<f64>,
    pub git_sha: String,
    pub llama_version: String,
    pub log_file: String,
    pub notes: String,
    #[serde(skip)]
    pub source: PathBuf,
}

/// A malformed row or file omitted from a result listing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchResultIssue {
    pub source: PathBuf,
    pub line: Option<usize>,
    pub error: String,
}

/// Bounded result discovery output suitable for a UI or CLI browser.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BenchResultLoad {
    pub results: Vec<BenchResult>,
    pub issues: Vec<BenchResultIssue>,
    pub files_read: usize,
    pub truncated_rows: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultSort {
    TimestampDesc,
    ModelAsc,
    PromptThroughputDesc,
    GenerationThroughputDesc,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResultComparison {
    pub left: BenchResult,
    pub right: BenchResult,
    pub metrics: BTreeMap<String, MetricDelta>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricDelta {
    pub left: Option<f64>,
    pub right: Option<f64>,
    pub delta: Option<f64>,
}

/// Discover results in the canonical repository location and the historical
/// `bench-results/` location used by the original TODO.
pub fn load_results(root: impl AsRef<Path>) -> Result<BenchResultLoad> {
    load_results_in(root.as_ref())
}

pub fn load_results_in(root: &Path) -> Result<BenchResultLoad> {
    let mut files = Vec::new();
    for relative in ["bench-models/logs/results", "bench-results"] {
        let directory = root.join(relative);
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to read result directory {}", directory.display())
                })
            }
        };
        for entry in entries {
            let entry = entry.with_context(|| {
                format!("failed to inspect result directory {}", directory.display())
            })?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            if matches!(
                extension.to_ascii_lowercase().as_str(),
                "jsonl" | "json" | "md" | "markdown"
            ) {
                files.push(path);
            }
        }
    }
    files.sort();
    files.dedup();

    let mut load = BenchResultLoad::default();
    for path in files {
        load.files_read += 1;
        let bytes = fs::read(&path)
            .with_context(|| format!("failed to read benchmark result {}", path.display()))?;
        if bytes.len() > MAX_RESULT_FILE_BYTES {
            load.issues.push(BenchResultIssue {
                source: path,
                line: None,
                error: format!("result file exceeds {MAX_RESULT_FILE_BYTES} bytes"),
            });
            continue;
        }
        let source = path.clone();
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default();
        if matches!(extension.to_ascii_lowercase().as_str(), "md" | "markdown") {
            parse_markdown_results(&String::from_utf8_lossy(&bytes), &source, &mut load);
        } else {
            parse_json_results(&bytes, &source, &mut load);
        }
        if load.results.len() >= MAX_RESULTS {
            load.truncated_rows += load.results.len() - MAX_RESULTS;
            load.results.truncate(MAX_RESULTS);
            break;
        }
    }
    sort_results(&mut load.results, ResultSort::TimestampDesc);
    Ok(load)
}

pub fn sort_results(results: &mut [BenchResult], sort: ResultSort) {
    results.sort_by(|left, right| {
        let ordering = match sort {
            ResultSort::TimestampDesc => right.ts.cmp(&left.ts),
            ResultSort::ModelAsc => left.model_key.cmp(&right.model_key),
            ResultSort::PromptThroughputDesc => compare_optional_f64(right.pp_ts, left.pp_ts),
            ResultSort::GenerationThroughputDesc => compare_optional_f64(right.tg_ts, left.tg_ts),
        };
        ordering
            .then_with(|| left.model_key.cmp(&right.model_key))
            .then_with(|| left.strategy.cmp(&right.strategy))
            .then_with(|| left.source.cmp(&right.source))
    });
}

pub fn compare_results(left: &BenchResult, right: &BenchResult) -> ResultComparison {
    let mut metrics = BTreeMap::new();
    for (name, left_value, right_value) in [
        ("pp_ts", left.pp_ts, right.pp_ts),
        ("tg_ts", left.tg_ts, right.tg_ts),
        ("pp_std", left.pp_std, right.pp_std),
        ("tg_std", left.tg_std, right.tg_std),
    ] {
        metrics.insert(
            name.to_owned(),
            MetricDelta {
                left: left_value,
                right: right_value,
                delta: left_value.zip(right_value).map(|(a, b)| b - a),
            },
        );
    }
    ResultComparison {
        left: left.clone(),
        right: right.clone(),
        metrics,
    }
}

fn parse_json_results(bytes: &[u8], source: &Path, load: &mut BenchResultLoad) {
    for (line_number, line) in bytes.split(|byte| *byte == b'\n').enumerate() {
        if line.len() > MAX_RESULT_LINE_BYTES {
            load.issues.push(BenchResultIssue {
                source: source.to_owned(),
                line: Some(line_number + 1),
                error: format!("result row exceeds {MAX_RESULT_LINE_BYTES} bytes"),
            });
            continue;
        }
        let line = String::from_utf8_lossy(line).trim().to_owned();
        if line.is_empty() {
            continue;
        }
        let value: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                load.issues.push(BenchResultIssue {
                    source: source.to_owned(),
                    line: Some(line_number + 1),
                    error: format!("invalid JSON: {error}"),
                });
                continue;
            }
        };
        let Some(object) = value.as_object() else {
            load.issues.push(BenchResultIssue {
                source: source.to_owned(),
                line: Some(line_number + 1),
                error: "result row must be a JSON object".to_owned(),
            });
            continue;
        };
        let mut result: BenchResult = match serde_json::from_value(Value::Object(object.clone())) {
            Ok(result) => result,
            Err(error) => {
                load.issues.push(BenchResultIssue {
                    source: source.to_owned(),
                    line: Some(line_number + 1),
                    error: format!("invalid result row: {error}"),
                });
                continue;
            }
        };
        result.source = source.to_owned();
        load.results.push(result);
    }
}

fn parse_markdown_results(text: &str, source: &Path, load: &mut BenchResultLoad) {
    let mut current = BenchResult {
        model_key: source
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_owned(),
        source: source.to_owned(),
        ..BenchResult::default()
    };
    for (line_number, line) in text.lines().enumerate() {
        if line.len() > MAX_RESULT_LINE_BYTES {
            load.issues.push(BenchResultIssue {
                source: source.to_owned(),
                line: Some(line_number + 1),
                error: format!("result row exceeds {MAX_RESULT_LINE_BYTES} bytes"),
            });
            continue;
        }
        if !line.trim_start().starts_with('|') || line.contains("---") {
            continue;
        }
        let cells = line
            .trim()
            .trim_matches('|')
            .split('|')
            .map(str::trim)
            .collect::<Vec<_>>();
        for index in 0..cells.len().saturating_sub(1) {
            let Some((kind, tokens)) = parse_test_cell(cells[index]) else {
                continue;
            };
            let Some(speed) = parse_speed(cells[index + 1]) else {
                continue;
            };
            if kind == "pp" {
                current.pp_tokens = Some(tokens);
                current.pp_ts = Some(speed.0);
                current.pp_std = speed.1;
            } else {
                current.tg_tokens = Some(tokens);
                current.tg_ts = Some(speed.0);
                current.tg_std = speed.1;
            }
        }
    }
    if current.pp_ts.is_some() || current.tg_ts.is_some() {
        load.results.push(current);
    } else {
        load.issues.push(BenchResultIssue {
            source: source.to_owned(),
            line: None,
            error: "no pp/tg throughput rows found".to_owned(),
        });
    }
}

fn parse_test_cell(cell: &str) -> Option<(&str, u64)> {
    let mut parts = cell.split_whitespace();
    let kind = parts.next()?;
    if kind != "pp" && kind != "tg" {
        return None;
    }
    Some((kind, parts.next()?.parse().ok()?))
}

fn parse_speed(cell: &str) -> Option<(f64, Option<f64>)> {
    let mut parts = cell.split('±');
    let value = parts.next()?.trim().parse().ok()?;
    let std = parts.next().and_then(|part| part.trim().parse().ok());
    Some((value, std))
}

fn compare_optional_f64(left: Option<f64>, right: Option<f64>) -> Ordering {
    left.zip(right).map_or_else(
        || right.is_some().cmp(&left.is_some()),
        |(a, b)| a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    )
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn loads_jsonl_and_reports_malformed_rows() {
        let temp = TempDir::new().unwrap();
        let directory = temp.path().join("bench-models/logs/results");
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("qwen.jsonl"),
            "{\"ts\":\"2026-08-10T00:00:00Z\",\"model_key\":\"qwen\",\"pp_ts\":12.5,\"tg_ts\":4.5}\nnot json\n",
        )
        .unwrap();
        let load = load_results_in(temp.path()).unwrap();
        assert_eq!(load.results.len(), 1);
        assert_eq!(load.results[0].model_key, "qwen");
        assert_eq!(load.results[0].pp_ts, Some(12.5));
        assert_eq!(load.issues.len(), 1);
    }

    #[test]
    fn parses_markdown_pp_and_tg_rows() {
        let temp = TempDir::new().unwrap();
        let directory = temp.path().join("bench-results");
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("run.md"),
            "| test | speed |\n| --- | --- |\n| pp 512 | 123.4 ± 1.2 |\n| tg 128 | 45.6 ± 0.7 |\n",
        )
        .unwrap();
        let load = load_results_in(temp.path()).unwrap();
        assert_eq!(load.results.len(), 1);
        assert_eq!(load.results[0].pp_tokens, Some(512));
        assert_eq!(load.results[0].tg_ts, Some(45.6));
    }

    #[test]
    fn comparison_uses_none_for_missing_metrics() {
        let left = BenchResult {
            pp_ts: Some(10.0),
            ..BenchResult::default()
        };
        let right = BenchResult::default();
        let comparison = compare_results(&left, &right);
        assert_eq!(comparison.metrics["pp_ts"].delta, None);
        assert_eq!(comparison.metrics["tg_ts"].left, None);
    }
}
