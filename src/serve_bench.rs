//! Deterministic checks for serving/benchmark flag drift.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const TRACKED_FLAGS: [&str; 2] = ["-ngl", "--override-tensor"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlagAudit {
    pub serving_flags: BTreeSet<String>,
    pub bench_flags: BTreeSet<String>,
    pub missing_in_bench: BTreeSet<String>,
    pub extra_in_bench: BTreeSet<String>,
    pub scripts_checked: Vec<PathBuf>,
}

impl FlagAudit {
    pub fn is_aligned(&self) -> bool {
        self.missing_in_bench.is_empty() && self.extra_in_bench.is_empty()
    }

    pub fn summary(&self) -> String {
        if self.is_aligned() {
            format!(
                "serve/bench flags aligned ({} script(s), {} tracked flag(s))",
                self.scripts_checked.len(),
                self.serving_flags.len()
            )
        } else {
            format!(
                "serve/bench drift: missing={} extra={}",
                join_flags(&self.missing_in_bench),
                join_flags(&self.extra_in_bench)
            )
        }
    }
}

pub fn audit_flags(
    serving_config: impl AsRef<Path>,
    bench_scripts: impl IntoIterator<Item = impl AsRef<Path>>,
) -> Result<FlagAudit> {
    let serving_text = fs::read_to_string(serving_config.as_ref())
        .with_context(|| format!("read serving config {}", serving_config.as_ref().display()))?;
    let serving_flags = extract_tracked_flags(&serving_text);
    let mut bench_flags = BTreeSet::new();
    let mut scripts_checked = Vec::new();
    for script in bench_scripts {
        let script = script.as_ref();
        let text = fs::read_to_string(script)
            .with_context(|| format!("read benchmark script {}", script.display()))?;
        bench_flags.extend(extract_tracked_flags(&text));
        scripts_checked.push(script.to_owned());
    }
    scripts_checked.sort();
    let missing_in_bench = serving_flags.difference(&bench_flags).cloned().collect();
    let extra_in_bench = bench_flags.difference(&serving_flags).cloned().collect();
    Ok(FlagAudit {
        serving_flags,
        bench_flags,
        missing_in_bench,
        extra_in_bench,
        scripts_checked,
    })
}

fn extract_tracked_flags(text: &str) -> BTreeSet<String> {
    TRACKED_FLAGS
        .iter()
        .filter(|flag| {
            text.lines().any(|line| {
                line.split_whitespace().any(|token| {
                    canonical_flag(
                        token.trim_matches(|character| character == '\"' || character == '\''),
                    ) == **flag
                })
            })
        })
        .map(|flag| (*flag).to_owned())
        .collect()
}

fn canonical_flag(flag: &str) -> &str {
    if flag == "-ot" {
        "--override-tensor"
    } else {
        flag
    }
}

fn join_flags(flags: &BTreeSet<String>) -> String {
    if flags.is_empty() {
        "none".to_owned()
    } else {
        flags.iter().cloned().collect::<Vec<_>>().join(",")
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn detects_missing_and_extra_flags_deterministically() {
        let temp = TempDir::new().unwrap();
        let serve = temp.path().join("llama-swap.yaml");
        let bench = temp.path().join("bench.sh");
        std::fs::write(&serve, "  -ngl 49\\n  --override-tensor foo\\n").unwrap();
        std::fs::write(&bench, "cmd -ngl 49\\n").unwrap();
        let audit = audit_flags(&serve, [&bench]).unwrap();
        assert_eq!(
            audit.missing_in_bench,
            BTreeSet::from(["--override-tensor".into()])
        );
        assert_eq!(audit.extra_in_bench, BTreeSet::new());
        assert!(!audit.is_aligned());
    }
}
