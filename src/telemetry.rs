use std::{
    collections::HashSet,
    process::{Command, Stdio},
};

use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct ResourceSnapshot {
    pub process_count: usize,
    pub cpu_percent: f64,
    pub rss_kib: u64,
    pub gpu_memory_mib: Option<u64>,
}

impl ResourceSnapshot {
    pub fn render(&self, subject: &str, elapsed_seconds: Option<u64>) -> String {
        let gpu = self
            .gpu_memory_mib
            .map(|memory| format!("{memory} MiB"))
            .unwrap_or_else(|| "n/a".to_owned());
        let elapsed = elapsed_seconds.map_or_else(String::new, |seconds| {
            format!(" elapsed={:02}:{:02}", seconds / 60, seconds % 60)
        });
        format!(
            "Resources: {subject}={} cpu={:.1}% ram={:.1} MiB gpu={gpu}{elapsed}",
            self.process_count,
            self.cpu_percent,
            self.rss_kib as f64 / 1024.0,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ProcessSample {
    pid: u32,
    parent_pid: u32,
    process_group: u32,
    cpu_percent: f64,
    rss_kib: u64,
    command: String,
}

pub fn snapshot_process_group(process_group: u32) -> Result<ResourceSnapshot> {
    let processes = read_process_table()?;
    let pids = processes
        .iter()
        .filter(|process| process.process_group == process_group)
        .map(|process| process.pid)
        .collect::<HashSet<_>>();
    Ok(aggregate(&processes, &pids))
}

pub fn snapshot_descendants(parent_pid: u32) -> Result<ResourceSnapshot> {
    let processes = read_process_table()?;
    let pids = descendant_pids(&processes, parent_pid);
    Ok(aggregate(&processes, &pids))
}

pub fn find_process_named(name: &str) -> Result<Option<u32>> {
    let processes = read_process_table()?;
    Ok(processes
        .iter()
        .find(|process| command_name(&process.command) == name)
        .map(|process| process.pid))
}

fn read_process_table() -> Result<Vec<ProcessSample>> {
    let output = Command::new("ps")
        .args(["-axo", "pid=,ppid=,pgid=,pcpu=,rss=,comm="])
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .context("run ps for resource telemetry")?;
    anyhow::ensure!(output.status.success(), "ps exited with {}", output.status);
    Ok(parse_process_table(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

fn parse_process_table(output: &str) -> Vec<ProcessSample> {
    output.lines().filter_map(parse_process_row).collect()
}

fn parse_process_row(row: &str) -> Option<ProcessSample> {
    let mut fields = row.split_whitespace();
    let pid = fields.next()?.parse().ok()?;
    let parent_pid = fields.next()?.parse().ok()?;
    let process_group = fields.next()?.parse().ok()?;
    let cpu_percent = fields.next()?.parse().ok()?;
    let rss_kib = fields.next()?.parse().ok()?;
    let command = fields.collect::<Vec<_>>().join(" ");
    if command.is_empty() {
        return None;
    }
    Some(ProcessSample {
        pid,
        parent_pid,
        process_group,
        cpu_percent,
        rss_kib,
        command,
    })
}

fn descendant_pids(processes: &[ProcessSample], parent_pid: u32) -> HashSet<u32> {
    let mut descendants = HashSet::new();
    loop {
        let before = descendants.len();
        for process in processes {
            if process.parent_pid == parent_pid || descendants.contains(&process.parent_pid) {
                descendants.insert(process.pid);
            }
        }
        if descendants.len() == before {
            return descendants;
        }
    }
}

fn aggregate(processes: &[ProcessSample], pids: &HashSet<u32>) -> ResourceSnapshot {
    let cpu_percent = processes
        .iter()
        .filter(|process| pids.contains(&process.pid))
        .map(|process| process.cpu_percent)
        .sum();
    let rss_kib = processes
        .iter()
        .filter(|process| pids.contains(&process.pid))
        .map(|process| process.rss_kib)
        .sum();
    ResourceSnapshot {
        process_count: pids.len(),
        cpu_percent,
        rss_kib,
        gpu_memory_mib: gpu_memory_mib(pids),
    }
}

fn gpu_memory_mib(pids: &HashSet<u32>) -> Option<u64> {
    if pids.is_empty() {
        return None;
    }
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(parse_gpu_memory(
        &String::from_utf8_lossy(&output.stdout),
        pids,
    ))
}

fn parse_gpu_memory(output: &str, pids: &HashSet<u32>) -> u64 {
    output
        .lines()
        .filter_map(|row| {
            let (pid, memory) = row.split_once(',')?;
            let pid = pid.trim().parse::<u32>().ok()?;
            let memory = memory.trim().parse::<u64>().ok()?;
            pids.contains(&pid).then_some(memory)
        })
        .sum()
}

fn command_name(command: &str) -> &str {
    command.rsplit('/').next().unwrap_or(command)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_processes() -> Vec<ProcessSample> {
        parse_process_table(
            " 100 1 100 1.5 1024 /usr/bin/llama-swap\n\
             101 100 100 12.5 4096 /opt/llama-server\n\
             102 101 100 3.0 2048 /opt/worker\n\
             200 1 200 99.0 9999 /bin/other\n\
             malformed row\n",
        )
    }

    #[test]
    fn parses_process_rows_and_skips_invalid_input() {
        let processes = sample_processes();
        assert_eq!(processes.len(), 4);
        assert_eq!(processes[0].pid, 100);
        assert_eq!(processes[1].cpu_percent, 12.5);
        assert_eq!(processes[2].command, "/opt/worker");
    }

    #[test]
    fn discovers_all_descendant_generations() {
        let descendants = descendant_pids(&sample_processes(), 100);
        assert_eq!(descendants, HashSet::from([101, 102]));
    }

    #[test]
    fn aggregates_only_selected_processes() {
        let processes = sample_processes();
        let snapshot = aggregate_without_gpu(&processes, &HashSet::from([100, 101, 102]));
        assert_eq!(snapshot.process_count, 3);
        assert!((snapshot.cpu_percent - 17.0).abs() < f64::EPSILON);
        assert_eq!(snapshot.rss_kib, 7168);
        assert_eq!(
            snapshot.render("procs", Some(65)),
            "Resources: procs=3 cpu=17.0% ram=7.0 MiB gpu=n/a elapsed=01:05"
        );
    }

    #[test]
    fn sums_nvidia_rows_for_selected_pids() {
        let pids = HashSet::from([101, 102]);
        assert_eq!(
            parse_gpu_memory("101, 1000\n102, 250\n200, 999\nbad\n", &pids),
            1250
        );
    }

    #[test]
    fn matches_executable_basename() {
        assert_eq!(command_name("/opt/bin/llama-swap"), "llama-swap");
        assert_eq!(command_name("llama-swap"), "llama-swap");
    }

    fn aggregate_without_gpu(processes: &[ProcessSample], pids: &HashSet<u32>) -> ResourceSnapshot {
        ResourceSnapshot {
            process_count: pids.len(),
            cpu_percent: processes
                .iter()
                .filter(|process| pids.contains(&process.pid))
                .map(|process| process.cpu_percent)
                .sum(),
            rss_kib: processes
                .iter()
                .filter(|process| pids.contains(&process.pid))
                .map(|process| process.rss_kib)
                .sum(),
            gpu_memory_mib: None,
        }
    }
}
