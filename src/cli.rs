//! Command-line launcher for the Rust L3MS application.

use std::env;
use std::ffi::{OsStr, OsString};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use clap::{ArgGroup, Parser, ValueEnum};

use crate::{
    llama_swap::{SwapClient, SwapModel},
    media::{interactive_media, load_manifest, print_profile_list},
    script_store::{collect_scripts_in, ScriptMode},
};

#[derive(Debug, Parser)]
#[command(
    name = "l3ms",
    version,
    about = "L3MS launcher (TUI + interactive model, bench, and media CLI)",
    group(
        ArgGroup::new("action")
            .args(["run", "bench", "media", "list", "quickstart"])
            .multiple(false)
    )
)]
struct Cli {
    /// Interactively select and load a llama-swap model.
    #[arg(
        long,
        value_name = "FILTER",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    run: Option<String>,

    /// Interactively select and execute a benchmark script.
    #[arg(
        long,
        value_name = "FILTER",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    bench: Option<String>,

    /// Interactively select and run a media-generation profile.
    #[arg(
        long,
        value_name = "FILTER",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    media: Option<String>,

    /// List llama-swap models, benchmark scripts, media profiles, or all.
    #[arg(long, value_name = "MODE")]
    list: Option<ListMode>,

    /// Print a quick-start guide and exit.
    #[arg(long)]
    quickstart: bool,

    /// Shell-style arguments appended to the selected benchmark or media script.
    #[arg(
        long,
        default_value = "",
        value_name = "ARGS",
        allow_hyphen_values = true
    )]
    extra: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ListMode {
    Run,
    Bench,
    Media,
    All,
}

#[derive(Debug, PartialEq, Eq)]
struct CommandSpec {
    program: OsString,
    args: Vec<OsString>,
}

/// Parse process arguments and launch the requested mode.
pub fn run() -> Result<u8> {
    dispatch(Cli::parse())
}

fn dispatch(cli: Cli) -> Result<u8> {
    if cli.quickstart {
        print_quickstart();
        return Ok(0);
    }

    if let Some(mode) = cli.list {
        if !cli.extra.trim().is_empty() {
            bail!("--extra is only valid with --bench or --media");
        }
        return list(mode);
    }

    if let Some(filter) = cli.run {
        if !cli.extra.trim().is_empty() {
            bail!("--extra is only valid with --bench or --media");
        }
        return interactive_run(&filter);
    }

    if let Some(filter) = cli.bench {
        return interactive_bench(&filter, &cli.extra);
    }

    if let Some(filter) = cli.media {
        return interactive_media(&repository_root()?, &filter, &cli.extra);
    }

    if !cli.extra.trim().is_empty() {
        bail!("--extra is only valid with --bench or --media");
    }

    crate::app::run_tui()?;
    Ok(0)
}

fn print_quickstart() {
    println!("L3MS quick start");
    println!();
    println!("  1) Open the TUI");
    println!("     l3ms");
    println!();
    println!("  2) Load a llama-swap model without entering the TUI");
    println!("     l3ms --run");
    println!();
    println!("  3) Run a benchmark script");
    println!("     l3ms --bench");
    println!();
    println!("  4) Discover available models and scripts");
    println!("     l3ms --list all");
    println!();
    println!("  5) Generate music or video with a configured media runtime");
    println!(r#"     l3ms --media --extra '--prompt "a warm analog synth loop" --instrumental'"#);
    println!();
    println!("  6) Pass extra arguments to a selected benchmark or media profile");
    println!(r#"     l3ms --bench qwen --extra "--ctx-size 32768""#);
}

fn list(mode: ListMode) -> Result<u8> {
    match mode {
        ListMode::Run => {
            let client = SwapClient::from_env()?;
            print_model_list(&client.list_models()?);
        }
        ListMode::Bench => {
            let root = repository_root()?;
            print_script_list(&root, &collect_bench_scripts(&root)?);
        }
        ListMode::Media => {
            let root = repository_root()?;
            let manifest = load_manifest(&root)?;
            print_profile_list(&manifest.profiles);
        }
        ListMode::All => {
            let client = SwapClient::from_env()?;
            print_model_list(&client.list_models()?);
            println!();
            let root = repository_root()?;
            print_script_list(&root, &collect_bench_scripts(&root)?);
            println!();
            let manifest = load_manifest(&root)?;
            print_profile_list(&manifest.profiles);
        }
    }
    Ok(0)
}

fn interactive_run(filter: &str) -> Result<u8> {
    let client = SwapClient::from_env()?;
    let models = filter_models(client.list_models()?, filter);
    if models.is_empty() {
        eprintln!("No llama-swap models found for filter: {filter:?}");
        return Ok(1);
    }

    print_model_list(&models);
    let Some(index) = choose_index(models.len(), "model")? else {
        println!("Cancelled.");
        return Ok(0);
    };

    let model = &models[index];
    println!("POST {}/models/load  model={}", client.base_url(), model.id);
    println!("{}", client.load_model(&model.id)?);
    Ok(0)
}

fn interactive_bench(filter: &str, extra: &str) -> Result<u8> {
    let extra_args = parse_extra_args(extra)?;
    let root = repository_root()?;
    let scripts = filter_scripts(collect_bench_scripts(&root)?, &root, filter);
    if scripts.is_empty() {
        eprintln!("No benchmark scripts found for filter: {filter:?}");
        return Ok(1);
    }

    print_script_list(&root, &scripts);
    let Some(index) = choose_index(scripts.len(), "script")? else {
        println!("Cancelled.");
        return Ok(0);
    };

    let command = command_for_script(&scripts[index], &extra_args);
    println!("$ {}", format_command(&command));
    let status = Command::new(&command.program)
        .args(&command.args)
        .current_dir(&root)
        .status()
        .with_context(|| format!("failed to execute {}", scripts[index].display()))?;
    let code = status.code().unwrap_or(1);
    println!("Exited with code {code}");
    Ok(u8::try_from(code).unwrap_or(1))
}

fn collect_bench_scripts(root: &Path) -> Result<Vec<PathBuf>> {
    collect_scripts_in(root, ScriptMode::Bench)
}

fn filter_models(models: Vec<SwapModel>, filter: &str) -> Vec<SwapModel> {
    let filter = filter.trim().to_ascii_lowercase();
    if filter.is_empty() {
        return models;
    }

    models
        .into_iter()
        .filter(|model| {
            [&model.id, &model.name, &model.description]
                .iter()
                .any(|value| value.to_ascii_lowercase().contains(&filter))
        })
        .collect()
}

fn filter_scripts(scripts: Vec<PathBuf>, root: &Path, filter: &str) -> Vec<PathBuf> {
    let filter = filter.trim().to_ascii_lowercase();
    if filter.is_empty() {
        return scripts;
    }

    scripts
        .into_iter()
        .filter(|script| {
            let relative = script.strip_prefix(root).unwrap_or(script);
            relative
                .to_string_lossy()
                .to_ascii_lowercase()
                .contains(&filter)
                || pretty_name(script).to_ascii_lowercase().contains(&filter)
        })
        .collect()
}

fn pretty_name(path: &Path) -> String {
    let stem = path.file_stem().and_then(OsStr::to_str).unwrap_or_default();
    for prefix in ["bench-ik-llama-cpp-", "bench-llama-cpp-", "bench-"] {
        if let Some(name) = stem.strip_prefix(prefix) {
            return name.to_owned();
        }
    }
    stem.to_owned()
}

fn print_model_list(models: &[SwapModel]) {
    if models.is_empty() {
        println!("No llama-swap models found");
        return;
    }

    println!("RUN models ({}):", models.len());
    for (index, model) in models.iter().enumerate() {
        println!(
            "  {:>2}. {:<10} {:<36} {}",
            index + 1,
            model.state,
            model.id,
            model.name
        );
    }
}

fn print_script_list(root: &Path, scripts: &[PathBuf]) {
    if scripts.is_empty() {
        println!("No benchmark scripts found");
        return;
    }

    println!("BENCH scripts ({}):", scripts.len());
    for (index, script) in scripts.iter().enumerate() {
        let relative = script.strip_prefix(root).unwrap_or(script);
        println!(
            "  {:>2}. {:<36} {}",
            index + 1,
            pretty_name(script),
            relative.display()
        );
    }
}

fn choose_index(count: usize, item_name: &str) -> Result<Option<usize>> {
    println!("Select {item_name} index, or 'q' to quit.");
    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        print!("> ");
        io::stdout().flush().context("failed to flush prompt")?;
        input.clear();
        let bytes = stdin
            .read_line(&mut input)
            .context("failed to read selection")?;
        if bytes == 0 {
            return Ok(None);
        }

        let value = input.trim();
        if value.is_empty() && count == 1 {
            return Ok(Some(0));
        }
        if matches!(value.to_ascii_lowercase().as_str(), "q" | "quit" | "exit") {
            return Ok(None);
        }
        if let Ok(index) = value.parse::<usize>() {
            if (1..=count).contains(&index) {
                return Ok(Some(index - 1));
            }
        }
        println!("Invalid selection: {value:?}. Enter 1-{count} or q.");
    }
}

fn parse_extra_args(raw: &str) -> Result<Vec<String>> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    shell_words::split(raw).context("invalid --extra value")
}

fn command_for_script(path: &Path, extra_args: &[String]) -> CommandSpec {
    let mut args = Vec::with_capacity(extra_args.len() + 1);
    args.push(path.as_os_str().to_owned());
    args.extend(extra_args.iter().map(OsString::from));
    CommandSpec {
        program: OsString::from("bash"),
        args,
    }
}

fn format_command(command: &CommandSpec) -> String {
    std::iter::once(command.program.as_os_str())
        .chain(command.args.iter().map(OsString::as_os_str))
        .map(quote_for_display)
        .collect::<Vec<_>>()
        .join(" ")
}

fn quote_for_display(value: &OsStr) -> String {
    let value = value.to_string_lossy();
    if !value.is_empty()
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_@%+=:,./-".contains(character))
    {
        return value.into_owned();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn repository_root() -> Result<PathBuf> {
    if let Some(root) = env::var_os("L3MS_ROOT").map(PathBuf::from) {
        if is_repository_root(&root) {
            return Ok(root);
        }
        bail!("L3MS_ROOT is not an L3MS repository: {}", root.display());
    }

    if let Ok(executable) = env::current_exe() {
        if let Some(root) = find_repository_root(&executable) {
            return Ok(root);
        }
    }
    if let Ok(cwd) = env::current_dir() {
        if let Some(root) = find_repository_root(&cwd) {
            return Ok(root);
        }
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(root) = find_repository_root(manifest_dir) {
        return Ok(root);
    }

    bail!("could not locate the L3MS repository; run from the checkout or set L3MS_ROOT")
}

fn find_repository_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|candidate| is_repository_root(candidate))
        .map(Path::to_path_buf)
}

fn is_repository_root(path: &Path) -> bool {
    path.join("Cargo.toml").is_file()
        && path.join("llama-swap.yaml").is_file()
        && path.join("bench-models").is_dir()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str, name: &str, description: &str) -> SwapModel {
        SwapModel {
            id: id.into(),
            state: "unknown".into(),
            name: name.into(),
            description: description.into(),
        }
    }

    #[test]
    fn parses_optional_filters_and_modes() {
        let cli = Cli::try_parse_from(["l3ms", "--run"]).unwrap();
        assert_eq!(cli.run.as_deref(), Some(""));

        let cli = Cli::try_parse_from(["l3ms", "--run", "qwen"]).unwrap();
        assert_eq!(cli.run.as_deref(), Some("qwen"));

        let cli = Cli::try_parse_from(["l3ms", "--bench"]).unwrap();
        assert_eq!(cli.bench.as_deref(), Some(""));

        let cli = Cli::try_parse_from(["l3ms", "--media", "h3"]).unwrap();
        assert_eq!(cli.media.as_deref(), Some("h3"));

        let cli = Cli::try_parse_from(["l3ms", "--list", "all"]).unwrap();
        assert_eq!(cli.list, Some(ListMode::All));

        let cli = Cli::try_parse_from(["l3ms", "--list", "media"]).unwrap();
        assert_eq!(cli.list, Some(ListMode::Media));

        let cli = Cli::try_parse_from(["l3ms", "--bench", "qwen", "--extra", "--ctx-size 32768"])
            .unwrap();
        assert_eq!(cli.extra, "--ctx-size 32768");

        assert!(Cli::try_parse_from(["l3ms", "--run", "--bench"]).is_err());
        assert!(Cli::try_parse_from(["l3ms", "--list", "invalid"]).is_err());
    }

    #[test]
    fn parses_shell_style_extra_arguments() {
        assert_eq!(
            parse_extra_args(r#"--ctx-size 32768 --prompt "hello world""#).unwrap(),
            ["--ctx-size", "32768", "--prompt", "hello world"]
        );
        assert!(parse_extra_args("'unterminated").is_err());
        assert!(parse_extra_args("  ").unwrap().is_empty());
    }

    #[test]
    fn filters_models_case_insensitively_across_metadata() {
        let models = vec![
            model("qwen3-coder", "Coder", "large coding model"),
            model("gemma", "Vision Model", "multimodal"),
        ];

        assert_eq!(filter_models(models.clone(), "QWEN")[0].id, "qwen3-coder");
        assert_eq!(filter_models(models.clone(), "vision")[0].id, "gemma");
        assert_eq!(filter_models(models.clone(), "MULTI")[0].id, "gemma");
        assert_eq!(filter_models(models.clone(), "").len(), 2);
        assert!(filter_models(models, "missing").is_empty());
    }

    #[test]
    fn filters_scripts_by_relative_path() {
        let root = Path::new("/repo");
        let scripts = vec![
            root.join("bench-models/bench-llama-cpp-qwen.sh"),
            root.join("bench-models/bench-llama-cpp-gemma.sh"),
        ];
        assert_eq!(
            filter_scripts(scripts.clone(), root, "QWEN"),
            vec![scripts[0].clone()]
        );
        assert_eq!(
            filter_scripts(scripts.clone(), root, "bench-models").len(),
            2
        );
        assert_eq!(filter_scripts(scripts, root, "").len(), 2);
    }

    #[test]
    fn creates_readable_benchmark_names() {
        assert_eq!(
            pretty_name(Path::new("bench-models/bench-llama-cpp-qwen3.sh")),
            "qwen3"
        );
        assert_eq!(
            pretty_name(Path::new("bench-models/bench-ik-llama-cpp-gpt-oss.sh")),
            "gpt-oss"
        );
        assert_eq!(
            pretty_name(Path::new("bench-models/bench-custom.sh")),
            "custom"
        );
    }

    #[test]
    fn builds_bash_command_without_reparsing_arguments() {
        let script = Path::new("/repo/bench-models/bench model.sh");
        let extra = vec!["--prompt".into(), "hello world".into()];
        let command = command_for_script(script, &extra);

        assert_eq!(command.program, OsString::from("bash"));
        assert_eq!(
            command.args,
            vec![
                OsString::from("/repo/bench-models/bench model.sh"),
                OsString::from("--prompt"),
                OsString::from("hello world")
            ]
        );
        assert_eq!(
            format_command(&command),
            "bash '/repo/bench-models/bench model.sh' --prompt 'hello world'"
        );
    }

    #[test]
    fn finds_repository_above_target_binary() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let executable = root.join("target/debug/deps/l3ms-test");
        assert_eq!(find_repository_root(&executable).as_deref(), Some(root));
    }
}
