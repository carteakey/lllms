use std::{
    collections::VecDeque,
    env,
    ffi::OsString,
    fs,
    io::{self, BufRead, BufReader, Stdout},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{Duration, Instant, SystemTime},
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use crate::{
    chat::{stream_completion, ChatCompletion, ChatRequest},
    commands::{command, search_all_commands, visible_commands, CommandContext, CommandId},
    config_store::{
        load_config, load_config_strict, save_config_in, validate_config, DownloadConfig,
    },
    llama_swap::{SwapClient, SwapModel},
    script_store::{collect_scripts_in, command_for_script, ScriptMode},
    telemetry::{find_process_named, snapshot_descendants, snapshot_process_group},
};
use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{
        Block, Borders, Cell, Clear, List, ListItem, ListState, Paragraph, Row, Table, TableState,
        Tabs, Wrap,
    },
    Frame, Terminal,
};

type Tui = Terminal<CrosstermBackend<Stdout>>;

const TAB_NAMES: [&str; 7] = [
    "Workbench",
    "Model Ops",
    "Chat",
    "Model Browser",
    "Download",
    "Jobs",
    "Maintenance",
];
const BACKGROUND_QUEUE_CAPACITY: usize = 512;
const EVENTS_PER_TICK: usize = 128;
const MAX_PROCESS_LINE_BYTES: usize = 16 * 1024;
const MAX_CHAT_PREVIEW_BYTES: usize = 64 * 1024;
const TELEMETRY_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Workbench,
    ModelOps,
    Chat,
    Browser,
    Download,
    Jobs,
    Maintenance,
}

impl Tab {
    fn index(self) -> usize {
        match self {
            Self::Workbench => 0,
            Self::ModelOps => 1,
            Self::Chat => 2,
            Self::Browser => 3,
            Self::Download => 4,
            Self::Jobs => 5,
            Self::Maintenance => 6,
        }
    }

    fn from_index(index: usize) -> Self {
        match index % TAB_NAMES.len() {
            0 => Self::Workbench,
            1 => Self::ModelOps,
            2 => Self::Chat,
            3 => Self::Browser,
            4 => Self::Download,
            5 => Self::Jobs,
            _ => Self::Maintenance,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpsMode {
    Run,
    Bench,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InputMode {
    Normal,
    ModelFilter,
    BrowserPath,
    ChatMessage,
    ChatSystemPrompt,
}

#[derive(Debug, Clone)]
struct GgufFile {
    path: PathBuf,
    size: u64,
    quantization: String,
    modified: Option<SystemTime>,
}

#[derive(Debug, Clone)]
struct JobRecord {
    id: u64,
    name: String,
    kind: String,
    status: String,
    command: Vec<String>,
    started: Instant,
    exit_code: Option<i32>,
}

struct RunningProcess {
    job_id: u64,
    process_group: u32,
    child: Arc<Mutex<Option<Child>>>,
}

enum BackgroundEvent {
    Models(Result<Vec<SwapModel>, String>),
    ModelAction {
        job_id: u64,
        model_id: String,
        load: bool,
        result: Result<String, String>,
    },
    BrowserScan(Result<Vec<GgufFile>, String>),
    ChatDelta(String),
    ChatFinished(Result<ChatCompletion, String>),
    ProcessLine {
        job_id: u64,
        line: String,
    },
    ProcessFinished {
        job_id: u64,
        exit_code: i32,
    },
    Telemetry(Result<Option<String>, String>),
}

struct App {
    root: PathBuf,
    tab: Tab,
    input_mode: InputMode,
    should_quit: bool,
    show_help: bool,
    show_palette: bool,
    palette_query: String,
    palette_state: TableState,
    status: String,
    log: VecDeque<String>,
    sender: mpsc::SyncSender<BackgroundEvent>,
    receiver: mpsc::Receiver<BackgroundEvent>,

    models: Vec<SwapModel>,
    model_filter: String,
    model_state: TableState,
    loading_models: bool,
    model_action_pending: bool,
    loaded_model_id: Option<String>,

    ops_mode: OpsMode,
    bench_scripts: Vec<PathBuf>,
    bench_state: ListState,

    chat_input: String,
    chat_history: Vec<(String, String)>,
    chat_streaming: String,
    chat_pending: bool,
    chat_system_prompt: String,
    chat_temperature: f64,
    chat_max_tokens: u32,
    chat_thinking: bool,

    browser_path: String,
    browser_files: Vec<GgufFile>,
    browser_state: TableState,
    browser_scanning: bool,

    config_path: PathBuf,
    config_error: Option<String>,
    download_config: DownloadConfig,
    download_state: TableState,
    download_dirty: bool,

    maintenance_scripts: Vec<PathBuf>,
    maintenance_state: ListState,

    jobs: Vec<JobRecord>,
    jobs_state: TableState,
    next_job_id: u64,
    running_process: Option<RunningProcess>,
    telemetry: String,
    telemetry_pending: bool,
    telemetry_last_started: Option<Instant>,
}

impl App {
    fn new(root: PathBuf) -> Self {
        let (sender, receiver) = mpsc::sync_channel(BACKGROUND_QUEUE_CAPACITY);
        let config_path = root.join("model_downloader/models_config.json");
        let (download_config, config_warning) = match load_config_strict(&config_path) {
            Ok(config) => (config, None),
            Err(error) => (
                load_config(&config_path),
                Some(format!("Download config warning: {error:#}")),
            ),
        };
        let browser_path = if download_config.base_models_dir.trim().is_empty() {
            root.join("models").display().to_string()
        } else {
            download_config.base_models_dir.clone()
        };
        let mut app = Self {
            root,
            tab: Tab::Workbench,
            input_mode: InputMode::Normal,
            should_quit: false,
            show_help: false,
            show_palette: false,
            palette_query: String::new(),
            palette_state: TableState::default().with_selected(Some(0)),
            status: "Starting Rust workbench…".into(),
            log: VecDeque::new(),
            sender,
            receiver,
            models: Vec::new(),
            model_filter: String::new(),
            model_state: TableState::default().with_selected(Some(0)),
            loading_models: false,
            model_action_pending: false,
            loaded_model_id: None,
            ops_mode: OpsMode::Run,
            bench_scripts: Vec::new(),
            bench_state: ListState::default().with_selected(Some(0)),
            chat_input: String::new(),
            chat_history: Vec::new(),
            chat_streaming: String::new(),
            chat_pending: false,
            chat_system_prompt: String::new(),
            chat_temperature: 0.8,
            chat_max_tokens: 2048,
            chat_thinking: false,
            browser_path,
            browser_files: Vec::new(),
            browser_state: TableState::default().with_selected(Some(0)),
            browser_scanning: false,
            config_path,
            config_error: config_warning.clone(),
            download_config,
            download_state: TableState::default().with_selected(Some(0)),
            download_dirty: false,
            maintenance_scripts: Vec::new(),
            maintenance_state: ListState::default().with_selected(Some(0)),
            jobs: Vec::new(),
            jobs_state: TableState::default().with_selected(Some(0)),
            next_job_id: 1,
            running_process: None,
            telemetry: "Resources: idle".into(),
            telemetry_pending: false,
            telemetry_last_started: None,
        };
        if let Some(warning) = config_warning {
            app.status = "Download config needs attention".into();
            app.push_log(warning);
        }
        app.refresh_local_inventories();
        app.refresh_models();
        app
    }

    fn run(&mut self, terminal: &mut Tui) -> Result<()> {
        while !self.should_quit {
            self.drain_background_events();
            self.refresh_telemetry_if_due();
            terminal.draw(|frame| self.draw(frame))?;
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == event::KeyEventKind::Press {
                        self.handle_key(key)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn refresh_local_inventories(&mut self) {
        match collect_scripts_in(&self.root, ScriptMode::Bench) {
            Ok(scripts) => self.bench_scripts = scripts,
            Err(error) => self.push_log(format!("Bench inventory failed: {error:#}")),
        }
        self.maintenance_scripts = collect_shell_scripts(&self.root.join("maintenance"), "");
        clamp_list_selection(&mut self.bench_state, self.bench_scripts.len());
        clamp_list_selection(&mut self.maintenance_state, self.maintenance_scripts.len());
    }

    fn refresh_models(&mut self) {
        if self.loading_models {
            return;
        }
        self.loading_models = true;
        self.status = "Refreshing llama-swap models…".into();
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result = SwapClient::from_env()
                .and_then(|client| client.list_models())
                .map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::Models(result));
        });
    }

    fn drain_background_events(&mut self) {
        for _ in 0..EVENTS_PER_TICK {
            let Ok(event) = self.receiver.try_recv() else {
                break;
            };
            match event {
                BackgroundEvent::Models(result) => {
                    self.loading_models = false;
                    match result {
                        Ok(models) => {
                            self.models = models;
                            if self.loaded_model_id.is_none() {
                                self.loaded_model_id = self
                                    .models
                                    .iter()
                                    .find(|model| model.state == "loaded")
                                    .map(|model| model.id.clone());
                            }
                            let visible_count = self.visible_model_count();
                            clamp_table_selection(&mut self.model_state, visible_count);
                            self.status = format!("llama-swap: {} model(s)", self.models.len());
                        }
                        Err(error) => {
                            self.status = "llama-swap unavailable".into();
                            self.push_log(format!("Model refresh failed: {error}"));
                        }
                    }
                }
                BackgroundEvent::ModelAction {
                    job_id,
                    model_id,
                    load,
                    result,
                } => {
                    self.model_action_pending = false;
                    match result {
                        Ok(message) => {
                            if load {
                                self.loaded_model_id = Some(model_id.clone());
                            } else if self.loaded_model_id.as_deref() == Some(model_id.as_str()) {
                                self.loaded_model_id = None;
                            }
                            self.finish_job(job_id, 0);
                            self.status = message.clone();
                            self.push_log(message);
                            self.refresh_models();
                        }
                        Err(error) => {
                            self.finish_job(job_id, 1);
                            self.status = format!(
                                "Could not {} {model_id}",
                                if load { "load" } else { "unload" }
                            );
                            self.push_log(error);
                        }
                    }
                }
                BackgroundEvent::BrowserScan(result) => {
                    self.browser_scanning = false;
                    match result {
                        Ok(files) => {
                            self.browser_files = files;
                            clamp_table_selection(
                                &mut self.browser_state,
                                self.browser_files.len(),
                            );
                            self.status =
                                format!("Found {} GGUF file(s)", self.browser_files.len());
                        }
                        Err(error) => {
                            self.status = "GGUF scan failed".into();
                            self.push_log(error);
                        }
                    }
                }
                BackgroundEvent::ChatDelta(delta) => {
                    append_bounded_text(&mut self.chat_streaming, &delta, MAX_CHAT_PREVIEW_BYTES);
                }
                BackgroundEvent::ChatFinished(result) => {
                    self.chat_pending = false;
                    match result {
                        Ok(completion) => {
                            let status = match (
                                completion.completion_tokens,
                                completion.tokens_per_second(),
                            ) {
                                (Some(tokens), Some(rate)) => {
                                    format!("Chat response: {tokens} tokens · {rate:.1} tok/s")
                                }
                                _ => format!(
                                    "Chat response received in {:.1}s",
                                    completion.elapsed.as_secs_f64()
                                ),
                            };
                            self.chat_history
                                .push(("assistant".into(), completion.content));
                            self.status = status;
                        }
                        Err(error) => {
                            self.status = "Chat request failed".into();
                            self.push_log(error);
                        }
                    }
                    self.chat_streaming.clear();
                }
                BackgroundEvent::ProcessLine { job_id, line } => {
                    if self.jobs.iter().any(|job| job.id == job_id) {
                        self.push_log(line);
                    }
                }
                BackgroundEvent::ProcessFinished { job_id, exit_code } => {
                    self.finish_job(job_id, exit_code);
                    if self
                        .running_process
                        .as_ref()
                        .is_some_and(|running| running.job_id == job_id)
                    {
                        self.running_process = None;
                    }
                }
                BackgroundEvent::Telemetry(result) => {
                    self.telemetry_pending = false;
                    self.telemetry = match result {
                        Ok(Some(snapshot)) => snapshot,
                        Ok(None) => "Resources: idle".into(),
                        Err(error) => format!("Resources: unavailable ({error})"),
                    };
                }
            }
        }
    }

    fn refresh_telemetry_if_due(&mut self) {
        if self.telemetry_pending
            || self
                .telemetry_last_started
                .is_some_and(|started| started.elapsed() < TELEMETRY_INTERVAL)
        {
            return;
        }

        let running_target = self.running_process.as_ref().map(|running| {
            let elapsed = self
                .jobs
                .iter()
                .find(|job| job.id == running.job_id)
                .map_or(0, |job| job.started.elapsed().as_secs());
            (running.process_group, elapsed)
        });
        let monitor_swap = self.loaded_model_id.is_some();
        if running_target.is_none() && !monitor_swap {
            self.telemetry = "Resources: idle".into();
            return;
        }

        self.telemetry_pending = true;
        self.telemetry_last_started = Some(Instant::now());
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result = if let Some((process_group, elapsed)) = running_target {
                snapshot_process_group(process_group)
                    .map(|snapshot| Some(snapshot.render("procs", Some(elapsed))))
            } else {
                find_process_named("llama-swap").and_then(|process| {
                    process.map_or(Ok(None), |process| {
                        snapshot_descendants(process)
                            .map(|snapshot| Some(snapshot.render("upstreams", None)))
                    })
                })
            }
            .map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::Telemetry(result));
        });
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<()> {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.should_quit = true;
            return Ok(());
        }
        if self.show_palette {
            return self.handle_palette_key(key);
        }
        if key.code == KeyCode::Char('p') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.open_palette();
            return Ok(());
        }
        if let Some(command_id) = self.modified_key_command(key) {
            return self.execute_command(command_id);
        }
        if let KeyCode::F(number @ 1..=7) = key.code {
            self.tab = Tab::from_index(number as usize - 1);
            self.input_mode = InputMode::Normal;
            self.show_help = false;
            return Ok(());
        }
        if key.modifiers.contains(KeyModifiers::ALT) {
            let destination = match key.code {
                KeyCode::Char('1') => Some(0),
                KeyCode::Char('2') => Some(1),
                KeyCode::Char('3') => Some(2),
                KeyCode::Char('4') => Some(3),
                KeyCode::Char('5') => Some(4),
                KeyCode::Char('6') => Some(5),
                KeyCode::Char('7') => Some(6),
                _ => None,
            };
            if let Some(destination) = destination {
                self.tab = Tab::from_index(destination);
                self.input_mode = InputMode::Normal;
                self.show_help = false;
                return Ok(());
            }
            match key.code {
                KeyCode::Left => {
                    self.tab =
                        Tab::from_index((self.tab.index() + TAB_NAMES.len() - 1) % TAB_NAMES.len());
                    self.input_mode = InputMode::Normal;
                    return Ok(());
                }
                KeyCode::Right => {
                    self.tab = Tab::from_index(self.tab.index() + 1);
                    self.input_mode = InputMode::Normal;
                    return Ok(());
                }
                _ => {}
            }
        }
        if self.show_help {
            self.show_help = false;
            return Ok(());
        }
        if self.input_mode != InputMode::Normal {
            return self.handle_input_key(key);
        }
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('?') => self.show_help = true,
            _ => self.handle_tab_key(key)?,
        }
        Ok(())
    }

    fn modified_key_command(&self, key: KeyEvent) -> Option<CommandId> {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match (self.tab, key.code, control, alt) {
            (Tab::Workbench, KeyCode::Char('r'), true, _) => Some(CommandId::WorkbenchLoadModel),
            (Tab::Workbench, KeyCode::Char('s'), true, _) => Some(CommandId::WorkbenchUnloadModel),
            (Tab::Workbench, KeyCode::Char('f'), true, _) => Some(CommandId::WorkbenchFocusFilter),
            (Tab::Workbench, KeyCode::Char('j'), true, _) => Some(CommandId::WorkbenchFocusTable),
            (Tab::ModelOps, KeyCode::Char('r'), true, _) => Some(CommandId::ModelOpsStart),
            (Tab::ModelOps, KeyCode::Char('s'), true, _) => Some(CommandId::ModelOpsStop),
            (Tab::ModelOps, KeyCode::Char('m'), true, _) => Some(CommandId::ModelOpsToggleMode),
            (Tab::ModelOps, KeyCode::Char('f'), true, _) => Some(CommandId::ModelOpsFocusFilter),
            (Tab::ModelOps, KeyCode::Char('j'), true, _) => Some(CommandId::ModelOpsFocusTable),
            (Tab::Chat, KeyCode::Char('x'), true, _) => Some(CommandId::ChatClear),
            (Tab::Browser, KeyCode::Char('r'), _, true) => Some(CommandId::BrowserScan),
            (Tab::Browser, KeyCode::Char('g'), _, true) => Some(CommandId::BrowserFocusPath),
            (Tab::Browser, KeyCode::Char('j'), _, true) => Some(CommandId::BrowserFocusTable),
            (Tab::Download, KeyCode::Char('d'), _, true) => Some(CommandId::DownloadSelected),
            (Tab::Download, KeyCode::Char('e'), _, true) => Some(CommandId::DownloadEnabled),
            (Tab::Download, KeyCode::Char('w'), _, true) => Some(CommandId::DownloadSaveConfig),
            (Tab::Download, KeyCode::Char('v'), _, true) => Some(CommandId::DownloadValidateConfig),
            (Tab::Download, KeyCode::Char('t'), _, true) => Some(CommandId::DownloadFocusTable),
            (Tab::Maintenance, KeyCode::Char('r'), true, _) => Some(CommandId::MaintenanceRun),
            (Tab::Maintenance, KeyCode::Char('s'), true, _) => Some(CommandId::MaintenanceStop),
            _ => None,
        }
    }

    fn open_palette(&mut self) {
        self.show_help = false;
        self.show_palette = true;
        self.palette_query.clear();
        self.palette_state.select(Some(0));
    }

    fn handle_palette_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Esc => self.show_palette = false,
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.show_palette = false;
            }
            KeyCode::Up => {
                let count = self.palette_commands().len();
                select_previous_table(&mut self.palette_state, count)
            }
            KeyCode::Down => {
                let count = self.palette_commands().len();
                select_next_table(&mut self.palette_state, count)
            }
            KeyCode::Backspace => {
                self.palette_query.pop();
                self.palette_state.select(Some(0));
            }
            KeyCode::Enter => {
                let command_id = self
                    .palette_state
                    .selected()
                    .and_then(|index| self.palette_commands().get(index).map(|spec| spec.id));
                self.show_palette = false;
                if let Some(command_id) = command_id {
                    self.execute_command(command_id)?;
                }
            }
            KeyCode::Char(character)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.palette_query.push(character);
                self.palette_state.select(Some(0));
            }
            _ => {}
        }
        let count = self.palette_commands().len();
        clamp_table_selection(&mut self.palette_state, count);
        Ok(())
    }

    fn palette_commands(&self) -> Vec<&'static crate::commands::CommandSpec> {
        search_all_commands(&self.palette_query)
    }

    fn handle_input_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Esc => self.input_mode = InputMode::Normal,
            KeyCode::Enter => match self.input_mode {
                InputMode::ChatMessage => self.send_chat_message(),
                InputMode::BrowserPath => {
                    self.input_mode = InputMode::Normal;
                    self.scan_browser();
                }
                InputMode::ChatSystemPrompt => self.input_mode = InputMode::Normal,
                _ => self.input_mode = InputMode::Normal,
            },
            KeyCode::Backspace => match self.input_mode {
                InputMode::ModelFilter => {
                    self.model_filter.pop();
                    self.model_state.select(Some(0));
                }
                InputMode::BrowserPath => {
                    self.browser_path.pop();
                }
                InputMode::ChatMessage => {
                    self.chat_input.pop();
                }
                InputMode::ChatSystemPrompt => {
                    self.chat_system_prompt.pop();
                }
                InputMode::Normal => {}
            },
            KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                match self.input_mode {
                    InputMode::ModelFilter => {
                        self.model_filter.push(character);
                        self.model_state.select(Some(0));
                    }
                    InputMode::BrowserPath => self.browser_path.push(character),
                    InputMode::ChatMessage => self.chat_input.push(character),
                    InputMode::ChatSystemPrompt => self.chat_system_prompt.push(character),
                    InputMode::Normal => {}
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_tab_key(&mut self, key: KeyEvent) -> Result<()> {
        match self.tab {
            Tab::Workbench => self.handle_workbench_key(key),
            Tab::ModelOps => self.handle_ops_key(key),
            Tab::Chat => self.handle_chat_key(key),
            Tab::Browser => self.handle_browser_key(key),
            Tab::Download => self.handle_download_key(key)?,
            Tab::Jobs => self.handle_jobs_key(key),
            Tab::Maintenance => self.handle_maintenance_key(key),
        }
        Ok(())
    }

    fn command_context(&self) -> CommandContext {
        match self.tab {
            Tab::Workbench => CommandContext::Workbench,
            Tab::ModelOps => CommandContext::ModelOps,
            Tab::Chat => CommandContext::Chat,
            Tab::Browser => CommandContext::Browser,
            Tab::Download => CommandContext::Download,
            Tab::Jobs => CommandContext::Jobs,
            Tab::Maintenance => CommandContext::Maintenance,
        }
    }

    fn execute_command(&mut self, command_id: CommandId) -> Result<()> {
        use CommandId::*;

        match command_id {
            Quit => self.should_quit = true,
            ShowHelp => self.show_help = true,
            ShowPalette => self.open_palette(),
            OpenWorkbench => self.tab = Tab::Workbench,
            OpenModelOps => self.tab = Tab::ModelOps,
            OpenChat => self.tab = Tab::Chat,
            OpenBrowser => self.tab = Tab::Browser,
            OpenDownload => self.tab = Tab::Download,
            OpenJobs => self.tab = Tab::Jobs,
            OpenMaintenance => self.tab = Tab::Maintenance,
            PreviousTab => {
                self.tab =
                    Tab::from_index((self.tab.index() + TAB_NAMES.len() - 1) % TAB_NAMES.len())
            }
            NextTab => self.tab = Tab::from_index(self.tab.index() + 1),
            WorkbenchSelectPrevious => {
                let count = self.visible_model_count();
                select_previous_table(&mut self.model_state, count);
            }
            WorkbenchSelectNext => {
                let count = self.visible_model_count();
                select_next_table(&mut self.model_state, count);
            }
            WorkbenchFocusFilter => self.input_mode = InputMode::ModelFilter,
            WorkbenchFocusTable => self.input_mode = InputMode::Normal,
            WorkbenchRefreshModels => self.refresh_models(),
            WorkbenchLoadModel => self.model_action(true),
            WorkbenchUnloadModel => self.model_action(false),
            ModelOpsToggleMode => {
                self.ops_mode = match self.ops_mode {
                    OpsMode::Run => OpsMode::Bench,
                    OpsMode::Bench => OpsMode::Run,
                }
            }
            ModelOpsSelectPrevious => match self.ops_mode {
                OpsMode::Run => {
                    let count = self.visible_model_count();
                    select_previous_table(&mut self.model_state, count);
                }
                OpsMode::Bench => {
                    select_previous_list(&mut self.bench_state, self.bench_scripts.len())
                }
            },
            ModelOpsSelectNext => match self.ops_mode {
                OpsMode::Run => {
                    let count = self.visible_model_count();
                    select_next_table(&mut self.model_state, count);
                }
                OpsMode::Bench => select_next_list(&mut self.bench_state, self.bench_scripts.len()),
            },
            ModelOpsStart => match self.ops_mode {
                OpsMode::Run => self.model_action(true),
                OpsMode::Bench => self.run_selected_bench(),
            },
            ModelOpsStop => match self.ops_mode {
                OpsMode::Run => self.model_action(false),
                OpsMode::Bench => self.stop_running_process(),
            },
            ModelOpsRefreshModels => self.refresh_models(),
            ModelOpsFocusFilter => {
                if self.ops_mode == OpsMode::Run {
                    self.input_mode = InputMode::ModelFilter;
                } else {
                    self.status = "Bench script filtering is not implemented yet".into();
                }
            }
            ModelOpsFocusTable => self.input_mode = InputMode::Normal,
            ChatCompose => self.input_mode = InputMode::ChatMessage,
            ChatSend => {
                if self.chat_input.trim().is_empty() {
                    self.input_mode = InputMode::ChatMessage;
                } else {
                    self.send_chat_message();
                }
            }
            ChatRefreshModels => self.refresh_models(),
            ChatClear => {
                self.chat_history.clear();
                self.chat_streaming.clear();
                self.status = "Chat cleared".into();
            }
            ChatEditSystemPrompt => self.input_mode = InputMode::ChatSystemPrompt,
            ChatToggleThinking => self.chat_thinking = !self.chat_thinking,
            ChatDecreaseTemperature => {
                self.chat_temperature = (self.chat_temperature - 0.1).max(0.0)
            }
            ChatIncreaseTemperature => {
                self.chat_temperature = (self.chat_temperature + 0.1).min(10.0)
            }
            ChatDecreaseMaxTokens => self.chat_max_tokens = (self.chat_max_tokens / 2).max(128),
            ChatIncreaseMaxTokens => {
                self.chat_max_tokens = self.chat_max_tokens.saturating_mul(2).min(65_536)
            }
            BrowserSelectPrevious => {
                select_previous_table(&mut self.browser_state, self.browser_files.len())
            }
            BrowserSelectNext => {
                select_next_table(&mut self.browser_state, self.browser_files.len())
            }
            BrowserScan => self.scan_browser(),
            BrowserFocusPath => self.input_mode = InputMode::BrowserPath,
            BrowserFocusTable => self.input_mode = InputMode::Normal,
            DownloadSelectPrevious => {
                select_previous_table(&mut self.download_state, self.download_config.models.len())
            }
            DownloadSelectNext => {
                select_next_table(&mut self.download_state, self.download_config.models.len())
            }
            DownloadFocusTable => self.input_mode = InputMode::Normal,
            DownloadToggleEnabled => {
                self.handle_download_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))?
            }
            DownloadSaveConfig => {
                self.handle_download_key(KeyEvent::new(KeyCode::Char('w'), KeyModifiers::NONE))?
            }
            DownloadValidateConfig => {
                self.handle_download_key(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::NONE))?
            }
            DownloadSelected => self.download_selected(),
            DownloadEnabled => self.download_enabled(),
            JobsSelectPrevious => select_previous_table(&mut self.jobs_state, self.jobs.len()),
            JobsSelectNext => select_next_table(&mut self.jobs_state, self.jobs.len()),
            JobsStop => self.stop_running_process(),
            JobsRetry => self.retry_selected_job(),
            JobsClear if self.running_process.is_none() => {
                self.jobs.clear();
                self.jobs_state.select(None);
                self.status = "Job history cleared".into();
            }
            MaintenanceSelectPrevious => {
                select_previous_list(&mut self.maintenance_state, self.maintenance_scripts.len())
            }
            MaintenanceSelectNext => {
                select_next_list(&mut self.maintenance_state, self.maintenance_scripts.len())
            }
            MaintenanceRun => self.run_selected_maintenance(),
            MaintenanceStop => self.stop_running_process(),
            _ => {
                let label = command(command_id)
                    .map(|spec| spec.palette_label)
                    .unwrap_or("Unknown command");
                self.status = format!("{label} is not implemented yet");
            }
        }
        if matches!(
            command_id,
            OpenWorkbench
                | OpenModelOps
                | OpenChat
                | OpenBrowser
                | OpenDownload
                | OpenJobs
                | OpenMaintenance
                | PreviousTab
                | NextTab
        ) {
            self.input_mode = InputMode::Normal;
            self.show_help = false;
        }
        Ok(())
    }

    fn handle_workbench_key(&mut self, key: KeyEvent) {
        let visible_count = self.visible_model_count();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                select_previous_table(&mut self.model_state, visible_count)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next_table(&mut self.model_state, visible_count)
            }
            KeyCode::Char('/') => self.input_mode = InputMode::ModelFilter,
            KeyCode::Char('r') => self.refresh_models(),
            KeyCode::Enter | KeyCode::Char('l') => self.model_action(true),
            KeyCode::Char('s') => self.model_action(false),
            _ => {}
        }
    }

    fn handle_ops_key(&mut self, key: KeyEvent) {
        if matches!(key.code, KeyCode::Char('m')) {
            self.ops_mode = match self.ops_mode {
                OpsMode::Run => OpsMode::Bench,
                OpsMode::Bench => OpsMode::Run,
            };
            return;
        }
        match self.ops_mode {
            OpsMode::Run => self.handle_workbench_key(key),
            OpsMode::Bench => match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    select_previous_list(&mut self.bench_state, self.bench_scripts.len())
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    select_next_list(&mut self.bench_state, self.bench_scripts.len())
                }
                KeyCode::Enter | KeyCode::Char('r') => self.run_selected_bench(),
                KeyCode::Char('s') => self.stop_running_process(),
                _ => {}
            },
        }
    }

    fn handle_chat_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('i') | KeyCode::Enter => self.input_mode = InputMode::ChatMessage,
            KeyCode::Char('x') => {
                self.chat_history.clear();
                self.chat_streaming.clear();
                self.status = "Chat cleared".into();
            }
            KeyCode::Char('r') => self.refresh_models(),
            KeyCode::Char('p') => self.input_mode = InputMode::ChatSystemPrompt,
            KeyCode::Char('t') => {
                self.chat_thinking = !self.chat_thinking;
                self.status = format!(
                    "Thinking prefix {}",
                    if self.chat_thinking {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }
            KeyCode::Char('[') => {
                self.chat_temperature = (self.chat_temperature - 0.1).max(0.0);
            }
            KeyCode::Char(']') => {
                self.chat_temperature = (self.chat_temperature + 0.1).min(10.0);
            }
            KeyCode::Char('-') => {
                self.chat_max_tokens = (self.chat_max_tokens / 2).max(128);
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.chat_max_tokens = self.chat_max_tokens.saturating_mul(2).min(65_536);
            }
            _ => {}
        }
    }

    fn handle_browser_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                select_previous_table(&mut self.browser_state, self.browser_files.len())
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next_table(&mut self.browser_state, self.browser_files.len())
            }
            KeyCode::Char('g') => self.input_mode = InputMode::BrowserPath,
            KeyCode::Char('r') | KeyCode::Enter => self.scan_browser(),
            _ => {}
        }
    }

    fn handle_download_key(&mut self, key: KeyEvent) -> Result<()> {
        let count = self.download_config.models.len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                select_previous_table(&mut self.download_state, count)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next_table(&mut self.download_state, count)
            }
            KeyCode::Char(' ') => {
                if !self.download_config_is_usable() {
                    return Ok(());
                }
                if let Some(index) = self.download_state.selected() {
                    if let Some(model) = self.download_config.models.get_mut(index) {
                        model.enabled = !model.enabled;
                        self.download_dirty = true;
                        self.status = format!(
                            "{} {}",
                            if model.enabled { "Enabled" } else { "Disabled" },
                            model.repo_id
                        );
                    }
                }
            }
            KeyCode::Char('v') => {
                if !self.download_config_is_usable() {
                    return Ok(());
                }
                let errors = validate_config(&self.download_config);
                if errors.is_empty() {
                    self.status = "Download config is valid".into();
                } else {
                    self.status = format!("{} validation error(s)", errors.len());
                    for error in errors {
                        self.push_log(error);
                    }
                }
            }
            KeyCode::Char('w') => {
                if !self.download_config_is_usable() {
                    return Ok(());
                }
                let versions_root = self.root.join(".toolkit/download_config_versions");
                match save_config_in(
                    &self.config_path,
                    &self.download_config,
                    "rust-tui",
                    &versions_root,
                ) {
                    Ok(()) => {
                        self.download_dirty = false;
                        self.status = "Saved config with snapshot".into();
                    }
                    Err(error) => {
                        self.status = "Could not save download config".into();
                        self.push_log(format!("{error:#}"));
                    }
                }
            }
            KeyCode::Char('d') => self.download_selected(),
            KeyCode::Char('e') => self.download_enabled(),
            _ => {}
        }
        Ok(())
    }

    fn handle_jobs_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                select_previous_table(&mut self.jobs_state, self.jobs.len())
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next_table(&mut self.jobs_state, self.jobs.len())
            }
            KeyCode::Char('s') => self.stop_running_process(),
            KeyCode::Char('r') => self.retry_selected_job(),
            KeyCode::Char('c') | KeyCode::Delete if self.running_process.is_none() => {
                self.jobs.clear();
                self.jobs_state.select(None);
                self.status = "Job history cleared".into();
            }
            _ => {}
        }
    }

    fn handle_maintenance_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                select_previous_list(&mut self.maintenance_state, self.maintenance_scripts.len())
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next_list(&mut self.maintenance_state, self.maintenance_scripts.len())
            }
            KeyCode::Enter | KeyCode::Char('r') => self.run_selected_maintenance(),
            KeyCode::Char('s') => self.stop_running_process(),
            _ => {}
        }
    }

    fn visible_models(&self) -> Vec<&SwapModel> {
        let filter = self.model_filter.to_ascii_lowercase();
        self.models
            .iter()
            .filter(|model| filter.is_empty() || model.id.to_ascii_lowercase().contains(&filter))
            .collect()
    }

    fn visible_model_count(&self) -> usize {
        self.visible_models().len()
    }

    fn selected_model(&self) -> Option<SwapModel> {
        let index = self.model_state.selected()?;
        self.visible_models()
            .get(index)
            .map(|model| (*model).clone())
    }

    fn model_action(&mut self, load: bool) {
        if self.model_action_pending {
            self.status = "A llama-swap action is already in progress".into();
            return;
        }
        let model_id = if load {
            self.selected_model().map(|model| model.id)
        } else {
            self.loaded_model_id
                .clone()
                .or_else(|| self.selected_model().map(|model| model.id))
        };
        let Some(model_id) = model_id else {
            self.status = if load {
                "No model selected".into()
            } else {
                "No loaded model is known".into()
            };
            return;
        };
        self.begin_model_action(model_id, load);
    }

    fn begin_model_action(&mut self, model_id: String, load: bool) {
        if self.model_action_pending {
            self.status = "A llama-swap action is already in progress".into();
            return;
        }
        self.model_action_pending = true;
        self.status = format!(
            "{} {}…",
            if load { "Loading" } else { "Unloading" },
            model_id
        );
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let verb = if load { "load" } else { "unload" };
        self.jobs.insert(
            0,
            JobRecord {
                id: job_id,
                name: model_id.clone(),
                kind: format!("model-{verb}"),
                status: "running".into(),
                command: vec!["llama-swap".into(), verb.into(), model_id.clone()],
                started: Instant::now(),
                exit_code: None,
            },
        );
        self.jobs_state.select(Some(0));
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result = SwapClient::from_env()
                .and_then(|client| {
                    if load {
                        client.load_model(&model_id)
                    } else {
                        client.unload_model(&model_id)
                    }
                })
                .map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::ModelAction {
                job_id,
                model_id,
                load,
                result,
            });
        });
    }

    fn scan_browser(&mut self) {
        if self.browser_scanning {
            return;
        }
        let path = PathBuf::from(expand_tilde(&self.browser_path));
        self.browser_scanning = true;
        self.status = format!("Scanning {}…", path.display());
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result = scan_gguf_files(&path).map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::BrowserScan(result));
        });
    }

    fn send_chat_message(&mut self) {
        self.input_mode = InputMode::Normal;
        let prompt = self.chat_input.trim().to_owned();
        if prompt.is_empty() || self.chat_pending {
            return;
        }
        let Some(model) = self
            .loaded_model_id
            .as_deref()
            .and_then(|loaded| self.models.iter().find(|model| model.id == loaded).cloned())
            .or_else(|| self.selected_model())
            .or_else(|| self.models.first().cloned())
        else {
            self.status = "Refresh models before chatting".into();
            return;
        };
        self.chat_input.clear();
        self.chat_history.push(("user".into(), prompt));
        self.chat_streaming.clear();
        self.chat_pending = true;
        self.status = format!("Waiting for {}…", model.id);
        let history = self.chat_history.clone();
        let system_prompt = self.chat_system_prompt.clone();
        let temperature = self.chat_temperature;
        let max_tokens = self.chat_max_tokens;
        let thinking = self.chat_thinking;
        let sender = self.sender.clone();
        thread::spawn(move || {
            let mut request = ChatRequest::new(model.id, history);
            request.system_prompt = system_prompt;
            request.temperature = temperature;
            request.max_tokens = max_tokens;
            request.thinking = thinking;
            let result = stream_completion(&request, |delta| {
                sender
                    .send(BackgroundEvent::ChatDelta(delta.to_owned()))
                    .is_ok()
            })
            .map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::ChatFinished(result));
        });
    }

    fn run_selected_bench(&mut self) {
        let Some(index) = self.bench_state.selected() else {
            return;
        };
        let Some(path) = self.bench_scripts.get(index).cloned() else {
            return;
        };
        let command = command_for_script(&path, &[]);
        self.start_process_from_parts(
            display_name(&path),
            "bench",
            command.into_iter().map(OsString::from).collect(),
        );
    }

    fn run_selected_maintenance(&mut self) {
        let Some(index) = self.maintenance_state.selected() else {
            return;
        };
        let Some(path) = self.maintenance_scripts.get(index).cloned() else {
            return;
        };
        let command = command_for_script(&path, &[]);
        self.start_process_from_parts(
            display_name(&path),
            "maintenance",
            command.into_iter().map(OsString::from).collect(),
        );
    }

    fn download_selected(&mut self) {
        if !self.download_config_is_usable() {
            return;
        }
        let Some(index) = self.download_state.selected() else {
            return;
        };
        match build_selected_download_command(&self.root, &self.download_config, index) {
            Ok((name, parts)) => self.start_process_from_parts(name, "download", parts),
            Err(error) => {
                self.status = "Could not build download command".into();
                self.push_log(format!("{error:#}"));
            }
        }
    }

    fn download_enabled(&mut self) {
        if !self.download_config_is_usable() {
            return;
        }
        if self.download_dirty {
            self.status = "Save the edited config before downloading enabled models".into();
            return;
        }
        let parts = vec![
            self.root
                .join("model_downloader/download_hf_model.py")
                .into_os_string(),
            "--config".into(),
            self.config_path.clone().into_os_string(),
        ];
        self.start_process_from_parts("enabled downloads".into(), "download", parts);
    }

    fn download_config_is_usable(&mut self) -> bool {
        let Some(error) = self.config_error.clone() else {
            return true;
        };
        self.status = "Download config is blocked until its load error is fixed".into();
        self.push_log(error);
        false
    }

    fn start_process_from_parts(&mut self, name: String, kind: &str, parts: Vec<OsString>) {
        if self.running_process.is_some() {
            self.status = "A process is already running; stop it first".into();
            return;
        }
        let Some((program, arguments)) = parts.split_first() else {
            return;
        };
        let command_text: Vec<String> = parts
            .iter()
            .map(|part| part.to_string_lossy().into_owned())
            .collect();
        let mut command = Command::new(program);
        command
            .args(arguments)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        command.process_group(0);

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                self.status = format!("Could not start {name}");
                self.push_log(format!("{error:#}"));
                return;
            }
        };

        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let process_group = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let child = Arc::new(Mutex::new(Some(child)));
        self.running_process = Some(RunningProcess {
            job_id,
            process_group,
            child: Arc::clone(&child),
        });
        self.jobs.insert(
            0,
            JobRecord {
                id: job_id,
                name: name.clone(),
                kind: kind.into(),
                status: "running".into(),
                command: command_text.clone(),
                started: Instant::now(),
                exit_code: None,
            },
        );
        self.jobs_state.select(Some(0));
        self.status = format!("Running {name}");
        self.push_log(format!("$ {}", command_text.join(" ")));

        if let Some(stdout) = stdout {
            stream_reader(stdout, job_id, self.sender.clone());
        }
        if let Some(stderr) = stderr {
            stream_reader(stderr, job_id, self.sender.clone());
        }
        let sender = self.sender.clone();
        thread::spawn(move || loop {
            let result = {
                let mut guard = match child.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                match guard.as_mut() {
                    Some(process) => process.try_wait(),
                    None => return,
                }
            };
            match result {
                Ok(Some(status)) => {
                    if let Ok(mut guard) = child.lock() {
                        *guard = None;
                    }
                    let _ = sender.send(BackgroundEvent::ProcessFinished {
                        job_id,
                        exit_code: status.code().unwrap_or(1),
                    });
                    return;
                }
                Ok(None) => thread::sleep(Duration::from_millis(100)),
                Err(error) => {
                    #[cfg(unix)]
                    {
                        let _ = signal_process_group(process_group, "TERM");
                        thread::sleep(Duration::from_millis(250));
                        let _ = signal_process_group(process_group, "KILL");
                    }
                    if let Ok(mut guard) = child.lock() {
                        if let Some(mut process) = guard.take() {
                            let _ = process.kill();
                            let _ = process.wait();
                        }
                    }
                    let _ = sender.send(BackgroundEvent::ProcessLine {
                        job_id,
                        line: format!("process monitor failed: {error}"),
                    });
                    let _ = sender.send(BackgroundEvent::ProcessFinished {
                        job_id,
                        exit_code: 1,
                    });
                    return;
                }
            }
        });
    }

    fn stop_running_process(&mut self) {
        let Some(running) = &self.running_process else {
            self.status = "No process is running".into();
            return;
        };
        #[cfg(unix)]
        let result = signal_process_group(running.process_group, "TERM").or_else(|group_error| {
            running
                .child
                .lock()
                .map_err(|_| "process lock is poisoned".to_owned())
                .and_then(|mut child| match child.as_mut() {
                    Some(child) => child
                        .kill()
                        .map_err(|error| format!("{group_error}; fallback kill failed: {error}")),
                    None => Ok(()),
                })
        });
        #[cfg(not(unix))]
        let result = running
            .child
            .lock()
            .map_err(|_| "process lock is poisoned".to_owned())
            .and_then(|mut child| match child.as_mut() {
                Some(child) => child.kill().map_err(|error| error.to_string()),
                None => Ok(()),
            });

        #[cfg(unix)]
        if result.is_ok() {
            let process_group = running.process_group;
            let child = Arc::clone(&running.child);
            thread::spawn(move || {
                thread::sleep(Duration::from_secs(3));
                let still_running = child
                    .lock()
                    .ok()
                    .and_then(|mut child| child.as_mut().and_then(|child| child.try_wait().ok()))
                    .is_some_and(|status| status.is_none());
                if still_running {
                    let _ = signal_process_group(process_group, "KILL");
                }
            });
        }
        self.status = match result {
            Ok(()) => "Stop requested".into(),
            Err(error) => format!("Stop failed: {error}"),
        };
    }

    fn retry_selected_job(&mut self) {
        let Some(index) = self.jobs_state.selected() else {
            return;
        };
        let Some(job) = self.jobs.get(index).cloned() else {
            return;
        };
        if matches!(job.kind.as_str(), "model-load" | "model-unload") {
            let Some(model_id) = job.command.get(2).cloned() else {
                self.status = "Model job has no model identifier".into();
                return;
            };
            self.begin_model_action(model_id, job.kind == "model-load");
            return;
        }
        self.start_process_from_parts(
            format!("{} (retry)", job.name),
            &job.kind,
            job.command.into_iter().map(OsString::from).collect(),
        );
    }

    fn push_log(&mut self, message: impl Into<String>) {
        self.log.push_back(message.into());
        while self.log.len() > 300 {
            self.log.pop_front();
        }
    }

    fn finish_job(&mut self, job_id: u64, exit_code: i32) {
        if let Some(job) = self.jobs.iter_mut().find(|job| job.id == job_id) {
            job.status = if exit_code == 0 {
                "done".into()
            } else {
                "failed".into()
            };
            job.exit_code = Some(exit_code);
            self.status = format!(
                "{} exited with {exit_code} after {}s",
                job.name,
                job.started.elapsed().as_secs()
            );
        }
    }

    fn draw(&mut self, frame: &mut Frame) {
        let areas = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(3),
                Constraint::Min(8),
                Constraint::Length(5),
                Constraint::Length(1),
            ])
            .split(frame.area());

        self.draw_header(frame, areas[0]);
        self.draw_tabs(frame, areas[1]);
        match self.tab {
            Tab::Workbench => self.draw_workbench(frame, areas[2]),
            Tab::ModelOps => self.draw_model_ops(frame, areas[2]),
            Tab::Chat => self.draw_chat(frame, areas[2]),
            Tab::Browser => self.draw_browser(frame, areas[2]),
            Tab::Download => self.draw_download(frame, areas[2]),
            Tab::Jobs => self.draw_jobs(frame, areas[2]),
            Tab::Maintenance => self.draw_maintenance(frame, areas[2]),
        }
        self.draw_log(frame, areas[3]);
        self.draw_footer(frame, areas[4]);
        if self.show_palette {
            self.draw_palette(frame);
        } else if self.show_help {
            self.draw_help(frame);
        }
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let endpoint =
            env::var("LLAMA_SWAP_URL").unwrap_or_else(|_| "http://localhost:8080".into());
        let line = Line::from(vec![
            Span::styled(
                " L3MS ",
                Style::default().fg(Color::Black).bg(Color::Cyan).bold(),
            ),
            Span::raw(format!(" Rust v{}  ", env!("CARGO_PKG_VERSION"))),
            Span::styled(endpoint, Style::default().fg(Color::DarkGray)),
        ]);
        frame.render_widget(Paragraph::new(line), area);
    }

    fn draw_tabs(&self, frame: &mut Frame, area: Rect) {
        let titles = TAB_NAMES
            .iter()
            .map(|name| Line::from(*name))
            .collect::<Vec<_>>();
        let tabs = Tabs::new(titles)
            .block(Block::default().borders(Borders::ALL))
            .select(self.tab.index())
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .divider("│");
        frame.render_widget(tabs, area);
    }

    fn draw_workbench(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(area);
        self.draw_models_table(frame, chunks[0], "Models");
        let model = self.selected_model();
        let text = if let Some(model) = model {
            Text::from(vec![
                Line::from(Span::styled(
                    model.id,
                    Style::default().fg(Color::Cyan).bold(),
                )),
                Line::from(format!("state: {}", model.state)),
                Line::from(format!("name: {}", value_or_dash(&model.name))),
                Line::from(""),
                Line::from(model.description),
                Line::from(""),
                Line::from("Enter/l load · s unload · r refresh"),
                Line::from("/ filter · F2 full operations · F3 chat"),
            ])
        } else {
            Text::from("No model selected\n\nr refreshes llama-swap")
        };
        frame.render_widget(
            Paragraph::new(text)
                .block(Block::default().title("Fast actions").borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
            chunks[1],
        );
    }

    fn draw_model_ops(&mut self, frame: &mut Frame, area: Rect) {
        let mode = match self.ops_mode {
            OpsMode::Run => "RUN · llama-swap",
            OpsMode::Bench => "BENCH · scripts",
        };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(4)])
            .split(area);
        frame.render_widget(
            Paragraph::new(format!(
                " {mode}   m toggle mode · r/Enter start · s stop\n {}",
                self.telemetry
            ))
            .style(Style::default().fg(Color::Yellow)),
            chunks[0],
        );
        match self.ops_mode {
            OpsMode::Run => self.draw_models_table(frame, chunks[1], "Servable models"),
            OpsMode::Bench => {
                let items = self
                    .bench_scripts
                    .iter()
                    .map(|path| ListItem::new(relative_display(&self.root, path)))
                    .collect::<Vec<_>>();
                let list = List::new(items)
                    .block(
                        Block::default()
                            .title("Bench scripts")
                            .borders(Borders::ALL),
                    )
                    .highlight_symbol("▶ ")
                    .highlight_style(Style::default().fg(Color::Cyan).bold());
                frame.render_stateful_widget(list, chunks[1], &mut self.bench_state);
            }
        }
    }

    fn draw_models_table(&mut self, frame: &mut Frame, area: Rect, title: &str) {
        let filter = if self.model_filter.is_empty() {
            String::new()
        } else {
            format!(" · filter: {}", self.model_filter)
        };
        let rows = self
            .visible_models()
            .into_iter()
            .map(|model| {
                Row::new(vec![
                    Cell::from(model.id.clone()),
                    Cell::from(model.state.clone()),
                ])
            })
            .collect::<Vec<_>>();
        let table = Table::new(
            rows,
            [Constraint::Percentage(76), Constraint::Percentage(24)],
        )
        .header(
            Row::new(["Model", "State"])
                .style(Style::default().fg(Color::Yellow).bold())
                .bottom_margin(1),
        )
        .block(
            Block::default()
                .title(format!("{title}{filter}"))
                .borders(Borders::ALL),
        )
        .row_highlight_style(Style::default().bg(Color::DarkGray).fg(Color::White))
        .highlight_symbol("▶ ");
        frame.render_stateful_widget(table, area, &mut self.model_state);
    }

    fn draw_chat(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(4),
                Constraint::Length(3),
            ])
            .split(area);
        frame.render_widget(
            Paragraph::new(format!(
                "system: {} │ temp {:.1} │ max {} │ thinking {}",
                self.chat_system_prompt,
                self.chat_temperature,
                self.chat_max_tokens,
                if self.chat_thinking { "on" } else { "off" }
            ))
            .block(
                Block::default()
                    .title("Params · p system · [/] temp · -/+ max tokens · t thinking")
                    .borders(Borders::ALL),
            ),
            chunks[0],
        );
        if self.input_mode == InputMode::ChatSystemPrompt {
            frame.set_cursor_position((
                chunks[0].x + 9 + self.chat_system_prompt.chars().count() as u16,
                chunks[0].y + 1,
            ));
        }
        let mut lines = Vec::new();
        for (role, content) in self.chat_history.iter().rev().take(14).rev() {
            let color = if role == "user" {
                Color::Cyan
            } else {
                Color::Green
            };
            lines.push(Line::from(Span::styled(
                format!("{role}:"),
                Style::default().fg(color).bold(),
            )));
            lines.push(Line::from(content.as_str()));
            lines.push(Line::from(""));
        }
        if !self.chat_streaming.is_empty() {
            lines.push(Line::from(Span::styled(
                "assistant (streaming):",
                Style::default().fg(Color::Green).bold(),
            )));
            lines.push(Line::from(self.chat_streaming.as_str()));
        }
        if lines.is_empty() {
            lines.push(Line::from("Press i or Enter to compose a message."));
            lines.push(Line::from("The selected Workbench model is used."));
        }
        frame.render_widget(
            Paragraph::new(lines)
                .block(Block::default().title("Conversation").borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
            chunks[1],
        );
        let marker = if self.chat_pending { " waiting…" } else { "" };
        frame.render_widget(
            Paragraph::new(format!("> {}{marker}", self.chat_input)).block(
                Block::default()
                    .title("Message · Enter send · Esc cancel")
                    .borders(Borders::ALL),
            ),
            chunks[2],
        );
        if self.input_mode == InputMode::ChatMessage {
            frame.set_cursor_position((
                chunks[2].x + 3 + self.chat_input.chars().count() as u16,
                chunks[2].y + 1,
            ));
        }
    }

    fn draw_browser(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(4)])
            .split(area);
        frame.render_widget(
            Paragraph::new(self.browser_path.as_str()).block(
                Block::default()
                    .title("GGUF path · g edit · r scan")
                    .borders(Borders::ALL),
            ),
            chunks[0],
        );
        if self.input_mode == InputMode::BrowserPath {
            frame.set_cursor_position((
                chunks[0].x + 1 + self.browser_path.chars().count() as u16,
                chunks[0].y + 1,
            ));
        }
        let rows = self.browser_files.iter().map(|file| {
            Row::new(vec![
                Cell::from(
                    file.path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
                ),
                Cell::from(file.quantization.clone()),
                Cell::from(format_bytes(file.size)),
                Cell::from(
                    file.modified
                        .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
                        .map(|duration| duration.as_secs().to_string())
                        .unwrap_or_else(|| "—".into()),
                ),
            ])
        });
        let table = Table::new(
            rows,
            [
                Constraint::Percentage(55),
                Constraint::Length(14),
                Constraint::Length(12),
                Constraint::Length(14),
            ],
        )
        .header(
            Row::new(["File", "Quant", "Size", "Modified (unix)"])
                .yellow()
                .bold(),
        )
        .block(
            Block::default()
                .title(format!("Inventory · {} file(s)", self.browser_files.len()))
                .borders(Borders::ALL),
        )
        .row_highlight_style(Style::default().bg(Color::DarkGray))
        .highlight_symbol("▶ ");
        frame.render_stateful_widget(table, chunks[1], &mut self.browser_state);
    }

    fn draw_download(&mut self, frame: &mut Frame, area: Rect) {
        let rows = self.download_config.models.iter().map(|model| {
            Row::new(vec![
                Cell::from(if model.enabled { "yes" } else { "no" }),
                Cell::from(model.repo_id.clone()),
                Cell::from(model.description.clone()),
            ])
        });
        let dirty = if self.download_dirty {
            " · unsaved"
        } else {
            ""
        };
        let table = Table::new(
            rows,
            [
                Constraint::Length(8),
                Constraint::Percentage(45),
                Constraint::Percentage(55),
            ],
        )
        .header(
            Row::new(["Enabled", "Repository", "Description"])
                .yellow()
                .bold(),
        )
        .block(
            Block::default()
                .title(format!(
                    "Download config{dirty} · Space toggle · w save · v validate · d/e download"
                ))
                .borders(Borders::ALL),
        )
        .row_highlight_style(Style::default().bg(Color::DarkGray))
        .highlight_symbol("▶ ");
        frame.render_stateful_widget(table, area, &mut self.download_state);
    }

    fn draw_jobs(&mut self, frame: &mut Frame, area: Rect) {
        let rows = self.jobs.iter().map(|job| {
            Row::new(vec![
                Cell::from(job.id.to_string()),
                Cell::from(job.kind.clone()),
                Cell::from(job.name.clone()),
                Cell::from(job.status.clone()),
                Cell::from(
                    job.exit_code
                        .map(|code| code.to_string())
                        .unwrap_or_else(|| "—".into()),
                ),
            ])
        });
        let table = Table::new(
            rows,
            [
                Constraint::Length(5),
                Constraint::Length(13),
                Constraint::Percentage(55),
                Constraint::Length(12),
                Constraint::Length(7),
            ],
        )
        .header(
            Row::new(["ID", "Kind", "Name", "Status", "Exit"])
                .yellow()
                .bold(),
        )
        .block(
            Block::default()
                .title("Jobs · s stop · r retry · c clear")
                .borders(Borders::ALL),
        )
        .row_highlight_style(Style::default().bg(Color::DarkGray))
        .highlight_symbol("▶ ");
        frame.render_stateful_widget(table, area, &mut self.jobs_state);
    }

    fn draw_maintenance(&mut self, frame: &mut Frame, area: Rect) {
        let items = self
            .maintenance_scripts
            .iter()
            .map(|path| ListItem::new(relative_display(&self.root, path)))
            .collect::<Vec<_>>();
        let list = List::new(items)
            .block(
                Block::default()
                    .title("Maintenance scripts · r/Enter run · s stop")
                    .borders(Borders::ALL),
            )
            .highlight_symbol("▶ ")
            .highlight_style(Style::default().fg(Color::Cyan).bold());
        frame.render_stateful_widget(list, area, &mut self.maintenance_state);
    }

    fn draw_log(&self, frame: &mut Frame, area: Rect) {
        let lines = self
            .log
            .iter()
            .rev()
            .take(area.height.saturating_sub(2) as usize)
            .rev()
            .map(|line| Line::from(line.as_str()))
            .collect::<Vec<_>>();
        frame.render_widget(
            Paragraph::new(lines)
                .block(Block::default().title("Activity").borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let input = match self.input_mode {
            InputMode::Normal => "",
            InputMode::ModelFilter => "  FILTER",
            InputMode::BrowserPath => "  PATH INPUT",
            InputMode::ChatMessage => "  CHAT INPUT",
            InputMode::ChatSystemPrompt => "  SYSTEM PROMPT INPUT",
        };
        let footer = format!(
            " {}{}  │ F1–F7 tabs │ Alt+←/→ cycle │ Ctrl+P palette │ ? help │ q quit ",
            self.status, input
        );
        frame.render_widget(
            Paragraph::new(footer)
                .style(Style::default().fg(Color::Black).bg(Color::Cyan))
                .alignment(Alignment::Left),
            area,
        );
    }

    fn draw_help(&self, frame: &mut Frame) {
        let area = centered_rect(82, 82, frame.area());
        frame.render_widget(Clear, area);
        let context = self.command_context();
        let mut help = vec![
            Line::from(Span::styled(
                "L3MS Rust key bindings",
                Style::default().fg(Color::Cyan).bold(),
            )),
            Line::from(""),
            Line::from("Global  F1–F7/Alt+1–7 tabs · Alt+←/→ cycle · Ctrl+P palette · q quit"),
            Line::from(""),
        ];
        help.push(Line::from(Span::styled(
            format!("{} commands", context),
            Style::default().fg(Color::Yellow).bold(),
        )));
        for spec in visible_commands(context)
            .into_iter()
            .filter(|spec| !spec.contexts.contains(&CommandContext::Global))
        {
            help.push(Line::from(vec![
                Span::styled(
                    format!("  {:<20}", spec.shortcut),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(spec.label),
            ]));
        }
        help.push(Line::from(""));
        help.push(Line::from("Press any key to close."));
        frame.render_widget(
            Paragraph::new(help)
                .block(Block::default().title(" Help ").borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn draw_palette(&mut self, frame: &mut Frame) {
        let area = centered_rect(84, 84, frame.area());
        frame.render_widget(Clear, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(4)])
            .split(area);
        frame.render_widget(
            Paragraph::new(self.palette_query.as_str()).block(
                Block::default()
                    .title(" Command palette · type to search · Esc close ")
                    .borders(Borders::ALL),
            ),
            chunks[0],
        );
        let commands = self.palette_commands();
        let rows = commands
            .iter()
            .map(|spec| Row::new([Cell::from(spec.shortcut), Cell::from(spec.palette_label)]));
        let table = Table::new(rows, [Constraint::Length(22), Constraint::Percentage(100)])
            .block(
                Block::default()
                    .title(format!(" {} command(s) · Enter run ", commands.len()))
                    .borders(Borders::ALL),
            )
            .row_highlight_style(Style::default().bg(Color::DarkGray).fg(Color::White))
            .highlight_symbol("▶ ");
        frame.render_stateful_widget(table, chunks[1], &mut self.palette_state);
        frame.set_cursor_position((
            chunks[0].x + 1 + self.palette_query.chars().count() as u16,
            chunks[0].y + 1,
        ));
    }
}

impl Drop for App {
    fn drop(&mut self) {
        if self.running_process.is_some() {
            self.stop_running_process();
        }
    }
}

pub fn run_tui() -> Result<()> {
    let root = repository_root()?;
    let mut session = TerminalSession::new()?;
    App::new(root).run(session.terminal_mut())
}

struct TerminalSession {
    terminal: Tui,
}

impl TerminalSession {
    fn new() -> Result<Self> {
        enable_raw_mode().context("enable terminal raw mode")?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(stdout, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(error).context("enter alternate screen");
        }
        let backend = CrosstermBackend::new(stdout);
        match Terminal::new(backend) {
            Ok(terminal) => Ok(Self { terminal }),
            Err(error) => {
                let _ = execute!(io::stdout(), LeaveAlternateScreen);
                let _ = disable_raw_mode();
                Err(error).context("initialize terminal")
            }
        }
    }

    fn terminal_mut(&mut self) -> &mut Tui {
        &mut self.terminal
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

pub(crate) fn repository_root() -> Result<PathBuf> {
    if let Some(root) = env::var_os("L3MS_ROOT").filter(|value| !value.is_empty()) {
        return validate_root(PathBuf::from(root));
    }
    if let Ok(current) = env::current_dir() {
        for candidate in current.ancestors() {
            if is_repository_root(candidate) {
                return Ok(candidate.to_path_buf());
            }
        }
    }
    if let Ok(executable) = env::current_exe() {
        for candidate in executable.ancestors() {
            if is_repository_root(candidate) {
                return Ok(candidate.to_path_buf());
            }
        }
    }
    let development_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    validate_root(development_root)
}

fn validate_root(path: PathBuf) -> Result<PathBuf> {
    let path = path
        .canonicalize()
        .with_context(|| format!("resolve L3MS root {}", path.display()))?;
    anyhow::ensure!(
        is_repository_root(&path),
        "{} is not an L3MS repository root",
        path.display()
    );
    Ok(path)
}

fn is_repository_root(path: &Path) -> bool {
    path.join("llama-swap.yaml").is_file() && path.join("bench-models").is_dir()
}

fn collect_shell_scripts(directory: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut scripts = fs::read_dir(directory)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.extension().is_some_and(|extension| extension == "sh")
                && path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with(prefix))
        })
        .collect::<Vec<_>>();
    scripts.sort();
    scripts
}

fn stream_reader(
    reader: impl io::Read + Send + 'static,
    job_id: u64,
    sender: mpsc::SyncSender<BackgroundEvent>,
) {
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut line = Vec::with_capacity(1024);
        let mut truncated = false;
        loop {
            let available = match reader.fill_buf() {
                Ok([]) => break,
                Ok(available) => available,
                Err(_) => return,
            };
            let newline = available.iter().position(|byte| *byte == b'\n');
            let consumed = newline.map_or(available.len(), |index| index + 1);
            let content = &available[..consumed];
            if line.len() < MAX_PROCESS_LINE_BYTES {
                let remaining = MAX_PROCESS_LINE_BYTES - line.len();
                let content_without_newline = content.strip_suffix(b"\n").unwrap_or(content);
                let keep = content_without_newline.len().min(remaining);
                line.extend_from_slice(&content_without_newline[..keep]);
                truncated |= keep < content_without_newline.len();
            } else {
                truncated = true;
            }
            reader.consume(consumed);

            if newline.is_some() {
                let mut rendered = String::from_utf8_lossy(&line).into_owned();
                if truncated {
                    rendered.push_str(" … [line truncated]");
                }
                if sender
                    .send(BackgroundEvent::ProcessLine {
                        job_id,
                        line: rendered,
                    })
                    .is_err()
                {
                    return;
                }
                line.clear();
                truncated = false;
            }
        }
        if !line.is_empty() || truncated {
            let mut rendered = String::from_utf8_lossy(&line).into_owned();
            if truncated {
                rendered.push_str(" … [line truncated]");
            }
            let _ = sender.send(BackgroundEvent::ProcessLine {
                job_id,
                line: rendered,
            });
        }
    });
}

fn append_bounded_text(target: &mut String, addition: &str, max_bytes: usize) {
    target.push_str(addition);
    if target.len() <= max_bytes {
        return;
    }
    if max_bytes == 0 {
        target.clear();
        return;
    }
    let mut remove = target.len() - max_bytes;
    while remove < target.len() && !target.is_char_boundary(remove) {
        remove += 1;
    }
    target.drain(..remove);
}

#[cfg(unix)]
fn signal_process_group(process_group: u32, signal: &str) -> std::result::Result<(), String> {
    let status = Command::new("/bin/kill")
        .args([
            format!("-{signal}"),
            "--".into(),
            format!("-{process_group}"),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "kill -{signal} for process group {process_group} exited with {status}"
        ))
    }
}

fn build_selected_download_command(
    root: &Path,
    config: &DownloadConfig,
    index: usize,
) -> Result<(String, Vec<OsString>)> {
    let model = config
        .models
        .get(index)
        .with_context(|| format!("download model index {index} is out of range"))?;
    anyhow::ensure!(
        !model.repo_id.trim().is_empty(),
        "selected model has no repo_id"
    );

    let mut parts = vec![
        root.join("model_downloader/download_hf_model.py")
            .into_os_string(),
        "--repo-id".into(),
        model.repo_id.clone().into(),
    ];
    if !model.allow_patterns.is_empty() {
        parts.push("--allow-patterns".into());
        parts.extend(model.allow_patterns.iter().map(OsString::from));
    }
    if !model.ignore_patterns.is_empty() {
        parts.push("--ignore-patterns".into());
        parts.extend(model.ignore_patterns.iter().map(OsString::from));
    }
    if !model.local_dir.is_empty() {
        parts.push("--local-dir".into());
        parts.push(model.local_dir.clone().into());
    } else if !config.base_models_dir.trim().is_empty() {
        parts.push("--base-models-dir".into());
        parts.push(config.base_models_dir.trim().into());
    }
    if !model.revision.is_empty() {
        parts.push("--revision".into());
        parts.push(model.revision.clone().into());
    }
    if model.force_download {
        parts.push("--force-download".into());
    }
    if let Some(workers) = model.max_workers {
        parts.push("--max-workers".into());
        parts.push(workers.to_string().into());
    }
    Ok((model.repo_id.clone(), parts))
}

fn scan_gguf_files(root: &Path) -> Result<Vec<GgufFile>> {
    anyhow::ensure!(root.is_dir(), "{} is not a directory", root.display());
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };
            if file_type.is_dir() && !file_type.is_symlink() {
                pending.push(path);
                continue;
            }
            if !file_type.is_file()
                || !path.extension().is_some_and(|extension| {
                    extension.to_string_lossy().eq_ignore_ascii_case("gguf")
                })
            {
                continue;
            }
            if let Ok(metadata) = entry.metadata() {
                files.push(GgufFile {
                    quantization: infer_quantization(&path),
                    path,
                    size: metadata.len(),
                    modified: metadata.modified().ok(),
                });
            }
        }
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(files)
}

fn infer_quantization(path: &Path) -> String {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_uppercase();
    for marker in [
        "MXFP4", "BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K", "Q4_K_M", "Q4_K", "Q3_K", "Q2_K",
    ] {
        if name.contains(marker) {
            return marker.into();
        }
    }
    "unknown".into()
}

fn display_name(path: &Path) -> String {
    path.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

fn relative_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn value_or_dash(value: &str) -> &str {
    if value.trim().is_empty() {
        "—"
    } else {
        value
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn expand_tilde(value: &str) -> String {
    if value == "~" {
        return env::var("HOME").unwrap_or_else(|_| value.into());
    }
    if let Some(rest) = value.strip_prefix("~/") {
        if let Ok(home) = env::var("HOME") {
            return Path::new(&home).join(rest).display().to_string();
        }
    }
    value.into()
}

fn clamp_table_selection(state: &mut TableState, count: usize) {
    if count == 0 {
        state.select(None);
    } else if state.selected().is_none_or(|index| index >= count) {
        state.select(Some(0));
    }
}

fn clamp_list_selection(state: &mut ListState, count: usize) {
    if count == 0 {
        state.select(None);
    } else if state.selected().is_none_or(|index| index >= count) {
        state.select(Some(0));
    }
}

fn select_previous_table(state: &mut TableState, count: usize) {
    if count == 0 {
        state.select(None);
        return;
    }
    let next = match state.selected() {
        Some(0) | None => count - 1,
        Some(index) => index - 1,
    };
    state.select(Some(next));
}

fn select_next_table(state: &mut TableState, count: usize) {
    if count == 0 {
        state.select(None);
        return;
    }
    state.select(Some(
        state.selected().map_or(0, |index| (index + 1) % count),
    ));
}

fn select_previous_list(state: &mut ListState, count: usize) {
    if count == 0 {
        state.select(None);
        return;
    }
    let next = match state.selected() {
        Some(0) | None => count - 1,
        Some(index) => index - 1,
    };
    state.select(Some(next));
}

fn select_next_list(state: &mut ListState, count: usize) {
    if count == 0 {
        state.select(None);
        return;
    }
    state.select(Some(
        state.selected().map_or(0, |index| (index + 1) % count),
    ));
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_store::ModelConfig;

    #[test]
    fn quantization_is_inferred_from_filename() {
        assert_eq!(
            infer_quantization(Path::new("model-UD-Q4_K_XL.gguf")),
            "Q4_K"
        );
        assert_eq!(infer_quantization(Path::new("model.gguf")), "unknown");
    }

    #[test]
    fn byte_formatter_is_readable() {
        assert_eq!(format_bytes(10), "10 B");
        assert_eq!(format_bytes(1536), "1.5 KiB");
    }

    #[test]
    fn shell_inventory_is_sorted_and_filtered() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("bench-z.sh"), "").unwrap();
        fs::write(directory.path().join("bench-a.sh"), "").unwrap();
        fs::write(directory.path().join("other.sh"), "").unwrap();
        let paths = collect_shell_scripts(directory.path(), "bench-");
        assert_eq!(paths.len(), 2);
        assert!(paths[0].ends_with("bench-a.sh"));
    }

    #[test]
    fn selected_download_keeps_all_patterns_and_config_base_directory() {
        let config = DownloadConfig {
            base_models_dir: "/models".into(),
            models: vec![ModelConfig {
                repo_id: "org/model".into(),
                allow_patterns: vec!["*Q4*".into(), "*mmproj*".into()],
                ignore_patterns: vec!["*old*".into(), "*debug*".into()],
                max_workers: Some(3),
                ..ModelConfig::default()
            }],
        };
        let (_, command) = build_selected_download_command(Path::new("/repo"), &config, 0)
            .expect("download command");
        let command = command
            .iter()
            .map(|part| part.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            command,
            vec![
                "/repo/model_downloader/download_hf_model.py",
                "--repo-id",
                "org/model",
                "--allow-patterns",
                "*Q4*",
                "*mmproj*",
                "--ignore-patterns",
                "*old*",
                "*debug*",
                "--base-models-dir",
                "/models",
                "--max-workers",
                "3",
            ]
        );
    }

    #[test]
    fn process_output_lines_are_bounded() {
        let mut bytes = vec![b'x'; MAX_PROCESS_LINE_BYTES * 2];
        bytes.push(b'\n');
        let (sender, receiver) = mpsc::sync_channel(2);
        stream_reader(io::Cursor::new(bytes), 42, sender);
        let event = receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("bounded output event");
        let BackgroundEvent::ProcessLine { job_id, line } = event else {
            panic!("unexpected background event");
        };
        assert_eq!(job_id, 42);
        assert!(line.len() < MAX_PROCESS_LINE_BYTES + 64);
        assert!(line.ends_with("[line truncated]"));
    }

    #[test]
    fn streaming_preview_is_bounded_at_utf8_boundaries() {
        let mut preview = "old".to_owned();
        append_bounded_text(&mut preview, "éclair", 6);
        assert_eq!(preview, "clair");
        append_bounded_text(&mut preview, "ignored", 0);
        assert!(preview.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn process_group_signal_stops_and_reaps_shell() {
        let mut command = Command::new("sh");
        command
            .args(["-c", "sleep 30 & wait"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0);
        let mut child = command.spawn().expect("spawn isolated shell");
        let process_group = child.id();
        thread::sleep(Duration::from_millis(50));
        signal_process_group(process_group, "TERM").expect("signal process group");

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if child.try_wait().expect("poll child").is_some() {
                break;
            }
            if Instant::now() >= deadline {
                let _ = signal_process_group(process_group, "KILL");
                let _ = child.wait();
                panic!("process group did not stop after TERM");
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
}
