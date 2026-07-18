//! UI-independent state for the Download tab's field editor.
//!
//! [`DownloadEditor`] remains the sole authority for typed configuration,
//! selection, validation, snapshots, and persistence. This module adds the
//! unapplied text fields and focus transitions needed by a terminal view
//! without depending on a rendering or input-event crate.

use std::{env, path::PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::{
    config_store::{csv_to_list, DownloadConfig, ModelConfig},
    download_editor::DownloadEditor,
    text_buffer::TextBuffer,
};

/// A persisted model field in legacy editor order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelField {
    Enabled,
    RepoId,
    Description,
    LocalDir,
    Revision,
    AllowPatterns,
    IgnorePatterns,
    ForceDownload,
    MaxWorkers,
}

impl ModelField {
    pub const ALL: [Self; 9] = [
        Self::Enabled,
        Self::RepoId,
        Self::Description,
        Self::LocalDir,
        Self::Revision,
        Self::AllowPatterns,
        Self::IgnorePatterns,
        Self::ForceDownload,
        Self::MaxWorkers,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::RepoId => "repo_id",
            Self::Description => "description",
            Self::LocalDir => "local_dir",
            Self::Revision => "revision",
            Self::AllowPatterns => "allow_patterns",
            Self::IgnorePatterns => "ignore_patterns",
            Self::ForceDownload => "force_download",
            Self::MaxWorkers => "max_workers",
        }
    }

    pub const fn is_boolean(self) -> bool {
        matches!(self, Self::Enabled | Self::ForceDownload)
    }
}

/// Keyboard focus within the Download tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadFocus {
    Table,
    ConfigPath,
    Versions,
    BaseModelsDir,
    SlowPreset,
    GlobalWorkers,
    SaveNote,
    Model(ModelField),
}

impl DownloadFocus {
    /// Focus traversal omits action buttons because every action has a direct
    /// command. This keeps Tab navigation deterministic in a keyboard UI.
    pub const ORDER: [Self; 16] = [
        Self::Table,
        Self::ConfigPath,
        Self::Versions,
        Self::BaseModelsDir,
        Self::SlowPreset,
        Self::GlobalWorkers,
        Self::SaveNote,
        Self::Model(ModelField::Enabled),
        Self::Model(ModelField::RepoId),
        Self::Model(ModelField::Description),
        Self::Model(ModelField::LocalDir),
        Self::Model(ModelField::Revision),
        Self::Model(ModelField::AllowPatterns),
        Self::Model(ModelField::IgnorePatterns),
        Self::Model(ModelField::ForceDownload),
        Self::Model(ModelField::MaxWorkers),
    ];
}

/// Unapplied form values for one model row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDraft {
    enabled: bool,
    repo_id: TextBuffer,
    description: TextBuffer,
    local_dir: TextBuffer,
    revision: TextBuffer,
    allow_patterns: TextBuffer,
    ignore_patterns: TextBuffer,
    force_download: bool,
    max_workers: TextBuffer,
}

impl Default for ModelDraft {
    fn default() -> Self {
        Self::from_model(None)
    }
}

impl ModelDraft {
    /// Build a canonical display draft. `None` produces the legacy blank row.
    pub fn from_model(model: Option<&ModelConfig>) -> Self {
        let default;
        let model = match model {
            Some(model) => model,
            None => {
                default = ModelConfig::default();
                &default
            }
        };

        Self {
            enabled: model.enabled,
            repo_id: TextBuffer::from_content(model.repo_id.clone()),
            description: TextBuffer::from_content(model.description.clone()),
            local_dir: TextBuffer::from_content(model.local_dir.clone()),
            revision: TextBuffer::from_content(model.revision.clone()),
            allow_patterns: TextBuffer::from_content(model.allow_patterns.join(", ")),
            ignore_patterns: TextBuffer::from_content(model.ignore_patterns.join(", ")),
            force_download: model.force_download,
            max_workers: TextBuffer::from_content(
                model
                    .max_workers
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
            ),
        }
    }

    /// Parse and normalize the draft at the same boundary as the Python form.
    pub fn parse(&self) -> Result<ModelConfig> {
        Ok(ModelConfig {
            enabled: self.enabled,
            repo_id: self.repo_id.content().trim().to_owned(),
            description: self.description.content().trim().to_owned(),
            local_dir: self.local_dir.content().trim().to_owned(),
            revision: self.revision.content().trim().to_owned(),
            allow_patterns: csv_to_list(self.allow_patterns.content()),
            ignore_patterns: csv_to_list(self.ignore_patterns.content()),
            force_download: self.force_download,
            max_workers: parse_optional_positive_integer(
                self.max_workers.content(),
                "model max_workers must be a positive integer or blank",
            )?,
        })
    }

    pub fn boolean(&self, field: ModelField) -> Option<bool> {
        match field {
            ModelField::Enabled => Some(self.enabled),
            ModelField::ForceDownload => Some(self.force_download),
            _ => None,
        }
    }

    pub fn set_boolean(&mut self, field: ModelField, value: bool) -> Result<()> {
        match field {
            ModelField::Enabled => self.enabled = value,
            ModelField::ForceDownload => self.force_download = value,
            _ => return Err(anyhow!("{} is not a boolean field", field.label())),
        }
        Ok(())
    }

    pub fn toggle_boolean(&mut self, field: ModelField) -> Result<bool> {
        let next = !self
            .boolean(field)
            .ok_or_else(|| anyhow!("{} is not a boolean field", field.label()))?;
        self.set_boolean(field, next)?;
        Ok(next)
    }

    pub fn buffer(&self, field: ModelField) -> Option<&TextBuffer> {
        match field {
            ModelField::RepoId => Some(&self.repo_id),
            ModelField::Description => Some(&self.description),
            ModelField::LocalDir => Some(&self.local_dir),
            ModelField::Revision => Some(&self.revision),
            ModelField::AllowPatterns => Some(&self.allow_patterns),
            ModelField::IgnorePatterns => Some(&self.ignore_patterns),
            ModelField::MaxWorkers => Some(&self.max_workers),
            ModelField::Enabled | ModelField::ForceDownload => None,
        }
    }

    pub fn buffer_mut(&mut self, field: ModelField) -> Option<&mut TextBuffer> {
        match field {
            ModelField::RepoId => Some(&mut self.repo_id),
            ModelField::Description => Some(&mut self.description),
            ModelField::LocalDir => Some(&mut self.local_dir),
            ModelField::Revision => Some(&mut self.revision),
            ModelField::AllowPatterns => Some(&mut self.allow_patterns),
            ModelField::IgnorePatterns => Some(&mut self.ignore_patterns),
            ModelField::MaxWorkers => Some(&mut self.max_workers),
            ModelField::Enabled | ModelField::ForceDownload => None,
        }
    }
}

/// Complete non-rendering state for the Download tab.
#[derive(Debug, Clone)]
pub struct DownloadUiState {
    editor: DownloadEditor,
    focus: DownloadFocus,
    config_path: TextBuffer,
    base_models_dir: TextBuffer,
    global_workers: TextBuffer,
    save_note: TextBuffer,
    slow: bool,
    draft: ModelDraft,
    versions: Vec<String>,
    selected_version: Option<usize>,
    history_warning: Option<String>,
}

impl DownloadUiState {
    /// Wrap an existing editor. Version discovery remains explicit so a TUI
    /// can still open the config when its optional history directory is bad.
    pub fn new(editor: DownloadEditor) -> Self {
        let config_path = TextBuffer::from_content(editor.config_path().display().to_string());
        let base_models_dir = TextBuffer::from_content(editor.config().base_models_dir.clone());
        let draft = ModelDraft::from_model(editor.selected_model());
        Self {
            editor,
            focus: DownloadFocus::Table,
            config_path,
            base_models_dir,
            global_workers: TextBuffer::new(),
            save_note: TextBuffer::new(),
            slow: true,
            draft,
            versions: Vec::new(),
            selected_version: None,
            history_warning: None,
        }
    }

    /// Strictly open a config. Snapshot discovery is secondary: a history I/O
    /// failure is retained as a warning without blocking a valid config.
    pub fn open(config_path: impl Into<PathBuf>, version_root: impl Into<PathBuf>) -> Result<Self> {
        let editor = DownloadEditor::open(config_path, version_root)?;
        let mut state = Self::new(editor);
        state.refresh_versions_secondary("config loaded");
        Ok(state)
    }

    pub fn editor(&self) -> &DownloadEditor {
        &self.editor
    }

    pub fn config(&self) -> &DownloadConfig {
        self.editor.config()
    }

    pub fn models(&self) -> &[ModelConfig] {
        self.editor.models()
    }

    pub fn selected_index(&self) -> Option<usize> {
        self.editor.selected_index()
    }

    pub fn selected_model(&self) -> Option<&ModelConfig> {
        self.editor.selected_model()
    }

    pub fn focus(&self) -> DownloadFocus {
        self.focus
    }

    /// Set focus and place a focused line buffer's cursor at its end.
    pub fn set_focus(&mut self, focus: DownloadFocus) {
        self.focus = focus;
        if let Some(buffer) = self.focused_buffer_mut() {
            buffer.move_end();
        }
    }

    pub fn focus_next(&mut self) -> DownloadFocus {
        let index = focus_index(self.focus);
        let next = DownloadFocus::ORDER[(index + 1) % DownloadFocus::ORDER.len()];
        self.set_focus(next);
        next
    }

    pub fn focus_previous(&mut self) -> DownloadFocus {
        let index = focus_index(self.focus);
        let previous = DownloadFocus::ORDER
            [(index + DownloadFocus::ORDER.len() - 1) % DownloadFocus::ORDER.len()];
        self.set_focus(previous);
        previous
    }

    pub fn config_path_buffer(&self) -> &TextBuffer {
        &self.config_path
    }

    pub fn config_path_buffer_mut(&mut self) -> &mut TextBuffer {
        &mut self.config_path
    }

    pub fn base_models_dir_buffer(&self) -> &TextBuffer {
        &self.base_models_dir
    }

    pub fn base_models_dir_buffer_mut(&mut self) -> &mut TextBuffer {
        &mut self.base_models_dir
    }

    pub fn global_workers_buffer(&self) -> &TextBuffer {
        &self.global_workers
    }

    pub fn global_workers_buffer_mut(&mut self) -> &mut TextBuffer {
        &mut self.global_workers
    }

    pub fn save_note_buffer(&self) -> &TextBuffer {
        &self.save_note
    }

    pub fn save_note_buffer_mut(&mut self) -> &mut TextBuffer {
        &mut self.save_note
    }

    pub fn draft(&self) -> &ModelDraft {
        &self.draft
    }

    pub fn draft_mut(&mut self) -> &mut ModelDraft {
        &mut self.draft
    }

    /// Return the active line buffer, excluding checkboxes, lists, and tables.
    pub fn focused_buffer(&self) -> Option<&TextBuffer> {
        match self.focus {
            DownloadFocus::ConfigPath => Some(&self.config_path),
            DownloadFocus::BaseModelsDir => Some(&self.base_models_dir),
            DownloadFocus::GlobalWorkers => Some(&self.global_workers),
            DownloadFocus::SaveNote => Some(&self.save_note),
            DownloadFocus::Model(field) => self.draft.buffer(field),
            DownloadFocus::Table | DownloadFocus::Versions | DownloadFocus::SlowPreset => None,
        }
    }

    pub fn focused_buffer_mut(&mut self) -> Option<&mut TextBuffer> {
        match self.focus {
            DownloadFocus::ConfigPath => Some(&mut self.config_path),
            DownloadFocus::BaseModelsDir => Some(&mut self.base_models_dir),
            DownloadFocus::GlobalWorkers => Some(&mut self.global_workers),
            DownloadFocus::SaveNote => Some(&mut self.save_note),
            DownloadFocus::Model(field) => self.draft.buffer_mut(field),
            DownloadFocus::Table | DownloadFocus::Versions | DownloadFocus::SlowPreset => None,
        }
    }

    pub fn slow(&self) -> bool {
        self.slow
    }

    pub fn set_slow(&mut self, slow: bool) {
        self.slow = slow;
    }

    pub fn toggle_slow(&mut self) -> bool {
        self.slow = !self.slow;
        self.slow
    }

    /// Parse the runtime worker override without changing state.
    pub fn global_max_workers(&self) -> Result<Option<u64>> {
        parse_optional_positive_integer(
            self.global_workers.content(),
            "max workers override must be a positive integer or blank",
        )
    }

    /// Legacy speed arguments: an explicit worker count wins over slow mode.
    pub fn speed_args(&self) -> Result<Vec<String>> {
        if let Some(workers) = self.global_max_workers()? {
            Ok(vec!["--max-workers".to_owned(), workers.to_string()])
        } else if self.slow {
            Ok(vec!["--slow".to_owned()])
        } else {
            Ok(Vec::new())
        }
    }

    /// Derived dirty state includes persisted edits and unapplied form values.
    /// Runtime speed, path, save-note, version, and focus changes are excluded.
    pub fn is_dirty(&self) -> bool {
        self.editor.is_dirty()
            || self.has_unapplied_model_changes()
            || self.has_unapplied_base_dir_change()
    }

    pub fn has_unapplied_model_changes(&self) -> bool {
        self.editor.selected_model().is_some()
            && self.draft != ModelDraft::from_model(self.editor.selected_model())
    }

    pub fn has_unapplied_base_dir_change(&self) -> bool {
        self.base_models_dir.content() != self.editor.config().base_models_dir
    }

    /// Select and load a row draft. Out-of-range indices clamp to the last row.
    pub fn select(&mut self, index: usize) -> Option<usize> {
        let selected = self.editor.select(index);
        self.sync_model_draft();
        selected
    }

    pub fn select_previous(&mut self) -> Option<usize> {
        let count = self.editor.models().len();
        if count == 0 {
            self.editor.clear_selection();
            self.sync_model_draft();
            return None;
        }
        let previous = match self.editor.selected_index() {
            Some(0) | None => count - 1,
            Some(index) => index - 1,
        };
        self.select(previous)
    }

    pub fn select_next(&mut self) -> Option<usize> {
        let count = self.editor.models().len();
        if count == 0 {
            self.editor.clear_selection();
            self.sync_model_draft();
            return None;
        }
        let next = self
            .editor
            .selected_index()
            .map_or(0, |index| (index + 1) % count);
        self.select(next)
    }

    /// Add the legacy blank row, select it, and focus its repository field.
    pub fn add_model(&mut self) -> usize {
        let index = self.editor.add_model();
        self.sync_model_draft();
        self.set_focus(DownloadFocus::Model(ModelField::RepoId));
        index
    }

    /// Parse and normalize the current draft into the selected model.
    pub fn apply_selected(&mut self) -> Result<usize> {
        let model = self.draft.parse()?;
        let index = self.editor.apply_selected(model)?;
        self.sync_model_draft();
        Ok(index)
    }

    pub fn delete_selected(&mut self) -> Option<ModelConfig> {
        let removed = self.editor.delete_selected();
        self.sync_model_draft();
        removed
    }

    /// Toggle the persisted row while retaining other unapplied draft fields.
    pub fn toggle_selected_enabled(&mut self) -> Result<bool> {
        let enabled = self.editor.toggle_selected_enabled()?;
        self.draft.set_boolean(ModelField::Enabled, enabled)?;
        Ok(enabled)
    }

    /// Apply pending fields and return deterministic typed validation errors.
    pub fn validate(&mut self) -> Result<Vec<String>> {
        self.apply_form()?;
        Ok(self.editor.validation_errors())
    }

    /// Apply pending fields and save with a snapshot.
    ///
    /// Snapshot discovery is secondary to the completed write. A listing
    /// failure leaves the editor clean and is available through
    /// [`Self::take_history_warning`].
    pub fn save(&mut self) -> Result<()> {
        self.apply_form()?;
        let note = self.save_note.content().trim();
        let note = if note.is_empty() { "manual-save" } else { note };
        self.editor.save(note)?;
        self.sync_persisted_fields();
        self.refresh_versions_secondary("config saved");
        Ok(())
    }

    /// Strictly replace the editor from the config-path field.
    ///
    /// Relative paths use the caller process directory for compatibility.
    /// Runtime applications should prefer [`Self::reload_path_in`] and pass
    /// their explicit repository root.
    pub fn reload_path(&mut self) -> Result<()> {
        let base = env::current_dir().context("resolve current directory for config path")?;
        self.reload_path_in(&base)
    }

    /// Strictly replace the editor, resolving relative paths against the
    /// runtime repository rather than the caller process directory.
    pub fn reload_path_in(&mut self, base: &std::path::Path) -> Result<()> {
        let candidate_path = expand_config_path(self.config_path.content(), base)?;
        let replacement =
            DownloadEditor::open(&candidate_path, self.editor.version_root().to_path_buf())?;
        let (replacement_versions, history_warning) = match replacement.versions() {
            Ok(versions) => (versions, None),
            Err(error) => (
                Vec::new(),
                Some(format!(
                    "config loaded, but snapshots could not be listed: {error:#}"
                )),
            ),
        };

        self.editor = replacement;
        self.config_path
            .set_content(candidate_path.display().to_string());
        self.versions = replacement_versions;
        self.selected_version = None;
        self.history_warning = history_warning;
        self.sync_persisted_fields();
        Ok(())
    }

    pub fn versions(&self) -> &[String] {
        &self.versions
    }

    pub fn selected_version_index(&self) -> Option<usize> {
        self.selected_version
    }

    pub fn selected_version(&self) -> Option<&str> {
        self.selected_version
            .and_then(|index| self.versions.get(index))
            .map(String::as_str)
    }

    pub fn refresh_versions(&mut self) -> Result<()> {
        let selected_name = self.selected_version().map(str::to_owned);
        let versions = self.editor.versions()?;
        self.selected_version = selected_name
            .as_ref()
            .and_then(|name| versions.iter().position(|candidate| candidate == name));
        self.versions = versions;
        self.history_warning = None;
        Ok(())
    }

    /// Consume the most recent non-fatal snapshot-list warning.
    pub fn take_history_warning(&mut self) -> Option<String> {
        self.history_warning.take()
    }

    pub fn select_version(&mut self, index: usize) -> Option<usize> {
        self.selected_version = if self.versions.is_empty() {
            None
        } else {
            Some(index.min(self.versions.len() - 1))
        };
        self.selected_version
    }

    pub fn clear_version_selection(&mut self) {
        self.selected_version = None;
    }

    pub fn select_previous_version(&mut self) -> Option<usize> {
        let count = self.versions.len();
        if count == 0 {
            self.selected_version = None;
            return None;
        }
        let previous = match self.selected_version {
            Some(0) | None => count - 1,
            Some(index) => index - 1,
        };
        self.selected_version = Some(previous);
        self.selected_version
    }

    pub fn select_next_version(&mut self) -> Option<usize> {
        let count = self.versions.len();
        if count == 0 {
            self.selected_version = None;
            return None;
        }
        self.selected_version = Some(self.selected_version.map_or(0, |index| (index + 1) % count));
        self.selected_version
    }

    pub fn restore_selected_version(&mut self) -> Result<String> {
        let version = self
            .selected_version()
            .ok_or_else(|| anyhow!("no version selected"))?
            .to_owned();
        self.restore_version(&version)?;
        Ok(version)
    }

    /// Restore a strictly valid snapshot and refresh persisted form fields.
    /// Invalid snapshot bytes leave disk, core state, and unapplied fields
    /// unchanged. A later snapshot-list failure is retained as a secondary
    /// warning and does not turn the completed restore into an error.
    pub fn restore_version(&mut self, version_name: &str) -> Result<()> {
        self.editor.restore(version_name)?;
        self.sync_persisted_fields();
        self.refresh_versions_secondary("config restored");
        self.selected_version = self
            .versions
            .iter()
            .position(|candidate| candidate == version_name);
        Ok(())
    }

    fn apply_form(&mut self) -> Result<()> {
        if self.editor.selected_model().is_some() {
            self.apply_selected()?;
        }
        let base_models_dir = self.base_models_dir.content().trim().to_owned();
        self.editor.set_base_models_dir(base_models_dir.clone());
        self.base_models_dir.set_content(base_models_dir);
        Ok(())
    }

    fn sync_model_draft(&mut self) {
        self.draft = ModelDraft::from_model(self.editor.selected_model());
    }

    fn sync_persisted_fields(&mut self) {
        self.config_path
            .set_content(self.editor.config_path().display().to_string());
        self.base_models_dir
            .set_content(self.editor.config().base_models_dir.clone());
        self.sync_model_draft();
    }

    fn refresh_versions_secondary(&mut self, context: &str) {
        match self.refresh_versions() {
            Ok(()) => {}
            Err(error) => {
                self.versions.clear();
                self.selected_version = None;
                self.history_warning = Some(format!(
                    "{context}, but snapshots could not be listed: {error:#}"
                ));
            }
        }
    }
}

fn focus_index(focus: DownloadFocus) -> usize {
    DownloadFocus::ORDER
        .iter()
        .position(|candidate| *candidate == focus)
        .expect("all DownloadFocus values are present in ORDER")
}

fn parse_optional_positive_integer(raw: &str, message: &str) -> Result<Option<u64>> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    if !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(anyhow!(message.to_owned()));
    }
    let value = raw
        .parse::<u64>()
        .map_err(|_| anyhow!(message.to_owned()))?;
    if value == 0 {
        return Err(anyhow!(message.to_owned()));
    }
    Ok(Some(value))
}

fn expand_config_path(raw: &str, base: &std::path::Path) -> Result<PathBuf> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(anyhow!("config path is required"));
    }

    if raw == "~" || raw.starts_with("~/") {
        let home = env::var_os("HOME")
            .or_else(|| env::var_os("USERPROFILE"))
            .ok_or_else(|| anyhow!("cannot expand config path: home directory is unavailable"))?;
        let mut path = PathBuf::from(home);
        if let Some(remainder) = raw.strip_prefix("~/") {
            path.push(remainder);
        }
        Ok(path)
    } else {
        let path = PathBuf::from(raw);
        if path.is_absolute() {
            Ok(path)
        } else {
            Ok(base.join(path))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    use tempfile::TempDir;

    use super::*;

    fn model(repo_id: &str) -> ModelConfig {
        ModelConfig {
            repo_id: repo_id.to_owned(),
            ..ModelConfig::default()
        }
    }

    fn config(base: &str, repos: &[&str]) -> DownloadConfig {
        DownloadConfig {
            base_models_dir: base.to_owned(),
            models: repos.iter().map(|repo| model(repo)).collect(),
        }
    }

    fn write_config(path: &std::path::Path, config: &DownloadConfig) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut bytes = serde_json::to_vec_pretty(config).unwrap();
        bytes.push(b'\n');
        fs::write(path, bytes).unwrap();
    }

    fn open_state(temp: &TempDir, repos: &[&str]) -> DownloadUiState {
        let path = temp.path().join("configs/models.json");
        write_config(&path, &config("/models", repos));
        DownloadUiState::open(&path, temp.path().join("versions")).unwrap()
    }

    #[cfg(unix)]
    fn only_version_directory(version_root: &std::path::Path) -> PathBuf {
        let entries = fs::read_dir(version_root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 1, "expected one config history directory");
        entries.into_iter().next().unwrap()
    }

    #[test]
    fn model_draft_round_trips_every_field_and_parses_csv_and_workers() {
        let source = ModelConfig {
            enabled: false,
            repo_id: "org/model".to_owned(),
            description: "demo".to_owned(),
            local_dir: "/models/demo".to_owned(),
            revision: "main".to_owned(),
            allow_patterns: vec!["*.gguf".to_owned(), "mmproj*".to_owned()],
            ignore_patterns: vec!["*f16*".to_owned()],
            force_download: true,
            max_workers: Some(4),
        };
        let mut draft = ModelDraft::from_model(Some(&source));
        assert_eq!(draft.parse().unwrap(), source);
        assert_eq!(draft.boolean(ModelField::Enabled), Some(false));
        assert_eq!(draft.boolean(ModelField::RepoId), None);

        draft
            .buffer_mut(ModelField::RepoId)
            .unwrap()
            .set_content(" org/changed ");
        draft
            .buffer_mut(ModelField::AllowPatterns)
            .unwrap()
            .set_content(" *.gguf, , mmproj* ");
        draft
            .buffer_mut(ModelField::MaxWorkers)
            .unwrap()
            .set_content("8");
        draft.toggle_boolean(ModelField::Enabled).unwrap();
        let parsed = draft.parse().unwrap();
        assert_eq!(parsed.repo_id, "org/changed");
        assert_eq!(parsed.allow_patterns, ["*.gguf", "mmproj*"]);
        assert_eq!(parsed.max_workers, Some(8));
        assert!(parsed.enabled);

        draft
            .buffer_mut(ModelField::MaxWorkers)
            .unwrap()
            .set_content("0");
        assert!(draft
            .parse()
            .unwrap_err()
            .to_string()
            .contains("positive integer or blank"));
        assert!(draft.set_boolean(ModelField::RepoId, true).is_err());
    }

    #[test]
    fn dirty_state_includes_unapplied_model_and_base_but_not_runtime_fields() {
        let temp = TempDir::new().unwrap();
        let mut state = open_state(&temp, &["org/one"]);
        assert!(!state.is_dirty());

        state
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("pending");
        assert!(state.has_unapplied_model_changes());
        assert!(state.is_dirty());
        state
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("");
        assert!(!state.is_dirty());

        state.base_models_dir_buffer_mut().set_content("/other");
        assert!(state.has_unapplied_base_dir_change());
        assert!(state.is_dirty());
        state.base_models_dir_buffer_mut().set_content("/models");
        assert!(!state.is_dirty());

        state.config_path_buffer_mut().set_content("elsewhere.json");
        state.global_workers_buffer_mut().set_content("3");
        state.save_note_buffer_mut().set_content("note");
        state.toggle_slow();
        state.set_focus(DownloadFocus::SaveNote);
        assert!(!state.is_dirty());
    }

    #[test]
    fn selection_add_apply_delete_and_toggle_stay_synchronized() {
        let temp = TempDir::new().unwrap();
        let mut state = open_state(&temp, &["org/one", "org/two"]);
        assert_eq!(state.select_previous(), Some(1));
        assert_eq!(state.draft().parse().unwrap().repo_id, "org/two");
        assert_eq!(state.select_next(), Some(0));

        state
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("discard me");
        state.select(1);
        assert_eq!(state.draft().parse().unwrap().description, "");

        assert_eq!(state.add_model(), 2);
        assert_eq!(state.focus(), DownloadFocus::Model(ModelField::RepoId));
        state
            .draft_mut()
            .buffer_mut(ModelField::RepoId)
            .unwrap()
            .set_content(" org/three ");
        state.apply_selected().unwrap();
        assert_eq!(state.selected_model().unwrap().repo_id, "org/three");

        assert!(!state.toggle_selected_enabled().unwrap());
        assert_eq!(state.draft().boolean(ModelField::Enabled), Some(false));
        assert_eq!(state.delete_selected().unwrap().repo_id, "org/three");
        assert_eq!(state.selected_index(), Some(1));
        assert_eq!(state.draft().parse().unwrap().repo_id, "org/two");
    }

    #[test]
    fn validation_applies_draft_and_base_and_parse_failure_is_non_mutating() {
        let temp = TempDir::new().unwrap();
        let mut state = open_state(&temp, &["org/one"]);
        state
            .draft_mut()
            .buffer_mut(ModelField::RepoId)
            .unwrap()
            .set_content("   ");
        state
            .base_models_dir_buffer_mut()
            .set_content(" /trimmed/base ");
        assert_eq!(state.validate().unwrap(), ["models[0].repo_id is required"]);
        assert_eq!(state.config().base_models_dir, "/trimmed/base");
        assert_eq!(state.base_models_dir_buffer().content(), "/trimmed/base");

        state
            .draft_mut()
            .buffer_mut(ModelField::MaxWorkers)
            .unwrap()
            .set_content("bad");
        state
            .base_models_dir_buffer_mut()
            .set_content("/not-applied");
        let before = state.config().clone();
        assert!(state.validate().is_err());
        assert_eq!(state.config(), &before);
    }

    #[test]
    fn save_normalizes_marks_clean_and_refreshes_snapshot_names() {
        let temp = TempDir::new().unwrap();
        let mut state = open_state(&temp, &["org/one"]);
        state
            .draft_mut()
            .buffer_mut(ModelField::RepoId)
            .unwrap()
            .set_content(" org/changed ");
        state
            .base_models_dir_buffer_mut()
            .set_content(" /new/base ");
        state.save_note_buffer_mut().set_content(" field edit! ");

        state.save().unwrap();

        assert!(!state.is_dirty());
        assert_eq!(state.selected_model().unwrap().repo_id, "org/changed");
        assert_eq!(state.config().base_models_dir, "/new/base");
        assert_eq!(state.versions().len(), 1);
        assert!(state.versions()[0].ends_with("__field-edit.json"));
    }

    #[test]
    fn valid_open_survives_an_unreadable_history_root_and_exposes_one_warning() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("history-is-a-file");
        let expected = config("/models", &["org/valid"]);
        write_config(&path, &expected);
        fs::write(&version_root, "not a directory").unwrap();

        let mut state = DownloadUiState::open(&path, &version_root).unwrap();

        assert_eq!(state.config(), &expected);
        assert_eq!(state.editor().config_path(), path);
        assert_eq!(
            state.config_path_buffer().content(),
            path.display().to_string()
        );
        assert!(!state.is_dirty());
        assert!(state.versions().is_empty());
        let warning = state.take_history_warning().unwrap();
        assert!(warning.contains("config loaded"));
        assert!(warning.contains("snapshots could not be listed"));
        assert!(state.take_history_warning().is_none());
    }

    #[test]
    fn relative_reload_uses_explicit_runtime_base_and_resynchronizes_clean_fields() {
        let temp = TempDir::new().unwrap();
        let runtime_root = temp.path().join("runtime-root");
        let relative_path = PathBuf::from("alternate/models.json");
        let target = runtime_root.join(&relative_path);
        write_config(&target, &config("/alternate", &["org/alternate"]));

        let mut state = open_state(&temp, &["org/original"]);
        state
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("discarded by successful reload");
        state.base_models_dir_buffer_mut().set_content("/pending");
        state
            .config_path_buffer_mut()
            .set_content(relative_path.display().to_string());
        assert!(state.is_dirty());

        state.reload_path_in(&runtime_root).unwrap();

        assert_eq!(state.editor().config_path(), target);
        assert_eq!(
            state.config_path_buffer().content(),
            target.display().to_string()
        );
        assert_eq!(state.config().base_models_dir, "/alternate");
        assert_eq!(state.base_models_dir_buffer().content(), "/alternate");
        assert_eq!(state.selected_model().unwrap().repo_id, "org/alternate");
        assert_eq!(state.draft().parse().unwrap().repo_id, "org/alternate");
        assert!(!state.is_dirty());
        assert!(state.take_history_warning().is_none());
    }

    #[test]
    fn relative_reload_keeps_valid_replacement_when_history_listing_fails() {
        let temp = TempDir::new().unwrap();
        let runtime_root = temp.path().join("runtime-root");
        let first_path = runtime_root.join("first.json");
        let second_path = runtime_root.join("nested/second.json");
        let version_root = temp.path().join("history-is-a-file");
        write_config(&first_path, &config("/first", &["org/first"]));
        write_config(&second_path, &config("/second", &["org/second"]));
        fs::write(&version_root, "not a directory").unwrap();
        let mut state = DownloadUiState::open(&first_path, &version_root).unwrap();
        let _ = state.take_history_warning();
        state
            .config_path_buffer_mut()
            .set_content("nested/second.json");

        state.reload_path_in(&runtime_root).unwrap();

        assert_eq!(state.editor().config_path(), second_path);
        assert_eq!(
            state.config_path_buffer().content(),
            second_path.display().to_string()
        );
        assert_eq!(state.selected_model().unwrap().repo_id, "org/second");
        assert!(!state.is_dirty());
        let warning = state.take_history_warning().unwrap();
        assert!(warning.contains("config loaded"));
        assert!(warning.contains("snapshots could not be listed"));
        assert!(state.take_history_warning().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn successful_save_remains_successful_when_post_write_history_listing_fails() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("versions");
        let mut state = open_state(&temp, &["org/original"]);
        state.save().unwrap();
        let history_dir = only_version_directory(&version_root);
        let original_permissions = fs::metadata(&history_dir).unwrap().permissions();
        fs::set_permissions(&history_dir, fs::Permissions::from_mode(0o300)).unwrap();
        state
            .draft_mut()
            .buffer_mut(ModelField::RepoId)
            .unwrap()
            .set_content("org/saved");

        let result = state.save();
        let warning = state.take_history_warning();
        fs::set_permissions(&history_dir, original_permissions).unwrap();

        result.unwrap();
        assert_eq!(state.selected_model().unwrap().repo_id, "org/saved");
        assert_eq!(
            crate::config_store::load_config_strict(&path).unwrap(),
            state.config().clone()
        );
        assert!(!state.is_dirty());
        assert!(state.versions().is_empty());
        let warning = warning.expect("history listing warning");
        assert!(warning.contains("config saved"));
        assert!(warning.contains("snapshots could not be listed"));
    }

    #[cfg(unix)]
    #[test]
    fn successful_restore_remains_successful_when_post_write_history_listing_fails() {
        let temp = TempDir::new().unwrap();
        let version_root = temp.path().join("versions");
        let mut state = open_state(&temp, &["org/original"]);
        state
            .draft_mut()
            .buffer_mut(ModelField::RepoId)
            .unwrap()
            .set_content("org/current");
        state.save().unwrap();
        state.select_version(0);
        let selected_version = state.selected_version().unwrap().to_owned();
        let history_dir = only_version_directory(&version_root);
        let original_permissions = fs::metadata(&history_dir).unwrap().permissions();
        // Restore writes an undo snapshot before replacing the config, so the
        // directory remains writable/executable while enumeration is denied.
        fs::set_permissions(&history_dir, fs::Permissions::from_mode(0o300)).unwrap();

        let result = state.restore_selected_version();
        let warning = state.take_history_warning();
        fs::set_permissions(&history_dir, original_permissions).unwrap();

        assert_eq!(result.unwrap(), selected_version);
        assert_eq!(state.selected_model().unwrap().repo_id, "org/original");
        assert!(!state.is_dirty());
        assert!(state.versions().is_empty());
        assert!(state.selected_version().is_none());
        let warning = warning.expect("history listing warning");
        assert!(warning.contains("config restored"));
        assert!(warning.contains("snapshots could not be listed"));
    }

    #[test]
    fn reload_path_is_strict_transactional_and_preserves_runtime_options() {
        let temp = TempDir::new().unwrap();
        let first_path = temp.path().join("first/models.json");
        let second_path = temp.path().join("second/models.json");
        write_config(&first_path, &config("/first", &["org/first"]));
        fs::create_dir_all(second_path.parent().unwrap()).unwrap();
        fs::write(&second_path, "not json").unwrap();
        let mut state = DownloadUiState::open(&first_path, temp.path().join("versions")).unwrap();
        state.global_workers_buffer_mut().set_content("7");
        state.save_note_buffer_mut().set_content("keep");
        state.set_slow(false);
        state
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("pending");
        state
            .config_path_buffer_mut()
            .set_content(second_path.display().to_string());

        let before_config = state.config().clone();
        let before_draft = state.draft().clone();
        assert!(state.reload_path().is_err());
        assert_eq!(state.config(), &before_config);
        assert_eq!(state.draft(), &before_draft);
        assert_eq!(state.editor().config_path(), first_path);

        write_config(&second_path, &config("/second", &["org/second"]));
        state.reload_path().unwrap();
        assert_eq!(state.editor().config_path(), second_path);
        assert_eq!(state.selected_model().unwrap().repo_id, "org/second");
        assert_eq!(state.base_models_dir_buffer().content(), "/second");
        assert_eq!(state.global_max_workers().unwrap(), Some(7));
        assert_eq!(state.save_note_buffer().content(), "keep");
        assert!(!state.slow());
        assert!(!state.is_dirty());
    }

    #[test]
    fn version_selection_and_restore_refresh_all_persisted_fields() {
        let temp = TempDir::new().unwrap();
        let mut state = open_state(&temp, &["org/original", "org/two"]);
        state.select(1);
        state
            .draft_mut()
            .buffer_mut(ModelField::RepoId)
            .unwrap()
            .set_content("org/changed");
        state.save_note_buffer_mut().set_content("change");
        state.save().unwrap();
        assert_eq!(state.select_version(99), Some(0));
        let version = state.selected_version().unwrap().to_owned();

        state.base_models_dir_buffer_mut().set_content("pending");
        state
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("pending");
        assert!(state.is_dirty());
        assert_eq!(state.restore_selected_version().unwrap(), version);

        assert!(!state.is_dirty());
        assert_eq!(
            state.config(),
            &config("/models", &["org/original", "org/two"])
        );
        assert_eq!(state.selected_index(), Some(1));
        assert_eq!(state.selected_model().unwrap().repo_id, "org/two");
        assert_eq!(state.selected_version(), Some(version.as_str()));
    }

    #[test]
    fn invalid_restore_preserves_disk_core_and_unapplied_form_state() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("versions");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "not json").unwrap();
        let current = config("/current", &["org/current"]);
        crate::config_store::save_config_in(&path, &current, "replace-invalid", &version_root)
            .unwrap();

        let mut state = DownloadUiState::open(&path, &version_root).unwrap();
        assert_eq!(state.select_version(0), Some(0));
        state
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("pending model edit");
        state
            .base_models_dir_buffer_mut()
            .set_content("/pending/base");
        let before_bytes = fs::read(&path).unwrap();
        let before_config = state.config().clone();
        let before_draft = state.draft().clone();

        let error = state.restore_selected_version().unwrap_err();

        assert!(error.to_string().contains("failed to parse config"));
        assert_eq!(fs::read(&path).unwrap(), before_bytes);
        assert_eq!(state.config(), &before_config);
        assert_eq!(state.draft(), &before_draft);
        assert_eq!(state.base_models_dir_buffer().content(), "/pending/base");
        assert!(state.is_dirty());
        assert_eq!(
            crate::config_store::load_config_strict(&path).unwrap(),
            before_config
        );
    }

    #[test]
    fn focus_traversal_and_focused_buffers_are_typed() {
        let temp = TempDir::new().unwrap();
        let mut state = open_state(&temp, &["org/one"]);
        assert_eq!(state.focus(), DownloadFocus::Table);
        assert_eq!(state.focus_next(), DownloadFocus::ConfigPath);
        assert!(state.focused_buffer().is_some());
        state.set_focus(DownloadFocus::Model(ModelField::Enabled));
        assert!(state.focused_buffer().is_none());
        state.set_focus(DownloadFocus::Model(ModelField::Description));
        assert!(state.focused_buffer_mut().is_some());
        assert_eq!(
            state.focus_previous(),
            DownloadFocus::Model(ModelField::RepoId)
        );
    }

    #[test]
    fn speed_arguments_validate_override_and_apply_precedence() {
        let temp = TempDir::new().unwrap();
        let mut state = open_state(&temp, &["org/one"]);
        assert_eq!(state.speed_args().unwrap(), ["--slow"]);
        state.global_workers_buffer_mut().set_content("12");
        assert_eq!(state.speed_args().unwrap(), ["--max-workers", "12"]);
        state.global_workers_buffer_mut().set_content("0");
        assert!(state.speed_args().is_err());
        state.global_workers_buffer_mut().set_content("");
        state.set_slow(false);
        assert!(state.speed_args().unwrap().is_empty());
    }
}
