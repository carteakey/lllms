//! UI-independent editing state for model downloader configurations.
//!
//! This module owns selection and dirty-state transitions while delegating all
//! schema normalization, validation, snapshots, path containment, and atomic
//! persistence to [`crate::config_store`].  Keeping those concerns separate
//! lets terminal surfaces edit the legacy JSON format without reimplementing
//! its storage rules.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

use crate::config_store::{
    list_versions_in, load_config_strict, normalize_download_config, restore_version_and_load_in,
    save_config_in, validate_config, DownloadConfig, ModelConfig,
};

/// Mutable downloader configuration plus its persistent editing context.
///
/// A configuration is considered dirty whenever it differs from the last
/// successful strict load, save, or restore. Selection changes do not affect
/// dirty state.
#[derive(Debug, Clone)]
pub struct DownloadEditor {
    config_path: PathBuf,
    version_root: PathBuf,
    config: DownloadConfig,
    clean_config: DownloadConfig,
    selected_index: Option<usize>,
}

impl DownloadEditor {
    /// Create an empty editor without reading the target path.
    ///
    /// Use [`Self::open`] when the config must already exist and malformed or
    /// unreadable files should be reported immediately.
    pub fn new(config_path: impl Into<PathBuf>, version_root: impl Into<PathBuf>) -> Self {
        Self::from_config(config_path, version_root, DownloadConfig::default())
    }

    /// Create an editor around an in-memory configuration and mark it clean.
    ///
    /// This is useful for callers that construct their state before choosing
    /// when to read or write disk. The supplied typed values are retained as-is
    /// so [`Self::validation_errors`] can report invalid typed input.
    pub fn from_config(
        config_path: impl Into<PathBuf>,
        version_root: impl Into<PathBuf>,
        config: DownloadConfig,
    ) -> Self {
        let selected_index = (!config.models.is_empty()).then_some(0);
        Self {
            config_path: config_path.into(),
            version_root: version_root.into(),
            clean_config: config.clone(),
            config,
            selected_index,
        }
    }

    /// Strictly load an existing config and select its first model, if any.
    pub fn open(config_path: impl Into<PathBuf>, version_root: impl Into<PathBuf>) -> Result<Self> {
        let config_path = config_path.into();
        let config = load_config_strict(&config_path)?;
        Ok(Self::from_config(config_path, version_root, config))
    }

    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    pub fn version_root(&self) -> &Path {
        &self.version_root
    }

    pub fn config(&self) -> &DownloadConfig {
        &self.config
    }

    pub fn models(&self) -> &[ModelConfig] {
        &self.config.models
    }

    pub fn selected_index(&self) -> Option<usize> {
        self.selected_index
    }

    pub fn selected_model(&self) -> Option<&ModelConfig> {
        self.selected_index
            .and_then(|index| self.config.models.get(index))
    }

    /// Whether the editable config differs from its last persisted baseline.
    pub fn is_dirty(&self) -> bool {
        self.config != self.clean_config
    }

    /// Select a model, clamping indices beyond the end to the final row.
    ///
    /// Empty configurations always produce no selection.
    pub fn select(&mut self, index: usize) -> Option<usize> {
        self.selected_index = clamp_index(Some(index), self.config.models.len());
        self.selected_index
    }

    /// Explicitly clear the current selection without changing the config.
    pub fn clear_selection(&mut self) {
        self.selected_index = None;
    }

    /// Update the root directory field. Normalization remains a save concern.
    pub fn set_base_models_dir(&mut self, base_models_dir: impl Into<String>) {
        self.config.base_models_dir = base_models_dir.into();
    }

    /// Append a blank legacy-compatible model row and select it.
    pub fn add_model(&mut self) -> usize {
        self.add_model_config(ModelConfig::default())
    }

    /// Append a typed model after applying the config store's normalization.
    pub fn add_model_config(&mut self, model: ModelConfig) -> usize {
        let index = self.config.models.len();
        self.config.models.push(normalize_model_config(model));
        self.selected_index = Some(index);
        index
    }

    /// Apply a typed editor value to the selected row.
    pub fn apply_selected(&mut self, model: ModelConfig) -> Result<usize> {
        let index = self
            .selected_index
            .ok_or_else(|| anyhow!("no model selected"))?;
        self.replace_model(index, model)?;
        Ok(index)
    }

    /// Replace a row by index and keep the existing selection valid.
    pub fn replace_model(&mut self, index: usize, model: ModelConfig) -> Result<()> {
        let model_count = self.config.models.len();
        let target = self.config.models.get_mut(index).ok_or_else(|| {
            anyhow!("model index {index} out of range (model count: {model_count})")
        })?;
        *target = normalize_model_config(model);
        self.selected_index = clamp_index(self.selected_index, self.config.models.len());
        Ok(())
    }

    /// Delete the selected row and select the nearest surviving row.
    pub fn delete_selected(&mut self) -> Option<ModelConfig> {
        let index = self.selected_index?;
        if index >= self.config.models.len() {
            self.selected_index = clamp_index(Some(index), self.config.models.len());
            return None;
        }

        let removed = self.config.models.remove(index);
        self.selected_index = clamp_index(Some(index), self.config.models.len());
        Some(removed)
    }

    /// Toggle the selected row and return its new enabled value.
    pub fn toggle_selected_enabled(&mut self) -> Result<bool> {
        let index = self
            .selected_index
            .ok_or_else(|| anyhow!("no model selected"))?;
        let model = self
            .config
            .models
            .get_mut(index)
            .ok_or_else(|| anyhow!("selected model index {index} is out of range"))?;
        model.enabled = !model.enabled;
        Ok(model.enabled)
    }

    /// Return validation errors in the config store's deterministic order.
    pub fn validation_errors(&self) -> Vec<String> {
        validate_config(&self.config)
    }

    pub fn is_valid(&self) -> bool {
        self.validation_errors().is_empty()
    }

    /// Reload from disk with no compatibility fallback.
    ///
    /// A failed read, parse, or validation leaves all in-memory state intact.
    /// On success, the prior selection is retained and clamped to the new row
    /// count; a previously empty/unselected editor selects the first row.
    pub fn reload(&mut self) -> Result<()> {
        let loaded = load_config_strict(&self.config_path)?;
        self.accept_persisted_config(loaded);
        Ok(())
    }

    /// Validate, normalize, save atomically, and snapshot previous disk bytes.
    pub fn save(&mut self, note: &str) -> Result<()> {
        let errors = self.validation_errors();
        if !errors.is_empty() {
            return Err(anyhow!("invalid config: {}", errors.join("; ")));
        }

        save_config_in(&self.config_path, &self.config, note, &self.version_root)?;

        // save_config_in writes this exact normalization. Updating from the
        // same helper prevents harmless whitespace from remaining dirty.
        let persisted = normalize_download_config(&self.config);
        self.accept_persisted_config(persisted);
        Ok(())
    }

    /// List snapshot names newest-first for this config path.
    pub fn versions(&self) -> Result<Vec<String>> {
        list_versions_in(&self.config_path, &self.version_root)
    }

    /// Restore a contained, strictly valid snapshot and update from the exact
    /// bytes atomically written by the config store.
    ///
    /// Invalid snapshot bytes leave both disk and this editor unchanged.
    pub fn restore(&mut self, version_name: &str) -> Result<()> {
        let restored =
            restore_version_and_load_in(&self.config_path, version_name, &self.version_root)?;
        self.accept_persisted_config(restored);
        Ok(())
    }

    fn accept_persisted_config(&mut self, config: DownloadConfig) {
        let preferred_index = self.selected_index.unwrap_or(0);
        self.selected_index = clamp_index(Some(preferred_index), config.models.len());
        self.clean_config = config.clone();
        self.config = config;
    }
}

fn normalize_model_config(model: ModelConfig) -> ModelConfig {
    normalize_download_config(&DownloadConfig {
        models: vec![model],
        ..DownloadConfig::default()
    })
    .models
    .into_iter()
    .next()
    .expect("one model is always normalized to one model")
}

fn clamp_index(index: Option<usize>, model_count: usize) -> Option<usize> {
    if model_count == 0 {
        None
    } else {
        index.map(|index| index.min(model_count - 1))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    fn model(repo_id: &str) -> ModelConfig {
        ModelConfig {
            repo_id: repo_id.to_owned(),
            ..ModelConfig::default()
        }
    }

    fn config(repos: &[&str]) -> DownloadConfig {
        DownloadConfig {
            base_models_dir: "/models".to_owned(),
            models: repos.iter().map(|repo| model(repo)).collect(),
        }
    }

    fn write_typed_config(path: &Path, config: &DownloadConfig) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut bytes = serde_json::to_vec_pretty(config).unwrap();
        bytes.push(b'\n');
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn editing_transitions_track_dirty_state_and_nearest_selection() {
        let temp = TempDir::new().unwrap();
        let mut editor = DownloadEditor::from_config(
            temp.path().join("models.json"),
            temp.path().join("versions"),
            config(&["org/one", "org/two"]),
        );

        assert_eq!(editor.selected_index(), Some(0));
        assert_eq!(editor.selected_model().unwrap().repo_id, "org/one");
        assert!(!editor.is_dirty());

        assert!(!editor.toggle_selected_enabled().unwrap());
        assert!(editor.is_dirty());
        assert!(editor.toggle_selected_enabled().unwrap());
        assert!(!editor.is_dirty(), "reverting an edit should be clean");

        let blank_index = editor.add_model();
        assert_eq!(blank_index, 2);
        assert_eq!(editor.selected_index(), Some(2));
        assert!(editor.is_dirty());

        editor
            .apply_selected(ModelConfig {
                repo_id: " org/three ".to_owned(),
                allow_patterns: vec![" *.gguf ".to_owned(), String::new()],
                ..ModelConfig::default()
            })
            .unwrap();
        assert_eq!(editor.selected_model().unwrap().repo_id, "org/three");
        assert_eq!(editor.selected_model().unwrap().allow_patterns, ["*.gguf"]);

        let removed = editor.delete_selected().unwrap();
        assert_eq!(removed.repo_id, "org/three");
        assert_eq!(editor.selected_index(), Some(1));
        assert!(!editor.is_dirty(), "add then delete restored the baseline");

        assert_eq!(editor.delete_selected().unwrap().repo_id, "org/two");
        assert_eq!(editor.selected_index(), Some(0));
        assert!(editor.is_dirty());
    }

    #[test]
    fn selection_clamps_across_direct_selection_deletion_and_reload() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("versions");
        write_typed_config(&path, &config(&["org/one", "org/two", "org/three"]));
        let mut editor = DownloadEditor::open(&path, &version_root).unwrap();

        assert_eq!(editor.select(usize::MAX), Some(2));
        assert_eq!(editor.selected_model().unwrap().repo_id, "org/three");

        write_typed_config(&path, &config(&["org/only"]));
        editor.reload().unwrap();
        assert_eq!(editor.selected_index(), Some(0));
        assert_eq!(editor.selected_model().unwrap().repo_id, "org/only");

        write_typed_config(&path, &config(&[]));
        editor.reload().unwrap();
        assert_eq!(editor.selected_index(), None);
        assert_eq!(editor.select(42), None);

        write_typed_config(&path, &config(&["org/new", "org/second"]));
        editor.reload().unwrap();
        assert_eq!(editor.selected_index(), Some(0));
        editor.clear_selection();
        assert_eq!(editor.selected_index(), None);
        assert!(!editor.is_dirty());
    }

    #[test]
    fn malformed_strict_reload_preserves_in_memory_edits_and_selection() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");
        let version_root = temp.path().join("versions");
        write_typed_config(&path, &config(&["org/one", "org/two"]));
        let mut editor = DownloadEditor::open(&path, &version_root).unwrap();
        editor.select(1);
        editor.toggle_selected_enabled().unwrap();

        let before = editor.config().clone();
        fs::write(&path, "{not json").unwrap();
        let error = editor.reload().unwrap_err();

        assert!(error.to_string().contains("failed to parse config"));
        assert_eq!(editor.config(), &before);
        assert_eq!(editor.selected_index(), Some(1));
        assert!(editor.is_dirty());
    }

    #[test]
    fn save_validates_normalizes_snapshots_and_marks_state_clean() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("versions");
        write_typed_config(&path, &config(&["org/original"]));
        let mut editor = DownloadEditor::open(&path, &version_root).unwrap();

        editor.set_base_models_dir(" /mnt/models ");
        editor
            .apply_selected(ModelConfig {
                repo_id: " org/changed ".to_owned(),
                local_dir: " changed/model ".to_owned(),
                ..ModelConfig::default()
            })
            .unwrap();
        assert!(editor.is_dirty());

        editor.save(" editor save! ").unwrap();

        assert!(!editor.is_dirty());
        assert_eq!(editor.config().base_models_dir, "/mnt/models");
        assert_eq!(editor.selected_model().unwrap().repo_id, "org/changed");
        let versions = editor.versions().unwrap();
        assert_eq!(versions.len(), 1);
        assert!(versions[0].ends_with("__editor-save.json"));
        assert_eq!(load_config_strict(&path).unwrap(), editor.config().clone());
    }

    #[test]
    fn save_rejects_invalid_state_without_touching_disk() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");
        let version_root = temp.path().join("versions");
        let mut editor = DownloadEditor::new(&path, &version_root);
        editor.add_model();

        assert_eq!(
            editor.validation_errors(),
            ["models[0].repo_id is required"]
        );
        assert!(!editor.is_valid());
        let error = editor.save("invalid").unwrap_err();
        assert!(error.to_string().contains("repo_id is required"));
        assert!(!path.exists());
        assert!(editor.is_dirty());
    }

    #[test]
    fn restore_updates_editor_and_clean_baseline_from_snapshot() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("versions");
        write_typed_config(&path, &config(&["org/original", "org/two"]));
        let mut editor = DownloadEditor::open(&path, &version_root).unwrap();
        editor.select(1);
        editor.replace_model(1, model("org/changed")).unwrap();
        editor.save("change").unwrap();
        let original_version = editor.versions().unwrap().remove(0);

        editor.set_base_models_dir("unsaved");
        assert!(editor.is_dirty());
        editor.restore(&original_version).unwrap();

        assert!(!editor.is_dirty());
        assert_eq!(editor.config(), &config(&["org/original", "org/two"]));
        assert_eq!(editor.selected_index(), Some(1));
        assert_eq!(editor.selected_model().unwrap().repo_id, "org/two");
        assert_eq!(load_config_strict(&path).unwrap(), editor.config().clone());
    }

    #[test]
    fn invalid_restore_keeps_disk_and_editor_on_the_same_config() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("versions");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "not json").unwrap();
        let current = config(&["org/current", "org/two"]);
        save_config_in(&path, &current, "replace-invalid", &version_root).unwrap();

        let mut editor = DownloadEditor::open(&path, &version_root).unwrap();
        editor.select(1);
        let invalid_version = editor.versions().unwrap().remove(0);
        let before_bytes = fs::read(&path).unwrap();
        let before_config = editor.config().clone();

        let error = editor.restore(&invalid_version).unwrap_err();

        assert!(error.to_string().contains("failed to parse config"));
        assert_eq!(fs::read(&path).unwrap(), before_bytes);
        assert_eq!(editor.config(), &before_config);
        assert_eq!(editor.selected_index(), Some(1));
        assert!(!editor.is_dirty());
        assert_eq!(load_config_strict(&path).unwrap(), before_config);
    }

    #[test]
    fn restore_rejects_paths_outside_the_version_directory() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");
        let version_root = temp.path().join("versions");
        write_typed_config(&path, &config(&["org/one"]));
        let mut editor = DownloadEditor::open(&path, &version_root).unwrap();

        let error = editor.restore("../outside.json").unwrap_err();
        assert!(error.to_string().contains("invalid version path"));
        assert_eq!(editor.selected_model().unwrap().repo_id, "org/one");
    }
}
