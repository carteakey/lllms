use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::script_store::{
    list_script_versions_in, load_script_in, resolve_script_in, restore_script_version_in,
    save_script_with_version_in,
};

const DEFAULT_VERSIONS_DIRECTORY: &str = ".toolkit/script_versions";

/// Editable state for one repository-contained script.
///
/// This type deliberately knows nothing about whether the caller presents a
/// bench or maintenance workflow. Both are ordinary repository scripts and use
/// the same snapshot-backed [`crate::script_store`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptEditorState {
    repository_root: PathBuf,
    versions_root: PathBuf,
    selected_path: Option<PathBuf>,
    content: String,
    persisted_content: String,
    versions: Vec<String>,
}

impl ScriptEditorState {
    /// Create empty editor state with explicit repository and snapshot roots.
    ///
    /// A relative `versions_root` is interpreted relative to the repository,
    /// which keeps runtime checkouts independent of the process working
    /// directory. The repository must already exist.
    pub fn new(repository_root: impl AsRef<Path>, versions_root: impl AsRef<Path>) -> Result<Self> {
        let supplied_root = repository_root.as_ref();
        let repository_root = fs::canonicalize(supplied_root).with_context(|| {
            format!(
                "failed to resolve repository root {}",
                supplied_root.display()
            )
        })?;
        if !repository_root.is_dir() {
            return Err(anyhow!(
                "repository root is not a directory: {}",
                repository_root.display()
            ));
        }

        let versions_root = versions_root.as_ref();
        let versions_root = if versions_root.is_absolute() {
            versions_root.to_path_buf()
        } else {
            repository_root.join(versions_root)
        };

        Ok(Self {
            repository_root,
            versions_root,
            selected_path: None,
            content: String::new(),
            persisted_content: String::new(),
            versions: Vec::new(),
        })
    }

    /// Create state using the repository's compatible script-version folder.
    pub fn for_repository(repository_root: impl AsRef<Path>) -> Result<Self> {
        Self::new(repository_root, DEFAULT_VERSIONS_DIRECTORY)
    }

    pub fn repository_root(&self) -> &Path {
        &self.repository_root
    }

    pub fn versions_root(&self) -> &Path {
        &self.versions_root
    }

    pub fn selected_path(&self) -> Option<&Path> {
        self.selected_path.as_deref()
    }

    /// Return the selected path relative to the canonical repository root.
    pub fn selected_relative_path(&self) -> Option<&Path> {
        self.selected_path
            .as_deref()
            .and_then(|path| path.strip_prefix(&self.repository_root).ok())
    }

    pub fn has_selection(&self) -> bool {
        self.selected_path.is_some()
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn versions(&self) -> &[String] {
        &self.versions
    }

    /// Whether the edit buffer differs from the last selected, reloaded,
    /// saved, or restored file content.
    pub fn is_dirty(&self) -> bool {
        self.content != self.persisted_content
    }

    /// Select and load a script. Failed selections leave the prior state
    /// unchanged; successful selections intentionally replace any dirty buffer.
    pub fn select(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = resolve_script_in(path.as_ref(), &self.repository_root)?;
        let content = load_script_in(&path, &self.repository_root)?;
        let versions = list_script_versions_in(&path, &self.repository_root, &self.versions_root)?;

        self.install_loaded(path, content, versions);
        Ok(())
    }

    /// Clear the selection and edit buffer, intentionally discarding edits.
    pub fn clear_selection(&mut self) {
        self.selected_path = None;
        self.content.clear();
        self.persisted_content.clear();
        self.versions.clear();
    }

    /// Replace the edit buffer for the selected script.
    pub fn set_content(&mut self, content: impl Into<String>) -> Result<()> {
        self.require_selection()?;
        self.content = content.into();
        Ok(())
    }

    /// Reload the selected script and its snapshot list, intentionally
    /// discarding unsaved edits. A failed reload leaves state unchanged.
    pub fn reload(&mut self) -> Result<()> {
        let path = self.require_selection()?.to_path_buf();
        let content = load_script_in(&path, &self.repository_root)?;
        let versions = list_script_versions_in(&path, &self.repository_root, &self.versions_root)?;

        self.install_loaded(path, content, versions);
        Ok(())
    }

    /// Refresh only the available snapshots without touching the edit buffer.
    pub fn refresh_versions(&mut self) -> Result<()> {
        let path = self.require_selection()?.to_path_buf();
        self.versions = list_script_versions_in(&path, &self.repository_root, &self.versions_root)?;
        Ok(())
    }

    /// Save the edit buffer atomically after snapshotting the displaced file.
    pub fn save(&mut self, note: &str) -> Result<()> {
        let path = self.require_selection()?.to_path_buf();
        save_script_with_version_in(
            &path,
            &self.content,
            note,
            &self.repository_root,
            &self.versions_root,
        )?;

        // The write is already committed at this point, so reflect that even
        // if refreshing the optional version list encounters an I/O error.
        self.persisted_content.clone_from(&self.content);
        self.versions = list_script_versions_in(&path, &self.repository_root, &self.versions_root)
            .context("script saved, but its version list could not be refreshed")?;
        Ok(())
    }

    /// Restore a named snapshot. The store first snapshots the displaced file,
    /// and a successful restore intentionally replaces any dirty edit buffer.
    pub fn restore(&mut self, version_name: &str) -> Result<()> {
        let path = self.require_selection()?.to_path_buf();
        restore_script_version_in(
            &path,
            version_name,
            &self.repository_root,
            &self.versions_root,
        )?;

        // Restore has already committed. Keep the buffer consistent with disk
        // before refreshing the secondary snapshot listing.
        let content = load_script_in(&path, &self.repository_root)
            .context("script restored, but its content could not be reloaded")?;
        self.content = content.clone();
        self.persisted_content = content;
        self.versions = list_script_versions_in(&path, &self.repository_root, &self.versions_root)
            .context("script restored, but its version list could not be refreshed")?;
        Ok(())
    }

    fn require_selection(&self) -> Result<&Path> {
        self.selected_path
            .as_deref()
            .ok_or_else(|| anyhow!("no script selected"))
    }

    fn install_loaded(&mut self, path: PathBuf, content: String, versions: Vec<String>) {
        self.selected_path = Some(path);
        self.content = content.clone();
        self.persisted_content = content;
        self.versions = versions;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use tempfile::TempDir;

    struct Fixture {
        _temp: TempDir,
        root: PathBuf,
        versions_root: PathBuf,
        bench: PathBuf,
        maintenance: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let temp = TempDir::new().unwrap();
            let unresolved_root = temp.path().join("repo");
            let versions_root = temp.path().join("state/script-versions");
            let bench = unresolved_root.join("bench-models/bench-example.sh");
            let maintenance = unresolved_root.join("maintenance/rebuild.sh");
            fs::create_dir_all(bench.parent().unwrap()).unwrap();
            fs::create_dir_all(maintenance.parent().unwrap()).unwrap();
            fs::write(&bench, "#!/bin/sh\necho bench\n").unwrap();
            fs::write(&maintenance, "#!/bin/sh\necho maintenance\n").unwrap();
            let root = fs::canonicalize(unresolved_root).unwrap();
            let bench = root.join("bench-models/bench-example.sh");
            let maintenance = root.join("maintenance/rebuild.sh");
            Self {
                _temp: temp,
                root,
                versions_root,
                bench,
                maintenance,
            }
        }

        fn editor(&self) -> ScriptEditorState {
            ScriptEditorState::new(&self.root, &self.versions_root).unwrap()
        }
    }

    #[test]
    fn one_state_type_loads_bench_and_maintenance_scripts() {
        let fixture = Fixture::new();
        let mut editor = fixture.editor();

        editor.select("bench-models/bench-example.sh").unwrap();
        assert_eq!(editor.selected_path(), Some(fixture.bench.as_path()));
        assert_eq!(
            editor.selected_relative_path(),
            Some(Path::new("bench-models/bench-example.sh"))
        );
        assert_eq!(editor.content(), "#!/bin/sh\necho bench\n");
        assert!(!editor.is_dirty());

        editor.select(&fixture.maintenance).unwrap();
        assert_eq!(editor.selected_path(), Some(fixture.maintenance.as_path()));
        assert_eq!(editor.content(), "#!/bin/sh\necho maintenance\n");
        assert!(!editor.is_dirty());
    }

    #[test]
    fn rejects_unsafe_selection_without_disturbing_current_edits() {
        let fixture = Fixture::new();
        let outside = fixture._temp.path().join("outside.sh");
        let unsupported = fixture.root.join("notes.txt");
        fs::write(&outside, "outside").unwrap();
        fs::write(&unsupported, "notes").unwrap();

        let mut editor = fixture.editor();
        editor.select(&fixture.bench).unwrap();
        editor.set_content("unsaved edit").unwrap();

        assert!(editor.select(&outside).is_err());
        assert!(editor.select("../outside.sh").is_err());
        assert!(editor.select(&unsupported).is_err());
        assert_eq!(editor.selected_path(), Some(fixture.bench.as_path()));
        assert_eq!(editor.content(), "unsaved edit");
        assert!(editor.is_dirty());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_selected_symlink_that_escapes_repository() {
        use std::os::unix::fs::symlink;

        let fixture = Fixture::new();
        let outside = fixture._temp.path().join("outside.sh");
        let link = fixture.root.join("maintenance/escape.sh");
        fs::write(&outside, "outside").unwrap();
        symlink(&outside, &link).unwrap();

        let mut editor = fixture.editor();
        assert!(editor.select(&link).is_err());
        assert!(!editor.has_selection());
    }

    #[test]
    fn dirty_state_tracks_edits_and_reload_discards_them() {
        let fixture = Fixture::new();
        let mut editor = fixture.editor();
        editor.select(&fixture.bench).unwrap();

        let original = editor.content().to_owned();
        editor.set_content(original.clone()).unwrap();
        assert!(!editor.is_dirty());
        editor.set_content("changed in editor").unwrap();
        assert!(editor.is_dirty());

        editor.reload().unwrap();
        assert_eq!(editor.content(), original);
        assert!(!editor.is_dirty());

        fs::write(&fixture.bench, "changed outside\n").unwrap();
        editor.set_content("another edit").unwrap();
        editor.reload().unwrap();
        assert_eq!(editor.content(), "changed outside\n");
        assert!(!editor.is_dirty());
    }

    #[test]
    fn save_writes_buffer_and_snapshots_previous_content() {
        let fixture = Fixture::new();
        let mut editor = fixture.editor();
        editor.select(&fixture.maintenance).unwrap();
        let original = editor.content().to_owned();

        editor
            .set_content("#!/bin/sh\necho updated maintenance\n")
            .unwrap();
        editor.save("maintenance edit").unwrap();

        assert_eq!(
            fs::read_to_string(&fixture.maintenance).unwrap(),
            editor.content()
        );
        assert!(!editor.is_dirty());
        assert_eq!(editor.versions().len(), 1);
        assert!(editor.versions()[0].contains("maintenance-edit"));

        let version_dir = crate::script_store::version_dir_for_script_in(
            &fixture.maintenance,
            &fixture.root,
            &fixture.versions_root,
        )
        .unwrap();
        assert_eq!(
            fs::read_to_string(version_dir.join(&editor.versions()[0])).unwrap(),
            original
        );
    }

    #[test]
    fn restore_reloads_buffer_and_snapshots_displaced_content() {
        let fixture = Fixture::new();
        let mut editor = fixture.editor();
        editor.select(&fixture.bench).unwrap();
        let original = editor.content().to_owned();

        editor.set_content("second\n").unwrap();
        editor.save("second").unwrap();
        editor.set_content("third\n").unwrap();
        editor.save("third").unwrap();

        let first_snapshot = editor
            .versions()
            .iter()
            .find(|name| name.contains("second"))
            .cloned()
            .unwrap();
        editor.set_content("unsaved text to discard").unwrap();
        editor.restore(&first_snapshot).unwrap();

        assert_eq!(editor.content(), original);
        assert_eq!(fs::read_to_string(&fixture.bench).unwrap(), original);
        assert!(!editor.is_dirty());
        assert_eq!(editor.versions().len(), 3);

        let version_dir = crate::script_store::version_dir_for_script_in(
            &fixture.bench,
            &fixture.root,
            &fixture.versions_root,
        )
        .unwrap();
        assert!(editor
            .versions()
            .iter()
            .any(|name| { fs::read_to_string(version_dir.join(name)).unwrap() == "third\n" }));
    }

    #[test]
    fn invalid_restore_preserves_dirty_buffer() {
        let fixture = Fixture::new();
        let mut editor = fixture.editor();
        editor.select(&fixture.bench).unwrap();
        editor.set_content("keep this edit").unwrap();

        assert!(editor.restore("../escape.sh").is_err());
        assert!(editor.restore("missing.sh").is_err());
        assert_eq!(editor.content(), "keep this edit");
        assert!(editor.is_dirty());
    }

    #[test]
    fn refresh_versions_does_not_touch_dirty_buffer() {
        let fixture = Fixture::new();
        let mut editor = fixture.editor();
        editor.select(&fixture.bench).unwrap();
        crate::script_store::save_script_with_version_in(
            &fixture.bench,
            "external change",
            "external",
            &fixture.root,
            &fixture.versions_root,
        )
        .unwrap();

        editor.set_content("local edit").unwrap();
        editor.refresh_versions().unwrap();
        assert_eq!(editor.content(), "local edit");
        assert!(editor.is_dirty());
        assert_eq!(editor.versions().len(), 1);
    }

    #[test]
    fn clear_and_unselected_operations_have_stable_behavior() {
        let fixture = Fixture::new();
        let mut editor = fixture.editor();

        assert!(editor.set_content("text").is_err());
        assert!(editor.reload().is_err());
        assert!(editor.refresh_versions().is_err());
        assert!(editor.save("note").is_err());
        assert!(editor.restore("version.sh").is_err());

        editor.select(&fixture.bench).unwrap();
        editor.set_content("dirty").unwrap();
        editor.clear_selection();
        assert!(!editor.has_selection());
        assert_eq!(editor.content(), "");
        assert!(editor.versions().is_empty());
        assert!(!editor.is_dirty());
    }

    #[test]
    fn relative_version_root_is_repository_relative() {
        let fixture = Fixture::new();
        let editor = ScriptEditorState::new(&fixture.root, "custom/versions").unwrap();
        assert_eq!(
            editor.versions_root(),
            fixture.root.join("custom/versions").as_path()
        );

        let default_editor = ScriptEditorState::for_repository(&fixture.root).unwrap();
        assert_eq!(
            default_editor.versions_root(),
            fixture.root.join(DEFAULT_VERSIONS_DIRECTORY).as_path()
        );
    }
}
