use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const DEFAULT_CONFIG_RELATIVE_PATH: &str = "model_downloader/models_config.json";
const VERSIONS_RELATIVE_PATH: &str = ".toolkit/download_config_versions";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A normalized model downloader entry.
///
/// Field order intentionally matches the Python store so serialized files keep
/// the same stable, human-friendly layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub enabled: bool,
    pub repo_id: String,
    pub local_dir: String,
    pub allow_patterns: Vec<String>,
    pub ignore_patterns: Vec<String>,
    pub revision: String,
    pub force_download: bool,
    pub max_workers: Option<u64>,
    pub description: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            repo_id: String::new(),
            local_dir: String::new(),
            allow_patterns: Vec::new(),
            ignore_patterns: Vec::new(),
            revision: String::new(),
            force_download: false,
            max_workers: None,
            description: String::new(),
        }
    }
}

/// The normalized downloader configuration persisted as JSON.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DownloadConfig {
    pub base_models_dir: String,
    pub models: Vec<ModelConfig>,
}

pub fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn default_config_path() -> PathBuf {
    repository_root().join(DEFAULT_CONFIG_RELATIVE_PATH)
}

pub fn versions_root() -> PathBuf {
    repository_root().join(VERSIONS_RELATIVE_PATH)
}

/// Convert a JSON model object to the canonical persisted representation.
/// Unknown keys are ignored, strings are trimmed, empty patterns are removed,
/// and non-positive or non-integral worker counts become `None`.
pub fn normalize_model(raw: &Value) -> ModelConfig {
    let Some(model) = raw.as_object() else {
        return ModelConfig::default();
    };

    ModelConfig {
        enabled: model.get("enabled").map(json_truthy).unwrap_or(true),
        repo_id: model
            .get("repo_id")
            .map(python_string)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        local_dir: model
            .get("local_dir")
            .map(python_string)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        allow_patterns: normalize_patterns(model.get("allow_patterns")),
        ignore_patterns: normalize_patterns(model.get("ignore_patterns")),
        revision: model
            .get("revision")
            .map(python_string)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        force_download: model
            .get("force_download")
            .map(json_truthy)
            .unwrap_or(false),
        max_workers: model
            .get("max_workers")
            .and_then(Value::as_u64)
            .filter(|workers| *workers > 0),
        description: model
            .get("description")
            .map(python_string)
            .unwrap_or_default()
            .trim()
            .to_owned(),
    }
}

/// Normalize arbitrary JSON using the same schema and defaults as the Python
/// implementation. Non-object roots and non-array `models` values yield an
/// empty configuration rather than failing a load.
pub fn normalize_config(raw: &Value) -> DownloadConfig {
    let Some(object) = raw.as_object() else {
        return DownloadConfig::default();
    };

    let models = object
        .get("models")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|item| item.is_object())
                .map(normalize_model)
                .collect()
        })
        .unwrap_or_default();

    DownloadConfig {
        base_models_dir: object
            .get("base_models_dir")
            .map(python_string)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        models,
    }
}

/// Normalize an already typed config before persistence.
pub fn normalize_download_config(config: &DownloadConfig) -> DownloadConfig {
    DownloadConfig {
        base_models_dir: config.base_models_dir.trim().to_owned(),
        models: config
            .models
            .iter()
            .map(|model| ModelConfig {
                enabled: model.enabled,
                repo_id: model.repo_id.trim().to_owned(),
                local_dir: model.local_dir.trim().to_owned(),
                allow_patterns: normalize_string_patterns(&model.allow_patterns),
                ignore_patterns: normalize_string_patterns(&model.ignore_patterns),
                revision: model.revision.trim().to_owned(),
                force_download: model.force_download,
                max_workers: model.max_workers.filter(|workers| *workers > 0),
                description: model.description.trim().to_owned(),
            })
            .collect(),
    }
}

/// Validate a typed configuration and return all errors in deterministic order.
pub fn validate_config(config: &DownloadConfig) -> Vec<String> {
    let mut errors = Vec::new();

    for (index, model) in config.models.iter().enumerate() {
        if model.repo_id.trim().is_empty() {
            errors.push(format!("models[{index}].repo_id is required"));
        }
        if matches!(model.max_workers, Some(0)) {
            errors.push(format!(
                "models[{index}].max_workers must be null or positive integer"
            ));
        }
    }

    errors
}

/// Validate unnormalized JSON. This is useful at API boundaries where the
/// caller needs the Python store's structural error messages before coercion.
pub fn validate_json_config(config: &Value) -> Vec<String> {
    let Some(object) = config.as_object() else {
        return vec!["Config must be a JSON object".to_owned()];
    };

    let Some(models) = object.get("models").and_then(Value::as_array) else {
        return vec!["models must be an array".to_owned()];
    };

    let mut errors = Vec::new();
    for (index, model) in models.iter().enumerate() {
        let Some(model) = model.as_object() else {
            errors.push(format!("models[{index}] must be an object"));
            continue;
        };

        let repo_id = model.get("repo_id").map(python_string).unwrap_or_default();
        if repo_id.trim().is_empty() {
            errors.push(format!("models[{index}].repo_id is required"));
        }

        for key in ["allow_patterns", "ignore_patterns"] {
            match model.get(key) {
                None | Some(Value::Null) => {}
                Some(Value::Array(entries)) => {
                    if entries.iter().any(|entry| !entry.is_string()) {
                        errors.push(format!("models[{index}].{key} entries must be strings"));
                    }
                }
                Some(_) => errors.push(format!("models[{index}].{key} must be an array")),
            }
        }

        if let Some(workers) = model.get("max_workers") {
            let valid = match workers {
                Value::Null => true,
                // Python's bool is an int subclass: True passes this check and
                // False fails it. Keep that compatibility for raw validation.
                Value::Bool(value) => *value,
                Value::Number(value) => {
                    value.as_i64().is_some_and(|number| number > 0)
                        || value.as_u64().is_some_and(|number| number > 0)
                }
                _ => false,
            };
            if !valid {
                errors.push(format!(
                    "models[{index}].max_workers must be null or positive integer"
                ));
            }
        }
    }

    errors
}

/// Load and normalize a JSON config. Missing, unreadable, malformed, and
/// non-object files intentionally produce an empty config, matching Python.
pub fn load_config(path: impl AsRef<Path>) -> DownloadConfig {
    let Ok(contents) = fs::read_to_string(path.as_ref()) else {
        return DownloadConfig::default();
    };
    let Ok(raw) = serde_json::from_str::<Value>(&contents) else {
        return DownloadConfig::default();
    };
    normalize_config(&raw)
}

/// Load a config without the compatibility fallback used by [`load_config`].
/// I/O, JSON syntax, root-shape, and schema validation errors are returned to
/// the caller so an interactive surface can make data loss or corruption
/// visible instead of presenting an apparently empty configuration.
pub fn load_config_strict(path: impl AsRef<Path>) -> Result<DownloadConfig> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read config {}", path.display()))?;
    let raw: Value = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse config {}", path.display()))?;
    let errors = validate_json_config(&raw);
    if !errors.is_empty() {
        return Err(anyhow!("invalid config: {}", errors.join("; ")));
    }
    Ok(normalize_config(&raw))
}

/// Save a normalized config and snapshot the previous bytes when the target
/// already exists.
pub fn save_config(path: impl AsRef<Path>, config: &DownloadConfig, note: &str) -> Result<()> {
    save_config_in(path.as_ref(), config, note, &versions_root())
}

/// List config snapshots newest-first by their stable filename ordering.
pub fn list_versions(path: impl AsRef<Path>) -> Result<Vec<String>> {
    list_versions_in(path.as_ref(), &versions_root())
}

/// Restore a config snapshot verbatim. Snapshot names must be a single path
/// component so callers cannot escape the version directory.
pub fn restore_version(path: impl AsRef<Path>, version_name: &str) -> Result<()> {
    restore_version_in(path.as_ref(), version_name, &versions_root())
}

pub fn csv_to_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Save using an explicit snapshot root. Runtime applications should prefer
/// this over [`save_config`] when they operate on a repository other than the
/// one used to compile the binary.
pub fn save_config_in(
    path: &Path,
    config: &DownloadConfig,
    note: &str,
    version_root: &Path,
) -> Result<()> {
    let normalized = normalize_download_config(config);
    let errors = validate_config(&normalized);
    if !errors.is_empty() {
        return Err(anyhow!(errors.join("; ")));
    }

    let target = resolve_allow_missing(path)
        .with_context(|| format!("failed to resolve config path {}", path.display()))?;
    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("config path has no parent: {}", target.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create config directory {}", parent.display()))?;

    #[cfg(unix)]
    let target_mode = fs::metadata(&target).ok().map(|metadata| {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode()
    });
    #[cfg(not(unix))]
    let target_mode = None;

    if target.is_file() {
        let previous = fs::read(&target)
            .with_context(|| format!("failed to read existing config {}", target.display()))?;
        let version_dir = version_dir_for_config_in(&target, version_root)?;
        fs::create_dir_all(&version_dir).with_context(|| {
            format!(
                "failed to create config version directory {}",
                version_dir.display()
            )
        })?;
        write_unique_snapshot(
            &version_dir,
            &safe_stamp(),
            &sanitize_note(note, "save", None),
            ".json",
            &previous,
        )?;
    }

    let mut serialized = serde_json::to_vec_pretty(&normalized)
        .context("failed to serialize normalized download config")?;
    serialized.push(b'\n');
    atomic_write(&target, &serialized, target_mode)
        .with_context(|| format!("failed to save config {}", target.display()))
}

/// List versions using an explicit snapshot root.
pub fn list_versions_in(path: &Path, version_root: &Path) -> Result<Vec<String>> {
    let version_dir = version_dir_for_config_in(path, version_root)?;
    let entries = match fs::read_dir(&version_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to read config version directory {}",
                    version_dir.display()
                )
            })
        }
    };

    let mut versions = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| {
            format!(
                "failed to inspect config version directory {}",
                version_dir.display()
            )
        })?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension == "json")
        {
            versions.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    versions.sort_unstable_by(|left, right| right.cmp(left));
    Ok(versions)
}

/// Restore a version from an explicit snapshot root.
pub fn restore_version_in(path: &Path, version_name: &str, version_root: &Path) -> Result<()> {
    ensure_single_component(version_name)?;
    let target = resolve_allow_missing(path)
        .with_context(|| format!("failed to resolve config path {}", path.display()))?;
    let version_dir = version_dir_for_config_in(&target, version_root)?;
    let source = version_dir.join(version_name);
    if !source.is_file() {
        return Err(anyhow!("version not found"));
    }

    let resolved_dir = resolve_allow_missing(&version_dir)?;
    let resolved_source = resolve_allow_missing(&source)?;
    if !resolved_source.starts_with(&resolved_dir) {
        return Err(anyhow!("invalid version path"));
    }

    let content = fs::read(&resolved_source)
        .with_context(|| format!("failed to read config version {}", source.display()))?;
    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("config path has no parent: {}", target.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create config directory {}", parent.display()))?;

    #[cfg(unix)]
    let mode = fs::metadata(&target).ok().map(|metadata| {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode()
    });
    #[cfg(not(unix))]
    let mode = None;

    atomic_write(&target, &content, mode)
        .with_context(|| format!("failed to restore config {}", target.display()))
}

fn version_dir_for_config_in(path: &Path, version_root: &Path) -> Result<PathBuf> {
    let resolved = resolve_allow_missing(path)
        .with_context(|| format!("failed to resolve config path {}", path.display()))?;
    Ok(version_root.join(config_path_key(&resolved)))
}

fn config_path_key(path: &Path) -> String {
    let raw = path.to_string_lossy();
    let mut sanitized = String::with_capacity(raw.len());
    let mut in_replacement = false;

    for character in raw.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '/' | '-') {
            sanitized.push(character);
            in_replacement = false;
        } else if !in_replacement {
            sanitized.push('-');
            in_replacement = true;
        }
    }

    let key = sanitized.trim_matches('/').replace('/', "__");
    if key.is_empty() {
        "models_config_json".to_owned()
    } else {
        key
    }
}

fn normalize_patterns(value: Option<&Value>) -> Vec<String> {
    match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(patterns)) => patterns
            .iter()
            .map(python_string)
            .map(|pattern| pattern.trim().to_owned())
            .filter(|pattern| !pattern.is_empty())
            .collect(),
        // Python iterates truthy strings and dictionaries here. Preserve that
        // compatibility while handling non-iterable malformed values safely.
        Some(Value::String(pattern)) => pattern
            .chars()
            .map(|character| character.to_string())
            .filter(|character| !character.trim().is_empty())
            .collect(),
        Some(Value::Object(patterns)) => patterns
            .keys()
            .map(|pattern| pattern.trim().to_owned())
            .filter(|pattern| !pattern.is_empty())
            .collect(),
        Some(value) if !json_truthy(value) => Vec::new(),
        Some(_) => Vec::new(),
    }
}

fn normalize_string_patterns(patterns: &[String]) -> Vec<String> {
    patterns
        .iter()
        .map(|pattern| pattern.trim())
        .filter(|pattern| !pattern.is_empty())
        .map(str::to_owned)
        .collect()
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_i64().map_or_else(
            || number.as_f64().is_some_and(|value| value != 0.0),
            |value| value != 0,
        ),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn sanitize_note(note: &str, fallback: &str, max_len: Option<usize>) -> String {
    let mut clean = String::with_capacity(note.len());
    let mut in_replacement = false;
    for character in note.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
            clean.push(character);
            in_replacement = false;
        } else if !in_replacement {
            clean.push('-');
            in_replacement = true;
        }
    }

    let clean = clean.trim_matches('-');
    let clean = if clean.is_empty() { fallback } else { clean };
    match max_len {
        Some(max_len) => clean.chars().take(max_len).collect(),
        None => clean.to_owned(),
    }
}

fn ensure_single_component(name: &str) -> Result<()> {
    let path = Path::new(name);
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => Err(anyhow!("invalid version path")),
    }
}

fn resolve_allow_missing(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to determine current directory")?
            .join(path)
    };

    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => resolved.push(prefix.as_os_str()),
            Component::RootDir => resolved.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(part) => {
                resolved.push(part);
                if fs::symlink_metadata(&resolved).is_ok() {
                    resolved = fs::canonicalize(&resolved).with_context(|| {
                        format!("failed to resolve path component {}", resolved.display())
                    })?;
                }
            }
        }
    }
    Ok(resolved)
}

fn write_unique_snapshot(
    directory: &Path,
    stamp: &str,
    note: &str,
    extension: &str,
    content: &[u8],
) -> Result<PathBuf> {
    for collision in 0_u32..10_000 {
        let stamp = if collision == 0 {
            stamp.to_owned()
        } else {
            // `~` sorts after the base name's `_`, so reverse filename order
            // still places a later same-second snapshot first.
            format!("{stamp}~{collision:04}")
        };
        let path = directory.join(format!("{stamp}__{note}{extension}"));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                if let Err(error) = write_and_sync(&mut file, content) {
                    drop(file);
                    let _ = fs::remove_file(&path);
                    return Err(error).with_context(|| {
                        format!("failed to write config snapshot {}", path.display())
                    });
                }
                return Ok(path);
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create config snapshot {}", path.display())
                })
            }
        }
    }
    Err(anyhow!("could not allocate a unique config snapshot name"))
}

fn atomic_write(path: &Path, content: &[u8], unix_mode: Option<u32>) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create directory {}", parent.display()))?;

    let file_name = path
        .file_name()
        .map_or_else(|| OsString::from("file"), OsString::from)
        .to_string_lossy()
        .into_owned();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    for attempt in 0_u32..100 {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{file_name}.l3ms-tmp-{}-{nonce}-{counter}-{attempt}",
            std::process::id()
        ));
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        };

        let result = (|| -> Result<()> {
            file.write_all(content)?;
            file.sync_all()?;
            drop(file);

            #[cfg(unix)]
            if let Some(mode) = unix_mode {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&temporary, fs::Permissions::from_mode(mode))?;
            }
            #[cfg(not(unix))]
            let _ = unix_mode;

            fs::rename(&temporary, path)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        return result;
    }

    Err(anyhow!("could not allocate a temporary file"))
}

fn write_and_sync(file: &mut File, content: &[u8]) -> std::io::Result<()> {
    file.write_all(content)?;
    file.sync_all()
}

fn safe_stamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_utc_seconds(seconds)
}

fn format_utc_seconds(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;

    // Gregorian civil date conversion derived from Howard Hinnant's
    // public-domain `civil_from_days` algorithm.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);

    format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn normalizes_models_and_discards_unknown_fields() {
        let raw = json!({
            "enabled": 0,
            "repo_id": "  org/model  ",
            "local_dir": " models/model ",
            "allow_patterns": [" *.gguf ", "", 17],
            "ignore_patterns": null,
            "revision": " main ",
            "force_download": "yes",
            "max_workers": -2,
            "description": " demo ",
            "unknown": "discard me"
        });

        let model = normalize_model(&raw);
        assert!(!model.enabled);
        assert_eq!(model.repo_id, "org/model");
        assert_eq!(model.local_dir, "models/model");
        assert_eq!(model.allow_patterns, ["*.gguf", "17"]);
        assert!(model.ignore_patterns.is_empty());
        assert_eq!(model.revision, "main");
        assert!(model.force_download);
        assert_eq!(model.max_workers, None);
        assert_eq!(model.description, "demo");
    }

    #[test]
    fn normalizes_config_shape_and_typed_values() {
        assert_eq!(normalize_config(&json!([])), DownloadConfig::default());
        assert_eq!(
            normalize_config(&json!({"base_models_dir": " /models ", "models": {}})),
            DownloadConfig {
                base_models_dir: "/models".to_owned(),
                models: Vec::new(),
            }
        );

        let config = DownloadConfig {
            base_models_dir: " /models ".to_owned(),
            models: vec![ModelConfig {
                repo_id: " org/model ".to_owned(),
                allow_patterns: vec![" *.gguf ".to_owned(), " ".to_owned()],
                max_workers: Some(0),
                ..ModelConfig::default()
            }],
        };
        let normalized = normalize_download_config(&config);
        assert_eq!(normalized.base_models_dir, "/models");
        assert_eq!(normalized.models[0].repo_id, "org/model");
        assert_eq!(normalized.models[0].allow_patterns, ["*.gguf"]);
        assert_eq!(normalized.models[0].max_workers, None);
    }

    #[test]
    fn validates_raw_structure_with_python_compatible_messages() {
        assert_eq!(
            validate_json_config(&json!([])),
            ["Config must be a JSON object"]
        );
        assert_eq!(
            validate_json_config(&json!({})),
            ["models must be an array"]
        );

        let errors = validate_json_config(&json!({
            "models": [
                null,
                {
                    "repo_id": " ",
                    "allow_patterns": "*.gguf",
                    "ignore_patterns": ["ok", 3],
                    "max_workers": 0
                },
                {"repo_id": "valid", "max_workers": true}
            ]
        }));
        assert_eq!(
            errors,
            [
                "models[0] must be an object",
                "models[1].repo_id is required",
                "models[1].allow_patterns must be an array",
                "models[1].ignore_patterns entries must be strings",
                "models[1].max_workers must be null or positive integer",
            ]
        );
    }

    #[test]
    fn typed_validation_finds_required_repo_and_zero_workers() {
        let config = DownloadConfig {
            models: vec![ModelConfig {
                max_workers: Some(0),
                ..ModelConfig::default()
            }],
            ..DownloadConfig::default()
        };
        assert_eq!(
            validate_config(&config),
            [
                "models[0].repo_id is required",
                "models[0].max_workers must be null or positive integer",
            ]
        );
    }

    #[test]
    fn load_returns_empty_for_missing_or_malformed_files_and_normalizes_valid_json() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");
        assert_eq!(load_config(&path), DownloadConfig::default());

        fs::write(&path, "not json").unwrap();
        assert_eq!(load_config(&path), DownloadConfig::default());

        fs::write(
            &path,
            r#"{"base_models_dir":" /models ","models":[{"repo_id":" org/model "},2]}"#,
        )
        .unwrap();
        let loaded = load_config(&path);
        assert_eq!(loaded.base_models_dir, "/models");
        assert_eq!(loaded.models.len(), 1);
        assert_eq!(loaded.models[0].repo_id, "org/model");
    }

    #[test]
    fn strict_load_surfaces_io_parse_and_schema_errors() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");

        assert!(load_config_strict(&path)
            .unwrap_err()
            .to_string()
            .contains("failed to read config"));

        fs::write(&path, "not json").unwrap();
        assert!(load_config_strict(&path)
            .unwrap_err()
            .to_string()
            .contains("failed to parse config"));

        fs::write(&path, r#"{"models":"wrong"}"#).unwrap();
        assert!(load_config_strict(&path)
            .unwrap_err()
            .to_string()
            .contains("models must be an array"));

        fs::write(&path, r#"{"base_models_dir":" /models ","models":[]}"#).unwrap();
        assert_eq!(
            load_config_strict(&path).unwrap().base_models_dir,
            "/models"
        );
    }

    #[test]
    fn save_snapshots_previous_bytes_and_restore_is_contained() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("versions");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let original = b"{\"legacy\":true}\n";
        fs::write(&path, original).unwrap();

        let config = DownloadConfig {
            base_models_dir: " /models ".to_owned(),
            models: vec![ModelConfig {
                repo_id: " org/model ".to_owned(),
                allow_patterns: vec![" *.gguf ".to_owned()],
                max_workers: Some(4),
                ..ModelConfig::default()
            }],
        };
        save_config_in(&path, &config, " first save! ", &version_root).unwrap();

        let saved = fs::read_to_string(&path).unwrap();
        assert!(saved.ends_with('\n'));
        let parsed: Value = serde_json::from_str(&saved).unwrap();
        assert_eq!(parsed["base_models_dir"], "/models");
        assert_eq!(parsed["models"][0]["repo_id"], "org/model");
        assert_eq!(parsed["models"][0]["allow_patterns"], json!(["*.gguf"]));

        let versions = list_versions_in(&path, &version_root).unwrap();
        assert_eq!(versions.len(), 1);
        assert!(versions[0].ends_with("__first-save.json"));
        let version_dir = version_dir_for_config_in(&path, &version_root).unwrap();
        assert_eq!(fs::read(version_dir.join(&versions[0])).unwrap(), original);

        restore_version_in(&path, &versions[0], &version_root).unwrap();
        assert_eq!(fs::read(&path).unwrap(), original);

        let outside = temp.path().join("outside.json");
        fs::write(&outside, "outside").unwrap();
        let error = restore_version_in(&path, "../outside.json", &version_root).unwrap_err();
        assert!(error.to_string().contains("invalid version path"));
    }

    #[test]
    fn rapid_saves_allocate_unique_snapshot_names() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");
        let version_root = temp.path().join("versions");
        let config = DownloadConfig {
            models: vec![ModelConfig {
                repo_id: "org/model".to_owned(),
                ..ModelConfig::default()
            }],
            ..DownloadConfig::default()
        };

        fs::write(&path, "one").unwrap();
        save_config_in(&path, &config, "manual", &version_root).unwrap();
        save_config_in(&path, &config, "manual", &version_root).unwrap();

        let versions = list_versions_in(&path, &version_root).unwrap();
        assert_eq!(versions.len(), 2);
        assert_ne!(versions[0], versions[1]);
    }

    #[test]
    fn context_aware_api_places_snapshots_under_supplied_root() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("runtime-repo/models.json");
        let supplied_root = temp.path().join("runtime-state/config-versions");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "previous").unwrap();
        let config = DownloadConfig {
            models: vec![ModelConfig {
                repo_id: "org/model".to_owned(),
                ..ModelConfig::default()
            }],
            ..DownloadConfig::default()
        };

        save_config_in(&path, &config, "runtime", &supplied_root).unwrap();

        let versions = list_versions_in(&path, &supplied_root).unwrap();
        assert_eq!(versions.len(), 1);
        let version_dir = version_dir_for_config_in(&path, &supplied_root).unwrap();
        assert!(version_dir.starts_with(&supplied_root));
        assert_eq!(
            fs::read_to_string(version_dir.join(&versions[0])).unwrap(),
            "previous"
        );
    }

    #[cfg(unix)]
    #[test]
    fn atomic_save_preserves_existing_unix_mode() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");
        let version_root = temp.path().join("versions");
        fs::write(&path, "old").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        let config = DownloadConfig {
            models: vec![ModelConfig {
                repo_id: "org/model".to_owned(),
                ..ModelConfig::default()
            }],
            ..DownloadConfig::default()
        };

        save_config_in(&path, &config, "mode", &version_root).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    #[test]
    fn csv_parser_trims_and_discards_empty_fields() {
        assert_eq!(csv_to_list(" one, ,two ,, three "), ["one", "two", "three"]);
    }

    #[test]
    fn utc_formatter_matches_epoch_and_known_leap_day() {
        assert_eq!(format_utc_seconds(0), "19700101T000000Z");
        assert_eq!(format_utc_seconds(1_709_164_800), "20240229T000000Z");
    }
}
