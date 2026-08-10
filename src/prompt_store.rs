//! Local, bounded system-prompt storage for the Chat view.

use std::fs::{self, OpenOptions};
use std::io::ErrorKind;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, Context, Result};

const PROMPTS_DIRECTORY: &str = "prompts";
const MAX_PROMPT_NAME_BYTES: usize = 96;
const MAX_PROMPT_BYTES: usize = 128 * 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    pub name: String,
    pub content: String,
}

pub fn prompts_path(data_root: impl AsRef<Path>) -> PathBuf {
    data_root.as_ref().join(PROMPTS_DIRECTORY)
}

pub fn list_prompts(data_root: impl AsRef<Path>) -> Result<Vec<String>> {
    let directory = prompts_path(data_root);
    let entries = match fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read prompt directory {}", directory.display()))
        }
    };
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|extension| extension == "md") {
            if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
                if validate_name(stem).is_ok() {
                    names.push(stem.to_owned());
                }
            }
        }
    }
    names.sort();
    Ok(names)
}

pub fn load_prompt(data_root: impl AsRef<Path>, name: &str) -> Result<Prompt> {
    validate_name(name)?;
    let path = prompt_path(&prompts_path(data_root), name);
    let bytes = fs::read(&path).with_context(|| format!("read prompt {name}"))?;
    if bytes.len() > MAX_PROMPT_BYTES {
        return Err(anyhow!("prompt {name} exceeds {MAX_PROMPT_BYTES} bytes"));
    }
    let content = String::from_utf8(bytes)
        .with_context(|| format!("prompt {name} is not UTF-8"))?;
    Ok(Prompt {
        name: name.to_owned(),
        content,
    })
}

pub fn save_prompt(data_root: impl AsRef<Path>, name: &str, content: &str) -> Result<()> {
    validate_name(name)?;
    if content.len() > MAX_PROMPT_BYTES {
        return Err(anyhow!("prompt {name} exceeds {MAX_PROMPT_BYTES} bytes"));
    }
    if content.contains('\0') {
        return Err(anyhow!("prompt content must not contain NUL"));
    }
    let directory = prompts_path(data_root);
    fs::create_dir_all(&directory)?;
    atomic_write(&prompt_path(&directory, name), content.as_bytes())
}

pub fn delete_prompt(data_root: impl AsRef<Path>, name: &str) -> Result<bool> {
    validate_name(name)?;
    let path = prompt_path(&prompts_path(data_root), name);
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).context("delete prompt"),
    }
}

fn prompt_path(directory: &Path, name: &str) -> PathBuf {
    directory.join(format!("{name}.md"))
}

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > MAX_PROMPT_NAME_BYTES || name.contains('\0') {
        return Err(anyhow!(
            "prompt name must be 1-{MAX_PROMPT_NAME_BYTES} bytes"
        ));
    }
    let path = Path::new(name);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "-_ .".contains(character))
    {
        return Err(anyhow!("prompt name contains unsupported path characters"));
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("prompt path has no parent"))?;
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp = parent.join(format!(".prompt-{nonce}.tmp"));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(temp, path).context("replace prompt")
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn saves_lists_loads_and_deletes_prompts() {
        let temp = TempDir::new().unwrap();
        save_prompt(temp.path(), "concise", "Be concise.").unwrap();
        assert_eq!(list_prompts(temp.path()).unwrap(), vec!["concise"]);
        assert_eq!(
            load_prompt(temp.path(), "concise").unwrap().content,
            "Be concise."
        );
        assert!(delete_prompt(temp.path(), "concise").unwrap());
        assert!(!delete_prompt(temp.path(), "concise").unwrap());
    }

    #[test]
    fn rejects_unsafe_names_and_unbounded_content() {
        let temp = TempDir::new().unwrap();
        assert!(save_prompt(temp.path(), "../secret", "x").is_err());
        assert!(save_prompt(temp.path(), "ok", &"x".repeat(MAX_PROMPT_BYTES + 1)).is_err());
    }
}
