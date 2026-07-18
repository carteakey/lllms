//! Portable command construction for the Python downloader compatibility boundary.

use std::{
    env,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

pub const DOWNLOADER_PYTHON_ENV: &str = "L3MS_DOWNLOADER_PYTHON";

const DOWNLOADER_SCRIPT: &str = "model_downloader/download_hf_model.py";
const FALLBACK_PYTHON: &str = "python3";

/// Build the executable and script prefix used for downloader commands.
///
/// `L3MS_DOWNLOADER_PYTHON` is treated as one executable path or command name,
/// not as a shell command containing arguments.
pub fn downloader_command_prefix(root: &Path) -> Vec<OsString> {
    let python_override = env::var_os(DOWNLOADER_PYTHON_ENV);
    downloader_command_prefix_with_override(root, python_override.as_deref())
}

/// Pure variant of [`downloader_command_prefix`] for callers and tests that
/// already resolved an interpreter override.
pub fn downloader_command_prefix_with_override(
    root: &Path,
    python_override: Option<&OsStr>,
) -> Vec<OsString> {
    let python = python_override
        .filter(|value| !value.is_empty())
        .map(OsString::from)
        .unwrap_or_else(|| repository_python(root));

    vec![python, root.join(DOWNLOADER_SCRIPT).into_os_string()]
}

fn repository_python(root: &Path) -> OsString {
    let candidate = repository_python_candidate(root);
    if candidate.is_file() {
        candidate.into_os_string()
    } else {
        FALLBACK_PYTHON.into()
    }
}

fn repository_python_candidate(root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        root.join(".venv").join("Scripts").join("python.exe")
    }

    #[cfg(not(windows))]
    {
        root.join(".venv").join("bin").join("python3")
    }
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsStr, fs};

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn explicit_non_empty_override_wins_over_repository_venv() {
        let repository = tempdir().unwrap();
        let venv_python = repository_python_candidate(repository.path());
        fs::create_dir_all(venv_python.parent().unwrap()).unwrap();
        fs::write(&venv_python, b"").unwrap();

        let prefix = downloader_command_prefix_with_override(
            repository.path(),
            Some(OsStr::new("custom-python")),
        );

        assert_eq!(prefix[0], OsString::from("custom-python"));
        assert_eq!(
            prefix[1],
            repository
                .path()
                .join("model_downloader/download_hf_model.py")
                .into_os_string()
        );
    }

    #[test]
    fn repository_venv_is_used_when_it_is_a_file() {
        let repository = tempdir().unwrap();
        let venv_python = repository_python_candidate(repository.path());
        fs::create_dir_all(venv_python.parent().unwrap()).unwrap();
        fs::write(&venv_python, b"").unwrap();

        let prefix =
            downloader_command_prefix_with_override(repository.path(), Some(OsStr::new("")));

        assert_eq!(prefix[0], venv_python.into_os_string());
    }

    #[test]
    fn missing_repository_venv_falls_back_to_python3() {
        let repository = tempdir().unwrap();

        let prefix = downloader_command_prefix_with_override(repository.path(), None);

        assert_eq!(prefix[0], OsString::from("python3"));
        assert_eq!(
            prefix[1],
            repository
                .path()
                .join("model_downloader/download_hf_model.py")
                .into_os_string()
        );
    }

    #[test]
    fn repository_venv_directory_is_not_treated_as_an_interpreter() {
        let repository = tempdir().unwrap();
        let venv_python = repository_python_candidate(repository.path());
        fs::create_dir_all(&venv_python).unwrap();

        let prefix = downloader_command_prefix_with_override(repository.path(), None);

        assert_eq!(prefix[0], OsString::from("python3"));
    }
}
