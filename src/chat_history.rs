use std::path::Path;

use anyhow::Result;

use crate::state_store::{self, ChatMessage, ChatSession, ChatSessionList, SavedChatSession};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatHistory {
    messages: Vec<ChatMessage>,
}

impl ChatHistory {
    pub fn records(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn push(&mut self, role: impl Into<String>, content: impl Into<String>) {
        self.messages.push(ChatMessage {
            role: role.into(),
            content: content.into(),
        });
    }

    pub fn request_messages(&self) -> Vec<(String, String)> {
        self.messages
            .iter()
            .map(|message| (message.role.clone(), message.content.clone()))
            .collect()
    }

    pub fn replace_with_session(&mut self, session: ChatSession) {
        self.messages = session.history;
    }

    pub fn save(&self) -> Result<SavedChatSession> {
        state_store::save_chat_session(&self.messages)
    }

    pub fn save_in(&self, data_root: &Path) -> Result<SavedChatSession> {
        state_store::save_chat_session_in(data_root, &self.messages)
    }

    pub fn list_sessions() -> Result<ChatSessionList> {
        state_store::list_chat_sessions()
    }

    pub fn list_sessions_in(data_root: &Path) -> Result<ChatSessionList> {
        state_store::list_chat_sessions_in(data_root)
    }

    pub fn load_session(session_file: impl AsRef<Path>) -> Result<ChatSession> {
        state_store::load_chat_session(session_file)
    }

    pub fn load_session_in(
        data_root: &Path,
        session_file: impl AsRef<Path>,
    ) -> Result<ChatSession> {
        state_store::load_chat_session_in(data_root, session_file)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn converts_between_tui_history_and_request_messages() {
        let mut history = ChatHistory::default();
        history.push("user", "hello");
        history.push("assistant", "hi");
        assert_eq!(
            history.request_messages(),
            [
                ("user".into(), "hello".into()),
                ("assistant".into(), "hi".into())
            ]
        );
        assert_eq!(history.records()[0].role, "user");
    }

    #[test]
    fn save_list_and_restore_use_legacy_compatible_store() {
        let data = TempDir::new().unwrap();
        let mut history = ChatHistory::default();
        history.push("user", "persist me");
        let saved = history.save_in(data.path()).unwrap();
        let listed = ChatHistory::list_sessions_in(data.path()).unwrap();
        assert_eq!(listed.sessions.len(), 1);
        assert_eq!(
            listed.sessions[0].file_name,
            saved.json_path.file_name().unwrap().to_string_lossy()
        );

        let session =
            ChatHistory::load_session_in(data.path(), Path::new(&listed.sessions[0].file_name))
                .unwrap();
        let mut restored = ChatHistory::default();
        restored.replace_with_session(session);
        assert_eq!(restored, history);
    }

    #[test]
    fn clear_and_empty_state_are_explicit() {
        let mut history = ChatHistory::default();
        assert!(history.is_empty());
        history.push("user", "message");
        history.clear();
        assert!(history.is_empty());
    }
}
