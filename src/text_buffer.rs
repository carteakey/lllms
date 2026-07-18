/// A zero-based logical cursor position.
///
/// Columns count Unicode scalar values rather than bytes. Terminal display
/// width (for example, for wide CJK characters) remains the renderer's concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorPosition {
    pub line: usize,
    pub column: usize,
}

/// A small UTF-8-safe multiline edit buffer.
///
/// The cursor is stored as a byte offset for efficient `String` mutation, but
/// it is always kept on a character boundary. Vertical movement remembers the
/// desired character column so moving through a short line and back to a long
/// one restores the original column.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TextBuffer {
    content: String,
    cursor_byte: usize,
    desired_column: Option<usize>,
}

impl TextBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a buffer at the start of the supplied content.
    pub fn from_content(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            cursor_byte: 0,
            desired_column: None,
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    /// Replace all content and return the cursor to the beginning.
    pub fn set_content(&mut self, content: impl Into<String>) {
        self.content = content.into();
        self.cursor_byte = 0;
        self.desired_column = None;
    }

    /// Current UTF-8 byte offset. This is always a valid character boundary.
    pub fn cursor_byte(&self) -> usize {
        self.cursor_byte
    }

    /// Set the cursor to a UTF-8 byte boundary.
    ///
    /// Returns `false` without changing state when the offset is out of bounds
    /// or falls inside a multi-byte character.
    pub fn set_cursor_byte(&mut self, offset: usize) -> bool {
        if offset > self.content.len() || !self.content.is_char_boundary(offset) {
            return false;
        }
        self.cursor_byte = offset;
        self.desired_column = None;
        true
    }

    /// Current zero-based line and character column.
    pub fn cursor_position(&self) -> CursorPosition {
        let line_start = self.current_line_start();
        CursorPosition {
            line: self.content[..line_start]
                .bytes()
                .filter(|byte| *byte == b'\n')
                .count(),
            column: self.content[line_start..self.cursor_byte].chars().count(),
        }
    }

    /// Number of logical lines, including an empty line after a trailing `\n`.
    pub fn line_count(&self) -> usize {
        self.content.bytes().filter(|byte| *byte == b'\n').count() + 1
    }

    pub fn insert_char(&mut self, character: char) {
        self.content.insert(self.cursor_byte, character);
        self.cursor_byte += character.len_utf8();
        self.desired_column = None;
    }

    pub fn insert_newline(&mut self) {
        self.insert_char('\n');
    }

    /// Remove the character before the cursor, including a newline when the
    /// cursor is at the start of a non-first line.
    pub fn backspace(&mut self) -> bool {
        self.desired_column = None;
        let Some(previous) = self.previous_boundary() else {
            return false;
        };
        self.content.drain(previous..self.cursor_byte);
        self.cursor_byte = previous;
        true
    }

    /// Remove the character under the cursor, including a newline at line end.
    pub fn delete_forward(&mut self) -> bool {
        self.desired_column = None;
        let Some(next) = self.next_boundary() else {
            return false;
        };
        self.content.drain(self.cursor_byte..next);
        true
    }

    pub fn move_left(&mut self) -> bool {
        self.desired_column = None;
        let Some(previous) = self.previous_boundary() else {
            return false;
        };
        self.cursor_byte = previous;
        true
    }

    pub fn move_right(&mut self) -> bool {
        self.desired_column = None;
        let Some(next) = self.next_boundary() else {
            return false;
        };
        self.cursor_byte = next;
        true
    }

    pub fn move_up(&mut self) -> bool {
        let current_start = self.current_line_start();
        if current_start == 0 {
            return false;
        }

        let desired = self.desired_column.unwrap_or_else(|| self.current_column());
        let previous_end = current_start - 1;
        let previous_start = line_start_at(&self.content, previous_end);
        self.cursor_byte = byte_at_column(&self.content, previous_start, previous_end, desired);
        self.desired_column = Some(desired);
        true
    }

    pub fn move_down(&mut self) -> bool {
        let current_end = self.current_line_end();
        if current_end == self.content.len() {
            return false;
        }

        let desired = self.desired_column.unwrap_or_else(|| self.current_column());
        let next_start = current_end + 1;
        let next_end = line_end_at(&self.content, next_start);
        self.cursor_byte = byte_at_column(&self.content, next_start, next_end, desired);
        self.desired_column = Some(desired);
        true
    }

    /// Move up by at most `lines` logical lines.
    ///
    /// This has the same desired-column behavior as repeated [`Self::move_up`]
    /// calls and stops at the first line. Returns whether the cursor moved.
    pub fn move_page_up(&mut self, lines: usize) -> bool {
        let mut moved = false;
        for _ in 0..lines {
            if !self.move_up() {
                break;
            }
            moved = true;
        }
        moved
    }

    /// Move down by at most `lines` logical lines.
    ///
    /// This has the same desired-column behavior as repeated
    /// [`Self::move_down`] calls and stops at the last line. Returns whether
    /// the cursor moved.
    pub fn move_page_down(&mut self, lines: usize) -> bool {
        let mut moved = false;
        for _ in 0..lines {
            if !self.move_down() {
                break;
            }
            moved = true;
        }
        moved
    }

    pub fn move_home(&mut self) -> bool {
        self.desired_column = None;
        let start = self.current_line_start();
        let moved = start != self.cursor_byte;
        self.cursor_byte = start;
        moved
    }

    pub fn move_end(&mut self) -> bool {
        self.desired_column = None;
        let end = self.current_line_end();
        let moved = end != self.cursor_byte;
        self.cursor_byte = end;
        moved
    }

    fn current_line_start(&self) -> usize {
        line_start_at(&self.content, self.cursor_byte)
    }

    fn current_line_end(&self) -> usize {
        line_end_at(&self.content, self.cursor_byte)
    }

    fn current_column(&self) -> usize {
        self.content[self.current_line_start()..self.cursor_byte]
            .chars()
            .count()
    }

    fn previous_boundary(&self) -> Option<usize> {
        self.content[..self.cursor_byte]
            .char_indices()
            .next_back()
            .map(|(offset, _)| offset)
    }

    fn next_boundary(&self) -> Option<usize> {
        self.content[self.cursor_byte..]
            .chars()
            .next()
            .map(|character| self.cursor_byte + character.len_utf8())
    }
}

fn line_start_at(content: &str, cursor: usize) -> usize {
    content[..cursor]
        .rfind('\n')
        .map_or(0, |newline| newline + 1)
}

fn line_end_at(content: &str, cursor: usize) -> usize {
    content[cursor..]
        .find('\n')
        .map_or(content.len(), |offset| cursor + offset)
}

fn byte_at_column(content: &str, start: usize, end: usize, column: usize) -> usize {
    content[start..end]
        .char_indices()
        .nth(column)
        .map_or(end, |(offset, _)| start + offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty_and_replacement_resets_cursor() {
        let mut buffer = TextBuffer::new();
        assert_eq!(buffer.content(), "");
        assert_eq!(buffer.cursor_byte(), 0);
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 0, column: 0 }
        );
        assert_eq!(buffer.line_count(), 1);

        buffer.set_content("first\nsecond\n");
        assert_eq!(buffer.cursor_byte(), 0);
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 0, column: 0 }
        );
        assert_eq!(buffer.line_count(), 3);
    }

    #[test]
    fn rejects_non_boundaries_without_moving_cursor() {
        let mut buffer = TextBuffer::from_content("aé🙂z");
        assert!(buffer.set_cursor_byte(1));
        assert!(!buffer.set_cursor_byte(2));
        assert!(!buffer.set_cursor_byte(buffer.content().len() + 1));
        assert_eq!(buffer.cursor_byte(), 1);
        assert!(buffer.content().is_char_boundary(buffer.cursor_byte()));
    }

    #[test]
    fn inserts_unicode_and_newlines_at_the_cursor() {
        let mut buffer = TextBuffer::from_content("ab");
        assert!(buffer.set_cursor_byte(1));
        buffer.insert_char('界');
        buffer.insert_newline();
        buffer.insert_char('🙂');

        assert_eq!(buffer.content(), "a界\n🙂b");
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 1 }
        );
        assert!(buffer.content().is_char_boundary(buffer.cursor_byte()));
    }

    #[test]
    fn left_and_right_cross_lines_by_whole_characters() {
        let mut buffer = TextBuffer::from_content("é\n🙂");
        assert!(buffer.move_right());
        assert_eq!(buffer.cursor_byte(), 'é'.len_utf8());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 0, column: 1 }
        );
        assert!(buffer.move_right());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 0 }
        );
        assert!(buffer.move_right());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 1 }
        );
        assert!(!buffer.move_right());

        assert!(buffer.move_left());
        assert!(buffer.move_left());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 0, column: 1 }
        );
    }

    #[test]
    fn backspace_and_delete_remove_complete_unicode_characters() {
        let mut buffer = TextBuffer::from_content("aé\n🙂z");
        assert!(buffer.set_cursor_byte("aé\n".len()));
        assert!(buffer.backspace());
        assert_eq!(buffer.content(), "aé🙂z");
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 0, column: 2 }
        );

        assert!(buffer.backspace());
        assert_eq!(buffer.content(), "a🙂z");
        assert!(buffer.delete_forward());
        assert_eq!(buffer.content(), "az");
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 0, column: 1 }
        );

        assert!(buffer.move_left());
        assert!(!buffer.move_left());
        assert!(buffer.delete_forward());
        assert_eq!(buffer.content(), "z");
        assert!(buffer.move_end());
        assert!(!buffer.delete_forward());
    }

    #[test]
    fn vertical_movement_preserves_desired_column_across_short_lines() {
        let mut buffer = TextBuffer::from_content("abcdef\nxy\n123456\nq");
        assert!(buffer.set_cursor_byte(5));
        assert!(buffer.move_down());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 2 }
        );
        assert!(buffer.move_down());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 2, column: 5 }
        );
        assert!(buffer.move_down());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 3, column: 1 }
        );
        assert!(!buffer.move_down());

        assert!(buffer.move_up());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 2, column: 5 }
        );
        assert!(buffer.move_up());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 2 }
        );
        assert!(buffer.move_up());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 0, column: 5 }
        );
        assert!(!buffer.move_up());
    }

    #[test]
    fn vertical_columns_count_unicode_scalars_not_bytes() {
        let mut buffer = TextBuffer::from_content("aé🙂z\n短\n12345");
        assert!(buffer.set_cursor_byte("aé🙂".len()));
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 0, column: 3 }
        );
        assert!(buffer.move_down());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 1 }
        );
        assert!(buffer.move_down());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 2, column: 3 }
        );
        assert!(buffer.content().is_char_boundary(buffer.cursor_byte()));
    }

    #[test]
    fn page_movement_preserves_column_through_utf8_and_short_lines() {
        let mut buffer = TextBuffer::from_content("aé🙂z\n短\n12345\nxy\nαβγδε\nq\nuvwxy");
        assert!(buffer.set_cursor_byte("aé🙂".len()));
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 0, column: 3 }
        );

        assert!(buffer.move_page_down(3));
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 3, column: 2 }
        );
        assert!(buffer.content().is_char_boundary(buffer.cursor_byte()));

        assert!(buffer.move_page_down(2));
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 5, column: 1 }
        );
        assert!(buffer.move_page_down(1));
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 6, column: 3 }
        );

        assert!(buffer.move_page_up(4));
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 2, column: 3 }
        );
        assert!(buffer.move_page_up(10));
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 0, column: 3 }
        );
        assert!(!buffer.move_page_up(1));
    }

    #[test]
    fn zero_line_pages_do_not_move_or_reset_desired_column() {
        let mut buffer = TextBuffer::from_content("abcdef\nx\n123456");
        assert!(buffer.set_cursor_byte(5));
        assert!(buffer.move_down());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 1 }
        );

        assert!(!buffer.move_page_up(0));
        assert!(!buffer.move_page_down(0));
        assert!(buffer.move_page_down(1));
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 2, column: 5 }
        );
        assert!(!buffer.move_page_down(usize::MAX));
    }

    #[test]
    fn horizontal_action_resets_desired_vertical_column() {
        let mut buffer = TextBuffer::from_content("abcdef\nx\n123456");
        assert!(buffer.set_cursor_byte(5));
        assert!(buffer.move_down());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 1 }
        );
        assert!(buffer.move_home());
        assert!(buffer.move_down());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 2, column: 0 }
        );

        assert!(buffer.move_end());
        assert!(buffer.move_up());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 1 }
        );
        assert!(buffer.move_up());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 0, column: 6 }
        );
    }

    #[test]
    fn home_and_end_target_the_current_logical_line() {
        let mut buffer = TextBuffer::from_content("first\né🙂z\nlast");
        assert!(buffer.set_cursor_byte("first\né".len()));
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 1 }
        );
        assert!(buffer.move_end());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 3 }
        );
        assert!(buffer.move_home());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 0 }
        );
        assert!(!buffer.move_home());
    }

    #[test]
    fn empty_and_trailing_lines_have_stable_boundaries() {
        let mut buffer = TextBuffer::from_content("a\n\n");
        assert_eq!(buffer.line_count(), 3);
        assert!(buffer.move_end());
        assert!(buffer.move_down());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 0 }
        );
        assert!(buffer.move_down());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 2, column: 0 }
        );
        assert!(!buffer.move_down());
        assert!(buffer.move_up());
        assert_eq!(
            buffer.cursor_position(),
            CursorPosition { line: 1, column: 0 }
        );
    }
}
