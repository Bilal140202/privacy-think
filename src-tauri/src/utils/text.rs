// Text utilities helper module

/// Safely slice a string up to `max_bytes` without splitting multi-byte UTF-8 character boundaries.
///
/// Prevents panic when slicing strings containing multi-byte characters (emoji, CJK, accented letters)
/// at non-boundary byte offsets.
pub fn safe_slice(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_slice_ascii() {
        assert_eq!(safe_slice("hello world", 5), "hello");
        assert_eq!(safe_slice("hello", 10), "hello");
    }

    #[test]
    fn test_safe_slice_utf8() {
        // '🚀' is 4 bytes
        let s = "hello 🚀 world";
        assert_eq!(safe_slice(s, 8), "hello ");
        assert_eq!(safe_slice(s, 10), "hello 🚀");
    }
}
