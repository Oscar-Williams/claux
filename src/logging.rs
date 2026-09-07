//! Defer bounded diagnostics while the alternate screen owns the terminal.
use std::io::{self, Write};
use std::sync::Mutex;

const LIMIT: usize = 64 * 1024;

#[derive(Default)]
struct LogState {
    active: bool,
    bytes: Vec<u8>,
    omitted: bool,
}

impl LogState {
    fn write(&mut self, bytes: &[u8], writer: &mut impl Write) -> io::Result<usize> {
        if !self.active {
            return writer.write(bytes);
        }
        let keep = bytes.len().min(LIMIT.saturating_sub(self.bytes.len()));
        self.bytes.extend_from_slice(&bytes[..keep]);
        self.omitted |= keep < bytes.len();
        Ok(bytes.len())
    }

    fn finish(&mut self, writer: &mut impl Write) -> io::Result<()> {
        self.active = false;
        let bytes = std::mem::take(&mut self.bytes);
        let omitted = std::mem::take(&mut self.omitted);
        if !bytes.is_empty() {
            writeln!(
                writer,
                "{}",
                crate::utils::sanitize_terminal_text(&String::from_utf8_lossy(&bytes))
            )?;
        }
        if omitted {
            writeln!(writer, "[additional TUI diagnostics omitted]")?;
        }
        writer.flush()
    }
}

static LOGS: Mutex<LogState> = Mutex::new(LogState {
    active: false,
    bytes: Vec::new(),
    omitted: false,
});

pub struct LogWriter;

pub fn writer() -> LogWriter {
    LogWriter
}

impl Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        LOGS.lock()
            .unwrap_or_else(|e| e.into_inner())
            .write(bytes, &mut io::stderr())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Create before TerminalGuard, so unwinding restores the screen before logs.
pub struct TuiLogs;

impl TuiLogs {
    pub fn defer() -> Self {
        LOGS.lock().unwrap_or_else(|e| e.into_inner()).active = true;
        Self
    }
}

impl Drop for TuiLogs {
    fn drop(&mut self) {
        let _ = LOGS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .finish(&mut io::stderr());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tui_logs_are_bounded_deferred_and_return_to_normal_after_cleanup() {
        let mut state = LogState {
            active: true,
            ..Default::default()
        };
        let mut terminal = Vec::new();
        let noise = vec![b'x'; LIMIT * 2];
        assert_eq!(state.write(&noise, &mut terminal).unwrap(), noise.len());
        assert!(terminal.is_empty());
        assert_eq!(state.bytes.len(), LIMIT);
        state.finish(&mut terminal).unwrap();
        assert!(String::from_utf8_lossy(&terminal).contains("diagnostics omitted"));
        assert!(state.bytes.is_empty());
        state.write(b"normal", &mut terminal).unwrap();
        assert!(terminal.ends_with(b"normal"));
    }
}
