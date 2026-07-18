use std::{
    env,
    io::{BufRead, BufReader, Read},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use reqwest::{
    blocking::{Client, Response},
    header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE},
    StatusCode,
};
use serde_json::{json, Value};

const MAX_SSE_LINE_BYTES: usize = 1024 * 1024;
const MAX_ERROR_BODY_BYTES: u64 = 16 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<(String, String)>,
    pub system_prompt: String,
    pub temperature: f64,
    pub max_tokens: u32,
    pub thinking: bool,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<(String, String)>) -> Self {
        Self {
            model: model.into(),
            messages,
            system_prompt: String::new(),
            temperature: 0.8,
            max_tokens: 2048,
            thinking: false,
        }
    }

    fn payload(&self) -> Value {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        if !self.system_prompt.trim().is_empty() {
            messages.push(json!({
                "role": "system",
                "content": self.system_prompt.trim(),
            }));
        }
        for (index, (role, content)) in self.messages.iter().enumerate() {
            let is_last = index + 1 == self.messages.len();
            let content = if self.thinking && is_last && role == "user" {
                format!("/think\n{content}")
            } else {
                content.clone()
            };
            messages.push(json!({"role": role, "content": content}));
        }
        json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream_options": {"include_usage": true},
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletion {
    pub content: String,
    pub completion_tokens: Option<u64>,
    pub elapsed: Duration,
}

impl ChatCompletion {
    pub fn tokens_per_second(&self) -> Option<f64> {
        let seconds = self.elapsed.as_secs_f64();
        (seconds > 0.0)
            .then(|| self.completion_tokens.map(|tokens| tokens as f64 / seconds))
            .flatten()
    }
}

pub fn stream_completion(
    request: &ChatRequest,
    mut on_delta: impl FnMut(&str) -> bool,
) -> Result<ChatCompletion> {
    validate_request(request)?;
    let base_url = env::var("LLAMA_SWAP_URL")
        .unwrap_or_else(|_| "http://localhost:8080".into())
        .trim_end_matches('/')
        .to_owned();
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(600))
        .build()
        .context("build chat HTTP client")?;
    let mut http_request = client
        .post(format!("{base_url}/v1/chat/completions"))
        .header(ACCEPT, "text/event-stream")
        .header(CONTENT_TYPE, "application/json")
        .json(&request.payload());
    if let Ok(api_key) = env::var("LLAMA_SWAP_API_KEY") {
        if !api_key.trim().is_empty() {
            http_request = http_request.header(AUTHORIZATION, format!("Bearer {}", api_key.trim()));
        }
    }

    let started = Instant::now();
    let response = http_request.send().context("send streaming chat request")?;
    let status = response.status();
    if !status.is_success() {
        return Err(response_error(status, response));
    }
    let mut completion = consume_sse(BufReader::new(response), &mut on_delta)?;
    completion.elapsed = started.elapsed();
    anyhow::ensure!(
        !completion.content.is_empty(),
        "chat stream completed without assistant content"
    );
    Ok(completion)
}

fn validate_request(request: &ChatRequest) -> Result<()> {
    anyhow::ensure!(
        !request.model.trim().is_empty(),
        "chat model cannot be empty"
    );
    anyhow::ensure!(
        request.temperature.is_finite() && request.temperature >= 0.0,
        "temperature must be a non-negative finite number"
    );
    anyhow::ensure!(
        request.max_tokens > 0,
        "max_tokens must be greater than zero"
    );
    anyhow::ensure!(!request.messages.is_empty(), "chat history cannot be empty");
    Ok(())
}

fn consume_sse(
    mut reader: impl BufRead,
    on_delta: &mut impl FnMut(&str) -> bool,
) -> Result<ChatCompletion> {
    let mut content = String::new();
    let mut completion_tokens = None;
    let mut line = Vec::with_capacity(1024);
    loop {
        line.clear();
        let bytes = reader
            .read_until(b'\n', &mut line)
            .context("read chat event stream")?;
        if bytes == 0 {
            break;
        }
        anyhow::ensure!(
            line.len() <= MAX_SSE_LINE_BYTES,
            "chat event line exceeds {MAX_SSE_LINE_BYTES} bytes"
        );
        let line = String::from_utf8_lossy(&line);
        let Some(data) = line.trim_end().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let event: Value = match serde_json::from_str(data) {
            Ok(event) => event,
            Err(_) => continue,
        };
        if let Some(tokens) = event
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
        {
            completion_tokens = Some(tokens);
        }
        let Some(delta) = event
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        else {
            continue;
        };
        if delta.is_empty() {
            continue;
        }
        content.push_str(delta);
        if !on_delta(delta) {
            anyhow::bail!("chat stream cancelled");
        }
    }
    Ok(ChatCompletion {
        content,
        completion_tokens,
        elapsed: Duration::ZERO,
    })
}

fn response_error(status: StatusCode, response: Response) -> anyhow::Error {
    let mut body = String::new();
    let _ = response
        .take(MAX_ERROR_BODY_BYTES)
        .read_to_string(&mut body);
    let summary = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if summary.is_empty() {
        anyhow::anyhow!("chat server returned HTTP {status}")
    } else {
        anyhow::anyhow!("chat server returned HTTP {status}: {summary}")
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn payload_includes_parameters_system_prompt_and_thinking_prefix() {
        let mut request = ChatRequest::new("model-a", vec![("user".into(), "explain this".into())]);
        request.system_prompt = " Be concise. ".into();
        request.temperature = 0.25;
        request.max_tokens = 512;
        request.thinking = true;
        let payload = request.payload();

        assert_eq!(payload["model"], "model-a");
        assert_eq!(payload["temperature"], 0.25);
        assert_eq!(payload["max_tokens"], 512);
        assert_eq!(payload["messages"][0]["role"], "system");
        assert_eq!(payload["messages"][0]["content"], "Be concise.");
        assert_eq!(payload["messages"][1]["content"], "/think\nexplain this");
    }

    #[test]
    fn consumes_tokens_and_usage_from_sse() {
        let input = concat!(
            ": keepalive\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}\n\n",
            "data: not-json\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n",
        );
        let mut deltas = Vec::new();
        let completion = consume_sse(Cursor::new(input), &mut |delta| {
            deltas.push(delta.to_owned());
            true
        })
        .unwrap();

        assert_eq!(deltas, ["hello ", "world"]);
        assert_eq!(completion.content, "hello world");
        assert_eq!(completion.completion_tokens, Some(2));
    }

    #[test]
    fn callback_can_cancel_a_stream() {
        let input = "data: {\"choices\":[{\"delta\":{\"content\":\"stop\"}}]}\n";
        let error = consume_sse(Cursor::new(input), &mut |_| false).unwrap_err();
        assert!(error.to_string().contains("cancelled"));
    }

    #[test]
    fn validates_user_controlled_parameters() {
        let mut request = ChatRequest::new("model", vec![("user".into(), "hi".into())]);
        request.temperature = f64::NAN;
        assert!(validate_request(&request).is_err());
        request.temperature = 0.8;
        request.max_tokens = 0;
        assert!(validate_request(&request).is_err());
    }
}
