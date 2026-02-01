// Stream processing for gRPC responses
//
// This module provides stream consumption and SSE formatting for both
// chat and completion endpoints.

use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::json;
use std::io;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::grpc::client::proto;
use crate::protocols::spec::{ChatMessageDelta, ChatCompletionStreamResponse, ChatStreamChoice};
use crate::routers::grpc::auto_cleanup_stream::AutoCleanupStream;
use crate::tokenizer::traits::Tokenizer;

/// Processor for streaming gRPC responses
#[derive(Clone)]
pub struct StreamProcessor {
    tokenizer: Arc<dyn Tokenizer>,
}

impl StreamProcessor {
    pub fn new(tokenizer: Arc<dyn Tokenizer>) -> Self {
        Self { tokenizer }
    }

    /// Process chat completion stream and return SSE response
    ///
    /// This spawns a background task to consume the gRPC stream and
    /// returns an SSE response immediately.
    pub fn process_chat_completion(
        self: Arc<Self>,
        stream: AutoCleanupStream,
        request_id: String,
        model: String,
        created: u64,
    ) -> Response {
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        info!("[STREAM_LIFECYCLE] request_id={} Creating SSE response and spawning background task", request_id);

        // Spawn background task to process stream
        let request_id_clone = request_id.clone();
        info!("[STREAM_LIFECYCLE] request_id={} About to spawn background task", request_id);
        tokio::spawn(async move {
            info!("[STREAM_LIFECYCLE] request_id={} Background task started", request_id);
            let result = self
                .consume_chat_stream(stream, &request_id, &model, created, &tx)
                .await;

            if let Err(e) = result {
                error!("[STREAM_LIFECYCLE] request_id={} Stream processing error: {}", request_id, e);
                let error_chunk = format!(
                    "data: {}\n\n",
                    json!({
                        "error": {
                            "message": e,
                            "type": "internal_error"
                        }
                    })
                );
                let _ = tx.send(Ok(Bytes::from(error_chunk)));
            }

            debug!("[STREAM_LIFECYCLE] request_id={} Sending [DONE] marker", request_id);
            let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
            info!("[STREAM_LIFECYCLE] request_id={} Background task completing, dropping tx", request_id);
        });

        debug!("[STREAM_LIFECYCLE] request_id={} Returning SSE response to handler", request_id_clone);
        build_sse_response(rx)
    }

    /// Process completion stream and return SSE response
    pub fn process_completion(
        self: Arc<Self>,
        stream: AutoCleanupStream,
        request_id: String,
        model: String,
        created: u64,
    ) -> Response {
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        tokio::spawn(async move {
            let result = self
                .consume_completion_stream(stream, &request_id, &model, created, &tx)
                .await;

            if let Err(e) = result {
                let error_chunk = format!(
                    "data: {}\n\n",
                    json!({"error": e})
                );
                let _ = tx.send(Ok(Bytes::from(error_chunk)));
            }

            let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
        });

        build_sse_response(rx)
    }

    /// Consume chat stream and send SSE chunks
    async fn consume_chat_stream(
        &self,
        mut stream: AutoCleanupStream,
        request_id: &str,
        model: &str,
        created: u64,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        info!("[STREAM_LIFECYCLE] request_id={} Starting to consume gRPC stream", request_id);
        let mut chunk_count = 0;
        let mut empty_chunk_count = 0;
        let mut token_chunk_count = 0;

        while let Some(response) = stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e))?;

            match gen_response.response {
                Some(proto::generate_response::Response::Chunk(chunk)) => {
                    chunk_count += 1;
                    // Decode tokens
                    let token_ids: Vec<u32> = chunk.token_ids.iter().map(|&id| id as u32).collect();
                    let text = self
                        .tokenizer
                        .decode(&token_ids, true)
                        .unwrap_or_else(|e| format!("[Decode error: {}]", e));

                    if text.is_empty() {
                        empty_chunk_count += 1;
                        continue;
                    }
                    token_chunk_count += 1;

                    // Build chat chunk
                    let delta = ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: Some(text),
                        function_call: None,
                        tool_calls: None,
                        reasoning_content: None,
                    };

                    let chunk_response = ChatCompletionStreamResponse {
                        id: request_id.to_string(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.to_string(),
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta,
                            logprobs: None,
                            finish_reason: None,
                        }],
                        usage: None,
                        system_fingerprint: None,
                    };

                    let sse_data = format!(
                        "data: {}\n\n",
                        serde_json::to_string(&chunk_response).unwrap()
                    );
                    if tx.send(Ok(Bytes::from(sse_data))).is_err() {
                        debug!("Client disconnected");
                        break;
                    }
                }
                Some(proto::generate_response::Response::Complete(complete)) => {
                    info!("[STREAM_LIFECYCLE] request_id={} Received Complete message after {} chunks", request_id, chunk_count);
                    // Send finish chunk
                    let delta = ChatMessageDelta {
                        role: None,
                        content: None,
                        function_call: None,
                        tool_calls: None,
                        reasoning_content: None,
                    };

                    let final_chunk = ChatCompletionStreamResponse {
                        id: request_id.to_string(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.to_string(),
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta,
                            logprobs: None,
                            finish_reason: Some(complete.finish_reason.clone()),
                        }],
                        usage: None,
                        system_fingerprint: None,
                    };

                    let sse_data = format!(
                        "data: {}\n\n",
                        serde_json::to_string(&final_chunk).unwrap()
                    );
                    let _ = tx.send(Ok(Bytes::from(sse_data)));

                    // Mark completed to prevent abort
                    stream.mark_completed();
                    info!("[STREAM_LIFECYCLE] request_id={} Stream marked as completed, breaking loop to recycle connection", request_id);

                    // Break immediately to recycle the gRPC connection
                    // Continuing to read would consume thousands of empty keepalive messages
                    break;
                }
                Some(proto::generate_response::Response::Error(error)) => {
                    return Err(error.message.clone());
                }
                None => continue,
            }
        }

        info!("[STREAM_LIFECYCLE] request_id={} Finished consuming chat gRPC stream, total_chunks={} (tokens={}, empty={})", request_id, chunk_count, token_chunk_count, empty_chunk_count);

        // Mark stream as completed to prevent abort on drop
        stream.mark_completed();
        info!("[STREAM_LIFECYCLE] request_id={} Stream marked as completed (end of stream)", request_id);

        Ok(())
    }

    /// Consume completion stream and send SSE chunks
    async fn consume_completion_stream(
        &self,
        mut stream: AutoCleanupStream,
        request_id: &str,
        model: &str,
        created: u64,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        let mut chunk_count = 0;
        let mut empty_chunk_count = 0;
        let mut token_chunk_count = 0;

        while let Some(response) = stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e))?;

            match gen_response.response {
                Some(proto::generate_response::Response::Chunk(chunk)) => {
                    chunk_count += 1;
                    let token_ids: Vec<u32> = chunk.token_ids.iter().map(|&id| id as u32).collect();
                    let text = self
                        .tokenizer
                        .decode(&token_ids, true)
                        .unwrap_or_else(|e| format!("[Decode error: {}]", e));

                    if text.is_empty() {
                        empty_chunk_count += 1;
                        continue;
                    }
                    token_chunk_count += 1;

                    let chunk_response = json!({
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "text": text,
                            "logprobs": null,
                            "finish_reason": null
                        }]
                    });

                    let sse_data = format!(
                        "data: {}\n\n",
                        serde_json::to_string(&chunk_response).unwrap()
                    );
                    if tx.send(Ok(Bytes::from(sse_data))).is_err() {
                        debug!("Client disconnected");
                        break;
                    }
                }
                Some(proto::generate_response::Response::Complete(complete)) => {
                    let final_chunk = json!({
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "text": "",
                            "logprobs": null,
                            "finish_reason": complete.finish_reason
                        }]
                    });

                    let sse_data = format!(
                        "data: {}\n\n",
                        serde_json::to_string(&final_chunk).unwrap()
                    );
                    let _ = tx.send(Ok(Bytes::from(sse_data)));

                    stream.mark_completed();
                    info!("[STREAM_LIFECYCLE] request_id={} Completion stream marked as completed, breaking loop to recycle connection", request_id);

                    // Break immediately to recycle the gRPC connection
                    break;
                }
                Some(proto::generate_response::Response::Error(error)) => {
                    return Err(error.message.clone());
                }
                None => continue,
            }
        }

        info!("[STREAM_LIFECYCLE] request_id={} Finished consuming completion gRPC stream, total_chunks={} (tokens={}, empty={})", request_id, chunk_count, token_chunk_count, empty_chunk_count);

        // Mark stream as completed to prevent abort on drop
        stream.mark_completed();

        Ok(())
    }
}

/// Build SSE response from channel
fn build_sse_response(rx: mpsc::UnboundedReceiver<Result<Bytes, io::Error>>) -> Response {
    use axum::body::Body;
    use axum::http::{HeaderValue, StatusCode};
    use http::header::CONTENT_TYPE;
    use tokio_stream::wrappers::UnboundedReceiverStream;

    let stream = UnboundedReceiverStream::new(rx);
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
        .headers_mut()
        .insert("Cache-Control", HeaderValue::from_static("no-cache"));
    response
        .headers_mut()
        .insert("Connection", HeaderValue::from_static("keep-alive"));
    response
}
