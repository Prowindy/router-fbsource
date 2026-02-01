// Stream processing module for gRPC streaming responses
//
// This module handles the consumption and processing of gRPC streams,
// converting them to SSE (Server-Sent Events) format for HTTP clients.
//
// Key design principles:
// 1. Stream ownership: Streams are consumed by value, allowing RAII cleanup
// 2. Early client release: Clients are released as soon as streaming completes
// 3. Async channels: Uses unbounded channels for SSE chunk delivery

use bytes::Bytes;
use futures::StreamExt;
use serde_json::json;
use std::io;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, error};

use crate::grpc::client::proto;
use crate::routers::grpc::stream_guard::StreamGuard;
use crate::tokenizer::Tokenizer;

/// Process a gRPC stream for chat completions and send chunks via SSE channel
///
/// This function takes ownership of the StreamGuard, ensuring proper cleanup.
/// The StreamGuard's client is released when mark_completed() is called.
pub async fn process_chat_stream(
    mut stream_guard: StreamGuard,
    request_id: String,
    model: String,
    created_time: u64,
    tokenizer: std::sync::Arc<Tokenizer>,
    tx: UnboundedSender<Result<Bytes, io::Error>>,
) {
    let mut got_complete = false;
    let mut chunk_count = 0;

    while let Some(result) = stream_guard.stream_mut().next().await {
        chunk_count += 1;
        match result {
            Ok(response) => {
                match response.response {
                    Some(proto::generate_response::Response::Chunk(chunk)) => {
                        // Convert token IDs to text
                        let token_ids: Vec<u32> = chunk.token_ids.iter().map(|&id| id as u32).collect();
                        let delta_text = match tokenizer.decode(&token_ids, true) {
                            Ok(text) => text,
                            Err(e) => {
                                error!("Detokenization failed: {}", e);
                                format!("[ERROR: {}]", e)
                            }
                        };

                        // Build SSE chunk
                        let chunk_response = json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": delta_text
                                },
                                "finish_reason": null
                            }]
                        });

                        let sse_data = format!("data: {}\n\n", serde_json::to_string(&chunk_response).unwrap());
                        if tx.send(Ok(Bytes::from(sse_data))).is_err() {
                            debug!("Client disconnected, stopping stream processing");
                            break;
                        }
                    }
                    Some(proto::generate_response::Response::Complete(complete)) => {
                        got_complete = true;

                        // Send final chunk with finish reason
                        let final_chunk = json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": complete.finish_reason
                            }]
                        });

                        let sse_data = format!("data: {}\n\n", serde_json::to_string(&final_chunk).unwrap());
                        let _ = tx.send(Ok(Bytes::from(sse_data)));

                        // Mark completed to release client immediately
                        stream_guard.mark_completed();

                        // Continue reading to EOF (don't break)
                    }
                    Some(proto::generate_response::Response::Error(err)) => {
                        error!("gRPC error: {}", err.message);
                        let error_data = json!({"error": err.message});
                        let sse_data = format!("data: {}\n\n", serde_json::to_string(&error_data).unwrap());
                        let _ = tx.send(Ok(Bytes::from(sse_data)));
                        // Continue reading to EOF
                    }
                    None => {}
                }
            }
            Err(e) => {
                error!("Stream error: {}", e);
                let error_data = json!({"error": format!("Stream error: {}", e)});
                let sse_data = format!("data: {}\n\n", serde_json::to_string(&error_data).unwrap());
                let _ = tx.send(Ok(Bytes::from(sse_data)));
                // Stream error usually means EOF
            }
        }
    }

    debug!("Stream processing complete: request_id={}, chunks={}, got_complete={}",
           request_id, chunk_count, got_complete);

    // Send [DONE] marker
    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));

    // StreamGuard is dropped here, client already released via mark_completed()
}

/// Process a gRPC stream for text completions and send chunks via SSE channel
pub async fn process_completion_stream(
    mut stream_guard: StreamGuard,
    request_id: String,
    model: String,
    created_time: u64,
    tokenizer: std::sync::Arc<Tokenizer>,
    tx: UnboundedSender<Result<Bytes, io::Error>>,
) {
    let mut got_complete = false;
    let mut chunk_count = 0;

    while let Some(result) = stream_guard.stream_mut().next().await {
        chunk_count += 1;
        match result {
            Ok(response) => {
                match response.response {
                    Some(proto::generate_response::Response::Chunk(chunk)) => {
                        let token_ids: Vec<u32> = chunk.token_ids.iter().map(|&id| id as u32).collect();
                        let delta_text = match tokenizer.decode(&token_ids, true) {
                            Ok(text) => text,
                            Err(e) => {
                                error!("Detokenization failed: {}", e);
                                format!("[ERROR: {}]", e)
                            }
                        };

                        let chunk_response = json!({
                            "id": request_id,
                            "object": "text_completion",
                            "created": created_time,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "text": delta_text,
                                "logprobs": null,
                                "finish_reason": null
                            }]
                        });

                        let sse_data = format!("data: {}\n\n", serde_json::to_string(&chunk_response).unwrap());
                        if tx.send(Ok(Bytes::from(sse_data))).is_err() {
                            debug!("Client disconnected, stopping stream processing");
                            break;
                        }
                    }
                    Some(proto::generate_response::Response::Complete(complete)) => {
                        got_complete = true;

                        let final_chunk = json!({
                            "id": request_id,
                            "object": "text_completion",
                            "created": created_time,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "text": "",
                                "logprobs": null,
                                "finish_reason": complete.finish_reason
                            }]
                        });

                        let sse_data = format!("data: {}\n\n", serde_json::to_string(&final_chunk).unwrap());
                        let _ = tx.send(Ok(Bytes::from(sse_data)));

                        // Mark completed to release client immediately
                        stream_guard.mark_completed();

                        // Continue reading to EOF
                    }
                    Some(proto::generate_response::Response::Error(err)) => {
                        error!("gRPC error: {}", err.message);
                        let error_data = json!({"error": err.message});
                        let sse_data = format!("data: {}\n\n", serde_json::to_string(&error_data).unwrap());
                        let _ = tx.send(Ok(Bytes::from(sse_data)));
                    }
                    None => {}
                }
            }
            Err(e) => {
                error!("Stream error: {}", e);
                let error_data = json!({"error": format!("Stream error: {}", e)});
                let sse_data = format!("data: {}\n\n", serde_json::to_string(&error_data).unwrap());
                let _ = tx.send(Ok(Bytes::from(sse_data)));
            }
        }
    }

    debug!("Completion stream complete: request_id={}, chunks={}, got_complete={}",
           request_id, chunk_count, got_complete);

    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
}
