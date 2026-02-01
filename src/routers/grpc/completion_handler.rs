// Completion request handling for gRPC router

use super::router::GrpcRouter;
use crate::grpc::client::proto;
use crate::protocols::spec::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamChoice,
    CompletionStreamResponse, Usage,
};
use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use serde_json::json;
use tracing::{debug, error};
use ulid::Ulid;

impl GrpcRouter {
    /// Handle /v1/completions request
    pub async fn handle_completion(
        &self,
        _headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let request_id = format!("cmpl-{}", Ulid::new().to_string());
        debug!("=== COMPLETION HANDLER CALLED === request_id={}", request_id);
        debug!("Completion request body: stream={:?}, max_tokens={:?}, temperature={:?}",
               body.stream, body.max_tokens, body.temperature);

        // Step 1: Tokenize the prompt
        let prompt_text = match &body.prompt {
            crate::protocols::spec::PromptInput::String(s) => s.clone(),
            crate::protocols::spec::PromptInput::StringArray(arr) => arr.join("\n"),
            crate::protocols::spec::PromptInput::IntArray(ids) => {
                // Convert i32 to u32
                let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
                match self.detokenize(&u32_ids, false) {
                    Ok(text) => text,
                    Err(e) => {
                        error!("Failed to detokenize prompt: {}", e);
                        return (
                            StatusCode::BAD_REQUEST,
                            axum::Json(json!({"error": format!("Invalid token IDs: {}", e)})),
                        )
                            .into_response();
                    }
                }
            }
            crate::protocols::spec::PromptInput::IntBatch(arr) => {
                if arr.is_empty() {
                    return (
                        StatusCode::BAD_REQUEST,
                        axum::Json(json!({"error": "Empty prompt array"})),
                    )
                        .into_response();
                }
                // Convert i32 to u32
                let u32_ids: Vec<u32> = arr[0].iter().map(|&id| id as u32).collect();
                match self.detokenize(&u32_ids, false) {
                    Ok(text) => text,
                    Err(e) => {
                        error!("Failed to detokenize prompt: {}", e);
                        return (
                            StatusCode::BAD_REQUEST,
                            axum::Json(json!({"error": format!("Invalid token IDs: {}", e)})),
                        )
                            .into_response();
                    }
                }
            }
        };

        let prompt_token_ids = match &body.prompt {
            crate::protocols::spec::PromptInput::IntArray(ids) => {
                ids.iter().map(|&id| id as u32).collect()
            }
            crate::protocols::spec::PromptInput::IntBatch(arr) if !arr.is_empty() => {
                arr[0].iter().map(|&id| id as u32).collect()
            }
            _ => match self.tokenize(&prompt_text) {
                Ok(ids) => ids,
                Err(e) => {
                    error!("Tokenization failed: {}", e);
                    return (StatusCode::INTERNAL_SERVER_ERROR, axum::Json(json!({"error": e})))
                        .into_response();
                }
            },
        };

        debug!("Tokenized prompt: {} tokens", prompt_token_ids.len());

        // Step 2: Select worker - pass prompt text for consistent hashing
        let worker = match self.select_worker(Some(&prompt_text)).await {
            Ok(w) => w,
            Err(e) => {
                error!("No worker available: {}", e);
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    axum::Json(json!({"error": "No workers available"})),
                )
                    .into_response();
            }
        };

        let worker_url = worker.url().to_string();
        debug!("Selected worker: {}", worker_url);

        // Parse worker URL to extract base URL and optional DP rank
        let (base_worker_url, dp_rank) = crate::routers::http::dp_utils::parse_worker_url(&worker_url);
        let worker_url_to_use = if self.intra_node_data_parallel_size > 1 {
            &base_worker_url
        } else {
            &worker_url
        };

        // Step 3: Get gRPC client
        let mut client = match self.get_grpc_client(worker_url_to_use).await {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to get gRPC client: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(json!({"error": "gRPC connection failed"})),
                )
                    .into_response();
            }
        };

        // Step 4: Build gRPC request
        let grpc_request = proto::GenerateRequest {
            request_id: request_id.clone(),
            tokenized: Some(proto::TokenizedInput {
                original_text: prompt_text.clone(),
                input_ids: prompt_token_ids.clone(),
            }),
            sampling_params: Some(proto::SamplingParams {
                temperature: body.temperature.unwrap_or(1.0),
                top_p: body.top_p.unwrap_or(1.0),
                top_k: body.top_k.unwrap_or(-1),
                max_tokens: body.max_tokens.map(|v| v as i32),
                stop: body.stop.clone().map(|s| s.to_vec()).unwrap_or_default(),
                ..Default::default()
            }),
            stream: body.stream,
            data_parallel_rank: dp_rank.map(|r| r as i32),
        };

        // Step 5: Call gRPC to get auto-cleanup stream
        let stream = match client.generate(grpc_request).await {
            Ok(s) => s,
            Err(e) => {
                error!("gRPC generate failed: {}", e);
                self.return_grpc_client(base_worker_url.clone(), client);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(json!({"error": format!("Generation failed: {}", e)})),
                )
                    .into_response();
            }
        };

        // Return client immediately - stream owns its own client reference
        self.return_grpc_client(base_worker_url, client);

        let created_time = Self::current_timestamp();
        let model_name = _model_id.unwrap_or("unknown").to_string();

        // Step 6: Handle streaming vs non-streaming
        if body.stream {
            let processor = super::stream_processor_v2::StreamProcessor::new(self.tokenizer.clone());
            std::sync::Arc::new(processor).process_completion(stream, request_id, model_name, created_time)
        } else {
            self.handle_completion_non_streaming(
                stream,
                request_id,
                created_time,
                model_name,
                prompt_token_ids,
                prompt_text,
                body.echo,
            )
            .await
        }
    }

    /// Handle non-streaming completion
    async fn handle_completion_non_streaming(
        &self,
        mut stream: super::auto_cleanup_stream::AutoCleanupStream,
        request_id: String,
        created_time: u64,
        model_name: String,
        prompt_token_ids: Vec<u32>,
        prompt_text: String,
        echo: bool,
    ) -> Response {
        let mut all_token_ids = Vec::new();
        let mut finish_reason = None;
        let mut completion_tokens = 0;

        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    match response.response {
                        Some(proto::generate_response::Response::Chunk(chunk)) => {
                            all_token_ids.extend_from_slice(&chunk.token_ids);
                            completion_tokens += chunk.token_ids.len() as i32;
                        }
                        Some(proto::generate_response::Response::Complete(complete)) => {
                            if !complete.output_ids.is_empty() {
                                all_token_ids = complete.output_ids.clone();
                            }
                            finish_reason = Some(complete.finish_reason);
                            completion_tokens = complete.completion_tokens;

                            // Mark stream as completed
                            stream.mark_completed();
                            break;
                        }
                        Some(proto::generate_response::Response::Error(err)) => {
                            error!("gRPC error: {}", err.message);
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                axum::Json(json!({"error": err.message})),
                            )
                                .into_response();
                        }
                        None => {}
                    }
                }
                Err(e) => {
                    error!("Stream error: {}", e);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        axum::Json(json!({"error": format!("Stream error: {}", e)})),
                    )
                        .into_response();
                }
            }
        }

        // Mark stream as completed if not already marked
        stream.mark_completed();

        // Stream dropped here, client already released via mark_completed()

        let output_text = match self.detokenize(&all_token_ids, true) {
            Ok(text) => text,
            Err(e) => {
                error!("Detokenization failed: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(json!({"error": format!("Detokenization failed: {}", e)})),
                )
                    .into_response();
            }
        };

        let final_text = if echo {
            format!("{}{}", prompt_text, output_text)
        } else {
            output_text
        };

        let choice = CompletionChoice {
            index: 0,
            text: final_text,
            logprobs: None,
            finish_reason: finish_reason.clone(),
            matched_stop: None,
            hidden_states: None,
        };

        let response = CompletionResponse {
            id: request_id,
            object: "text_completion".to_string(),
            created: created_time,
            model: model_name,
            choices: vec![choice],
            usage: Some(Usage {
                prompt_tokens: prompt_token_ids.len() as u32,
                completion_tokens: completion_tokens as u32,
                total_tokens: (prompt_token_ids.len() as u32 + completion_tokens as u32),
                completion_tokens_details: None,
            }),
            system_fingerprint: None,
        };

        (StatusCode::OK, axum::Json(response)).into_response()
    }
}

