// Chat completion request handling for gRPC router

use super::router::GrpcRouter;
use crate::grpc::client::proto;
use crate::protocols::spec::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse,
    ChatMessage, ChatMessageDelta, ChatStreamChoice, Usage,
};
use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use serde_json::json;
use tracing::{debug, error, info};
use ulid::Ulid;

impl GrpcRouter {
    /// Handle /v1/chat/completions request
    pub async fn handle_chat_completion(
        &self,
        _headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let request_start = std::time::Instant::now();
        let request_id = format!("chatcmpl-{}", Ulid::new().to_string());
        info!("[PERF] request_id={} lifecycle_start", request_id);

        // Step 1: Apply chat template to messages
        let step1_start = std::time::Instant::now();
        let prompt_text = match self.apply_chat_template(&body.messages) {
            Ok(text) => text,
            Err(e) => {
                error!("Chat template failed: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(json!({"error": format!("Chat template error: {}", e)})),
                )
                    .into_response();
            }
        };
        info!("[PERF] request_id={} step1_chat_template={}ms", request_id, step1_start.elapsed().as_millis());

        // Step 2: Tokenize
        let step2_start = std::time::Instant::now();
        let prompt_token_ids = match self.tokenize(&prompt_text) {
            Ok(ids) => ids,
            Err(e) => {
                error!("Tokenization failed: {}", e);
                return (StatusCode::INTERNAL_SERVER_ERROR, axum::Json(json!({"error": e})))
                    .into_response();
            }
        };
        info!("[PERF] request_id={} step2_tokenize={}ms tokens={}", request_id, step2_start.elapsed().as_millis(), prompt_token_ids.len());

        // Step 3: Select worker - pass prompt text for consistent hashing
        let step3_start = std::time::Instant::now();
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
        info!("[PERF] request_id={} step3_select_worker={}ms worker={}", request_id, step3_start.elapsed().as_millis(), worker.url());

        let worker_url = worker.url().to_string();

        // Parse worker URL to extract base URL and optional DP rank
        let (base_worker_url, dp_rank) = crate::routers::http::dp_utils::parse_worker_url(&worker_url);
        let worker_url_to_use = if self.intra_node_data_parallel_size > 1 {
            &base_worker_url
        } else {
            &worker_url
        };

        // Step 4: Get gRPC client
        let step4_start = std::time::Instant::now();
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
        info!("[PERF] request_id={} step4_get_grpc_client={}ms", request_id, step4_start.elapsed().as_millis());

        // Step 5: Build gRPC request
        // Support both max_completion_tokens (new) and max_tokens (legacy)
        let max_tokens_val = body.max_completion_tokens
            .or(body.max_tokens)
            .map(|v| v as i32);
        info!("[PERF] request_id={} max_tokens from request: {:?} (max_completion_tokens={:?}, max_tokens={:?})",
              request_id, max_tokens_val, body.max_completion_tokens, body.max_tokens);

        let grpc_request = proto::GenerateRequest {
            request_id: request_id.clone(),
            tokenized: Some(proto::TokenizedInput {
                original_text: prompt_text.clone(),
                input_ids: prompt_token_ids.clone(),
            }),
            sampling_params: Some(proto::SamplingParams {
                temperature: body.temperature.unwrap_or(1.0),
                top_p: body.top_p.unwrap_or(1.0),
                max_tokens: max_tokens_val,
                stop: body.stop.clone().map(|s| s.to_vec()).unwrap_or_default(),
                ignore_eos: body.ignore_eos,
                ..Default::default()
            }),
            stream: body.stream,
            data_parallel_rank: dp_rank.map(|r| r as i32),
        };

        // Step 6: Call gRPC to get auto-cleanup stream
        let step6_start = std::time::Instant::now();
        let stream = match client.generate(grpc_request).await {
            Ok(s) => s,
            Err(e) => {
                error!("gRPC generate failed: {}", e);
                self.return_grpc_client(base_worker_url, client);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(json!({"error": format!("Generation failed: {}", e)})),
                )
                    .into_response();
            }
        };
        info!("[PERF] request_id={} step6_grpc_generate_call={}ms", request_id, step6_start.elapsed().as_millis());

        // Return client immediately - stream owns its own client reference
        let step7_start = std::time::Instant::now();
        self.return_grpc_client(base_worker_url, client);
        info!("[PERF] request_id={} step7_return_client={}ms", request_id, step7_start.elapsed().as_millis());

        let created_time = Self::current_timestamp();
        let model_name = _model_id.unwrap_or("unknown").to_string();

        info!("[PERF] request_id={} total_before_stream_processing={}ms", request_id, request_start.elapsed().as_millis());

        // Step 8: Handle streaming vs non-streaming
        if body.stream {
            let processor = super::stream_processor_v2::StreamProcessor::new(self.tokenizer.clone());
            std::sync::Arc::new(processor).process_chat_completion(
                stream,
                request_id,
                model_name,
                created_time,
            )
        } else {
            self.handle_chat_non_streaming(
                stream,
                request_id,
                created_time,
                model_name,
                prompt_token_ids,
            )
            .await
        }
    }

    /// Handle non-streaming chat completion
    async fn handle_chat_non_streaming(
        &self,
        mut stream: super::auto_cleanup_stream::AutoCleanupStream,
        request_id: String,
        created_time: u64,
        model_name: String,
        prompt_token_ids: Vec<u32>,
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

        let message = ChatMessage::Assistant {
            role: "assistant".to_string(),
            content: Some(output_text),
            name: None,
            tool_calls: None,
            function_call: None,
            reasoning_content: None,
        };

        let choice = ChatChoice {
            index: 0,
            message,
            logprobs: None,
            finish_reason: finish_reason.clone(),
            matched_stop: None,
            hidden_states: None,
        };

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
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
