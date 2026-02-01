// gRPC Router Implementation

use crate::config::types::RetryConfig;
use crate::core::{
    BasicWorker, CircuitBreakerConfig, HealthChecker, HealthConfig, Worker, WorkerType,
};
use crate::grpc::VllmSchedulerClient;
use crate::metrics::RouterMetrics;
use crate::policies::LoadBalancingPolicy;
use crate::protocols::spec::ChatMessage;
use crate::routers::{RouterTrait, WorkerManagement};
use crate::tokenizer::traits::Tokenizer;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// gRPC router implementation for VLLM
#[allow(dead_code)] // Fields will be used once implementation is complete
pub struct GrpcRouter {
    /// Worker connections
    workers: Arc<RwLock<Vec<Arc<dyn Worker>>>>,
    /// Shared gRPC clients for each worker (lock-free via Arc cloning)
    /// tonic clients are internally Arc-wrapped, making clone() essentially free
    pub(super) grpc_clients: Arc<HashMap<String, VllmSchedulerClient>>,
    /// Load balancing policy
    policy: Arc<dyn LoadBalancingPolicy>,
    /// Tokenizer for handling text encoding/decoding
    pub(super) tokenizer: Arc<dyn Tokenizer>,
    /// Worker health checker
    _health_checker: Option<HealthChecker>,
    /// Configuration
    timeout_secs: u64,
    interval_secs: u64,
    pub(super) intra_node_data_parallel_size: usize,
    api_key: Option<String>,
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
}

impl GrpcRouter {
    /// Create a new gRPC router
    pub async fn new(
        worker_urls: Vec<String>,
        policy: Arc<dyn LoadBalancingPolicy>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Automatically expand to DP-aware workers when intra_node_data_parallel_size > 1
        let worker_urls = if ctx.router_config.intra_node_data_parallel_size > 1 {
            // worker address now in the format of "localhost:50051@dp_rank"
            crate::routers::http::dp_utils::get_dp_aware_workers(
                &worker_urls,
                &ctx.router_config.api_key,
                ctx.router_config.intra_node_data_parallel_size,
            )
            .await?
        } else {
            worker_urls
        };

        // Update metrics
        RouterMetrics::set_active_workers(worker_urls.len());

        // Extract necessary components from context
        let tokenizer = ctx
            .tokenizer
            .as_ref()
            .ok_or_else(|| "gRPC router requires tokenizer".to_string())?
            .clone();

        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Create gRPC clients for each worker with retry logic
        // vLLM may take time to start, so we retry for up to 30 minutes
        let max_wait_duration = Duration::from_secs(30 * 60); // 30 minutes
        let start_time = std::time::Instant::now();
        let mut grpc_clients = HashMap::new();
        let mut retry_count = 0;

        info!(
            "Waiting for gRPC workers to become ready (timeout: {} minutes)...",
            max_wait_duration.as_secs() / 60
        );

        // Extract unique base URLs for gRPC connections
        // When DP is enabled, multiple worker URLs may share the same base URL
        let mut base_urls = std::collections::HashSet::new();
        for url in &worker_urls {
            let (base_url, _) = crate::routers::http::dp_utils::parse_worker_url(url);
            base_urls.insert(base_url);
        }
        let base_urls: Vec<String> = base_urls.into_iter().collect();

        // Keep trying until we connect to at least one worker or timeout
        while grpc_clients.is_empty() && start_time.elapsed() < max_wait_duration {
            for base_url in &base_urls {
                // Skip if already connected
                if grpc_clients.contains_key(base_url) {
                    continue;
                }

                match VllmSchedulerClient::connect(base_url).await {
                    Ok(client) => {
                        grpc_clients.insert(base_url.clone(), client);
                        info!("Connected to gRPC worker at {}", base_url);
                    }
                    Err(e) => {
                        if retry_count == 0 {
                            info!(
                                "Waiting for gRPC worker at {} to start (this is normal during vLLM initialization)...",
                                base_url
                            );
                        }
                        // Log detailed errors only on first attempt or every 10th attempt
                        if retry_count == 0 || retry_count % 10 == 0 {
                            warn!(
                                "Connection attempt {} to {}: {} (elapsed: {:.0}s)",
                                retry_count + 1,
                                base_url,
                                e,
                                start_time.elapsed().as_secs()
                            );
                        }
                    }
                }
            }

            if grpc_clients.is_empty() {
                retry_count += 1;
                // Exponential backoff with cap at 5 seconds
                let delay = std::cmp::min(
                    Duration::from_millis(100 * 2_u64.pow(std::cmp::min(retry_count, 5))),
                    Duration::from_secs(5),
                );
                tokio::time::sleep(delay).await;
            }
        }

        if grpc_clients.is_empty() {
            return Err(format!(
                "Failed to connect to any gRPC workers after {} seconds. \
                 Ensure vLLM gRPC server is starting or already running.",
                start_time.elapsed().as_secs()
            ));
        }

        info!(
            "Successfully connected to {} gRPC worker(s) after {:.1}s",
            grpc_clients.len(),
            start_time.elapsed().as_secs_f64()
        );

        // Create Worker trait objects with gRPC connection mode
        let mut workers: Vec<Arc<dyn Worker>> = Vec::new();

        // Create a worker for each DP-aware URL
        // Multiple workers may share the same gRPC client (when DP is enabled)
        for url in &worker_urls {
            // Parse worker URL to extract base URL for gRPC client lookup
            let (base_url, _) = crate::routers::http::dp_utils::parse_worker_url(url);

            if let Some(client) = grpc_clients.get(&base_url) {
                let worker = BasicWorker::with_connection_mode(
                    url.clone(),  // Use full DP-aware URL for worker identification
                    WorkerType::Regular,
                    crate::core::ConnectionMode::Grpc { port: None },
                )
                .with_circuit_breaker_config(core_cb_config.clone())
                .with_health_config(HealthConfig {
                    timeout_secs: ctx.router_config.health_check.timeout_secs,
                    check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                    endpoint: ctx.router_config.health_check.endpoint.clone(),
                    failure_threshold: ctx.router_config.health_check.failure_threshold,
                    success_threshold: ctx.router_config.health_check.success_threshold,
                })
                .with_grpc_client(client.clone());

                workers.push(Arc::new(worker) as Arc<dyn Worker>);
            } else {
                warn!("No gRPC client for worker {} (base: {}), skipping", url, base_url);
            }
        }

        // Initialize policy with workers if needed
        if let Some(cache_aware) = policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            cache_aware.init_workers(&workers);
        }

        let workers = Arc::new(RwLock::new(workers));
        let health_checker = crate::core::start_health_checker(
            Arc::clone(&workers),
            ctx.router_config.worker_startup_check_interval_secs,
        );

        Ok(GrpcRouter {
            workers,
            grpc_clients: Arc::new(grpc_clients),
            policy,
            tokenizer,
            _health_checker: Some(health_checker),
            timeout_secs: ctx.router_config.worker_startup_timeout_secs,
            interval_secs: ctx.router_config.worker_startup_check_interval_secs,
            intra_node_data_parallel_size: ctx.router_config.intra_node_data_parallel_size,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            circuit_breaker_config: core_cb_config,
        })
    }

    /// Helper: Tokenize a prompt string into token IDs
    pub(super) fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, String> {
        debug!("Tokenizing prompt: {} chars", prompt.len());

        let encoding = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let token_ids: Vec<u32> = encoding
            .token_ids()
            .iter()
            .map(|id| *id as u32)
            .collect();

        debug!("Tokenized to {} tokens", token_ids.len());
        Ok(token_ids)
    }

    /// Helper: Detokenize token IDs back to text
    pub(super) fn detokenize(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| format!("Detokenization failed: {}", e))
    }

    /// Helper: Apply chat template to messages
    pub(super) fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String, String> {
        // Convert ChatMessage to a format the tokenizer can use
        // For now, simple concatenation - this should use the actual chat template
        let mut prompt = String::new();

        for msg in messages {
            match msg {
                ChatMessage::System { content, .. } => {
                    prompt.push_str(&format!("System: {}\n", content));
                }
                ChatMessage::User { content, .. } => {
                    // content can be string or array of content parts
                    let text = match content {
                        crate::protocols::spec::UserMessageContent::Text(s) => s.clone(),
                        crate::protocols::spec::UserMessageContent::Parts(parts) => {
                            parts
                                .iter()
                                .filter_map(|p| match p {
                                    crate::protocols::spec::ContentPart::Text { text, .. } => {
                                        Some(text.clone())
                                    }
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join(" ")
                        }
                    };
                    prompt.push_str(&format!("User: {}\n", text));
                }
                ChatMessage::Assistant { content, .. } => {
                    if let Some(text) = content {
                        prompt.push_str(&format!("Assistant: {}\n", text));
                    }
                }
                _ => {}
            }
        }

        prompt.push_str("Assistant:");
        Ok(prompt)
    }

    /// Helper: Get current Unix timestamp in seconds
    pub(super) fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Helper: Select a worker using the load balancing policy
    pub(super) async fn select_worker(&self, request_text: Option<&str>) -> Result<Arc<dyn Worker>, String> {
        let workers = self.workers.read().unwrap();

        if workers.is_empty() {
            return Err("No workers available".to_string());
        }

        // Use policy to select worker (returns index)
        let worker_idx = self
            .policy
            .select_worker(&workers, request_text)
            .ok_or_else(|| "No healthy worker available".to_string())?;

        Ok(workers[worker_idx].clone())
    }

    /// Helper: Get gRPC client for a worker (lock-free via cheap clone)
    /// Since tonic clients are Arc-wrapped internally, cloning is essentially
    /// just incrementing a reference count - no actual connection duplication.
    pub(super) async fn get_grpc_client(&self, worker_url: &str) -> Result<VllmSchedulerClient, String> {
        // Lock-free read from shared client map
        self.grpc_clients
            .get(worker_url)
            .cloned()  // Cheap clone - just Arc increment
            .ok_or_else(|| format!("No gRPC client configured for {}", worker_url))
    }

    /// Helper: No-op for compatibility with streaming handlers
    /// With lock-free client management, we don't need to return clients to a pool.
    /// The client is shared via Arc and will be automatically cleaned up when
    /// all references are dropped.
    #[inline]
    pub(super) fn return_grpc_client(&self, _worker_url: String, _client: VllmSchedulerClient) {
        // No-op: clients are shared via Arc, no pooling needed
    }
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcRouter")
            .field("workers_count", &self.workers.read().unwrap().len())
            .field("timeout_secs", &self.timeout_secs)
            .field("interval_secs", &self.interval_secs)
            .field(
                "intra_node_data_parallel_size",
                &self.intra_node_data_parallel_size,
            )
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.handle_chat_completion(headers, body, model_id).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.handle_completion(headers, body, model_id).await
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn flush_cache(&self) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_worker_loads(&self) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    fn router_type(&self) -> &'static str {
        "grpc"
    }

    fn readiness(&self) -> Response {
        (StatusCode::SERVICE_UNAVAILABLE).into_response()
    }
}

#[async_trait]
impl WorkerManagement for GrpcRouter {
    async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        Err("Not implemented".to_string())
    }

    fn remove_worker(&self, _worker_url: &str) {}

    fn get_worker_urls(&self) -> Vec<String> {
        self.workers
            .read()
            .unwrap()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }
}
