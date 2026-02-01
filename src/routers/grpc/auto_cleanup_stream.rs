// Automatic stream cleanup using RAII pattern
//
// This wrapper ensures gRPC streams are properly aborted when dropped,
// preventing wasted computation on the backend.

use crate::grpc::client::proto;
use crate::grpc::VllmSchedulerClient;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use tonic::Streaming;
use tracing::{debug, warn};

/// Wrapper around gRPC streaming responses that automatically sends abort on drop
///
/// When the stream completes successfully, call `mark_completed()` to prevent
/// unnecessary abort RPCs.
pub struct AutoCleanupStream {
    inner: Streaming<proto::GenerateResponse>,
    request_id: String,
    client: VllmSchedulerClient,
    completed: Arc<AtomicBool>,
}

impl AutoCleanupStream {
    /// Create a new auto-cleanup stream wrapper
    pub fn new(
        stream: Streaming<proto::GenerateResponse>,
        request_id: String,
        client: VllmSchedulerClient,
    ) -> Self {
        debug!("Created AutoCleanupStream for request {}", request_id);
        Self {
            inner: stream,
            request_id,
            client,
            completed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Mark the stream as completed to prevent abort on drop
    ///
    /// Call this when the request completes successfully to avoid unnecessary abort RPC.
    pub fn mark_completed(&self) {
        let was_completed = self.completed.swap(true, Ordering::AcqRel);
        if !was_completed {
            debug!("Request {} marked as completed", self.request_id);
        } else {
            debug!("Request {} already marked as completed", self.request_id);
        }
    }

    /// Get the request ID
    pub fn request_id(&self) -> &str {
        &self.request_id
    }
}

impl Drop for AutoCleanupStream {
    fn drop(&mut self) {
        // Atomically check and set the completed flag
        // If already completed, skip abort
        if self
            .completed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            debug!(
                "AutoCleanupStream dropped for request {} - already marked completed, skipping abort",
                self.request_id
            );
            return;
        }

        let mut client = self.client.clone();
        let request_id = self.request_id.clone();

        // Spawn background task to send abort (since Drop is sync but abort is async)
        tokio::spawn(async move {
            warn!(
                "Stream dropped without completion for request {}, sending abort",
                request_id
            );
            let request_id_for_log = request_id.clone();
            if let Err(e) = client
                .abort_request(request_id, "Stream dropped".to_string())
                .await
            {
                warn!(
                    "Failed to send abort on drop for request {}: {}",
                    request_id_for_log, e
                );
            } else {
                warn!(
                    "Successfully sent abort for request {}",
                    request_id_for_log
                );
            }
        });
    }
}

// Implement Stream trait to make AutoCleanupStream work like the original Streaming
impl futures::Stream for AutoCleanupStream {
    type Item = Result<proto::GenerateResponse, tonic::Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Delegate to the inner stream
        Pin::new(&mut self.inner).poll_next(cx)
    }
}
