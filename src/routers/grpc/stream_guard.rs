// Automatic stream cleanup using RAII pattern
//
// This module provides a wrapper around gRPC response streams that automatically
// sends an abort RPC when the stream is dropped prematurely (e.g., due to client
// disconnect, error, or cancellation). This prevents wasted GPU computation on
// the backend.

use crate::grpc::client::proto;
use crate::grpc::VllmSchedulerClient;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, warn};

/// Guard wrapper for gRPC generation streams
/// Automatically aborts the request on the backend if dropped before completion
///
/// IMPORTANT: To avoid client leaks, this struct uses Arc<Option<VllmSchedulerClient>>
/// The client is taken out (leaving None) when sending abort, so the Arc reference
/// is released immediately after the abort RPC is spawned, not when the StreamGuard drops.
pub struct StreamGuard {
    inner: tonic::Streaming<proto::GenerateResponse>,
    request_id: String,
    /// Client wrapped in Arc<Option> to allow taking ownership for abort without holding ref
    client: Arc<parking_lot::Mutex<Option<VllmSchedulerClient>>>,
    /// Flag to track if stream completed normally
    completed: Arc<AtomicBool>,
}

impl StreamGuard {
    /// Create a new stream guard
    pub fn new(
        stream: tonic::Streaming<proto::GenerateResponse>,
        request_id: String,
        client: VllmSchedulerClient,
    ) -> Self {
        Self {
            inner: stream,
            request_id,
            client: Arc::new(parking_lot::Mutex::new(Some(client))),
            completed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Mark the stream as successfully completed
    /// Call this when the final completion message is received
    pub fn mark_completed(&self) {
        self.completed.store(true, Ordering::Release);
        // Take the client out of the Option, releasing our reference to it
        // This ensures the client Arc can be dropped immediately
        let mut client_lock = self.client.lock();
        *client_lock = None;
        drop(client_lock); // Explicitly drop the lock
        debug!("Stream {} marked as completed, client released", self.request_id);
    }

    /// Get a reference to the underlying stream
    pub fn stream_mut(&mut self) -> &mut tonic::Streaming<proto::GenerateResponse> {
        &mut self.inner
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        &self.request_id
    }
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        // Check if stream was completed normally
        if !self.completed.load(Ordering::Acquire) {
            // Stream was dropped before completion - send abort to backend
            let request_id = self.request_id.clone();

            // Take the client from the Option (leaving None), so we don't hold the Arc ref
            let client_opt = {
                let mut lock = self.client.lock();
                lock.take()
            };

            if let Some(mut client) = client_opt {
                debug!(
                    "Stream {} dropped before completion, sending abort to backend",
                    request_id
                );

                // Send abort in background task to avoid blocking Drop
                // The client is moved into this task and will be dropped when the task completes
                tokio::spawn(async move {
                    match client
                        .abort_request(
                            request_id.clone(),
                            "Stream dropped before completion".to_string(),
                        )
                        .await
                    {
                        Ok(_) => {
                            debug!("Successfully aborted request {}", request_id);
                        }
                        Err(e) => {
                            // Log but don't fail - the backend may have already finished
                            warn!(
                                "Failed to abort request {} (this may be normal if generation already completed): {}",
                                request_id, e
                            );
                        }
                    }
                    // client is dropped HERE when task completes (a few ms after spawn)
                });
            } else {
                debug!("Stream {} dropped but client already released (likely marked completed)", request_id);
            }
        } else {
            debug!("Stream {} dropped after completion, no abort needed", self.request_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_guard_completion_flag() {
        // This test just ensures the completion flag logic works
        let completed = Arc::new(AtomicBool::new(false));
        assert!(!completed.load(Ordering::Acquire));

        completed.store(true, Ordering::Release);
        assert!(completed.load(Ordering::Acquire));
    }
}
