use core::num::NonZeroU8;

use wasm_bindgen::prelude::*;

/// Browser-facing wrapper around [`crate::gpu::AnubisGpuSolver`].
#[wasm_bindgen]
pub struct WasmAnubisGpuSolver {
    inner: crate::gpu::AnubisGpuSolver,
}

/// Result returned by the browser WebGPU solver.
#[wasm_bindgen]
pub struct WasmGpuSolution {
    nonce: u64,
    hash: String,
    dispatched_hashes: u64,
}

#[wasm_bindgen(js_name = createAnubisGpuSolver)]
pub async fn create_anubis_gpu_solver() -> Result<WasmAnubisGpuSolver, JsError> {
    let inner = crate::gpu::AnubisGpuSolver::create()
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(WasmAnubisGpuSolver { inner })
}

#[wasm_bindgen]
impl WasmAnubisGpuSolver {
    /// WebGPU adapter/backend description.
    #[wasm_bindgen(getter)]
    pub fn adapter(&self) -> String {
        self.inner.adapter().to_owned()
    }

    /// Force lazy WebGPU pipeline/driver setup before the first real challenge.
    #[wasm_bindgen(js_name = warmUp)]
    pub async fn warm_up(&mut self) -> Result<(), JsError> {
        self.inner
            .warm_up()
            .await
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Solve an Anubis challenge with browser WebGPU.
    #[wasm_bindgen(js_name = solve)]
    pub async fn solve(
        &mut self,
        prefix: &str,
        difficulty: u8,
    ) -> Result<WasmGpuSolution, JsError> {
        let difficulty = NonZeroU8::new(difficulty)
            .ok_or_else(|| JsError::new("difficulty must be non-zero"))?;
        let solution = self
            .inner
            .solve(prefix.as_bytes(), difficulty)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        let mut hash = [0_u8; 64];
        crate::encode_hex(&mut hash, solution.hash);
        let hash = String::from_utf8(hash.to_vec())
            .map_err(|_| JsError::new("GPU returned invalid hash encoding"))?;
        Ok(WasmGpuSolution {
            nonce: solution.nonce,
            hash,
            dispatched_hashes: solution.dispatched_hashes,
        })
    }
}

#[wasm_bindgen]
impl WasmGpuSolution {
    /// Decimal nonce. wasm-bindgen exposes `u64` as a JavaScript `BigInt`.
    #[wasm_bindgen(getter)]
    pub fn nonce(&self) -> u64 {
        self.nonce
    }

    /// Hex SHA-256 digest.
    #[wasm_bindgen(getter)]
    pub fn hash(&self) -> String {
        self.hash.clone()
    }

    /// Number of candidate hashes submitted through the winning batch.
    #[wasm_bindgen(getter, js_name = dispatchedHashes)]
    pub fn dispatched_hashes(&self) -> u64 {
        self.dispatched_hashes
    }
}
