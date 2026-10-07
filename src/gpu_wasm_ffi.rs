use wasm_bindgen::prelude::*;

use crate::{
    gpu::{
        AltchaSha256Solver, CerberusSolver, DecimalSolver, GoAwaySolver, GpuContext, GpuSolution,
    },
    solver::{SOLVE_TYPE_GT, SOLVE_TYPE_LT, SOLVE_TYPE_MASK},
};

/// Browser-facing wrapper around the reusable GPU context.
#[wasm_bindgen(js_name = GpuSolver)]
pub struct WasmGpuSolver {
    inner: GpuContext,
}

/// Result returned by the browser GPU solver.
#[wasm_bindgen(js_name = GpuSolution)]
pub struct WasmGpuSolution {
    subtype: String,
    nonce: u64,
    hash: String,
    response: String,
    dispatched_hashes: u64,
    delay: u32,
}

#[wasm_bindgen(js_name = createGpuSolver)]
pub async fn create_gpu_solver() -> Result<WasmGpuSolver, JsError> {
    let inner = GpuContext::create()
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(WasmGpuSolver { inner })
}

#[wasm_bindgen]
impl WasmGpuSolver {
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

    /// Solve a descriptor/configuration JSON object using the matching GPU adapter.
    ///
    /// Supported here: Anubis, Cerberus, GoAway, mCaptcha and Cap.js. Altcha
    /// needs an explicit target and therefore uses `solveAltchaJson`.
    #[wasm_bindgen(js_name = solveJson)]
    pub async fn solve_json(&mut self, input: &str) -> Result<WasmGpuSolution, JsError> {
        if let Ok(descriptor) =
            serde_json::from_str::<crate::adapter::cerberus::ChallengeDescriptor>(input)
        {
            return self.solve_cerberus(&descriptor).await;
        }
        if let Ok(descriptor) =
            serde_json::from_str::<crate::adapter::anubis::ChallengeDescriptor>(input)
        {
            return self.solve_anubis(&descriptor).await;
        }
        if let Ok(config) = serde_json::from_str::<crate::adapter::goaway::GoAwayConfig>(input) {
            return self.solve_goaway(&config).await;
        }
        if let Ok(config) = serde_json::from_str::<crate::adapter::mcaptcha::PoWConfig>(input) {
            return self.solve_mcaptcha(&config).await;
        }
        if let Ok(descriptor) =
            serde_json::from_str::<crate::adapter::capjs::ChallengeDescriptor>(input)
        {
            return self.solve_capjs(descriptor).await;
        }
        Err(JsError::new(
            "unsupported or invalid GPU challenge descriptor",
        ))
    }

    /// Solve an Altcha descriptor. Altcha keeps target/mask outside its descriptor,
    /// so they are explicit here. `solve_type` is one of `1` (LT), `2` (GT), `4` (MASK).
    #[wasm_bindgen(js_name = solveAltchaJson)]
    pub async fn solve_altcha_json(
        &mut self,
        input: &str,
        target: u64,
        solve_type: u8,
        mask: u64,
    ) -> Result<WasmGpuSolution, JsError> {
        let descriptor = serde_json::from_str::<crate::adapter::altcha::ChallengeDescriptor>(input)
            .map_err(|e| JsError::new(&format!("invalid Altcha descriptor: {e}")))?;
        let mut solver = AltchaSha256Solver::from(crate::message::AltchaMessage::from(descriptor));
        let solution =
            solve_by_type(&mut solver, &mut self.inner, target, solve_type, mask).await?;
        Ok(solution_to_wasm("altcha", solution, false, 0))
    }

    async fn solve_anubis(
        &mut self,
        descriptor: &crate::adapter::anubis::ChallengeDescriptor,
    ) -> Result<WasmGpuSolution, JsError> {
        if !descriptor.supported() {
            return Err(JsError::new("unsupported Anubis algorithm"));
        }
        if descriptor.rules().algorithm() == "preact" {
            let (result, _) = descriptor.solve();
            let Some((nonce, hash)) = result else {
                return Err(JsError::new("preact solve failed"));
            };
            let hash = encode_hash(hash, false);
            return Ok(WasmGpuSolution {
                subtype: "anubis".into(),
                nonce,
                response: hash.clone(),
                hash,
                dispatched_hashes: 0,
                delay: descriptor.delay() as u32,
            });
        }

        let difficulty = core::num::NonZeroU8::new(descriptor.rules().difficulty())
            .ok_or_else(|| JsError::new("Anubis difficulty must be non-zero"))?;
        let mask = crate::compute_mask_anubis(difficulty);
        let mut attempted = 0_u64;
        for bank in 0.. {
            let Some(message) = crate::message::DecimalMessage::new(
                descriptor.challenge().as_ref().as_bytes(),
                bank,
            ) else {
                break;
            };
            let mut solver = DecimalSolver::from(message);
            let result = solver
                .solve::<SOLVE_TYPE_MASK>(&mut self.inner, 0, mask)
                .await
                .map_err(gpu_js)?;
            attempted = attempted.saturating_add(solver.get_attempted_nonces());
            if let Some(mut solution) = result {
                solution.dispatched_hashes = attempted;
                return Ok(solution_to_wasm(
                    "anubis",
                    solution,
                    false,
                    descriptor.delay() as u32,
                ));
            }
        }
        Err(JsError::new("Anubis GPU search exhausted"))
    }

    async fn solve_cerberus(
        &mut self,
        descriptor: &crate::adapter::cerberus::ChallengeDescriptor,
    ) -> Result<WasmGpuSolution, JsError> {
        let mut attempted = 0_u64;
        for bank in 0.. {
            let Some(message) = descriptor.build_msg(bank) else {
                break;
            };
            let mut solver = CerberusSolver::from(message);
            let result = solver
                .solve::<SOLVE_TYPE_MASK>(&mut self.inner, 0, descriptor.mask())
                .await
                .map_err(gpu_js)?;
            attempted = attempted.saturating_add(solver.get_attempted_nonces());
            if let Some(mut solution) = result {
                solution.dispatched_hashes = attempted;
                return Ok(solution_to_wasm("cerberus", solution, true, 0));
            }
        }
        Err(JsError::new("Cerberus GPU search exhausted"))
    }

    async fn solve_goaway(
        &mut self,
        config: &crate::adapter::goaway::GoAwayConfig,
    ) -> Result<WasmGpuSolution, JsError> {
        let challenge: &[u8; 64] = config
            .challenge()
            .as_bytes()
            .try_into()
            .map_err(|_| JsError::new("GoAway challenge must contain 64 hex bytes"))?;
        let mut message = crate::message::GoAwayMessage::new_hex(challenge, 0)
            .ok_or_else(|| JsError::new("invalid GoAway hex challenge"))?;
        let mask = crate::compute_mask_goaway(config.difficulty());
        let mut attempted = 0_u64;
        for high in 0_u32.. {
            message.set_high_word(high);
            let mut solver = GoAwaySolver::from(message.clone());
            let result = solver
                .solve::<SOLVE_TYPE_MASK>(&mut self.inner, 0, mask)
                .await
                .map_err(gpu_js)?;
            attempted = attempted.saturating_add(solver.get_attempted_nonces());
            if let Some(mut solution) = result {
                solution.dispatched_hashes = attempted;
                let mut response = config.challenge().as_bytes().to_vec();
                for byte in solution.nonce.to_be_bytes() {
                    response.extend(hex_byte(byte));
                }
                let response = String::from_utf8(response)
                    .map_err(|_| JsError::new("invalid GoAway response encoding"))?;
                let mut wasm = solution_to_wasm("goaway", solution, false, 0);
                wasm.response = response;
                return Ok(wasm);
            }
        }
        Err(JsError::new("GoAway GPU search exhausted"))
    }

    async fn solve_mcaptcha(
        &mut self,
        config: &crate::adapter::mcaptcha::PoWConfig,
    ) -> Result<WasmGpuSolution, JsError> {
        if config.difficulty_factor == 0 {
            return Err(JsError::new("mCaptcha difficulty_factor must be non-zero"));
        }
        let mut prefix = Vec::new();
        crate::build_mcaptcha_prefix(&mut prefix, &config.string, &config.salt);
        let target = crate::compute_target_mcaptcha(u64::from(config.difficulty_factor));
        let mut attempted = 0_u64;
        for bank in 0.. {
            let Some(message) = crate::message::DecimalMessage::new(&prefix, bank) else {
                break;
            };
            let mut solver = DecimalSolver::from(message);
            let result = solver
                .solve::<SOLVE_TYPE_GT>(&mut self.inner, target, u64::MAX)
                .await
                .map_err(gpu_js)?;
            attempted = attempted.saturating_add(solver.get_attempted_nonces());
            if let Some(mut solution) = result {
                solution.dispatched_hashes = attempted;
                let mut wasm = solution_to_wasm("mcaptcha", solution, false, 0);
                wasm.response = crate::extract128_be(solution.hash).to_string();
                return Ok(wasm);
            }
        }
        Err(JsError::new("mCaptcha GPU search exhausted"))
    }

    async fn solve_capjs(
        &mut self,
        descriptor: crate::adapter::capjs::ChallengeDescriptor,
    ) -> Result<WasmGpuSolution, JsError> {
        let rules = *descriptor.rules();
        if rules.difficulty > 16 {
            return Err(JsError::new("Cap.js difficulty must be <= 16"));
        }
        let token = descriptor.token.clone();
        let emitter = crate::message::CapJSEmitter::new(token.as_bytes());
        let mut attempted = 0_u64;
        let mut solutions = Vec::with_capacity(rules.count);
        for i in 0..rules.count {
            let mut salt = vec![0_u8; rules.salt_length];
            let mut targets = [0_u32; 2];
            emitter.emit(&mut salt, &mut targets, i as u32 + 1);
            let mask = if rules.difficulty == 0 {
                0
            } else {
                u64::MAX << (64 - u64::from(rules.difficulty) * 4)
            };
            let (message, fixup) = crate::message::DecimalMessage::new_f64(&salt, 0)
                .ok_or_else(|| JsError::new("Cap.js message cannot be represented"))?;
            let mut solver = DecimalSolver::from(message);
            let result = solver
                .solve::<SOLVE_TYPE_MASK>(
                    &mut self.inner,
                    (u64::from(targets[0]) << 32) | u64::from(targets[1]),
                    mask,
                )
                .await
                .map_err(gpu_js)?
                .ok_or_else(|| JsError::new("Cap.js GPU search exhausted"))?;
            attempted = attempted.saturating_add(solver.get_attempted_nonces());
            let nonce = fixup
                .map(|prefix| prefix.fixup(result.nonce))
                .unwrap_or(result.nonce as f64);
            solutions.push(nonce);
        }
        let response = serde_json::to_string(&serde_json::json!({
            "token": token,
            "solutions": solutions,
        }))
        .map_err(|e| JsError::new(&format!("serialize Cap.js response: {e}")))?;
        Ok(WasmGpuSolution {
            subtype: "capjs".into(),
            nonce: 0,
            hash: String::new(),
            response,
            dispatched_hashes: attempted,
            delay: 0,
        })
    }
}

async fn solve_by_type(
    solver: &mut AltchaSha256Solver,
    gpu: &mut GpuContext,
    target: u64,
    solve_type: u8,
    mask: u64,
) -> Result<GpuSolution, JsError> {
    let result = match solve_type {
        SOLVE_TYPE_LT => solver.solve::<SOLVE_TYPE_LT>(gpu, target, mask).await,
        SOLVE_TYPE_GT => solver.solve::<SOLVE_TYPE_GT>(gpu, target, mask).await,
        SOLVE_TYPE_MASK => solver.solve::<SOLVE_TYPE_MASK>(gpu, target, mask).await,
        _ => {
            return Err(JsError::new(
                "solve_type must be 1 (LT), 2 (GT), or 4 (MASK)",
            ));
        }
    }
    .map_err(gpu_js)?;
    result.ok_or_else(|| JsError::new("Altcha GPU search exhausted"))
}

fn solution_to_wasm(
    subtype: &str,
    solution: GpuSolution,
    little_endian_hash: bool,
    delay: u32,
) -> WasmGpuSolution {
    let hash = encode_hash(solution.hash, little_endian_hash);
    WasmGpuSolution {
        subtype: subtype.into(),
        nonce: solution.nonce,
        response: hash.clone(),
        hash,
        dispatched_hashes: solution.dispatched_hashes,
        delay,
    }
}

fn encode_hash(hash: [u32; 8], little_endian: bool) -> String {
    let mut encoded = [0_u8; 64];
    if little_endian {
        crate::encode_hex_le(&mut encoded, hash);
    } else {
        crate::encode_hex(&mut encoded, hash);
    }
    String::from_utf8(encoded.to_vec()).expect("hex encoder is ASCII")
}

fn hex_byte(byte: u8) -> [u8; 2] {
    fn nibble(value: u8) -> u8 {
        if value < 10 {
            b'0' + value
        } else {
            b'a' + value - 10
        }
    }
    [nibble(byte >> 4), nibble(byte & 0x0f)]
}

fn gpu_js(error: crate::gpu::GpuError) -> JsError {
    JsError::new(&error.to_string())
}

#[wasm_bindgen]
impl WasmGpuSolution {
    /// Solver/protocol identifier.
    #[wasm_bindgen(getter)]
    pub fn subtype(&self) -> String {
        self.subtype.clone()
    }

    /// Solver nonce. wasm-bindgen exposes `u64` as JavaScript `BigInt`.
    #[wasm_bindgen(getter)]
    pub fn nonce(&self) -> u64 {
        self.nonce
    }

    /// Canonical digest encoding when the protocol has one.
    #[wasm_bindgen(getter)]
    pub fn hash(&self) -> String {
        self.hash.clone()
    }

    /// Protocol-facing response payload/value.
    #[wasm_bindgen(getter)]
    pub fn response(&self) -> String {
        self.response.clone()
    }

    /// Number of candidate hashes submitted through the winning batch(es).
    #[wasm_bindgen(getter, js_name = dispatchedHashes)]
    pub fn dispatched_hashes(&self) -> u64 {
        self.dispatched_hashes
    }

    /// Required post-solve delay in milliseconds, if any.
    #[wasm_bindgen(getter)]
    pub fn delay(&self) -> u32 {
        self.delay
    }
}
