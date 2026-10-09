use sha2::digest::generic_array::GenericArray;

use crate::{
    Align16, Align64, SWAP_DWORD_BYTE_ORDER, decompose_blocks_mut,
    message::{
        AltchaMessage, BinaryMessage, CerberusMessage, DecimalMessage, DoubleBlockMessage,
        GoAwayMessage, SingleBlockMessage,
    },
};
use core::arch::x86_64::*;

static LANE_ID_MSB_STR: Align64<[u32; 11 * 8]> = Align64({
    let bytes = * b"1111111111222222222233333333334444444444555555555566666666667777777777888888888899999999";
    let mut out = [0; 11 * 8];
    let mut i = 0;
    while i < 11 * 8 {
        out[i] = bytes[i] as u32;
        i += 1;
    }
    out
});

static LANE_ID_LSB_STR: Align64<[u32; 11 * 8]> = Align64({
    let bytes = * b"0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567";
    let mut out = [0; 11 * 8];
    let mut i = 0;
    while i < 11 * 8 {
        out[i] = bytes[i] as u32;
        i += 1;
    }
    out
});

static LANE_ID_STR_COMBINED_LE_HI: Align64<[u32; 1000 / 8 * 8]> = {
    let mut out = [0; 1000 / 8 * 8];
    let mut i = 0;
    while i < 1000 / 8 * 8 {
        let mut copy = i;
        let mut ds = [0; 4];
        let mut j = 0;
        while j < 3 {
            ds[j] = (copy % 10) as u8 + b'0';
            copy /= 10;
            j += 1;
        }
        out[i] = u32::from_be_bytes(ds);
        i += 1;
    }
    Align64(out)
};

#[expect(dead_code)]
mod static_asserts {
    use super::*;

    const ASSERT_LANE_ID_STR_COMBINED_LE_HI_0: [(); 1] =
        [(); (LANE_ID_STR_COMBINED_LE_HI.0[0] == u32::from_be_bytes(*b"000\x00")) as usize];

    const ASSERT_LANE_ID_STR_COMBINED_LE_HI_1: [(); 1] =
        [(); (LANE_ID_STR_COMBINED_LE_HI.0[1] == u32::from_be_bytes(*b"100\x00")) as usize];

    const ASSERT_LANE_ID_STR_COMBINED_LE_HI_123: [(); 1] =
        [(); (LANE_ID_STR_COMBINED_LE_HI.0[123] == u32::from_be_bytes(*b"321\x00")) as usize];
}

#[cfg(feature = "compare-64bit")]
const INDEX_REMAP_PUNPCKLDQ: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];

#[inline]
#[target_feature(enable = "avx2")]
fn find_sha256_target_lane<const TYPE: u8>(
    word0: __m256i,
    word1: __m256i,
    target: u64,
    mask: u64,
) -> Option<usize> {
    #[cfg(not(feature = "compare-64bit"))]
    let _ = word1;

    #[cfg(feature = "compare-64bit")]
    {
        let bias = _mm256_set1_epi64x(i64::MIN);
        let cmp = |x: __m256i| {
            if TYPE == crate::solver::SOLVE_TYPE_GT {
                _mm256_cmpgt_epi64(
                    _mm256_add_epi64(x, bias),
                    _mm256_add_epi64(_mm256_set1_epi64x(target as _), bias),
                )
            } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                _mm256_cmpgt_epi64(
                    _mm256_add_epi64(_mm256_set1_epi64x(target as _), bias),
                    _mm256_add_epi64(x, bias),
                )
            } else {
                _mm256_cmpeq_epi64(
                    _mm256_and_si256(x, _mm256_set1_epi64x(mask as _)),
                    _mm256_set1_epi64x((target & mask) as _),
                )
            }
        };
        let lo = cmp(_mm256_unpacklo_epi32(word1, word0));
        let hi = cmp(_mm256_unpackhi_epi32(word1, word0));
        if _mm256_testz_si256(lo, lo) != 0 && _mm256_testz_si256(hi, hi) != 0 {
            return None;
        }
        let mut dump = Align64([0u64; 8]);
        unsafe {
            _mm256_store_si256(dump.as_mut_ptr().cast(), lo);
            _mm256_store_si256(dump.as_mut_ptr().add(4).cast(), hi);
        }
        let packed_lane = dump.0.iter().position(|x| *x != 0)?;
        Some(INDEX_REMAP_PUNPCKLDQ[packed_lane])
    }
    #[cfg(not(feature = "compare-64bit"))]
    {
        let bias = _mm256_set1_epi32(i32::MIN);
        let target_hi = (target >> 32) as u32;
        let mask_hi = (mask >> 32) as u32;
        let met = if TYPE == crate::solver::SOLVE_TYPE_GT {
            _mm256_cmpgt_epi32(
                _mm256_add_epi32(word0, bias),
                _mm256_add_epi32(_mm256_set1_epi32(target_hi as _), bias),
            )
        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
            _mm256_cmpgt_epi32(
                _mm256_add_epi32(_mm256_set1_epi32(target_hi as _), bias),
                _mm256_add_epi32(word0, bias),
            )
        } else {
            _mm256_cmpeq_epi32(
                _mm256_and_si256(word0, _mm256_set1_epi32(mask_hi as _)),
                _mm256_set1_epi32((target_hi & mask_hi) as _),
            )
        };
        if _mm256_testz_si256(met, met) != 0 {
            return None;
        }
        let mut dump = Align64([0u32; 8]);
        unsafe { _mm256_store_si256(dump.as_mut_ptr().cast(), met) };
        dump.0.iter().position(|x| *x != 0)
    }
}

cpufeatures::new!(avx2, "avx2");

#[derive(Debug, Copy, Clone)]
/// Required features for AVX-2 solver.
pub struct RequiredFeatures;

impl Default for RequiredFeatures {
    fn default() -> Self {
        Self
    }
}

impl crate::solver::CpuIDToken for RequiredFeatures {
    fn get() -> bool {
        avx2::get()
    }
}

/// AVX-2 decimal nonce single block solver.
///
///
/// Current implementation: 8 way SIMD with 1-round hotstart granularity.
pub struct SingleBlockSolver {
    message: SingleBlockMessage,

    attempted_nonces: u64,

    limit: u64,
}

impl From<super::safe::SingleBlockSolver> for SingleBlockSolver {
    fn from(solver: super::safe::SingleBlockSolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
        }
    }
}

impl From<SingleBlockMessage> for SingleBlockSolver {
    fn from(message: SingleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

const MUTATION_TYPE_UNALIGNED: u8 = 0;
const MUTATION_TYPE_ALIGNED: u8 = 1;
const MUTATION_TYPE_OCTAL: u8 = 2;
const MUTATION_TYPE_ALIGNED_OCTAL: u8 = MUTATION_TYPE_ALIGNED | MUTATION_TYPE_OCTAL;
const MUTATION_TYPE_UNALIGNED_OCTAL: u8 = MUTATION_TYPE_UNALIGNED | MUTATION_TYPE_OCTAL;

impl SingleBlockSolver {
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_impl<
        const DIGIT_WORD_IDX0: usize,
        const DIGIT_WORD_IDX1_INCREMENT: bool,
        const TYPE: u8,
        const MUTATION_TYPE: u8,
    >(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<u64> {
        let mut partial_state = self.message.prefix_state;
        crate::sha256::ingest_message_prefix::<DIGIT_WORD_IDX0>(
            &mut partial_state,
            core::array::from_fn(|i| self.message.message[i]),
        );

        // zero out the nonce portion to prevent incorrect results if solvers are reused
        for (ix, i) in (self.message.digit_index..).take(9).enumerate() {
            let message = decompose_blocks_mut(&mut self.message.message);
            message[SWAP_DWORD_BYTE_ORDER[i]] =
                if ix >= 2 && MUTATION_TYPE & MUTATION_TYPE_OCTAL != 0 {
                    b'1'
                } else {
                    b'0'
                };
        }

        let lane_id_0_byte_idx = self.message.digit_index % 4;
        let lane_id_1_byte_idx = (self.message.digit_index + 1) % 4;

        for prefix_set_index in 0..11 {
            if self.attempted_nonces >= self.limit {
                return None;
            }

            let mut inner_key_buf = if MUTATION_TYPE & MUTATION_TYPE_OCTAL != 0 {
                Align16(*b"1111\x80111")
            } else {
                Align16(*b"0000\x80000")
            };

            unsafe {
                let lane_id_0_or_value = _mm256_sll_epi32(
                    _mm256_load_si256(LANE_ID_MSB_STR.as_ptr().add(prefix_set_index * 8).cast()),
                    _mm_cvtsi64x_si128(((3 - lane_id_0_byte_idx) * 8) as _),
                );
                let lane_id_1_or_value = _mm256_sll_epi32(
                    _mm256_load_si256(LANE_ID_LSB_STR.as_ptr().add(prefix_set_index * 8).cast()),
                    _mm_cvtsi64x_si128(((3 - lane_id_1_byte_idx) * 8) as _),
                );

                let lane_id_0_or_value_v = if !DIGIT_WORD_IDX1_INCREMENT {
                    _mm256_or_si256(lane_id_0_or_value, lane_id_1_or_value)
                } else {
                    lane_id_0_or_value
                };

                let mut inner_iteration_end = if MUTATION_TYPE & MUTATION_TYPE_OCTAL != 0 {
                    0o10_000_000
                } else {
                    10_000_000
                };

                // clamp it to the number of remaining iterations
                inner_iteration_end = self
                    .limit
                    .saturating_sub(self.attempted_nonces)
                    .div_ceil(8)
                    .min(inner_iteration_end as u64) as u32;

                // soft pipeline this to compute the new message after the hash
                // LLVM seems to handle cases where high register pressure work happens first better
                // so this prevents some needless register spills
                // doesn't seem to affect performance on my Zen4 but dirty so avoid
                // on the last iteration simd_itoa(10_000_000) is unit-tested to convert to 0000\x80000
                // so no fixup is needed-saves a branch on LLVM codegen
                for next_inner_key in 1..=inner_iteration_end {
                    macro_rules! fetch_msg {
                        ($idx:expr) => {
                            if $idx == DIGIT_WORD_IDX0 {
                                _mm256_or_si256(
                                    _mm256_set1_epi32(self.message.message[$idx] as _),
                                    lane_id_0_or_value_v,
                                )
                            } else if DIGIT_WORD_IDX1_INCREMENT && $idx == DIGIT_WORD_IDX0 + 1 {
                                _mm256_or_si256(
                                    _mm256_set1_epi32(self.message.message[$idx] as _),
                                    lane_id_1_or_value,
                                )
                            } else if (MUTATION_TYPE_ALIGNED & MUTATION_TYPE != 0)
                                && $idx == DIGIT_WORD_IDX0 + 1
                            {
                                _mm256_set1_epi32(
                                    (inner_key_buf.as_ptr().cast::<u32>().read()) as _,
                                )
                            } else if (MUTATION_TYPE_ALIGNED & MUTATION_TYPE != 0)
                                && $idx == DIGIT_WORD_IDX0 + 2
                            {
                                _mm256_set1_epi32(
                                    (inner_key_buf.as_ptr().add(4).cast::<u32>().read()) as _,
                                )
                            } else {
                                _mm256_set1_epi32(self.message.message[$idx] as _)
                            }
                        };
                    }
                    let mut blocks = [
                        fetch_msg!(0),
                        fetch_msg!(1),
                        fetch_msg!(2),
                        fetch_msg!(3),
                        fetch_msg!(4),
                        fetch_msg!(5),
                        fetch_msg!(6),
                        fetch_msg!(7),
                        fetch_msg!(8),
                        fetch_msg!(9),
                        fetch_msg!(10),
                        fetch_msg!(11),
                        fetch_msg!(12),
                        fetch_msg!(13),
                        fetch_msg!(14),
                        fetch_msg!(15),
                    ];

                    let mut state =
                        core::array::from_fn(|i| _mm256_set1_epi32(partial_state[i] as _));

                    // do 8-way SHA-256 without feedback so as not to force the compiler to save 8 registers
                    // we already have them in scalar form, this allows more registers to be reused in the next iteration
                    crate::sha256::avx2::multiway_arx::<DIGIT_WORD_IDX0>(&mut state, &mut blocks);

                    state[0] = _mm256_add_epi32(
                        state[0],
                        _mm256_set1_epi32(self.message.prefix_state[0] as _),
                    );

                    #[cfg(feature = "compare-64bit")]
                    {
                        state[1] = _mm256_add_epi32(
                            state[1],
                            _mm256_set1_epi32(self.message.prefix_state[1] as _),
                        );
                    }

                    #[cfg(not(feature = "compare-64bit"))]
                    let cmp_fn = |x: __m256i, y: __m256i| {
                        let bias = _mm256_set1_epi32(i32::MIN);
                        if TYPE == crate::solver::SOLVE_TYPE_GT {
                            _mm256_cmpgt_epi32(_mm256_add_epi32(x, bias), _mm256_add_epi32(y, bias))
                        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                            _mm256_cmpgt_epi32(_mm256_add_epi32(y, bias), _mm256_add_epi32(x, bias))
                        } else {
                            _mm256_cmpeq_epi32(
                                _mm256_and_si256(x, _mm256_set1_epi32((mask >> 32) as _)),
                                y,
                            )
                        }
                    };

                    #[cfg(feature = "compare-64bit")]
                    let cmp64_fn = |x: __m256i, y: __m256i| {
                        let bias = _mm256_set1_epi64x(i64::MIN);
                        if TYPE == crate::solver::SOLVE_TYPE_GT {
                            _mm256_cmpgt_epi64(_mm256_add_epi64(x, bias), _mm256_add_epi64(y, bias))
                        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                            _mm256_cmpgt_epi64(_mm256_add_epi64(y, bias), _mm256_add_epi64(x, bias))
                        } else {
                            _mm256_cmpeq_epi64(
                                _mm256_and_si256(x, _mm256_set1_epi64x(mask as _)),
                                y,
                            )
                        }
                    };

                    #[cfg(not(feature = "compare-64bit"))]
                    let met_target = cmp_fn(state[0], _mm256_set1_epi32((target >> 32) as _));

                    #[cfg(feature = "compare-64bit")]
                    let result_ab_lo = _mm256_unpacklo_epi32(state[1], state[0]);
                    #[cfg(feature = "compare-64bit")]
                    let result_ab_hi = _mm256_unpackhi_epi32(state[1], state[0]);
                    #[cfg(feature = "compare-64bit")]
                    let (met_target_hi, met_target_lo) = {
                        let ab_met_target_lo =
                            cmp64_fn(result_ab_lo, _mm256_set1_epi64x(target as _));
                        let ab_met_target_high =
                            cmp64_fn(result_ab_hi, _mm256_set1_epi64x(target as _));
                        (ab_met_target_high, ab_met_target_lo)
                    };

                    #[cfg(feature = "compare-64bit")]
                    let nothit = _mm256_testz_si256(met_target_hi, met_target_hi)
                        & _mm256_testz_si256(met_target_lo, met_target_lo);
                    #[cfg(not(feature = "compare-64bit"))]
                    let nothit = _mm256_testz_si256(met_target, met_target);

                    self.attempted_nonces += 8;

                    if nothit == 0 {
                        crate::unlikely();

                        #[cfg(not(feature = "compare-64bit"))]
                        let success_lane_idx = {
                            let mut dump = Align64([0u32; 8]);
                            _mm256_store_si256(dump.as_mut_ptr().cast(), met_target);
                            dump.0.iter().position(|x| *x != 0).unwrap()
                        };

                        #[cfg(feature = "compare-64bit")]
                        let success_lane_idx = INDEX_REMAP_PUNPCKLDQ[{
                            let mut dump = Align64([0u64; 8]);
                            _mm256_store_si256(dump.as_mut_ptr().cast(), met_target_lo);
                            _mm256_store_si256(dump.as_mut_ptr().add(4).cast(), met_target_hi);
                            dump.0.iter().position(|x| *x != 0).unwrap()
                        }];

                        let nonce_prefix = 8 * prefix_set_index + success_lane_idx + 10;

                        if MUTATION_TYPE & MUTATION_TYPE_ALIGNED != 0 {
                            self.message.message[DIGIT_WORD_IDX0 + 1] =
                                inner_key_buf.as_ptr().cast::<u32>().read();
                            self.message.message[DIGIT_WORD_IDX0 + 2] =
                                inner_key_buf.as_ptr().add(4).cast::<u32>().read();
                        }

                        // stamp the lane ID back onto the message
                        {
                            let message_bytes = decompose_blocks_mut(&mut self.message.message);
                            *message_bytes.get_unchecked_mut(
                                *SWAP_DWORD_BYTE_ORDER.get_unchecked(self.message.digit_index),
                            ) = (nonce_prefix / 10) as u8 + b'0';
                            *message_bytes.get_unchecked_mut(
                                *SWAP_DWORD_BYTE_ORDER.get_unchecked(self.message.digit_index + 1),
                            ) = (nonce_prefix % 10) as u8 + b'0';
                        }

                        let mut decimal_inner_key = next_inner_key as u64 - 1;
                        if MUTATION_TYPE & MUTATION_TYPE_OCTAL != 0 {
                            decimal_inner_key = 0;
                            let mut key_octal = next_inner_key - 1;
                            for m in (0..7u32).map(|i| 10u64.pow(i)) {
                                let output = (key_octal % 8) + 1;
                                key_octal /= 8;
                                decimal_inner_key += output as u64 * m;
                            }
                            let mut message_be = [0u8; 64];
                            for i in 0..16 {
                                message_be[i * 4..][..4]
                                    .copy_from_slice(&self.message.message[i].to_be_bytes());
                            }
                        }

                        // the nonce is the 7 digits in the message, plus the first two digits recomputed from the lane index
                        return Some(nonce_prefix as u64 * 10u64.pow(7) + decimal_inner_key);
                    }

                    if MUTATION_TYPE == MUTATION_TYPE_ALIGNED_OCTAL {
                        crate::strings::to_octal_7::<true, 0x80, 1>(
                            &mut inner_key_buf,
                            next_inner_key,
                        )
                    } else if MUTATION_TYPE == MUTATION_TYPE_ALIGNED {
                        crate::strings::simd_itoa8::<7, true, 0x80>(
                            &mut inner_key_buf,
                            next_inner_key,
                        );
                    } else if MUTATION_TYPE == MUTATION_TYPE_UNALIGNED_OCTAL {
                        let message_bytes = decompose_blocks_mut(&mut self.message.message);
                        let mut key_copy = next_inner_key;

                        for i in (0..7).rev() {
                            let output = key_copy % 8;
                            key_copy /= 8;
                            *message_bytes.get_unchecked_mut(
                                *SWAP_DWORD_BYTE_ORDER
                                    .get_unchecked(self.message.digit_index + i + 2),
                            ) = output as u8 + b'1';
                        }
                    } else {
                        let message_bytes = decompose_blocks_mut(&mut self.message.message);
                        let mut key_copy = next_inner_key;

                        for i in (0..7).rev() {
                            let output = key_copy % 10;
                            key_copy /= 10;
                            *message_bytes.get_unchecked_mut(
                                *SWAP_DWORD_BYTE_ORDER
                                    .get_unchecked(self.message.digit_index + i + 2),
                            ) = output as u8 + b'0';
                        }
                    }
                }
            }
        }

        crate::unlikely();
        None
    }
}

impl crate::solver::Solver for SingleBlockSolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve_nonce_only<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        if self.attempted_nonces >= self.limit {
            return None;
        }
        let target = target & mask;

        // the official default difficulty is 5e6, so we design for 1e8
        // and there should almost always be a valid solution within our supported solution space
        // pgeom(5 * 16e7, 1/5e7, lower=F) = 0.03%
        // pgeom(16e7, 1/5e7, lower=F) = 20%, which is too much so we need the prefix to change as well

        // pre-compute an OR to apply to the message to add the lane ID
        let lane_id_0_word_idx = self.message.digit_index / 4;
        let lane_id_1_word_idx = (self.message.digit_index + 1) / 4;

        macro_rules! dispatch {
            ($idx0:literal, $idx1_inc:literal) => {
                unsafe {
                    if self.message.digit_index % 4 == 2 {
                        // if we have to much search space it doesn't matter
                        // use the octal kernel
                        if self.message.no_trailing_zeros
                            || self.message.approx_working_set_count.get() >= 100
                        {
                            self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_ALIGNED_OCTAL>(
                                target, mask,
                            )
                        } else {
                            self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_ALIGNED>(
                                target, mask,
                            )
                        }
                    } else if self.message.no_trailing_zeros {
                        self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_UNALIGNED_OCTAL>(
                            target, mask,
                        )
                    } else {
                        self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_UNALIGNED>(
                            target, mask,
                        )
                    }
                }
            };
            ($idx0:literal) => {
                if lane_id_0_word_idx == lane_id_1_word_idx {
                    dispatch!($idx0, false)
                } else {
                    dispatch!($idx0, true)
                }
            };
        }

        let nonce = match lane_id_0_word_idx {
            0 => dispatch!(0),
            1 => dispatch!(1),
            2 => dispatch!(2),
            3 => dispatch!(3),
            4 => dispatch!(4),
            5 => dispatch!(5),
            6 => dispatch!(6),
            7 => dispatch!(7),
            8 => dispatch!(8),
            9 => dispatch!(9),
            10 => dispatch!(10),
            11 => dispatch!(11),
            12 => dispatch!(12),
            13 => dispatch!(13),
            _ => unsafe { core::hint::unreachable_unchecked() },
        }?;

        Some(nonce + self.message.nonce_addend)
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        let nonce = self.solve_nonce_only::<TYPE>(target, mask)?;

        // recompute the hash from the beginning
        // this prevents the compiler from having to compute the final B-H registers alive in tight loops
        let mut final_sha_state = self.message.prefix_state;
        crate::sha256::digest_block(&mut final_sha_state, &self.message.message);

        Some((nonce, final_sha_state))
    }
}

/// AVX2 decimal nonce double block solver.
///
///
/// Current implementation: 8 way SIMD with 1-round hotstart granularity.
pub struct DoubleBlockSolver {
    message: DoubleBlockMessage,
    attempted_nonces: u64,

    limit: u64,
}

impl From<super::safe::DoubleBlockSolver> for DoubleBlockSolver {
    fn from(solver: super::safe::DoubleBlockSolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
        }
    }
}

impl From<DoubleBlockMessage> for DoubleBlockSolver {
    fn from(message: DoubleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl crate::solver::Solver for DoubleBlockSolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    #[inline]
    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        unsafe { self.solve_impl::<TYPE>(target, mask) }
    }
}

impl DoubleBlockSolver {
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_impl<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        let target = target & mask;

        if self.attempted_nonces >= self.limit {
            return None;
        }

        for (ix, i) in (DoubleBlockMessage::DIGIT_IDX as usize..)
            .take(9)
            .enumerate()
        {
            let message = decompose_blocks_mut(&mut self.message.message);
            message[SWAP_DWORD_BYTE_ORDER[i]] = b'0';
            if ix >= 2 {
                message[SWAP_DWORD_BYTE_ORDER[i]] = b'1';
            }
        }

        let mut partial_state = self.message.prefix_state;
        crate::sha256::sha2_arx::<0>(&mut partial_state, &self.message.message[..13]);

        let mut terminal_message_schedule = Align16([0; 64]);
        terminal_message_schedule[14] = ((self.message.message_length * 8) >> 32) as u32;
        terminal_message_schedule[15] = (self.message.message_length * 8) as u32;
        crate::sha256::do_message_schedule_k_w(&mut terminal_message_schedule);

        let mut itoa_buf = Align16(*b"1111\x80111");
        // the addend is definitely not zero for double block solver, so we can start at 0
        // to recoup some lost search space from using octal digits
        for prefix_set_index in 0..(LANE_ID_LSB_STR.len() / 8) {
            unsafe {
                let lane_id_0_or_value = _mm256_slli_epi32(
                    _mm256_load_si256(LANE_ID_MSB_STR.as_ptr().add(prefix_set_index * 8).cast()),
                    8,
                );
                let lane_id_1_or_value =
                    _mm256_load_si256(LANE_ID_LSB_STR.as_ptr().add(prefix_set_index * 8).cast());

                let lane_index_value_v = _mm256_or_si256(
                    _mm256_set1_epi32(self.message.message[13] as _),
                    _mm256_or_epi32(lane_id_0_or_value, lane_id_1_or_value),
                );

                for next_inner_key in 1..=0o10_000_000 {
                    let cum0 = itoa_buf.as_ptr().cast::<u32>().read();
                    let cum1 = itoa_buf.as_ptr().add(4).cast::<u32>().read();

                    let mut state =
                        core::array::from_fn(|i| _mm256_set1_epi32(partial_state[i] as _));

                    {
                        let mut blocks = [
                            _mm256_set1_epi32(self.message.message[0] as _),
                            _mm256_set1_epi32(self.message.message[1] as _),
                            _mm256_set1_epi32(self.message.message[2] as _),
                            _mm256_set1_epi32(self.message.message[3] as _),
                            _mm256_set1_epi32(self.message.message[4] as _),
                            _mm256_set1_epi32(self.message.message[5] as _),
                            _mm256_set1_epi32(self.message.message[6] as _),
                            _mm256_set1_epi32(self.message.message[7] as _),
                            _mm256_set1_epi32(self.message.message[8] as _),
                            _mm256_set1_epi32(self.message.message[9] as _),
                            _mm256_set1_epi32(self.message.message[10] as _),
                            _mm256_set1_epi32(self.message.message[11] as _),
                            _mm256_set1_epi32(self.message.message[12] as _),
                            lane_index_value_v,
                            _mm256_set1_epi32(cum0 as _),
                            _mm256_set1_epi32(cum1 as _),
                        ];

                        crate::sha256::avx2::multiway_arx::<13>(&mut state, &mut blocks);

                        // we have to do feedback now
                        state
                            .iter_mut()
                            .zip(self.message.prefix_state.iter())
                            .for_each(|(state, prefix_state)| {
                                *state =
                                    _mm256_add_epi32(*state, _mm256_set1_epi32(*prefix_state as _));
                            });
                    }

                    // save only A register for comparison
                    let save_a = state[0];

                    #[cfg(feature = "compare-64bit")]
                    let save_b = state[1];

                    crate::sha256::avx2::bcst_multiway_arx::<14>(
                        &mut state,
                        &terminal_message_schedule,
                    );
                    #[cfg(not(feature = "compare-64bit"))]
                    let cmp_fn = |x: __m256i, y: __m256i| {
                        let bias = _mm256_set1_epi32(i32::MIN);
                        if TYPE == crate::solver::SOLVE_TYPE_GT {
                            _mm256_cmpgt_epi32(_mm256_add_epi32(x, bias), _mm256_add_epi32(y, bias))
                        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                            _mm256_cmpgt_epi32(_mm256_add_epi32(y, bias), _mm256_add_epi32(x, bias))
                        } else {
                            _mm256_cmpeq_epi32(
                                _mm256_and_si256(x, _mm256_set1_epi32((mask >> 32) as _)),
                                y,
                            )
                        }
                    };

                    #[cfg(feature = "compare-64bit")]
                    let cmp64_fn = |x: __m256i, y: __m256i| {
                        let bias = _mm256_set1_epi64x(i64::MIN);
                        if TYPE == crate::solver::SOLVE_TYPE_GT {
                            _mm256_cmpgt_epi64(_mm256_add_epi64(x, bias), _mm256_add_epi64(y, bias))
                        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                            _mm256_cmpgt_epi64(_mm256_add_epi64(y, bias), _mm256_add_epi64(x, bias))
                        } else {
                            _mm256_cmpeq_epi64(
                                _mm256_and_si256(x, _mm256_set1_epi64x(mask as _)),
                                y,
                            )
                        }
                    };
                    state[0] = _mm256_add_epi32(state[0], save_a);

                    #[cfg(feature = "compare-64bit")]
                    {
                        state[1] = _mm256_add_epi32(state[1], save_b);
                    }

                    #[cfg(not(feature = "compare-64bit"))]
                    let met_target = (cmp_fn)(state[0], _mm256_set1_epi32((target >> 32) as _));

                    #[cfg(feature = "compare-64bit")]
                    let result_ab_lo = _mm256_unpacklo_epi32(state[1], state[0]);
                    #[cfg(feature = "compare-64bit")]
                    let result_ab_hi = _mm256_unpackhi_epi32(state[1], state[0]);
                    #[cfg(feature = "compare-64bit")]
                    let (met_target_hi, met_target_lo) = {
                        let ab_met_target_lo =
                            cmp64_fn(result_ab_lo, _mm256_set1_epi64x(target as _));
                        let ab_met_target_hi =
                            cmp64_fn(result_ab_hi, _mm256_set1_epi64x(target as _));
                        (ab_met_target_hi, ab_met_target_lo)
                    };

                    #[cfg(feature = "compare-64bit")]
                    let nothit = _mm256_testz_si256(met_target_hi, met_target_hi)
                        & _mm256_testz_si256(met_target_lo, met_target_lo);
                    #[cfg(not(feature = "compare-64bit"))]
                    let nothit = _mm256_testz_si256(met_target, met_target);

                    if nothit == 0 {
                        crate::unlikely();

                        #[cfg(not(feature = "compare-64bit"))]
                        let success_lane_idx = {
                            let mut dump = Align64([0u32; 8]);
                            _mm256_store_si256(dump.as_mut_ptr().cast(), met_target);
                            dump.0.iter().position(|x| *x != 0).unwrap()
                        };

                        #[cfg(feature = "compare-64bit")]
                        let success_lane_idx = INDEX_REMAP_PUNPCKLDQ[{
                            let mut dump = Align64([0u64; 8]);
                            _mm256_store_si256(dump.as_mut_ptr().cast(), met_target_lo);
                            _mm256_store_si256(dump.as_mut_ptr().add(4).cast(), met_target_hi);
                            dump.0.iter().position(|x| *x != 0).unwrap()
                        }];

                        let nonce_prefix = 10 + 8 * prefix_set_index + success_lane_idx;

                        self.message.message[14] = cum0;
                        self.message.message[15] = cum1;
                        {
                            let message_bytes = decompose_blocks_mut(&mut self.message.message);
                            *message_bytes.get_unchecked_mut(
                                *SWAP_DWORD_BYTE_ORDER
                                    .get_unchecked(DoubleBlockMessage::DIGIT_IDX as usize),
                            ) = (nonce_prefix / 10) as u8 + b'0';
                            *message_bytes.get_unchecked_mut(
                                *SWAP_DWORD_BYTE_ORDER
                                    .get_unchecked(DoubleBlockMessage::DIGIT_IDX as usize + 1),
                            ) = (nonce_prefix % 10) as u8 + b'0';
                        }

                        // recompute the hash from the beginning
                        // this prevents the compiler from having to compute the final B-H registers alive in tight loops
                        let mut final_sha_state = self.message.prefix_state;
                        crate::sha256::digest_block(&mut final_sha_state, &self.message.message);
                        let mut terminal_message = [0; 16];
                        terminal_message[14] = ((self.message.message_length * 8) >> 32) as u32;
                        terminal_message[15] = (self.message.message_length * 8) as u32;
                        crate::sha256::digest_block(&mut final_sha_state, &terminal_message);

                        let mut decimal_inner_key = 0;
                        let mut key_octal = next_inner_key - 1;
                        for m in (0..7u32).map(|i| 10u64.pow(i)) {
                            let output = (key_octal % 8) + 1;
                            key_octal /= 8;
                            decimal_inner_key += output as u64 * m;
                        }

                        let computed_nonce = nonce_prefix as u64 * 10u64.pow(7)
                            + decimal_inner_key
                            + self.message.nonce_addend;

                        // the nonce is the 8 digits in the message, plus the first two digits recomputed from the lane index
                        return Some((computed_nonce, *final_sha_state));
                    }

                    self.attempted_nonces += 8;

                    if self.attempted_nonces >= self.limit {
                        return None;
                    }

                    crate::strings::to_octal_7::<true, 0x80, 1>(&mut itoa_buf, next_inner_key);
                }
            }
        }

        crate::unlikely();

        None
    }
}

#[macro_use]
#[path = "impl_decimal_solver.rs"]
mod impl_decimal_solver;

impl_decimal_solver!(
    [SingleBlockSolver, DoubleBlockSolver] => DecimalSolver
);

/// AVX2 binary nonce solver.
///
/// Output: nonce in little endian order
///
/// Current implementation: 8 way SIMD with 1-round hotstart granularity.
pub struct BinarySolver {
    message: BinaryMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<BinaryMessage> for BinarySolver {
    fn from(message: BinaryMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl From<crate::solver::safe::BinarySolver> for BinarySolver {
    fn from(solver: crate::solver::safe::BinarySolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
        }
    }
}

impl BinarySolver {
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_impl<
        const TYPE: u8,
        const FIRST_NONCE_WORD_IDX: usize,
        const NEED_SECOND_BLOCK: bool,
    >(
        &mut self,
        prefix_state: [u32; 8],
        first_block: &Align64<[u32; 16]>,
        second_block_schedule: &[u32; 64],
        nonce_byte_offset: usize,
        nonce_byte_count: core::num::NonZeroU8,
        target: u64,
        mask: u64,
    ) -> Option<u64> {
        let target = target & mask;

        // at most 3 words may need to be patched (i.e. 96 bits)
        // from which word to poke the nonce?
        let poke_word_base = FIRST_NONCE_WORD_IDX.min(16 - 4);
        let poke_word_byte_base = poke_word_base * 4;
        let mut poke_word_tbl = Align16([!0u8; 16]);
        let nonce_byte_count_decr = nonce_byte_count.get() as usize - 1;
        for (i, ix) in ((nonce_byte_offset + 1)..)
            .take(nonce_byte_count_decr)
            .enumerate()
        {
            poke_word_tbl.0[unsafe {
                *crate::SWAP_DWORD_BYTE_ORDER.get_unchecked(ix - poke_word_byte_base)
            }] = i as u8;
        }
        let poke_word_tbl = unsafe { _mm_load_si128(poke_word_tbl.as_ptr().cast()) };
        let lane_id_byte_remainder = 3 - nonce_byte_offset % 4;
        let mut lane_id_base = Align64([0u32; 8]);
        for i in 0..8 {
            lane_id_base[i as usize] = i << (lane_id_byte_remainder * 8);
        }

        let mut memo_state = prefix_state;
        crate::sha256::ingest_message_prefix::<FIRST_NONCE_WORD_IDX>(
            &mut memo_state,
            first_block[..FIRST_NONCE_WORD_IDX].try_into().unwrap(),
        );

        let candidate_space = 256u64.saturating_pow(nonce_byte_count.get() as u32);
        let search_end = self.limit.min(candidate_space);
        let mut cursor = self.attempted_nonces.min(search_end);
        while cursor < search_end {
            let x = cursor >> 8;
            unsafe {
                let mut block_tpl = *first_block;
                let xm = _mm_cvtsi64x_si128(x as _);
                let xmd = _mm_shuffle_epi8(xm, poke_word_tbl);
                let loadd = _mm_loadu_si128(block_tpl.as_ptr().add(poke_word_base).cast());
                _mm_storeu_si128(
                    block_tpl.as_mut_ptr().add(poke_word_base).cast(),
                    _mm_or_si128(loadd, xmd),
                );

                let mut lane_base = (cursor & 0xff) as usize;
                while lane_base < 256 && cursor < search_end {
                    let valid_lanes =
                        (search_end - cursor).min(8).min((256 - lane_base) as u64) as usize;
                    let lane_id_v = _mm256_add_epi32(
                        _mm256_load_si256(lane_id_base.as_ptr().cast()),
                        _mm256_set1_epi32(((lane_base as u32) << lane_id_byte_remainder * 8) as _),
                    );
                    macro_rules! get_msg {
                        ($idx:expr) => {
                            if $idx == FIRST_NONCE_WORD_IDX {
                                _mm256_or_epi32(_mm256_set1_epi32(block_tpl[$idx] as _), lane_id_v)
                            } else {
                                _mm256_set1_epi32(block_tpl[$idx] as _)
                            }
                        };
                    }

                    let mut state = core::array::from_fn(|i| _mm256_set1_epi32(memo_state[i] as _));
                    let mut msg = [
                        get_msg!(0),
                        get_msg!(1),
                        get_msg!(2),
                        get_msg!(3),
                        get_msg!(4),
                        get_msg!(5),
                        get_msg!(6),
                        get_msg!(7),
                        get_msg!(8),
                        get_msg!(9),
                        get_msg!(10),
                        get_msg!(11),
                        get_msg!(12),
                        get_msg!(13),
                        get_msg!(14),
                        get_msg!(15),
                    ];

                    crate::sha256::avx2::multiway_arx::<FIRST_NONCE_WORD_IDX>(&mut state, &mut msg);

                    if NEED_SECOND_BLOCK {
                        for i in 0..8 {
                            state[i] =
                                _mm256_add_epi32(state[i], _mm256_set1_epi32(prefix_state[i] as _));
                        }
                        let save_a = state[0];
                        #[cfg(feature = "compare-64bit")]
                        let save_b = state[1];

                        crate::sha256::avx2::bcst_multiway_arx::<0>(
                            &mut state,
                            second_block_schedule,
                        );
                        state[0] = _mm256_add_epi32(state[0], save_a);
                        #[cfg(feature = "compare-64bit")]
                        {
                            state[1] = _mm256_add_epi32(state[1], save_b);
                        }
                    } else {
                        state[0] =
                            _mm256_add_epi32(state[0], _mm256_set1_epi32(prefix_state[0] as _));
                        #[cfg(feature = "compare-64bit")]
                        {
                            state[1] =
                                _mm256_add_epi32(state[1], _mm256_set1_epi32(prefix_state[1] as _));
                        }
                    }

                    #[cfg(not(feature = "compare-64bit"))]
                    let cmp_fn = |x: __m256i, y: __m256i| {
                        let bias = _mm256_set1_epi32(i32::MIN);
                        if TYPE == crate::solver::SOLVE_TYPE_GT {
                            _mm256_cmpgt_epi32(_mm256_add_epi32(x, bias), _mm256_add_epi32(y, bias))
                        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                            _mm256_cmpgt_epi32(_mm256_add_epi32(y, bias), _mm256_add_epi32(x, bias))
                        } else {
                            _mm256_cmpeq_epi32(
                                _mm256_and_si256(x, _mm256_set1_epi32((mask >> 32) as _)),
                                y,
                            )
                        }
                    };

                    #[cfg(feature = "compare-64bit")]
                    let cmp64_fn = |x: __m256i, y: __m256i| {
                        let bias = _mm256_set1_epi64x(i64::MIN);
                        if TYPE == crate::solver::SOLVE_TYPE_GT {
                            _mm256_cmpgt_epi64(_mm256_add_epi64(x, bias), _mm256_add_epi64(y, bias))
                        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                            _mm256_cmpgt_epi64(_mm256_add_epi64(y, bias), _mm256_add_epi64(x, bias))
                        } else {
                            _mm256_cmpeq_epi64(
                                _mm256_and_si256(x, _mm256_set1_epi64x(mask as _)),
                                y,
                            )
                        }
                    };

                    #[cfg(not(feature = "compare-64bit"))]
                    let met_target = cmp_fn(state[0], _mm256_set1_epi32((target >> 32) as _));

                    #[cfg(feature = "compare-64bit")]
                    let result_ab_lo = _mm256_unpacklo_epi32(state[1], state[0]);
                    #[cfg(feature = "compare-64bit")]
                    let result_ab_hi = _mm256_unpackhi_epi32(state[1], state[0]);
                    #[cfg(feature = "compare-64bit")]
                    let (met_target_hi, met_target_lo) = {
                        let lo = cmp64_fn(result_ab_lo, _mm256_set1_epi64x(target as _));
                        let hi = cmp64_fn(result_ab_hi, _mm256_set1_epi64x(target as _));
                        (hi, lo)
                    };

                    #[cfg(feature = "compare-64bit")]
                    let nothit = _mm256_testz_si256(met_target_hi, met_target_hi)
                        & _mm256_testz_si256(met_target_lo, met_target_lo);
                    #[cfg(not(feature = "compare-64bit"))]
                    let nothit = _mm256_testz_si256(met_target, met_target);

                    cursor += valid_lanes as u64;
                    self.attempted_nonces = cursor;

                    if nothit == 0 {
                        crate::unlikely();
                        let mut a = Align64([0u32; 8]);
                        _mm256_store_si256(a.as_mut_ptr().cast(), state[0]);
                        #[cfg(feature = "compare-64bit")]
                        let mut b = Align64([0u32; 8]);
                        #[cfg(feature = "compare-64bit")]
                        _mm256_store_si256(b.as_mut_ptr().cast(), state[1]);

                        let success_lane_idx = (0..valid_lanes).find(|&lane| {
                            #[cfg(feature = "compare-64bit")]
                            {
                                let value = (u64::from(a[lane]) << 32) | u64::from(b[lane]);
                                if TYPE == crate::solver::SOLVE_TYPE_GT {
                                    value > target
                                } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                                    value < target
                                } else {
                                    value & mask == target & mask
                                }
                            }
                            #[cfg(not(feature = "compare-64bit"))]
                            {
                                let value = a[lane];
                                let target_hi = (target >> 32) as u32;
                                let mask_hi = (mask >> 32) as u32;
                                if TYPE == crate::solver::SOLVE_TYPE_GT {
                                    value > target_hi
                                } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                                    value < target_hi
                                } else {
                                    value & mask_hi == target_hi & mask_hi
                                }
                            }
                        });

                        if let Some(success_lane_idx) = success_lane_idx {
                            return Some((x << 8) | (lane_base + success_lane_idx) as u64);
                        }
                    }

                    lane_base += valid_lanes;
                }
            }
        }

        None
    }
}

impl crate::solver::Solver for BinarySolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        // A one-byte search has only 256 candidates, and a nonce crossing a SHA block boundary
        // needs a distinct shuffle layout. Keep those rare cases on the scalar fallback instead of
        // burdening the hot SIMD path with extra setup and branches.
        if (self.message.nonce_byte_count.get() == 1)
            || (self.message.salt_residual_len + self.message.nonce_byte_count.get() as usize > 64)
        {
            crate::unlikely();
            let mut solver = crate::solver::safe::BinarySolver::from(self.message.clone());
            return solver.solve::<TYPE>(target, mask);
        }

        let salt = &self.message.salt_residual[..self.message.salt_residual_len];
        let mut blocks = [GenericArray::default(); 2];
        blocks[0][..salt.len()].copy_from_slice(salt);
        let mut ptr = salt.len();
        let mut cur_block = 0;

        for _ in 0..self.message.nonce_byte_count.get() {
            blocks[cur_block][ptr] = 0;
            ptr += 1;
            if ptr == 64 {
                cur_block = 1;
                ptr = 0;
            }
        }
        blocks[cur_block][ptr] = 0x80;
        ptr += 1;
        if ptr + 8 > 64 {
            cur_block = 1;
        }
        blocks[cur_block][(64 - 8)..]
            .copy_from_slice(&(self.message.message_length * 8).to_be_bytes());

        let used_blocks = &mut blocks[..=cur_block];

        let mut block_template_be = Align64([0; 16]);
        for i in 0..16 {
            block_template_be[i] =
                u32::from_be_bytes(used_blocks[0][i * 4..][..4].try_into().unwrap());
        }

        let mut second_block_schedule = [0; 64];
        if cur_block == 1 {
            for i in 0..16 {
                second_block_schedule[i] =
                    u32::from_be_bytes(used_blocks[1][i * 4..][..4].try_into().unwrap());
            }
            crate::sha256::do_message_schedule_k_w(&mut second_block_schedule);
        }

        macro_rules! dispatch {
            ($skipped_rounds:expr) => {
                unsafe {
                    if cur_block == 1 {
                        if let Some(nonce) = self.solve_impl::<TYPE, { $skipped_rounds }, true>(
                            *self.message.prefix_state,
                            &block_template_be,
                            &second_block_schedule,
                            self.message.salt_residual_len,
                            self.message.nonce_byte_count,
                            target,
                            mask,
                        ) {
                            let mut final_sha_state = self.message.prefix_state;
                            for i in 0..self.message.nonce_byte_count.get() as usize {
                                used_blocks[0][self.message.salt_residual_len + i] =
                                    nonce.to_le_bytes()[i];
                            }
                            sha2::compress256(&mut final_sha_state, &used_blocks);
                            return Some((nonce, final_sha_state.0));
                        }
                    } else {
                        if let Some(nonce) = self.solve_impl::<TYPE, { $skipped_rounds }, false>(
                            *self.message.prefix_state,
                            &block_template_be,
                            &[0; 64],
                            self.message.salt_residual_len,
                            self.message.nonce_byte_count,
                            target,
                            mask,
                        ) {
                            let mut final_sha_state = self.message.prefix_state;
                            for i in 0..self.message.nonce_byte_count.get() as usize {
                                used_blocks[0][self.message.salt_residual_len + i] =
                                    nonce.to_le_bytes()[i];
                            }
                            sha2::compress256(&mut final_sha_state, &used_blocks);
                            return Some((nonce, final_sha_state.0));
                        }
                    }
                }
            };
        }

        match self.message.salt_residual_len / 4 {
            0 => dispatch!(0),
            1 => dispatch!(1),
            2 => dispatch!(2),
            3 => dispatch!(3),
            4 => dispatch!(4),
            5 => dispatch!(5),
            6 => dispatch!(6),
            7 => dispatch!(7),
            8 => dispatch!(8),
            9 => dispatch!(9),
            10 => dispatch!(10),
            11 => dispatch!(11),
            12 => dispatch!(12),
            13 => dispatch!(13),
            14 => dispatch!(14),
            15 => dispatch!(15),
            _ => unreachable!(),
        }

        crate::unlikely();

        None
    }
}

/// AVX2 GoAway solver.
///
///
/// Current implementation: 8 way SIMD with 1-round hotstart granularity.
pub struct GoAwaySolver {
    message: GoAwayMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<super::safe::GoAwaySolver> for GoAwaySolver {
    fn from(solver: super::safe::GoAwaySolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
        }
    }
}

impl From<GoAwayMessage> for GoAwaySolver {
    fn from(message: GoAwayMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl GoAwaySolver {
    const MSG_LEN: u32 = 10 * 4 * 8;

    #[inline(never)]
    #[target_feature(enable = "avx2")]
    unsafe fn solve_nonce_only_impl<const TYPE: u8>(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<u64> {
        let lane_id_v = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

        let target = target & mask;

        let mut prefix_state = crate::sha256::IV;
        crate::sha256::ingest_message_prefix(&mut prefix_state, self.message.challenge);

        let search_end = self.limit.min(u64::from(u32::MAX) + 1);

        unsafe {
            let mut partial_state = prefix_state;
            crate::sha256::sha2_arx::<8>(&mut partial_state, &[self.message.high_word]);

            let mut low_word = self.attempted_nonces.min(search_end);
            while low_word < search_end {
                let valid_lanes = (search_end - low_word).min(8) as usize;
                let low_word_u32 = low_word as u32;
                let mut state = core::array::from_fn(|i| _mm256_set1_epi32(partial_state[i] as _));

                let mut msg = [
                    _mm256_set1_epi32(self.message.challenge[0] as _),
                    _mm256_set1_epi32(self.message.challenge[1] as _),
                    _mm256_set1_epi32(self.message.challenge[2] as _),
                    _mm256_set1_epi32(self.message.challenge[3] as _),
                    _mm256_set1_epi32(self.message.challenge[4] as _),
                    _mm256_set1_epi32(self.message.challenge[5] as _),
                    _mm256_set1_epi32(self.message.challenge[6] as _),
                    _mm256_set1_epi32(self.message.challenge[7] as _),
                    _mm256_set1_epi32(self.message.high_word as _),
                    _mm256_add_epi32(_mm256_set1_epi32(low_word_u32 as _), lane_id_v),
                    _mm256_set1_epi32(u32::from_be_bytes([0x80, 0, 0, 0]) as _),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_set1_epi32(Self::MSG_LEN as _),
                ];
                crate::sha256::avx2::multiway_arx::<9>(&mut state, &mut msg);

                state[0] = _mm256_add_epi32(state[0], _mm256_set1_epi32(crate::sha256::IV[0] as _));

                #[cfg(feature = "compare-64bit")]
                {
                    state[1] =
                        _mm256_add_epi32(state[1], _mm256_set1_epi32(crate::sha256::IV[1] as _));
                }

                #[cfg(not(feature = "compare-64bit"))]
                let cmp_fn = |x: __m256i, y: __m256i| {
                    let bias = _mm256_set1_epi32(i32::MIN);
                    if TYPE == crate::solver::SOLVE_TYPE_GT {
                        _mm256_cmpgt_epi32(_mm256_add_epi32(x, bias), _mm256_add_epi32(y, bias))
                    } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                        _mm256_cmpgt_epi32(_mm256_add_epi32(y, bias), _mm256_add_epi32(x, bias))
                    } else {
                        _mm256_cmpeq_epi32(
                            _mm256_and_si256(x, _mm256_set1_epi32((mask >> 32) as _)),
                            y,
                        )
                    }
                };

                #[cfg(feature = "compare-64bit")]
                let cmp64_fn = |x: __m256i, y: __m256i| {
                    let bias = _mm256_set1_epi64x(i64::MIN);
                    if TYPE == crate::solver::SOLVE_TYPE_GT {
                        _mm256_cmpgt_epi64(_mm256_add_epi64(x, bias), _mm256_add_epi64(y, bias))
                    } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                        _mm256_cmpgt_epi64(_mm256_add_epi64(y, bias), _mm256_add_epi64(x, bias))
                    } else {
                        _mm256_cmpeq_epi64(_mm256_and_si256(x, _mm256_set1_epi64x(mask as _)), y)
                    }
                };

                #[cfg(not(feature = "compare-64bit"))]
                let met_target = cmp_fn(state[0], _mm256_set1_epi32((target >> 32) as _));

                #[cfg(feature = "compare-64bit")]
                let result_ab_lo = _mm256_unpacklo_epi32(state[1], state[0]);
                #[cfg(feature = "compare-64bit")]
                let result_ab_hi = _mm256_unpackhi_epi32(state[1], state[0]);
                #[cfg(feature = "compare-64bit")]
                let (met_target_hi, met_target_lo) = {
                    let ab_met_target_lo = cmp64_fn(result_ab_lo, _mm256_set1_epi64x(target as _));
                    let ab_met_target_high =
                        cmp64_fn(result_ab_hi, _mm256_set1_epi64x(target as _));
                    (ab_met_target_high, ab_met_target_lo)
                };

                #[cfg(feature = "compare-64bit")]
                let nothit = _mm256_testz_si256(met_target_hi, met_target_hi)
                    & _mm256_testz_si256(met_target_lo, met_target_lo);
                #[cfg(not(feature = "compare-64bit"))]
                let nothit = _mm256_testz_si256(met_target, met_target);

                self.attempted_nonces += valid_lanes as u64;

                if nothit == 0 {
                    crate::unlikely();
                    let mut a = Align64([0u32; 8]);
                    _mm256_store_si256(a.as_mut_ptr().cast(), state[0]);
                    #[cfg(feature = "compare-64bit")]
                    let mut b = Align64([0u32; 8]);
                    #[cfg(feature = "compare-64bit")]
                    _mm256_store_si256(b.as_mut_ptr().cast(), state[1]);

                    let success_lane_idx = (0..valid_lanes).find(|&lane| {
                        #[cfg(feature = "compare-64bit")]
                        let value = (u64::from(a[lane]) << 32) | u64::from(b[lane]);
                        #[cfg(not(feature = "compare-64bit"))]
                        let value = a[lane];
                        #[cfg(not(feature = "compare-64bit"))]
                        let target = (target >> 32) as u32;
                        #[cfg(not(feature = "compare-64bit"))]
                        let mask = (mask >> 32) as u32;
                        if TYPE == crate::solver::SOLVE_TYPE_GT {
                            value > target
                        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                            value < target
                        } else {
                            value & mask == target & mask
                        }
                    });

                    if let Some(success_lane_idx) = success_lane_idx {
                        let final_low_word = low_word_u32.wrapping_add(success_lane_idx as u32);
                        return Some((self.message.high_word as u64) << 32 | final_low_word as u64);
                    }
                }

                low_word += valid_lanes as u64;
            }
        }
        None
    }
}

impl crate::solver::Solver for GoAwaySolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    #[inline(always)]
    fn solve_nonce_only<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        unsafe { self.solve_nonce_only_impl::<TYPE>(target, mask) }
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        let mut output_msg = [0; 16];
        let nonce = self.solve_nonce_only::<TYPE>(target, mask)?;
        output_msg[..8].copy_from_slice(&self.message.challenge);
        output_msg[8] = (nonce >> 32) as u32;
        output_msg[9] = nonce as u32;
        output_msg[10] = u32::from_be_bytes([0x80, 0, 0, 0]);
        output_msg[15] = Self::MSG_LEN as _;

        let mut final_sha_state = crate::sha256::IV;
        crate::sha256::digest_block(&mut final_sha_state, &output_msg);

        Some((nonce, final_sha_state))
    }
}

/// AVX-2 Cerberus solver.
///
/// Current implementation: 9-digit out-of-order kernel with dual-wavefront 8 way SIMD and quarter-round hotstart granularity.
pub struct CerberusSolver {
    message: CerberusMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<CerberusMessage> for CerberusSolver {
    fn from(message: CerberusMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: !0,
        }
    }
}

impl CerberusSolver {
    /// Set the limit.
    pub fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Get the attempted nonces.
    pub fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }
}

impl CerberusSolver {
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_decimal_impl<
        const CENTER_WORD_IDX: usize,
        const LANE_ID_WORD_IDX: usize,
        const CONSTANT_WORD_COUNT: usize,
    >(
        &mut self,
        msg_tpl: Align64<[u32; 16]>,
        target: u64,
        mask: u64,
    ) -> Option<(u64, u64)> {
        debug_assert_eq!(target, 0);

        let CerberusMessage::Decimal(message) = &self.message else {
            return None;
        };

        // inform LLVM that padding is guaranteed to be zero
        let mut msg = Align64([0u32; 16]);
        msg.0[..=CENTER_WORD_IDX + 1].copy_from_slice(&msg_tpl.0[..=CENTER_WORD_IDX + 1]);
        let prepared_state = crate::blake3::ingest_message_prefix(
            *message.prefix_state,
            &msg[..CONSTANT_WORD_COUNT],
            0,
            message.salt_residual_len as u32 + 9,
            message.flags,
        );

        for lane_id_idx in 0..(LANE_ID_STR_COMBINED_LE_HI.len() / 8) {
            if self.attempted_nonces >= self.limit {
                return None;
            }
            unsafe {
                let mut lane_id_value = _mm256_load_si256(
                    LANE_ID_STR_COMBINED_LE_HI
                        .as_ptr()
                        .add(lane_id_idx * 8)
                        .cast(),
                );
                if CENTER_WORD_IDX < LANE_ID_WORD_IDX {
                    lane_id_value = _mm256_srli_epi32(lane_id_value, 8);
                }

                let state_base =
                    core::array::from_fn(|i| _mm256_set1_epi32(prepared_state[i] as _));
                let patch =
                    _mm256_or_epi32(_mm256_set1_epi32(msg[LANE_ID_WORD_IDX] as _), lane_id_value);
                let maskv = _mm256_set1_epi32((mask >> 32) as _);

                let lut = &crate::strings::DIGIT_LUT_10000_LE_EVEN;
                let remaining = self.limit - self.attempted_nonces;
                let full_pairs = ((remaining / 16) as usize).min(lut.len());

                for (i, word) in lut.iter().take(full_pairs).enumerate() {
                    msg[CENTER_WORD_IDX] = *word;

                    let mut state = state_base;

                    crate::blake3::avx2::compress_mb8::<CONSTANT_WORD_COUNT, LANE_ID_WORD_IDX>(
                        &mut state, &msg, patch,
                    );

                    let s0 = state[0];
                    let sm0 = _mm256_and_si256(s0, maskv);
                    let cmp0 = _mm256_cmpeq_epi32(sm0, _mm256_setzero_si256());
                    let nothit0 = _mm256_testz_si256(cmp0, cmp0);
                    self.attempted_nonces += 8;

                    if nothit0 == 0 {
                        crate::unlikely();
                        let mut dump = Align64([0u32; 8]);
                        let word_idx = i as u64 * 2;

                        _mm256_store_si256(dump.as_mut_ptr().cast(), sm0);

                        let success_lane_idx = dump.0.iter().position(|x| *x == 0).unwrap();

                        return Some((word_idx, lane_id_idx as u64 * 8 + success_lane_idx as u64));
                    }

                    msg[CENTER_WORD_IDX] |= u32::from_be_bytes([1, 0, 0, 0]);

                    state = state_base;
                    crate::blake3::avx2::compress_mb8::<CONSTANT_WORD_COUNT, LANE_ID_WORD_IDX>(
                        &mut state, &msg, patch,
                    );

                    let s1 = state[0];
                    let sm1 = _mm256_and_si256(s1, maskv);
                    let cmp1 = _mm256_cmpeq_epi32(sm1, _mm256_setzero_si256());
                    let nothit1 = _mm256_testz_si256(cmp1, cmp1);
                    self.attempted_nonces += 8;

                    if nothit1 == 0 {
                        crate::unlikely();
                        let mut dump = Align64([0u32; 8]);
                        let word_idx = i as u64 * 2 + 1;

                        _mm256_store_si256(dump.as_mut_ptr().cast(), sm1);

                        let success_lane_idx = dump.0.iter().position(|x| *x == 0).unwrap();

                        return Some((word_idx, lane_id_idx as u64 * 8 + success_lane_idx as u64));
                    }
                }

                if full_pairs < lut.len() {
                    let tail = (self.limit - self.attempted_nonces) as usize;
                    if tail == 0 {
                        return None;
                    }
                    debug_assert!(tail < 16);
                    let i = full_pairs;
                    let word = lut[i];
                    msg[CENTER_WORD_IDX] = word;
                    let mut state = state_base;
                    crate::blake3::avx2::compress_mb8::<CONSTANT_WORD_COUNT, LANE_ID_WORD_IDX>(
                        &mut state, &msg, patch,
                    );
                    let sm0 = _mm256_and_si256(state[0], maskv);
                    let valid0 = tail.min(8);
                    self.attempted_nonces += valid0 as u64;
                    let mut dump = Align64([0u32; 8]);
                    _mm256_store_si256(dump.as_mut_ptr().cast(), sm0);
                    if let Some(success_lane_idx) = dump.0[..valid0].iter().position(|x| *x == 0) {
                        return Some((
                            i as u64 * 2,
                            lane_id_idx as u64 * 8 + success_lane_idx as u64,
                        ));
                    }
                    if valid0 < 8 {
                        return None;
                    }
                    let valid1 = tail - 8;
                    if valid1 == 0 {
                        return None;
                    }
                    msg[CENTER_WORD_IDX] = word | u32::from_be_bytes([1, 0, 0, 0]);
                    state = state_base;
                    crate::blake3::avx2::compress_mb8::<CONSTANT_WORD_COUNT, LANE_ID_WORD_IDX>(
                        &mut state, &msg, patch,
                    );
                    let sm1 = _mm256_and_si256(state[0], maskv);
                    self.attempted_nonces += valid1 as u64;
                    _mm256_store_si256(dump.as_mut_ptr().cast(), sm1);
                    if let Some(success_lane_idx) = dump.0[..valid1].iter().position(|x| *x == 0) {
                        return Some((
                            i as u64 * 2 + 1,
                            lane_id_idx as u64 * 8 + success_lane_idx as u64,
                        ));
                    }
                    return None;
                }
            }
        }
        None
    }

    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_binary_impl(&mut self, target: u64, mask: u64) -> Option<u64> {
        debug_assert_eq!(target, 0);

        let CerberusMessage::Binary(message) = &self.message else {
            return None;
        };

        let mut msg = [0; 16];
        msg[0] = message.first_word;
        let prepared_state = crate::blake3::ingest_message_prefix(
            *message.midstate,
            &msg[..1],
            0,
            8,
            crate::blake3::FLAG_CHUNK_END | crate::blake3::FLAG_ROOT,
        );
        unsafe {
            let state_base = core::array::from_fn(|i| _mm256_set1_epi32(prepared_state[i] as _));
            let mut nonce = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            let increment_nonce = _mm256_set1_epi32(8);
            let masks = (mask >> 32) as u32;
            let maskv = _mm256_set1_epi32(masks as i32);
            for rep in 0..=(u32::MAX / 8) {
                let mut state = state_base;
                crate::blake3::avx2::compress_mb8::<1, 1>(&mut state, &msg, nonce);
                self.attempted_nonces += 8;
                let m = _mm256_and_si256(state[0], maskv);
                let cmp = _mm256_cmpeq_epi32(m, _mm256_setzero_si256());
                let nothit = _mm256_testz_si256(cmp, cmp);
                if nothit == 0 {
                    crate::unlikely();
                    let mut dump = Align64([0u32; 8]);
                    _mm256_store_si256(dump.as_mut_ptr().cast(), state[0]);
                    let success_lane_idx = dump.0.iter().position(|x| *x & masks == 0).unwrap();
                    return Some(
                        (rep * 8 + success_lane_idx as u32) as u64
                            | (message.first_word as u64) << 32,
                    );
                }
                nonce = _mm256_add_epi32(nonce, increment_nonce);
                if self.attempted_nonces >= self.limit {
                    return None;
                }
            }
        }
        None
    }
}

impl crate::solver::Solver for CerberusSolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve_nonce_only<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        match &self.message {
            CerberusMessage::Binary(_) => unsafe { self.solve_binary_impl(target, mask) },
            CerberusMessage::Decimal(message) => {
                // two digits as lane ID, N=\x00, ? is prefix
                // position % 4 =0: |1234|5678|NNN9
                // position % 4 =1: |123?|4567|NN89
                // position % 4 =2: |12??|3456|N789
                // position % 4 =3: |1???|2345|6789

                let center_word_idx = message.salt_residual_len / 4 + 1;
                let nonce_addend = message.nonce_addend;
                let salt_residual = message.salt_residual;
                let salt_residual_len = message.salt_residual_len;
                let position_mod = message.salt_residual_len % 4;

                for resid0 in 0..10u64 {
                    for resid1 in 0..10u64 {
                        if self.attempted_nonces >= self.limit {
                            return None;
                        }
                        let mut msg = salt_residual;

                        match position_mod {
                            0 => {
                                msg[salt_residual_len] = resid0 as u8 + b'0';
                                msg[salt_residual_len + 8] = resid1 as u8 + b'0';
                            }
                            1 => {
                                msg[salt_residual_len + 7] = resid0 as u8 + b'0';
                                msg[salt_residual_len + 8] = resid1 as u8 + b'0';
                            }
                            2 => {
                                msg[salt_residual_len] = resid0 as u8 + b'0';
                                msg[salt_residual_len + 1] = resid1 as u8 + b'0';
                            }
                            3 => {
                                msg[salt_residual_len] = resid0 as u8 + b'0';
                                msg[salt_residual_len + 8] = resid1 as u8 + b'0';
                            }
                            _ => unreachable!(),
                        }

                        let msg = Align64(core::array::from_fn(|i| {
                            u32::from_le_bytes([
                                msg[i * 4],
                                msg[i * 4 + 1],
                                msg[i * 4 + 2],
                                msg[i * 4 + 3],
                            ])
                        }));

                        macro_rules! dispatch {
                            ($center_word_idx:literal) => {
                                if position_mod < 2 {
                                    unsafe {
                                        self.solve_decimal_impl::<$center_word_idx, { $center_word_idx - 1 }, {$center_word_idx - 1}>(
                                        msg, target, mask,
                                        )
                                    }
                                } else {
                                    unsafe {
                                        self.solve_decimal_impl::<$center_word_idx, { $center_word_idx + 1 }, $center_word_idx>(
                                        msg, target, mask,
                                        )
                                    }
                                }
                            };
                        }

                        if let Some((middle_word, success_lane_idx)) = match center_word_idx {
                            1 => dispatch!(1),
                            2 => dispatch!(2),
                            3 => dispatch!(3),
                            4 => dispatch!(4),
                            5 => dispatch!(5),
                            6 => dispatch!(6),
                            7 => dispatch!(7),
                            8 => dispatch!(8),
                            9 => dispatch!(9),
                            10 => dispatch!(10),
                            11 => dispatch!(11),
                            12 => dispatch!(12),
                            13 => dispatch!(13),
                            14 => dispatch!(14),
                            15 => dispatch!(15),
                            _ => unreachable!(),
                        } {
                            let output_nonce = nonce_addend
                                + match position_mod {
                                    0 => {
                                        10 * middle_word
                                            + 100_000 * success_lane_idx
                                            + 100_000_000 * resid0
                                            + resid1
                                    }
                                    1 => {
                                        100 * middle_word
                                            + 1_000_000 * success_lane_idx
                                            + 10 * resid0
                                            + resid1
                                    }
                                    2 => {
                                        1000 * middle_word
                                            + success_lane_idx
                                            + 100_000_000 * resid0
                                            + 10_000_000 * resid1
                                    }
                                    3 => {
                                        10000 * middle_word
                                            + 10 * success_lane_idx
                                            + 100_000_000 * resid0
                                            + resid1
                                    }
                                    _ => unreachable!(),
                                };

                            return Some(output_nonce);
                        }
                    }
                }

                None
            }
        }
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if let Some(nonce) = self.solve_nonce_only::<TYPE>(target, mask) {
            match &self.message {
                CerberusMessage::Decimal(message) => {
                    let mut msg = message.salt_residual;

                    let mut nonce_copy = nonce;
                    for i in (0..9).rev() {
                        msg[message.salt_residual_len + i] = (nonce_copy % 10) as u8 + b'0';
                        nonce_copy /= 10;
                    }

                    let mut msg = core::array::from_fn(|i| {
                        u32::from_le_bytes([
                            msg[i * 4],
                            msg[i * 4 + 1],
                            msg[i * 4 + 2],
                            msg[i * 4 + 3],
                        ])
                    });

                    let hash = crate::blake3::compress8(
                        &message.prefix_state,
                        &mut msg,
                        0,
                        message.salt_residual_len as u32 + 9,
                        message.flags,
                    );

                    Some((nonce, hash))
                }
                CerberusMessage::Binary(message) => {
                    let mut msg = [0; 16];
                    msg[0] = message.first_word;
                    msg[1] = nonce as u32;
                    let hash = crate::blake3::compress8(
                        &message.midstate,
                        &msg,
                        0,
                        8,
                        crate::blake3::FLAG_CHUNK_END | crate::blake3::FLAG_ROOT,
                    );
                    Some((msg[1] as u64 | (msg[0] as u64) << 32, hash))
                }
            }
        } else {
            None
        }
    }
}

/// AVX2 Altcha SHA-256 solver
pub struct AltchaSha256Solver {
    pub(super) message: AltchaMessage,
    pub(super) attempted_nonces: u64,
    pub(super) limit: u64,
}

impl From<AltchaMessage> for AltchaSha256Solver {
    fn from(message: AltchaMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl AltchaSha256Solver {
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_nested_impl<const TYPE: u8, const KW: usize>(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])> {
        for counter in (0u32..).step_by(8) {
            if self.attempted_nonces >= self.limit {
                return None;
            }

            let mut blocks = [
                _mm256_set1_epi32(i32::from_be_bytes(
                    self.message.salt[0..4].try_into().unwrap(),
                )),
                _mm256_set1_epi32(i32::from_be_bytes(
                    self.message.salt[4..8].try_into().unwrap(),
                )),
                _mm256_set1_epi32(i32::from_be_bytes(
                    self.message.salt[8..12].try_into().unwrap(),
                )),
                _mm256_set1_epi32(i32::from_be_bytes(
                    self.message.salt[12..16].try_into().unwrap(),
                )),
                _mm256_set1_epi32(i32::from_be_bytes(
                    self.message.nonce[0..4].try_into().unwrap(),
                )),
                _mm256_set1_epi32(i32::from_be_bytes(
                    self.message.nonce[4..8].try_into().unwrap(),
                )),
                _mm256_set1_epi32(i32::from_be_bytes(
                    self.message.nonce[8..12].try_into().unwrap(),
                )),
                _mm256_set1_epi32(i32::from_be_bytes(
                    self.message.nonce[12..16].try_into().unwrap(),
                )),
                _mm256_xor_si256(
                    _mm256_set1_epi32(counter as _),
                    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                ),
                _mm256_set1_epi32(i32::from_be_bytes([0x80, 0, 0, 0])),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_set1_epi32((32 + 4) * 8),
            ];

            //core::arch::asm!("# LLVM-MCA-BEGIN altcha_nested_impl",);

            for r in 0..self.message.cost.get() {
                // key truncation if needed (generally not)
                if KW < 8 && r > 0 {
                    blocks[KW] = _mm256_set1_epi32(0x80);
                    for i in (KW + 1)..8 {
                        blocks[i] = _mm256_setzero_si256();
                    }
                }
                let mut state =
                    core::array::from_fn(|i| _mm256_set1_epi32(crate::sha256::IV[i] as _));
                crate::sha256::avx2::multiway_arx::<0>(&mut state, &mut blocks);

                // vpaddd blocks, state, dword bcst [] ; result goes in blocks[0..8], overwrite entire variable to defuse loop dependency
                blocks = [
                    _mm256_add_epi32(state[0], _mm256_set1_epi32(crate::sha256::IV[0] as _)),
                    _mm256_add_epi32(state[1], _mm256_set1_epi32(crate::sha256::IV[1] as _)),
                    _mm256_add_epi32(state[2], _mm256_set1_epi32(crate::sha256::IV[2] as _)),
                    _mm256_add_epi32(state[3], _mm256_set1_epi32(crate::sha256::IV[3] as _)),
                    _mm256_add_epi32(state[4], _mm256_set1_epi32(crate::sha256::IV[4] as _)),
                    _mm256_add_epi32(state[5], _mm256_set1_epi32(crate::sha256::IV[5] as _)),
                    _mm256_add_epi32(state[6], _mm256_set1_epi32(crate::sha256::IV[6] as _)),
                    _mm256_add_epi32(state[7], _mm256_set1_epi32(crate::sha256::IV[7] as _)),
                    _mm256_set1_epi32(i32::from_be_bytes([0x80, 0, 0, 0])),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_set1_epi32((KW * 4 * 8) as i32),
                ];
            }

            self.attempted_nonces += 8;
            if let Some(success_lane_idx) =
                find_sha256_target_lane::<TYPE>(blocks[0], blocks[1], target, mask)
            {
                crate::unlikely();

                let perm = _mm256_set1_epi32(success_lane_idx as _);
                for i in 0..8 {
                    blocks[i] = _mm256_permutevar8x32_epi32(blocks[i], perm);
                }
                return Some((
                    counter as u64 + success_lane_idx as u64,
                    core::array::from_fn(|i| {
                        _mm_extract_epi32(_mm256_castsi256_si128(blocks[i]), 0) as u32
                    }),
                ));
            }

            //core::arch::asm!("# LLVM-MCA-END altcha_nested_impl",);
        }
        None
    }

    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_pbkdf2_impl<const TYPE: u8>(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])> {
        for counter in (0u32..).step_by(8) {
            if self.attempted_nonces >= self.limit {
                return None;
            }

            // first compute HMAC midstate
            let hmac_state = {
                let mut midstate_ipad =
                    core::array::from_fn(|i| _mm256_set1_epi32(crate::sha256::IV[i] as _));
                let mut blocks =
                    core::array::from_fn(|_| _mm256_set1_epi32(crate::solver::HMAC_IPAD32 as _));
                for i in 0..4 {
                    blocks[i] = _mm256_xor_si256(
                        blocks[i],
                        _mm256_set1_epi32(i32::from_be_bytes(
                            self.message.nonce[i * 4..][..4].try_into().unwrap(),
                        )),
                    );
                }
                blocks[4] = _mm256_xor_si256(
                    blocks[4],
                    _mm256_xor_si256(
                        _mm256_set1_epi32(counter as _),
                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                    ),
                );
                crate::sha256::avx2::multiway_arx::<0>(&mut midstate_ipad, &mut blocks);
                let mut midstate_opad =
                    core::array::from_fn(|i| _mm256_set1_epi32(crate::sha256::IV[i] as _));
                blocks =
                    core::array::from_fn(|_| _mm256_set1_epi32(crate::solver::HMAC_OPAD32 as _));
                for i in 0..4 {
                    blocks[i] = _mm256_xor_si256(
                        blocks[i],
                        _mm256_set1_epi32(i32::from_be_bytes(
                            self.message.nonce[i * 4..][..4].try_into().unwrap(),
                        )),
                    );
                }
                blocks[4] = _mm256_xor_si256(
                    blocks[4],
                    _mm256_xor_si256(
                        _mm256_set1_epi32(counter as _),
                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                    ),
                );

                crate::sha256::avx2::multiway_arx::<0>(&mut midstate_opad, &mut blocks);

                for i in 0..8 {
                    midstate_ipad[i] = _mm256_add_epi32(
                        midstate_ipad[i],
                        _mm256_set1_epi32(crate::sha256::IV[i] as _),
                    );
                    midstate_opad[i] = _mm256_add_epi32(
                        midstate_opad[i],
                        _mm256_set1_epi32(crate::sha256::IV[i] as _),
                    );
                }

                [midstate_ipad, midstate_opad]
            };

            // load salt
            let mut blocks = [
                _mm256_set1_epi32(
                    i32::from_be_bytes(self.message.salt[0..4].try_into().unwrap()) as _,
                ),
                _mm256_set1_epi32(
                    i32::from_be_bytes(self.message.salt[4..8].try_into().unwrap()) as _,
                ),
                _mm256_set1_epi32(
                    i32::from_be_bytes(self.message.salt[8..12].try_into().unwrap()) as _,
                ),
                _mm256_set1_epi32(
                    i32::from_be_bytes(self.message.salt[12..16].try_into().unwrap()) as _,
                ),
                _mm256_set1_epi32(1), // pbkdf counter
                _mm256_set1_epi32(i32::from_be_bytes([0x80, 0, 0, 0]) as _),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_set1_epi32(512 + ((16 + 4) * 8) as i32),
            ];

            let mut result: [__m256i; 8] = core::array::from_fn(|_| _mm256_setzero_si256());

            //core::arch::asm!("# LLVM-MCA-BEGIN altcha_pbkdf2_impl",);

            for r in 0..(2 * self.message.cost.get()) {
                let midstate = &hmac_state[(r % 2) as usize];

                let mut state = *midstate;

                crate::sha256::avx2::multiway_arx::<0>(&mut state, &mut blocks);

                blocks = [
                    _mm256_add_epi32(state[0], midstate[0]),
                    _mm256_add_epi32(state[1], midstate[1]),
                    _mm256_add_epi32(state[2], midstate[2]),
                    _mm256_add_epi32(state[3], midstate[3]),
                    _mm256_add_epi32(state[4], midstate[4]),
                    _mm256_add_epi32(state[5], midstate[5]),
                    _mm256_add_epi32(state[6], midstate[6]),
                    _mm256_add_epi32(state[7], midstate[7]),
                    _mm256_set1_epi32(i32::from_be_bytes([0x80, 0, 0, 0]) as _),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_setzero_si256(),
                    _mm256_set1_epi32(512 + 256),
                ];

                if r % 2 == 1 {
                    for i in 0..8 {
                        result[i] = _mm256_xor_si256(result[i], blocks[i]);
                    }
                }
            }

            self.attempted_nonces += 8;
            if let Some(success_lane_idx) =
                find_sha256_target_lane::<TYPE>(result[0], result[1], target, mask)
            {
                crate::unlikely();
                let perm = _mm256_set1_epi32(success_lane_idx as _);
                for i in 0..8 {
                    result[i] = _mm256_permutevar8x32_epi32(result[i], perm);
                }
                return Some((
                    counter as u64 + success_lane_idx as u64,
                    core::array::from_fn(|i| {
                        _mm_extract_epi32(_mm256_castsi256_si128(result[i]), 0) as u32
                    }),
                ));
            }

            //core::arch::asm!("# LLVM-MCA-END altcha_pbkdf2_impl",);
        }

        None
    }

    fn solve_fallback<const TYPE: u8>(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])> {
        let mut fallback = crate::solver::safe::AltchaSha256Solver::from(self.message.clone());
        crate::solver::Solver::set_limit(
            &mut fallback,
            self.limit.saturating_sub(self.attempted_nonces),
        );
        let result = crate::solver::Solver::solve::<TYPE>(&mut fallback, target, mask);
        self.attempted_nonces = self
            .attempted_nonces
            .saturating_add(crate::solver::Solver::get_attempted_nonces(&fallback));
        result
    }
}

impl crate::solver::Solver for AltchaSha256Solver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if !<RequiredFeatures as crate::solver::CpuIDToken>::get() {
            return self.solve_fallback::<TYPE>(target, mask);
        }
        if self.message.pbkdf2 {
            // SAFETY: <RequiredFeatures as crate::solver::CpuIDToken>::get() above proves AVX2 is available.
            unsafe { self.solve_pbkdf2_impl::<TYPE>(target, mask) }
        } else {
            // SAFETY: <RequiredFeatures as crate::solver::CpuIDToken>::get() above proves AVX2 is available.
            unsafe {
                match self.message.key_length.get().min(32) {
                    4 => self.solve_nested_impl::<TYPE, 1>(target, mask),
                    8 => self.solve_nested_impl::<TYPE, 2>(target, mask),
                    12 => self.solve_nested_impl::<TYPE, 3>(target, mask),
                    16 => self.solve_nested_impl::<TYPE, 4>(target, mask),
                    20 => self.solve_nested_impl::<TYPE, 5>(target, mask),
                    24 => self.solve_nested_impl::<TYPE, 6>(target, mask),
                    28 => self.solve_nested_impl::<TYPE, 7>(target, mask),
                    32 => self.solve_nested_impl::<TYPE, 8>(target, mask),
                    _ => return self.solve_fallback::<TYPE>(target, mask),
                }
            }
        }
    }
}

#[cfg(target_feature = "avx2")]
#[cfg(test)]
mod tests {
    use crate::message::{CerberusBinaryMessage, CerberusDecimalMessage};
    use crate::solver::Solver as _;

    use super::*;
    #[test]
    fn test_permute_lane_extraction_order() {
        let values = unsafe { _mm256_setr_epi32(10, 11, 12, 13, 14, 15, 16, 17) };
        for lane in 0..8 {
            let index = unsafe { _mm256_set1_epi32(lane) };
            let selected = unsafe { _mm256_permutevar8x32_epi32(values, index) };
            assert_eq!(unsafe { _mm256_extract_epi32::<0>(selected) }, 10 + lane);
        }
    }

    #[test]
    fn test_decimal_lane_prefix_mapping() {
        for prefix_set_index in 0..11 {
            for lane in 0..8 {
                let table_index = prefix_set_index * 8 + lane;
                let expected = u64::from(LANE_ID_MSB_STR[table_index] - u32::from(b'0')) * 10
                    + u64::from(LANE_ID_LSB_STR[table_index] - u32::from(b'0'));
                assert_eq!(
                    (10 + 8 * prefix_set_index + lane) as u64,
                    expected,
                    "prefix set {prefix_set_index}, lane {lane}"
                );
            }
        }
    }

    #[test]
    fn test_single_block_attempt_accounting() {
        let message = SingleBlockMessage::new(&[b'a'; 64], 0).expect("single block");
        let mut solver = SingleBlockSolver::from(message);
        solver.set_limit(8);
        assert!(
            solver
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(solver.get_attempted_nonces(), 8);
    }

    #[test]
    fn test_vector_tail_attempt_limits() {
        fn check<S: crate::solver::Solver<Output = [u32; 8]>>(mut solver: S) {
            solver.set_limit(1);
            assert!(
                solver
                    .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                    .is_none()
            );
            assert_eq!(solver.get_attempted_nonces(), 1);

            solver.set_limit(9);
            assert!(
                solver
                    .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                    .is_none()
            );
            assert_eq!(solver.get_attempted_nonces(), 9);
        }

        let salt = [b'a'; 32];
        check(BinarySolver::from(BinaryMessage::new(
            &salt,
            4.try_into().unwrap(),
        )));
        check(GoAwaySolver::from(GoAwayMessage::new_bytes(&salt, 0)));
    }

    #[cfg(feature = "compare-64bit")]
    #[test]
    fn test_full64_lt_comparisons() {
        const TARGET: u64 = 1u64 << 63;
        fn check<S: crate::solver::Solver<Output = [u32; 8]>>(mut solver: S) {
            solver.set_limit(1_000_000);
            let (_, hash) = solver
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(TARGET, u64::MAX)
                .expect("full-width LT solver should find an easy target");
            let value = (u64::from(hash[0]) << 32) | u64::from(hash[1]);
            assert!(value < TARGET, "{value:#018x} is not below {TARGET:#018x}");
        }

        let salt = [b'a'; 64];
        check(SingleBlockSolver::from(
            SingleBlockMessage::new(&salt, 0).expect("single block"),
        ));
        let double = (0..64)
            .find_map(|len| DoubleBlockMessage::new(&salt[..len], 0))
            .expect("double block shape");
        check(DoubleBlockSolver::from(double));
        check(BinarySolver::from(BinaryMessage::new(
            &salt,
            4.try_into().unwrap(),
        )));
        check(GoAwaySolver::from(GoAwayMessage::new_bytes(
            (&salt[..32]).try_into().unwrap(),
            0,
        )));
    }

    #[test]
    fn test_solve_decimal() {
        crate::solver::tests::test_decimal_validator::<DecimalSolver, _>(|prefix, search_space| {
            if let Some(solver) = SingleBlockMessage::new(prefix, search_space).map(Into::into) {
                Some(DecimalSolver::SingleBlock(solver))
            } else {
                DoubleBlockMessage::new(prefix, search_space).map(Into::into)
            }
        });
    }

    #[test]
    fn test_solve_cerberus_decimal() {
        for i in 0..=1 {
            crate::solver::tests::test_cerberus_decimal_validator::<CerberusSolver, _>(|prefix| {
                Some(CerberusMessage::Decimal(CerberusDecimalMessage::new(prefix, i)?).into())
            });
        }
    }

    #[test]
    fn test_cerberus_decimal_exact_limit() {
        let message = CerberusDecimalMessage::new(b"cerberus-decimal-benchmark|", 0)
            .expect("valid Cerberus decimal message");
        let mut solver = CerberusSolver::from(CerberusMessage::Decimal(message));
        solver.set_limit(17);
        assert!(
            solver
                .solve::<{ crate::solver::SOLVE_TYPE_MASK }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(solver.get_attempted_nonces(), 17);
    }

    #[test]
    fn test_solve_cerberus_binary() {
        for i in 0..=1 {
            crate::solver::tests::test_cerberus_binary_validator::<CerberusSolver, _>(|prefix| {
                Some(CerberusMessage::Binary(CerberusBinaryMessage::new(prefix, i)).into())
            });
        }
    }
    #[test]
    fn test_solve_binary() {
        crate::solver::tests::test_binary_validator::<BinarySolver, _>(
            |prefix, nonce_byte_count| {
                BinarySolver::from(BinaryMessage::new(prefix, nonce_byte_count))
            },
        )
    }

    #[test]
    fn test_solve_goaway() {
        crate::solver::tests::test_goaway_validator::<GoAwaySolver, _>(|prefix| {
            GoAwaySolver::from(GoAwayMessage::new(
                core::array::from_fn(|i| {
                    u32::from_be_bytes([
                        prefix[i * 4],
                        prefix[i * 4 + 1],
                        prefix[i * 4 + 2],
                        prefix[i * 4 + 3],
                    ])
                }),
                0,
            ))
        });
    }

    #[test]
    fn test_solve_altcha() {
        crate::solver::tests::test_altcha_validator::<AltchaSha256Solver, _>(|message| {
            Some(AltchaSha256Solver::from(message))
        });
    }
}
