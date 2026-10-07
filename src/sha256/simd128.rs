//! Multi-way SHA-256 wrapper using the shared Portable-SIMD core.
use core::arch::wasm32::*;
use core::simd::u32x4;

use super::portable;

#[inline(always)]
pub(crate) fn multiway_arx<const BEGIN_ROUND: usize>(
    state: &mut [v128; 8],
    block: &mut [v128; 16],
) {
    let mut state_simd: [u32x4; 8] = core::array::from_fn(|i| state[i].into());
    let mut block_simd: [u32x4; 16] = core::array::from_fn(|i| block[i].into());
    portable::multiway_arx::<4, BEGIN_ROUND>(&mut state_simd, &mut block_simd);
    for i in 0..8 {
        state[i] = state_simd[i].into();
    }
    for i in 0..16 {
        block[i] = block_simd[i].into();
    }
}

#[inline(always)]
pub(crate) fn bcst_multiway_arx<const LEADING_ZEROES: usize>(
    state: &mut [v128; 8],
    w_k: &[u32; 64],
) {
    let mut state_simd: [u32x4; 8] = core::array::from_fn(|i| state[i].into());
    portable::bcst_multiway_arx::<4, LEADING_ZEROES>(&mut state_simd, w_k);
    for i in 0..8 {
        state[i] = state_simd[i].into();
    }
}

#[cfg(test)]
#[inline(always)]
fn u32x4_ror(x: v128, shift: u32) -> v128 {
    let v: u32x4 = x.into();
    ((v >> shift) | (v << (32 - shift))).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::IV;

    #[test]
    fn test_simd128_ror() {
        unsafe {
            for amount in 0..32 {
                let input = [0x12345678, 0x9abcdef0, 0x0c0c0c0c, 0xffffeeee];
                let x = u32x4(input[0], input[1], input[2], input[3]);
                let y = u32x4_ror(x, amount);
                let mut ys = [0u32; 4];
                v128_store(ys.as_mut_ptr().cast(), y);
                let expected = core::array::from_fn(|i| input[i].rotate_right(amount));
                assert_eq!(
                    ys, expected,
                    "amount: {}, x: {:08x?}, y: {:08x?}",
                    amount, input, y
                );
            }
        }
    }

    #[test]
    fn test_sha256_simd128_single_block() {
        // Test vector from NIST FIPS 180-4
        // Input: "abc" repeated 16 times
        let input_block = [
            0x61626380, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000018,
        ];

        // Create 16 identical blocks for SIMD128 processing
        let mut block_simd128: [v128; 16] = core::array::from_fn(|i| u32x4_splat(input_block[i]));
        let state_save: [v128; 8] = core::array::from_fn(|i| u32x4_splat(IV[i]));

        // Process the blocks
        let mut state = state_save;
        multiway_arx::<0>(&mut state, &mut block_simd128);
        for i in 0..8 {
            state[i] = u32x4_add(state_save[i], state[i]);
        }

        // Expected output hash for "abc"
        let expected = [
            0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
            0xf20015ad,
        ];

        let mut results: [[u32; 4]; 8] = unsafe { core::mem::zeroed() };
        for i in 0..8 {
            unsafe {
                v128_store(results[i].as_mut_ptr().cast(), state[i]);
            }
        }

        // Verify all 4 results match the expected hash
        for i in 0..4 {
            let result = [
                results[0][i],
                results[1][i],
                results[2][i],
                results[3][i],
                results[4][i],
                results[5][i],
                results[6][i],
                results[7][i],
            ];
            assert_eq!(
                result, expected,
                "SHA-256 SIMD128 hash mismatch at index {}",
                i
            );
        }
    }
}
