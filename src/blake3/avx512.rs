use super::portable;
use core::arch::x86_64::*;
use core::simd::u32x16;

#[inline(always)]
pub(crate) fn compress_mb16<const CONSTANT_WORD_COUNT: usize, const PATCH_1: usize>(
    v: &mut [__m512i; 16],
    block_template: &[u32; 16],
    patch_1: __m512i,
) {
    let mut state: [u32x16; 16] = core::array::from_fn(|i| v[i].into());
    let patch: u32x16 = patch_1.into();
    portable::compress::<16, CONSTANT_WORD_COUNT, PATCH_1>(&mut state, block_template, patch);
    for i in 0..16 {
        v[i] = state[i].into();
    }
}

#[cfg(all(test, target_feature = "avx512f"))]
#[inline(always)]
fn g4(a: &mut __m512i, b: &mut __m512i, c: &mut __m512i, d: &mut __m512i, x: __m512i, y: __m512i) {
    let mut aa: u32x16 = (*a).into();
    let mut bb: u32x16 = (*b).into();
    let mut cc: u32x16 = (*c).into();
    let mut dd: u32x16 = (*d).into();
    portable::g(&mut aa, &mut bb, &mut cc, &mut dd, x.into(), y.into());
    *a = aa.into();
    *b = bb.into();
    *c = cc.into();
    *d = dd.into();
}

#[cfg(target_feature = "avx512f")]
#[cfg(test)]
mod tests {
    use blake3::Hasher;

    use super::*;
    use crate::blake3::g;

    #[test]
    fn test_g_function() {
        let mut state = core::array::from_fn(|i| crate::sha256::IV[i % 8].wrapping_add(i as u32));
        let mut state_v: [_; 16] =
            core::array::from_fn(|i| unsafe { _mm512_set1_epi32(state[i] as _) });
        g(
            &mut state,
            0,
            4,
            8,
            12,
            crate::sha256::IV[0],
            crate::sha256::IV[1],
        );
        let [va, vb, vc, vd] = state_v.get_disjoint_mut([0, 4, 8, 12]).unwrap();
        g4(
            va,
            vb,
            vc,
            vd,
            unsafe { _mm512_set1_epi32(crate::sha256::IV[0] as _) },
            unsafe { _mm512_set1_epi32(crate::sha256::IV[1] as _) },
        );

        for i in 0..16 {
            assert_eq!(
                unsafe { _mm_extract_epi32(_mm512_castsi512_si128(state_v[i]), 0) as u32 },
                state[i],
                "word {}: expected: {:08x}, results: {:08x}",
                i,
                state[i],
                unsafe { _mm_extract_epi32(_mm512_castsi512_si128(state_v[i]), 0) as u32 }
            );
        }
    }

    #[test]
    fn test_compress_mb16() {
        let mut v = [0u32; 16];
        v[..8].copy_from_slice(&crate::blake3::IV);
        v[8..12].copy_from_slice(&crate::blake3::IV[..4]);
        v[12] = 0;
        v[13] = 0;
        v[14] = 4;
        v[15] = 0x0b;
        let mut v = core::array::from_fn(|i| unsafe { _mm512_set1_epi32(v[i] as _) });
        let mut block = [0u32; 16];
        block[0] = u32::from_le_bytes(*b"IETF");
        compress_mb16::<0, 16>(&mut v, &block, unsafe { _mm512_setzero_epi32() });
        let expected = [
            0x1edea283, 0xabe6f4e6, 0x24896868, 0xcfc04e8f, 0x9470c54c, 0xff82a646, 0xd6b4cbd1,
            0xe2815116,
        ];
        let mut results = [0u32; 8];
        for i in 0..8 {
            unsafe {
                results[i] = _mm_extract_epi32(_mm512_castsi512_si128(v[i]), 0) as u32;
            }
        }
        assert_eq!(
            results, expected,
            "expected: {:08x?}, results: {:08x?}",
            expected, results
        );
        let mut hasher = Hasher::new();
        hasher.update(b"IETF");
        let hash = hasher.finalize();
        let hash = hash.as_bytes();
        let mut expected = [0u32; 8];
        for i in 0..8 {
            expected[i] = u32::from_le_bytes(hash[i * 4..i * 4 + 4].try_into().unwrap());
        }
        assert_eq!(results, expected);
    }
}
