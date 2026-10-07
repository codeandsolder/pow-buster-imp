//! Portable-SIMD BLAKE3 compression core shared by architecture wrappers.
use super::MESSAGE_SCHEDULE;
use core::simd::Simd;
#[macro_use]
#[path = "loop_macros.rs"]
mod loop_macros;
#[inline(always)]
fn ror<const N: usize>(x: Simd<u32, N>, s: u32) -> Simd<u32, N> {
    (x >> s) | (x << (32 - s))
}
#[inline(always)]
pub(super) fn g<const N: usize>(
    a: &mut Simd<u32, N>,
    b: &mut Simd<u32, N>,
    c: &mut Simd<u32, N>,
    d: &mut Simd<u32, N>,
    x: Simd<u32, N>,
    y: Simd<u32, N>,
) {
    *a = *a + *b + x;
    *d = ror(*d ^ *a, 16);
    *c = *c + *d;
    *b = ror(*b ^ *c, 12);
    *a = *a + *b + y;
    *d = ror(*d ^ *a, 8);
    *c = *c + *d;
    *b = ror(*b ^ *c, 7);
}
#[inline(always)]
pub(super) fn compress<const N: usize, const CONSTANT_WORD_COUNT: usize, const PATCH: usize>(
    v: &mut [Simd<u32, N>; 16],
    tpl: &[u32; 16],
    patch: Simd<u32, N>,
) {
    repeat7!(round, {
        macro_rules! mix {
            ($a:literal,$b:literal,$c:literal,$d:literal,$x:literal,$y:literal) => {{
                let ix = MESSAGE_SCHEDULE[round][$x];
                let iy = MESSAGE_SCHEDULE[round][$y];
                let x = if ix == PATCH {
                    patch
                } else {
                    Simd::splat(tpl[ix])
                };
                let y = if iy == PATCH {
                    patch
                } else {
                    Simd::splat(tpl[iy])
                };
                let [a, b, c, d] = v
                    .get_disjoint_mut([$a, $b, $c, $d])
                    .expect("BLAKE3 G indices are disjoint");
                g(a, b, c, d, x, y);
            }};
        }
        if round > 0 || CONSTANT_WORD_COUNT < 2 {
            mix!(0, 4, 8, 12, 0, 1);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 4 {
            mix!(1, 5, 9, 13, 2, 3);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 6 {
            mix!(2, 6, 10, 14, 4, 5);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 8 {
            mix!(3, 7, 11, 15, 6, 7);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 10 {
            mix!(0, 5, 10, 15, 8, 9);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 12 {
            mix!(1, 6, 11, 12, 10, 11);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 14 {
            mix!(2, 7, 8, 13, 12, 13);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 16 {
            mix!(3, 4, 9, 14, 14, 15);
        }
    });
    repeat8!(i, {
        v[i] = v[i] ^ v[i + 8];
    });
}
