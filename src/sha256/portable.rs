//! Portable-SIMD SHA-256 compression core shared by architecture wrappers.
use super::K32;
use core::simd::Simd;
#[macro_use]
#[path = "loop_macros.rs"]
mod loop_macros;
#[inline(always)]
fn ror<const N: usize>(x: Simd<u32, N>, s: u32) -> Simd<u32, N> {
    (x >> s) | (x << (32 - s))
}
#[inline(always)]
fn ss0<const N: usize>(x: Simd<u32, N>) -> Simd<u32, N> {
    ror(x, 7) ^ ror(x, 18) ^ (x >> 3)
}
#[inline(always)]
fn ss1<const N: usize>(x: Simd<u32, N>) -> Simd<u32, N> {
    ror(x, 17) ^ ror(x, 19) ^ (x >> 10)
}
#[inline(always)]
fn bs0<const N: usize>(x: Simd<u32, N>) -> Simd<u32, N> {
    ror(x, 2) ^ ror(x, 13) ^ ror(x, 22)
}
#[inline(always)]
fn bs1<const N: usize>(x: Simd<u32, N>) -> Simd<u32, N> {
    ror(x, 6) ^ ror(x, 11) ^ ror(x, 25)
}
#[inline(always)]
fn ch<const N: usize>(x: Simd<u32, N>, y: Simd<u32, N>, z: Simd<u32, N>) -> Simd<u32, N> {
    (x & y) ^ ((!x) & z)
}
#[inline(always)]
fn maj<const N: usize>(x: Simd<u32, N>, y: Simd<u32, N>, z: Simd<u32, N>) -> Simd<u32, N> {
    (x & y) ^ (x & z) ^ (y & z)
}
#[inline(always)]
pub(super) fn multiway_arx<const N: usize, const BEGIN: usize>(
    state: &mut [Simd<u32, N>; 8],
    block: &mut [Simd<u32, N>; 16],
) {
    let [a, b, c, d, e, f, g, h] = &mut *state;
    repeat64!(i, {
        if i >= BEGIN {
            let w = if i < 16 {
                block[i]
            } else {
                let x = ss1(block[(i - 2) % 16])
                    + block[(i - 7) % 16]
                    + ss0(block[(i - 15) % 16])
                    + block[i % 16];
                block[i % 16] = x;
                x
            };
            let t1 = *h + bs1(*e) + ch(*e, *f, *g) + Simd::splat(K32[i]) + w;
            let t2 = bs0(*a) + maj(*a, *b, *c);
            *h = *g;
            *g = *f;
            *f = *e;
            *e = *d + t1;
            *d = *c;
            *c = *b;
            *b = *a;
            *a = t1 + t2;
        }
    });
}
#[inline(always)]
pub(super) fn bcst_multiway_arx<const N: usize, const LEAD: usize>(
    state: &mut [Simd<u32, N>; 8],
    wk: &[u32; 64],
) {
    let [a, b, c, d, e, f, g, h] = &mut *state;
    repeat64!(i, {
        let w = Simd::splat(if i < LEAD { K32[i] } else { wk[i] });
        let t1 = *h + bs1(*e) + ch(*e, *f, *g) + w;
        let t2 = bs0(*a) + maj(*a, *b, *c);
        *h = *g;
        *g = *f;
        *f = *e;
        *e = *d + t1;
        *d = *c;
        *c = *b;
        *b = *a;
        *a = t1 + t2;
    });
}
