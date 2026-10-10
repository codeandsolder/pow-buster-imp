#!/bin/sh

set -e

RUSTFLAGS='-Ctarget-feature=+simd128' wasm-pack build --no-opt --target web -d pkg --no-default-features --features json

for file in pkg/*.wasm; do
  optimized="${file%.wasm}.opt.wasm"
  wasm-opt "$file" -o "$optimized" -O4 --enable-bulk-memory --enable-simd
  mv "$optimized" "$file"
  gzip -9knf "$file"
done
