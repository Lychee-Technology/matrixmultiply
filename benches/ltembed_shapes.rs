// benches/ltembed_shapes.rs — LTEmbed-shaped GEMM microbenchmarks.
//
// Shapes derived from the e5-small-v2 BERT forward pass as implemented in
// LTEmbed (src/models/bert.rs). All calls use matrixmultiply::sgemm with the
// exact strides that LTEmbed produces at runtime.
//
// Model dimensions (e5-small-v2):
//   hidden        = 384
//   intermediate  = 1536   (FFN expansion factor 4×)
//   num_heads     = 12
//   head_dim      = 32     (hidden / num_heads)
//
// Representative sequence lengths (batch = 1):
//   T =   8  (~"query: Hello, world!")
//   T =  25  (~medium query)
//   T = 128  (~long passage, "quick brown fox" × 30)
//   T = 512  (max_position_embeddings — edge case)
//
// GEMM hotspots benchmarked here:
//
//   proj_384_384   C[T,384]  = A[T,384]  @ B[384,384]   (Q/K/V/out projections)
//   ffn_up         C[T,1536] = A[T,384]  @ B[384,1536]  (FFN up-projection)
//   ffn_down       C[T,384]  = A[T,1536] @ B[1536,384]  (FFN down-projection)
//   attn_qk        C[T,T]    = A[T,32]   @ K^T[32,T]    (per-head attention scores;
//                                                         B uses column-major strides
//                                                         rs=1, cs=384)
//   attn_sv        C[T,32]   = A[T,T]    @ B[T,32]      (per-head weighted value sum;
//                                                         B uses row-stride=384 into
//                                                         the full value tensor)
//
// Run with:
//   cargo bench --bench ltembed_shapes
//
// For ARM64 (AWS Graviton):
//   cargo bench --bench ltembed_shapes --target aarch64-unknown-linux-gnu

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use matrixmultiply::sgemm;

const HIDDEN: usize = 384;
const INTER: usize = 1536;
const HEAD_DIM: usize = 32;
const SEQ_LENS: &[usize] = &[8, 25, 128, 512];

// ── proj_384_384 ─────────────────────────────────────────────────────────────
// C[T, 384] = A[T, 384] @ B[384, 384]
// All row-major. Called 4× per transformer layer (Q, K, V, output projections).
fn bench_proj_384_384(c: &mut Criterion) {
    let mut g = c.benchmark_group("proj_384_384");
    for &t in SEQ_LENS {
        let a = vec![0.5f32; t * HIDDEN];
        let b = vec![0.5f32; HIDDEN * HIDDEN];
        let mut out = vec![0.0f32; t * HIDDEN];
        g.throughput(Throughput::Elements((2 * t * HIDDEN * HIDDEN) as u64));
        g.bench_with_input(BenchmarkId::new("T", t), &t, |bench, &_| {
            bench.iter(|| unsafe {
                sgemm(
                    t,
                    HIDDEN,
                    HIDDEN,
                    1.0,
                    a.as_ptr(),
                    HIDDEN as isize,
                    1,
                    b.as_ptr(),
                    HIDDEN as isize,
                    1,
                    0.0,
                    out.as_mut_ptr(),
                    HIDDEN as isize,
                    1,
                )
            });
        });
    }
    g.finish();
}

// ── ffn_up ───────────────────────────────────────────────────────────────────
// C[T, 1536] = A[T, 384] @ B[384, 1536]
// All row-major. Called once per transformer layer.
fn bench_ffn_up(c: &mut Criterion) {
    let mut g = c.benchmark_group("ffn_up");
    for &t in SEQ_LENS {
        let a = vec![0.5f32; t * HIDDEN];
        let b = vec![0.5f32; HIDDEN * INTER];
        let mut out = vec![0.0f32; t * INTER];
        g.throughput(Throughput::Elements((2 * t * HIDDEN * INTER) as u64));
        g.bench_with_input(BenchmarkId::new("T", t), &t, |bench, &_| {
            bench.iter(|| unsafe {
                sgemm(
                    t,
                    HIDDEN,
                    INTER,
                    1.0,
                    a.as_ptr(),
                    HIDDEN as isize,
                    1,
                    b.as_ptr(),
                    INTER as isize,
                    1,
                    0.0,
                    out.as_mut_ptr(),
                    INTER as isize,
                    1,
                )
            });
        });
    }
    g.finish();
}

// ── ffn_down ─────────────────────────────────────────────────────────────────
// C[T, 384] = A[T, 1536] @ B[1536, 384]
// All row-major. Called once per transformer layer.
fn bench_ffn_down(c: &mut Criterion) {
    let mut g = c.benchmark_group("ffn_down");
    for &t in SEQ_LENS {
        let a = vec![0.5f32; t * INTER];
        let b = vec![0.5f32; INTER * HIDDEN];
        let mut out = vec![0.0f32; t * HIDDEN];
        g.throughput(Throughput::Elements((2 * t * INTER * HIDDEN) as u64));
        g.bench_with_input(BenchmarkId::new("T", t), &t, |bench, &_| {
            bench.iter(|| unsafe {
                sgemm(
                    t,
                    INTER,
                    HIDDEN,
                    1.0,
                    a.as_ptr(),
                    INTER as isize,
                    1,
                    b.as_ptr(),
                    HIDDEN as isize,
                    1,
                    0.0,
                    out.as_mut_ptr(),
                    HIDDEN as isize,
                    1,
                )
            });
        });
    }
    g.finish();
}

// ── attn_qk ──────────────────────────────────────────────────────────────────
// C[T, T] = A[T, 32] @ K^T[32, T]
//
// This is one head's attention score computation. In LTEmbed, Q and K are
// stored as [T, hidden=384] row-major, and each head accesses an interleaved
// slice with stride (rs=hidden, cs=1). K is accessed in transposed order:
//   B strides: rsb=1, csb=hidden=384   (column-major within the head slice)
//
// This non-standard B stride is the distinguishing feature of this shape —
// it exercises the packing path for column-major B.
fn bench_attn_qk(c: &mut Criterion) {
    let mut g = c.benchmark_group("attn_qk");
    for &t in SEQ_LENS {
        // Allocate full Q/K tensors as LTEmbed does: [T, hidden]
        let q = vec![0.5f32; t * HIDDEN];
        let k = vec![0.5f32; t * HIDDEN];
        let mut scores = vec![0.0f32; t * t];
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        g.throughput(Throughput::Elements((2 * t * HEAD_DIM * t) as u64));
        g.bench_with_input(BenchmarkId::new("T", t), &t, |bench, &_| {
            bench.iter(|| unsafe {
                // Benchmark head 0 (pointer offset = 0 * HEAD_DIM = 0)
                sgemm(
                    t,
                    HEAD_DIM,
                    t,
                    scale,
                    q.as_ptr(),              // Q head slice: rs=hidden, cs=1
                    HIDDEN as isize,
                    1,
                    k.as_ptr(),              // K^T: rs=1, cs=hidden (column-major)
                    1,
                    HIDDEN as isize,
                    0.0,
                    scores.as_mut_ptr(),
                    t as isize,
                    1,
                )
            });
        });
    }
    g.finish();
}

// ── attn_sv ──────────────────────────────────────────────────────────────────
// C[T, 32] = A[T, T] @ B[T, 32]
//
// One head's weighted value summation. scores[T,T] is row-major; V is accessed
// as an interleaved slice of [T, hidden=384] with rs=hidden, cs=1. The output
// is also interleaved: rs=hidden, cs=1.
fn bench_attn_sv(c: &mut Criterion) {
    let mut g = c.benchmark_group("attn_sv");
    for &t in SEQ_LENS {
        let scores = vec![0.5f32; t * t];
        let v = vec![0.5f32; t * HIDDEN]; // V: [T, hidden]
        let mut out = vec![0.0f32; t * HIDDEN]; // output: [T, hidden]
        g.throughput(Throughput::Elements((2 * t * t * HEAD_DIM) as u64));
        g.bench_with_input(BenchmarkId::new("T", t), &t, |bench, &_| {
            bench.iter(|| unsafe {
                // Benchmark head 0
                sgemm(
                    t,
                    t,
                    HEAD_DIM,
                    1.0,
                    scores.as_ptr(),
                    t as isize,
                    1,
                    v.as_ptr(),              // V head slice: rs=hidden, cs=1
                    HIDDEN as isize,
                    1,
                    0.0,
                    out.as_mut_ptr(),        // out head slice: rs=hidden, cs=1
                    HIDDEN as isize,
                    1,
                )
            });
        });
    }
    g.finish();
}

criterion_group!(
    ltembed,
    bench_proj_384_384,
    bench_ffn_up,
    bench_ffn_down,
    bench_attn_qk,
    bench_attn_sv,
);
criterion_main!(ltembed);
