# LLMetal

LLMetal is an experimental Rust runtime for visible LLM inference on Apple Metal.

The point is not to hide the model behind a polished serving stack. The point is to keep the inference path close enough to the surface that you can point at every major operation: GGUF loading, token flow, quantized matmul, RMSNorm, RoPE, KV cache, attention, SwiGLU, logits, sampling.

This is not meant as a tutorial project or a toy pet repo. It is a place to try current inference ideas from industry and research while keeping the machinery readable enough to change quickly. Some of those ideas will already exist in production runtimes; others may be fresh enough that projects like mistral.rs or llama.cpp have not picked them up yet.

The long-term target is a fast local inference runtime that can eventually be compared with serious baselines such as llama.cpp. The current target is smaller and more honest: keep the repository readable enough to discuss, experiment, and replace pieces one at a time.

## Shape

```text
src/
  main.rs          small CLI entrypoint
  gguf_loader.rs   GGUF metadata loading and architecture summary
  inference.rs     deliberately exposed inference trace
  metal.rs         Metal device boundary
  tokenizer.rs     tokenizer boundary, not a fake tokenizer

docs/
  inference-path.md        readable walkthrough of the transformer path
  notes/                   older sketches and GPU notes kept out of the root
```

## Commands

```bash
cargo run -- inspect <model.gguf>
cargo run -- trace <model.gguf> "your prompt"
```

`inspect` prints the model family, tensor count, file type, and architecture values pulled from GGUF metadata.

`trace` prints the intended transparent inference path for a prompt. It is a scaffold for the runtime, not a claim that generation is implemented.

## Design Bias

LLMetal should stay boring in the right places:

- one obvious path through inference
- minimal configuration surface
- no generic framework of maps, registries, or runtime wiring until it pays for itself
- Metal as the hardware target
- Rust as the implementation language
- room to test new inference techniques without waiting for a larger runtime to absorb them
- comments and docs that explain what is happening without pretending the code is finished

The style is closer to an annotated engine bay than a library API. If a future optimization makes the path faster but opaque, it should earn that opacity.

## Current Status

The runtime loads and runs real GGUF models (Devstral Small 22B Q8_0 verified). The inference path is complete: GGUF mmap, Q8_0 dequant, tokenizer, KV cache, RoPE, GQA attention, SwiGLU FFN, logit sampling with repetition penalty. The GPU kernel achieves 57 GB/s effective bandwidth (84% of M1 base peak) in isolation.

End-to-end decode speed is well below llama.cpp. Three reasons, ranked by impact:

**1. 280 serial GPU round-trips per token.** Each of the 280 matmuls (40 layers × 7 weights) gets its own command buffer: encode → commit → `waitUntilCompleted`. That last call blocks the CPU until the GPU finishes. Then the CPU does a small amount of work (RMSNorm, RoPE, attention) and submits the next job. The GPU sits idle during all that CPU work. The GPU is being fed one small job at a time instead of a continuous stream.

**2. Temporary Metal buffer allocation per dispatch.** `buf_zeros(n)` allocates a new Metal buffer for every matmul output — that is a kernel trap into the IOKit GPU subsystem per call, plus teardown when it is dropped. This happens 280 times per token.

**3. The kernel does not use Apple's hardware matrix units.** The kernel does scalar float arithmetic in a loop. Apple Silicon has dedicated `simdgroup_matrix_multiply` instructions (8×8 hardware tiles) that llama.cpp exploits. These get far higher throughput per clock than scalar ops.

The core problem is #1. The isolated GPU benchmark (synthetic buffers, no CPU round-trips) measured 57 GB/s — 84% of M1 peak. The kernel itself is fine once it is running. The GPU is just idle most of the time waiting for the CPU to issue the next job. Batching a full layer into one command buffer and keeping the GPU fed improves speed dramatically before touching the kernel at all.

Old CUDA/YALM notes and early inference sketches live in `docs/notes/`. They are references, not active implementation.
