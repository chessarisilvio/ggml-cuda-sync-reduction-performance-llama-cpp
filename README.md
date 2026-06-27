# GGML CUDA Sync Reduction for llama.cpp

This project applies CUDA synchronization reduction techniques from recent llama.cpp commits to improve inference performance on NVIDIA P40 (sm_61) and RTX 3050 GPUs. The focus is on minimizing GPU‑CPU synchronization points and introducing asynchronous CPU‑to‑CUDA memory copies.

## Problem
The stock llama.cpp CUDA backend contains several sources of overhead:
- Explicit `cudaDeviceSynchronize()` calls after each kernel launch.
- Synchronous `cudaMemcpy()` operations that stall the CPU while waiting for GPU completion.
- Underutilization of GPU compute pipelines due to frequent synchronization.
These issues are especially pronounced for Mixture‑of‑Experts (MoE) models such as Qwen3.6‑35B, where many small kernels are launched per token, leading to significant idle time on the CPU and poor GPU occupancy.

## Solution
We adopt the following strategies from upstream llama.cpp optimizations:
1. **Asynchronous Memory Transfers** – Replace blocking `cudaMemcpy` with `cudaMemcpyAsync` using dedicated CUDA streams, allowing the CPU to continue work while data moves to the GPU.
2. **Stream Synchronization Reduction** – Organize computation and transfers into separate streams, synchronizing only when host‑side results are required.
3. **CUDA Graphs (Optional)** – Capture repeated kernel launch patterns (e.g., the decode loop) into CUDA graphs to reduce launch overhead.
4. **Batched Operations** – Where possible, group multiple small copies or computes into fewer asynchronous calls to improve throughput.

## Implementation
The core changes reside in `src/ggml/ggml-cuda.cu`, which is a direct copy of the optimized file from the `llama-pr17400` branch. Key modifications include:
- Allocation of CUDA streams per backend context.
- Replacement of `cudaMemcpy` with `cudaMemcpyAsync` for model tensors and intermediate buffers.
- Insertion of `cudaStreamSynchronize` only before host‑side accesses (e.g., after a token generation step).
- Review of kernel launch wrappers to remove unnecessary `cudaDeviceSynchronize`.
- Placeholder hooks for CUDA graph capture, enabled via compile‑time flags.

The project builds with a standard CUDA 11+ toolkit. No changes to the build system are required; the existing Makefile/CMake configuration of llama.cpp works unchanged.

## Expected Impact
- Reduced CPU idle time waiting for GPU completion.
- Better utilization of GPU compute cores, particularly for memory‑bound workloads.
- Lower latency per token, especially at batch size = 1.
- Potential throughput increase of 10‑30% on MoE models where synchronization overhead dominates.
- Improved scalability when scaling to multiple GPU streams or pipelines.

## References
- llama.cpp pull requests and commit logs concerning CUDA sync reduction (post‑September 2024).
- NVIDIA CUDA Streams and Async API documentation.
- GGML backend integration guide.

## Usage
1. Clone the llama.cpp repository and apply the changes from `src/ggml/ggml-cuda.cu` to the corresponding file in your local copy.
2. Rebuild llama.cpp with CUDA support enabled.
3. Run your usual inference commands (e.g., `./main -m model.gguf -p "prompt"` ) and observe reduced latency.
4. For benchmarking, use the provided scripts in the `benchmark/` directory (if present) or run custom timing loops.

## License
This project is released under the MIT License. See the LICENSE file for details.