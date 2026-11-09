## What the numbers mean

specmaxx estimates the **theoretical maximum** speed of a model on a hardware stack: what the chips, memory and links allow, not what any inference engine achieves. Real engines land below it.

Every rate is quoted with the context it applies to, like **33.9 tok/s @ 8k**. Prefill @ P is a prompt of P tokens processed from an empty cache. Decode @ N is one new token generated after N tokens are already cached. A tok/s figure without its context length means nothing, because attention gets more expensive as the context grows.

More: [scope and conventions](simulation-model.md#1-scope-and-conventions), [context of each rate](simulation-model.md#pipeline-and-dp).

## From model to operations

The model is read from its Hugging Face `config.json` and turned into a list of **operations** with real shapes and number formats: matrix multiplies (projections, MLPs, the LM head), attention, norms, activations, embedding lookups, mixture-of-experts routing and, with several devices, communication. Layer types that are not plain attention (latent attention, sparse or sliding-window attention, linear-attention and state-space layers, n-gram tables) become the operations they physically need.

Each operation counts the arithmetic it needs (FLOPs) and the bytes it must read and write: each weight, cache entry and activation is touched once. The shapes depend on the batch size and on the context length of the phase being estimated.

More: [model IR](simulation-model.md#2-model-ir), [operation catalogue](simulation-model.md#3-op-catalogue).

## Hardware

A device is described by its peak arithmetic rate **per number format** (bf16, fp8, fp4, int8 and so on, with a separate sparse rate where the hardware has one), its memory bandwidth and capacity, and its links to other devices.

Peak rates are only reached by large, well-shaped matrices. Matrix units work on fixed tiles and the chip works in waves of tiles, so a small or awkward shape leaves part of the hardware idle. The estimate applies this tile and wave rounding to every matrix multiply, which is why single-token decode never reaches the headline TFLOPS.

Presets cite their vendor sources. Some values are marked as estimated where no vendor publishes them.

More: [hardware](simulation-model.md#4-hardware), [shape efficiency](simulation-model.md#shape-efficiency), [preset library](simulation-model.md#preset-library).

## How time is computed

Each operation takes the longer of two times (the **roofline**):

- **compute time**: its FLOPs divided by the achievable arithmetic rate, and
- **memory time**: its bytes divided by memory bandwidth.

If compute is longer the op is *compute bound*, otherwise *bandwidth bound*. The time of a step is the sum of its operations (on the busiest device). Communication is added using a latency-plus-bandwidth cost, and by default the bandwidth part may overlap with compute while the latency part may not.

More: [per-op time](simulation-model.md#per-op-time), [stage time and overlap](simulation-model.md#stage-time-and-overlap).

## Prefill and decode

**Prefill** processes the whole prompt at once. There is plenty of work per weight read, so it is usually compute bound. Its rate is prompt tokens divided by the time to process them.

**Decode** produces one token per sequence per step. Every weight and the whole KV cache must be read for very little arithmetic, so it is usually bandwidth bound, and it slows as the context grows. Its rate is one token per step time, times the batch size.

More: [pipeline and DP](simulation-model.md#pipeline-and-dp).

## Multiple devices

Work is split across devices three ways: **tensor parallel** (TP) shards each layer's matrices, **pipeline parallel** (PP) puts consecutive layers on consecutive devices, and **data parallel** (DP) runs independent replicas. Mixture-of-experts layers can also spread their experts (EP).

The estimate assumes a perfectly balanced split and costs one representative device per stage. It then adds the traffic the split requires: all-reduces for TP, a hand-off of activations between pipeline stages, and expert exchange for EP. A link has a bandwidth (per direction) and a latency, and a message costs latency plus size over bandwidth. Every network latency is measured the same way, half the round trip of a small-message ping-pong, so RDMA and TCP over the same cable compare fairly (RDMA is never slower).

A PCIe link carries data in packets, and each packet's header and checksum take part of the wire. So every PCIe bandwidth here (between GPUs, to host memory for offloading, to an SSD) is the line rate times 256/280 ≈ 0.91, the payload share of full 256-byte packets. Other links (NVLink, InfiniBand, Ethernet, Thunderbolt) use the vendor's data rate. More: [PCIe payload efficiency](simulation-model.md#links-and-topologies).

Between machines the network shape matters: a switch, a full mesh, a ring or a chain. Each shape gets its own routing and bandwidth rules. Where the hardware cannot pass data straight between devices, as with some consumer GPUs, or between a GPU and a network card, the traffic is staged through host memory and pays for it. Over TCP, the operating system also copies every message into and out of its socket buffers at each end; RDMA skips that copy.

More: [parallelism](simulation-model.md#5-parallelism), [collectives](simulation-model.md#6-collectives), [links and topologies](simulation-model.md#links-and-topologies), [fabric shapes](simulation-model.md#fabric-shapes), [host bounce](simulation-model.md#host-bounce), [inter-node host staging](simulation-model.md#inter-node-host-staging), [TCP socket copy](simulation-model.md#tcp-socket-copy).

## Does it fit in memory?

Per device, the estimate adds up the weights it holds, the cache and state for every sequence in the batch (KV cache, recurrent state), and the largest temporary activation buffer. If the total exceeds the memory the GPU may use, the setup does not fit. It also reports how many sequences would fit.

Long prompts are prefilled in chunks (8192 tokens by default), so the activation buffer is the one of a single chunk, not of the whole prompt. The prefill speed is not affected.

On unified memory the operating system lets the GPU use only part of it: about 75% on a Mac, about 50% by default on Linux with Strix Halo, nearly all of it on DGX Spark. The fit checks against that share; a raised limit can be entered under Advanced. An optional runtime reserve (framework and driver memory, 0 by default) can be added there too.

More: [memory fit](simulation-model.md#memory-fit), [activation peak](simulation-model.md#activation-peak), [GPU-usable capacity](simulation-model.md#gpu-usable-capacity), [runtime reserve](simulation-model.md#runtime-reserve).

## Offloading

Optional, off by default. Part of the weights can live in host RAM or on an SSD instead of device memory: experts of mixture-of-experts layers, whole layers, or n-gram lookup tables. Whatever is needed and not resident is fetched over the host link each step, which adds transfer time and frees device memory.

Two parameters shape the result:

- **Hit ratio**: how often a needed expert is already on the device. It depends on how much is kept resident and how skewed the model's routing is.
- **Prefetch accuracy**: how often a predictor guesses the next layer's experts early enough to hide the transfer behind compute.

With both set, the result is still an upper bound given those numbers.

More: [offloading](simulation-model.md#12-offloading), [expert hit ratio](simulation-model.md#expert-hit-ratio), [expert prefetch](simulation-model.md#expert-prefetch), [host and storage tiers](simulation-model.md#host-and-storage-tiers).

## Speculative decoding

Optional, off by default. A small draft produces several candidate tokens and the full model checks them all in one step. The step costs a little more than normal decode, because it processes several positions and runs the draft, but each step can yield more than one token.

How many tokens a step yields on average follows from the **acceptance rate** you set per drafted position. The result is the tokens per step divided by the step time, and the speedup against plain decode.

The default acceptance per method and use case is fitted to published measurements, at the draft lengths they were taken at. At any other draft length the app marks the default with ⚠, because the real acceptance per position need not stay constant as the draft grows.

More: [speculative decoding](simulation-model.md#13-speculative-decoding), [accepted tokens](simulation-model.md#accepted-tokens), [acceptance defaults](simulation-model.md#acceptance-defaults).

## Quantization

Weights, activations and the KV cache each have their own number format. Storage per element is the format's bits plus block scales and group overhead, so a 4-bit format with group scales costs a little more than 4 bits. Fractional-bit formats such as EXL3 (for example 2.5 bits per weight) are costed from their actual layout.

The format also picks the arithmetic rate. When weights are narrower than activations they are dequantized first, so the multiply runs at the wider format's rate: quantization saves memory traffic, not arithmetic.

More: [dtypes](simulation-model.md#dtypes-math-and-accumulate), [EXL3](simulation-model.md#exl3-exllamav3-fractional-bits-per-weight).

## What is not modelled

Anything that depends on a particular inference engine: paged attention, continuous batching, chunked prefill (except its effect on the memory fit), scheduling and kernel-launch overhead, CUDA graphs. Also left out: dequantization arithmetic, softmax cost, cache reuse beyond reading each operand once, clock throttling, in-network reduction, link contention, and framework memory unless you set a runtime reserve.

The reference lists every exclusion and each approximation that remains.

More: [deliberately excluded, and approximations](simulation-model.md#9-deliberately-excluded-and-approximations).
