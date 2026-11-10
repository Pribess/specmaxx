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

Peak rates are only reached by large, well-shaped matrices. Matrix units work on fixed tiles and the chip works in waves of tiles, so a small or awkward shape leaves part of the hardware idle. The estimate applies this tile and wave rounding to every matrix multiply, which is why single-token decode never reaches the headline TFLOPS. On Apple GPUs, single-token matrix-vector products run on the plain vector units without that rounding, as Apple's MLX and llama.cpp kernels do.

Presets cite their vendor sources. Some values are marked as estimated where no vendor publishes them.

More: [hardware](simulation-model.md#4-hardware), [shape efficiency](simulation-model.md#shape-efficiency), [preset library](simulation-model.md#preset-library).

## How time is computed

Each operation takes the longer of two times (the **roofline**):

- **compute time**: its FLOPs divided by the achievable arithmetic rate, and
- **memory time**: its bytes divided by memory bandwidth.

If compute is longer the op is *compute bound*, otherwise *bandwidth bound*. The time of a step is the sum of its operations (on the busiest device). Communication is added using a latency-plus-bandwidth cost, and by default the bandwidth part may overlap with compute while the latency part may not.

Attention also pays for its softmax: one exponential per score on the special-function units, plus a few vector operations, overlapped with its matrix multiplies. This only shows on long prompts on fast datacenter GPUs, where the exponentials can take longer than the matrix work.

Two what-if settings sit next to this. With several devices, prefill is also shown **without overlap**, as if no communication hid behind compute, since most engines do not overlap it fully. An optional **per-op latency floor** (0 by default) makes every operation take at least that long; it shows how much tiny operations could cost, but no measurement fixes its size yet.

More: [per-op time](simulation-model.md#per-op-time), [attention softmax](simulation-model.md#attention-softmax), [stage time and overlap](simulation-model.md#stage-time-and-overlap), [prefill without overlap](simulation-model.md#prefill-without-overlap), [per-op latency floor](simulation-model.md#per-op-latency-floor).

## Prefill and decode

**Prefill** processes the whole prompt at once. There is plenty of work per weight read, so it is usually compute bound. Its rate is prompt tokens divided by the time to process them.

**Decode** produces one token per sequence per step. Every weight and the whole KV cache must be read for very little arithmetic, so it is usually bandwidth bound, and it slows as the context grows. Its rate is one token per step time, times the batch size.

More: [pipeline and DP](simulation-model.md#pipeline-and-dp).

## Multiple devices

Work is split across devices three ways: **tensor parallel** (TP) shards each layer's matrices, **pipeline parallel** (PP) puts consecutive layers on consecutive devices, and **data parallel** (DP) runs independent replicas. Mixture-of-experts layers can also spread their experts (EP). The UI offers TP, PP, DP and, for mixture-of-experts models, EP; the device count is TP × PP × DP.

**DP** runs DP full copies of the model, each on its own TP × PP devices, side by side. The batch is split evenly between them (so it must be a multiple of DP), and the copies never talk to each other. Each copy therefore runs exactly like a single model instance serving batch / DP sequences: the aggregate rate is DP times that copy's rate, and the per-sequence rate is that copy's. Memory is checked per copy: every device holds its share of the whole model, so DP needs a model that fits on TP × PP devices. DP never makes a single sequence faster. Against TP on the same devices it trades the TP all-reduces for reading the full weights on every device: while decode is bandwidth bound TP usually gives the higher aggregate rate (each device reads only its slice of the weights), and DP wins where the all-reduces cost more than that, as in prefill, over slow networks, and at large batches ([pipeline and DP](simulation-model.md#pipeline-and-dp); `src/sim/engine/dp.test.ts`). With offloading, every copy on a machine keeps its own offloaded weights in that machine's host RAM or SSD.

**EP in the UI** adds no devices. With DP = 1 it uses the TP devices: instead of slicing every expert across the TP group, each device holds whole experts: E/EP of them, each sliced TP/EP ways. So EP must divide TP and the expert count. After attention every TP device already holds every token, so no tokens are exchanged: each device runs its own experts on the tokens routed to them, and their outputs are summed by the same TP all-reduce that TP alone needs. EP therefore leaves the communication unchanged, except on models with an expert-output norm, where it removes the collectives between the devices sharing an expert. What it changes is the work per device: the busiest device gets 1.2× the mean expert work (below), it reads and writes only its own experts' token rows instead of every row, and in decode it reads whole experts (the busiest device needs ⌈tokens × top-k / EP⌉ of them). With DP > 1, EP can also span several copies: a multiple of TP that divides TP × DP. Then each device holds whole, unsliced experts and the tokens are exchanged between the copies with an all-to-all ([MoE modes A/B](simulation-model.md#moe-modes-ab)).

**DP attention + EP** is a named layout for mixture-of-experts models, the one large deployments use (vLLM's DEP, `--data-parallel-size N --enable-expert-parallel` ([vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)); SGLang's `--enable-dp-attention` ([SGLang](https://docs.sglang.io/advanced_features/server_arguments.html)); DeepSeek's EP32 and EP144, SGLang's EP72 ([LMSYS](https://lmsys.org/blog/2025-05-05-large-scale-ep/))). It is TP 1 with DP = EP = the devices of one pipeline stage (all devices without PP; EP × PP = the device count): every device runs attention for its own sequences only, so it keeps only their KV cache, and a latent-attention (MLA) cache is not copied to other devices as it is under TP. The experts are spread over all the devices of the stage, and each layer sends every token to the devices holding its experts and back with two all-to-alls. The dense layers (the non-expert feed-forward layers and the LM head) can run two ways. **DP**, the default: whole on every device, with no extra traffic (SGLang `--moe-dense-tp-size 1 --enable-dp-lm-head`). **TP**: split over all EP devices, so each holds 1/EP of their weights; the tokens of all devices are gathered before each such layer and the results scattered back after it (SGLang's default with DP attention). TP saves memory and weight reads for two collectives per dense layer; with few dense layers (DeepSeek-V3 has 3 of 61) the speed difference is small. TP needs EP to divide the dense layers' width, which the app checks ([DP attention + EP](simulation-model.md#dp-attention--ep)).

The estimate assumes a perfectly balanced split and costs one representative device per stage. The one exception is EP: the busiest device gets more than the mean expert work, and the step waits for it ([EP load imbalance](simulation-model.md#ep-load-imbalance)). It then adds the traffic the split requires: all-reduces for TP (which also carry the expert outputs under EP), a hand-off of activations between pipeline stages, and, only when EP spans DP replicas, the token exchange for EP. A link has a bandwidth (per direction) and a latency, and a message costs latency plus size over bandwidth. Every network latency is measured the same way, half the round trip of a small-message ping-pong, so RDMA and TCP over the same cable compare fairly (RDMA is never slower).

**EP imbalance** (shown at EP > 1) is that busiest device's expert load divided by the mean device's. It scales the busiest device's expert compute; it does not change which expert weights are read (hot experts draw more tokens, not more experts). The default is 1.2 with no redundant experts. **Imbalance on all-to-all** (off by default) also applies it to the token exchange when EP spans DP replicas: on, the busiest device sends and receives its larger share; off, every device moves the mean volume, as before these settings existed. Measured values for DeepSeek-R1: 2.56 at EP 32 without a load balancer and 1.11 with one (EPLB) and one redundant expert per GPU, 288 expert slots for 256 experts ([TensorRT-LLM wide-EP blog, Table 1](https://nvidia.github.io/TensorRT-LLM/1.0.0rc3/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.html), which lists (max − mean) / mean: 1.564 and 0.109); about 1.47 on 8 GPUs without one, rising with the GPU count ([SGLang large-scale EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/)). So: about 1.5 (8 GPUs) to 2.5 (32 GPUs) without a balancer, about 1.1 with one. **Redundant experts** are the extra expert copies a balancer keeps on each GPU: each adds one expert's weights per MoE layer to every GPU's memory (about 2.6 GB over DeepSeek-V3's 58 MoE layers in FP8). The balancer itself is not modelled: redundant experts add memory but do not lower the imbalance on their own, so set EP imbalance to the load your balancer reaches ([EP load imbalance](simulation-model.md#ep-load-imbalance), [redundant experts](simulation-model.md#redundant-experts)).

When several tokens go through a mixture-of-experts layer at once (a batch, or a speculative verify step), the experts whose weights are read are the number expected if every token picked its experts at random. This is an expected value, not a bound: real routing often repeats experts and can touch fewer ([expected distinct experts](simulation-model.md#expected-distinct-experts)).

A PCIe link carries data in packets, and each packet's header and checksum take part of the wire. So every PCIe bandwidth here (between GPUs, to host memory for offloading, to an SSD) is the line rate times 256/280 ≈ 0.91, the payload share of full 256-byte packets. Other links (NVLink, InfiniBand, Ethernet, Thunderbolt) use the vendor's data rate. More: [PCIe payload efficiency](simulation-model.md#links-and-topologies).

Between machines the network shape matters: a switch, a full mesh, a ring or a chain. Each shape gets its own routing and bandwidth rules. Where the hardware cannot pass data straight between devices, as with some consumer GPUs, or between a GPU and a network card, the traffic is staged through host memory and pays for it. Over TCP, the operating system also copies every message into and out of its socket buffers at each end; RDMA skips that copy.

A "machine" can also be a whole rack. GB200 and GB300 NVL72 join 72 GPUs through NVLink switches at 900 GB/s per GPU in each direction ([NVIDIA](https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf)); GB200 NVL4 joins four on an NVLink bridge whose layout NVIDIA does not publish, so it is estimated. AMD MI300X, MI325X and MI355X join eight GPUs with one link per pair, 64, 64 and 72 GB/s each way ([MI325X](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf), [MI355X](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-miI355x-platform-brochure.pdf)); the estimate gives every GPU the sum of its seven links, which flatters groups of two or four. A group must split the machine evenly and the experts evenly, so a 256-expert model uses 64 of an NVL72's 72 GPUs for expert parallelism.

More: [parallelism](simulation-model.md#5-parallelism), [collectives](simulation-model.md#6-collectives), [links and topologies](simulation-model.md#links-and-topologies), [scale-up domains](simulation-model.md#scale-up-domains), [rank layout](simulation-model.md#rank-layout), [fabric shapes](simulation-model.md#fabric-shapes), [host bounce](simulation-model.md#host-bounce), [inter-node host staging](simulation-model.md#inter-node-host-staging), [TCP socket copy](simulation-model.md#tcp-socket-copy).

## Does it fit in memory?

Per device, the estimate adds up the weights it holds, the cache and state for every sequence in the batch (KV cache, recurrent state), and the largest temporary activation buffer. If the total exceeds the memory the GPU may use, the setup does not fit. It also reports how many sequences would fit.

Long prompts are prefilled in chunks (8192 tokens by default), so the activation buffer is the one of a single chunk, not of the whole prompt. The prefill speed is not affected.

On unified memory the operating system lets the GPU use only part of it: about 75% on a Mac, about 50% by default on Linux with Strix Halo, nearly all of it on DGX Spark. The fit checks against that share; a raised limit can be entered under Advanced. An optional runtime reserve (framework and driver memory, 0 by default) can be added there too.

More: [memory fit](simulation-model.md#memory-fit), [activation peak](simulation-model.md#activation-peak), [GPU-usable capacity](simulation-model.md#gpu-usable-capacity), [runtime reserve](simulation-model.md#runtime-reserve).

## Offloading

Optional, off by default. Part of the weights can live in host RAM or on an SSD instead of device memory: experts of mixture-of-experts layers, or n-gram lookup tables. Each can be split by layer range, e.g. the experts of the first layers in host RAM and the rest on the SSD. Whatever is needed and not resident is fetched over the host link each step, which adds transfer time and frees device memory.

On GB200 and GB300 the host link is NVLink-C2C to the Grace CPU, taken as 450 GB/s per GPU in each direction ([NVIDIA](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)); NVIDIA's wording leaves open whether a superchip's two GPUs share it. Each Grace adds 480 GB of host RAM at 384 GB/s for two GPUs ([NVIDIA](https://docs.nvidia.com/grace-perf-tuning-guide/index.html)), pooled across the machine.

The token-embedding table can move off the device too. Each token reads only one row of it, so this frees its whole size (about 2.5 GB for a 27B model) at almost no speed cost; llama.cpp keeps it in host RAM by default ([source](https://github.com/ggml-org/llama.cpp/blob/a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5/src/llama-model.cpp#L1780-L1782)). Models that reuse this table for their output (tied embeddings) keep it on the device.

Whole-layer offload (copying entire layers to the GPU every step) was removed: local setups don't run that way (llama.cpp computes the layers it keeps off the GPU on the CPU instead), and for a MoE model it reads every expert each step. Partial offload of dense layers would return as a CPU-compute mode, not as streaming.

Two parameters shape the result:

- **Hit ratio**: how often a needed expert is already on the device. It depends on how much is kept resident and how skewed the model's routing is.
- **Prefetch accuracy**: how often a predictor guesses the next layer's experts early enough to hide the transfer behind compute.

With both set, the result is still an upper bound given those numbers.

When one GPU reads from both host RAM and the SSD, the prefetched part of the two streams runs at the same time, limited by the PCIe link, host RAM and the SSD; experts that are only known after the router are fetched one after another.

**SSD read rate** defaults to the unchanged datasheet ceiling; the optional sustained scenario uses measured low-queue large-read fractions for expert reads, not small lookup rows. These workload-specific measurements are not a tighter proven upper bound.

More: [offloading](simulation-model.md#12-offloading), [expert hit ratio](simulation-model.md#expert-hit-ratio), [expert prefetch](simulation-model.md#expert-prefetch), [overlap across tiers](simulation-model.md#overlap-across-tiers), [token embedding off-device](simulation-model.md#token-embedding-off-device), [host and storage tiers](simulation-model.md#host-and-storage-tiers).

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

Anything that depends on a particular inference engine: paged attention, continuous batching, chunked prefill (except its effect on the memory fit), scheduling and kernel-launch overhead, CUDA graphs. Also left out: dequantization arithmetic, exponentials computed on the ordinary arithmetic units instead of the special-function units (a trick some newer attention kernels use), cache reuse beyond reading each operand once, clock throttling, in-network reduction, link contention, and framework memory unless you set a runtime reserve.

The one exception for engine overhead is the **engine dispatch overlay**, opt-in only and off by default. Turned on, it adds about 2 µs for launching each kernel (one per operation, nothing fused) to a decode step and shows the result as a second, labelled decode figure. The headline figure never includes it. More: [engine dispatch overlay](simulation-model.md#engine-dispatch-overlay).

The reference lists every exclusion and each approximation that remains.

More: [deliberately excluded, and approximations](simulation-model.md#9-deliberately-excluded-and-approximations).
