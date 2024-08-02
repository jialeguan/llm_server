# Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

## Dual-Phase Process of LLM Services

An LLM service responds to a user query in two phases.

- The prefill phase processes a user’s prompt, composed of a sequence of tokens, to generate the first token of the response in one step.
- The decoding phase sequentially generates subsequent tokens in multiple steps; each decoding step generates a new token based on tokens generated in previous steps, until reaching a termination token.

## Performance Metrics

| Metrics | Formula | Description | Related phase |
| --- | --- | --- | --- |
| TTFT | time to first token | Initial responsiveness of the model | Prefill |
| TBT | time between tokens | Responsiveness of the model | Decode|
| Throughput | queries per second | Efficiency of the model | Both |

## Characteristics of Each Phase

![alt text](png/sarathi/image.png)

## Optimization Strategies: Batch

![batch size](png/sarathi/image-2.png)

![alt text](png/sarathi/image-1.png)

**Improve memory efficiency**: Batching multiple queries together can improve throughput by amortizing the overhead of fetching model parameters and intermediate states over multiple queries. This process is consistent across varying input sizes.

## Existing Approaches

Larger batch sizes can lead to increased latency.

![alt text](png/sarathi/image-4.png)

![approches](png/sarathi/image-3.png)

Batching can suffer from bubble latency, where the last query in a batch experiences higher latency than the rest.

![alt text](png/sarathi/image-5.png)

Maybe this diagram depicts Tensor-Parallelism?

## Optimization 1: Chucked-Prefill

Naively, this can be done by creating hybrid batches which combine the memory bound decodes along with compute bound prefills

![alt text](png/sarathi/image-6.png)

![alt text](png/sarathi/image-8.png)

## Optimization 2: Stall-free batching

First calculates the budget of maximum number of tokens that can be executed in a batch based on user specified SLO.

Only after all the running requests have been accommodated, we admit new requests. By restricting the computational load in every iteration, stall-free batching ensures that decodes never experience a generation stall due to a co-running prefill chunk.

Pros:

- reduce bubble sizes.
- largely free of latency spikes.

Cons:

- overhead: increased memory reads from the GPU HBM, due to repeated KV-cache access in the attention operation.

  - even at small chunk sizes attention prefill operation is compute bound operation

  ![alt text](png/sarathi/image-7.png)

Solution:

one-time profiling of batches with different number of tokens and setting the token budget to maximum number of tokens that can be packed in a batch without violating TBT SLO.
> Note: matmuls achieve maximum GPU utilization when the matrix dimensions are divisible by the tile size, e.g., in some cases, using chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256.

## Evaluation

<https://github.com/microsoft/sarathi-serve>

![†](png/sarathi/image-9.png)

![alt text](png/sarathi/image-10.png)

![alt text](png/sarathi/image-12.png)

Line 6-8: pack all the running decodes in the next batch

Line 9-12: include any partially completed prefill

Line 13-20: check if the next chunk of tokens can be accommodated in the GPU memory

<https://github.com/microsoft/sarathi-serve/blob/main/png/sarathi/core/scheduler/sarathi_scheduler.py>

There are two cases:

1. The sequence group has incomplete prefill. The routine
remains identical to the one in sarathi scheduler for such sequences.
1. The sequence group has completed prefill. In this case, we need to
check for memory availability for the next chunk of decode tokens, and preempt
some sequence groups if necessary. Note that, the preempted sequence groups
might belong to either of the two categories.

### W/ Tensor Parallelism

![alt text](png/sarathi/image-11.png)
