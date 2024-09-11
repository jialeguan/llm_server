#import "@preview/touying:0.4.2": *
#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge
#import "@preview/tablex:0.0.8": tablex, hlinex, vlinex, colspanx, rowspanx

#let s = themes.dewdrop.register(
  aspect-ratio: "16-9",
  footer: [PCC],
  navigation: "mini-slides",
  // navigation: none,
)
#let s = (
  s.methods.info
)(
  self: s,
  title: [Private Cloud Computing],
  // subtitle: [Subtitle],
  author: [Jiale Guan],
  date: datetime.today(),
  // institution: [Institution],
)

#(
  s.methods.outline-slide = (self: none, ..args) => {
    (self.methods.slide)(
      self: self,
      setting: columns.with(2),
      heading(level: 2, self.outline-title) + parbreak() + (self.methods.touying-outline)(self: self, cover: false),
    )
  }
)
#(
  s.methods.touying-new-section-slide = (self: none, section) => {
    (self.methods.slide)(
      self: self,
      setting: columns.with(2),
      section: section,
      heading(level: 2, self.outline-title) + parbreak() + (self.methods.touying-outline)(self: self),
    )
  }
)

#let (init, slides, touying-outline, alert, speaker-note) = utils.methods(s)
#show: init

#show strong: alert

#let (slide, empty-slide, title-slide, new-section-slide, focus-slide) = utils.slides(s)
#show: slides

#let footnote_link(content, url) = {
  footnote(numbering: "*")[
    #set text(size: 12pt)
    #link(url)[#content]
  ]
}


= Private Cloud Compute
== Taxonomy

#slide[
  #set text(size: 12pt)
  #tablex(
  columns: 3,
  align: center + horizon,
  auto-vlines: false,
  repeat-header: true,

  /* --- header --- */
  [*Requirements*], [*Threat*], [*Guarantees*],
  /* -------------- */

  [Stateless computation], [Trace of data after processing\ e.g. Logging, debugging], [(Purpose) Only use user data to perform requested operations\
  (Transient) Delete the data after fulfilling the request\
  (Scope) Not available to even Apple staff\
  ],

  [Enforceable guarantees], [Technical enforceability\ e.g. External TLS-terminating load balancer], [
    (Hardware) Secure Enclave, Secure Boot\
    (System) Signed System Volume, Swift on Server\
    (Software) Code Signing, Sandboxing
  ],

  [No privileged runtime access], [Privileged interfaces\ e.g. Shell access by SREs], [
    No remote shell. Only pre-specified, structured, and audited \
    logs/metrics can leave the node\
    User data is reviewed by multiple indepedent layers\
  ],

  [Non-targetability], [Targeted attack\ e.g. Steer request to compromised nodes], [
    (Hardware) Hardened supply chain\
    // Revalidation before being provisioned for PCC\
    (Scheduler) Requests cannot be user/content-specific routed\
    (Anonymity) OHTTP Relay, RSA Blind Signature\
    (Scope) No system-wide encryption
  ],

  [Verifiable transparency], [Uninspected code], [
    Every production build of PCC publicly available
  ],
  )
]

== Requirements

=== Stateless computation

#slide[
  Private Cloud Compute must use the personal user data that it receives exclusively for the purpose of fulfilling the user's request. This data must never be available to anyone other than the user, not even to Apple staff, not even during active processing. And *this data must not be retained*, including via logging or for debugging, after the response is returned to the user. In other words, we want a strong form of stateless data processing where *personal data leaves no trace* in the PCC system.
]

=== Enforceable guarantees

#slide[
  Security and privacy guarantees are strongest when they are entirely technically enforceable, which means it must be possible to *constrain and analyze all the components* that critically contribute to the guarantees of the overall Private Cloud Compute system. To use our example from earlier, it's very difficult to reason about what a TLS-terminating load balancer may do with user data during a debugging session. Therefore, PCC must not depend on such external components for its core security and privacy guarantees. Similarly, operational requirements such as collecting server metrics and error logs must be supported with mechanisms that do not undermine privacy protections.
]

=== No privileged runtime access

#slide[
  Private Cloud Compute must not contain privileged interfaces that would enable Apple's site reliability staff to bypass PCC privacy guarantees, even when working to resolve an outage or other severe incident. This also means that PCC must not support a mechanism by which the privileged access envelope could be enlarged at runtime, such as by loading additional software.
]

=== Non-targetability

#slide[
  An attacker should not be able to attempt to compromise personal data that belongs to specific, targeted Private Cloud Compute users without attempting a broad compromise of the entire PCC system. This must hold true even for exceptionally sophisticated attackers who can attempt physical attacks on PCC nodes in the supply chain or attempt to obtain malicious access to PCC data centers. In other words, a limited PCC compromise must not allow the attacker to *steer requests from specific users to compromised nodes*; targeting users should require a wide attack that's likely to be detected. To understand this more intuitively, contrast it with a traditional cloud service design where every application server is provisioned with database credentials for the entire application database, so a compromise of a single application server is sufficient to access any user's data, even if that user doesn't have any active sessions with the compromised application server.
]

=== Verifiable transparency

#slide[
  Security researchers need to be able to verify, with a high degree of confidence, that our privacy and security guarantees for Private Cloud Compute match our public promises. We already have an earlier requirement for our guarantees to be enforceable. Hypothetically, then, if security researchers had sufficient access to the system, they would be able to verify the guarantees. But this last requirement, verifiable transparency, goes one step further and does away with the hypothetical: security researchers must be able to verify the security and privacy guarantees of Private Cloud Compute, and they must be able to verify that the software that's running in the PCC production environment is the same as the software they inspected when verifying the guarantees.
]

= LLM Serving Systems

// == Architecture

// #let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

// #let architecture_graph = fletcher-diagram(
//   let main_row = 1,
//   let upper_row = 0,
//   let lower_row = 2,
//   let left_col = 0,
//   let middle_col = 3,
//   let right_col = 6,

//   let user = (left_col, main_row),
//   let scheduler = (middle_col, main_row),
//   let worker_a = (right_col, upper_row),
//   let worker_b = (right_col, lower_row),

//   node-corner-radius: 4pt,
//   node(user, [*User*]),
//   node(scheduler, [*Scheduler*]),
//   node(worker_a, [*Worker A*]),
//   node(worker_b, [*Worker B*]),

//   edge(user, scheduler, "-|>", [Input], shift: 3pt, label-side: left),
//   edge(scheduler, user, "-|>", [Output], shift: 3pt, label-side: left),
//   edge(scheduler, worker_a, "-|>", [Task], shift: 3pt, label-side: left),
//   edge(worker_a, scheduler, "-|>", [Result], shift: 3pt, label-side: left),
//   edge(scheduler, worker_b, "-|>", [Task], shift: 3pt, label-side: left),
//   edge(worker_b, scheduler, "-|>", [Result], shift: 3pt, label-side: left),
//   edge(worker_a, worker_b, "-|>", [Data], shift: 3pt, label-side: left),
//   edge(worker_b, worker_a, "-|>", [Data], shift: 3pt, label-side: left),
// )
// #slide[
//   #architecture_graph
// ]

=== LLM Inference

#slide[
  Most of the popular decoder-only LLMs (GPT-3, for example) are pretrained on the causal modeling objective, essentially as next-word predictors. These LLMs take a series of tokens as inputs, and generate subsequent tokens autoregressively until they meet a stopping criteria.
]

== Prefill Phase: Processing the input

In the prefill phase, the LLM processes the input tokens to compute the intermediate states (keys and values), which are used to generate the “first” new token. Each new token depends on all the previous tokens, but because the full extent of the input is known, at a high level this is a matrix-matrix operation that's *highly parallelized*. It effectively *saturates GPU utilization*.

== Decode Phase: Generating the output

#slide[
  #set text(size: 14pt)
  In the decode phase, the LLM generates output tokens autoregressively one at a time, until a stopping criteria is met. Each sequential output token needs to know all the previous iterations' output states (keys and values). This is like a matrix-vector operation that underutilizes the GPU compute ability compared to the prefill phase. The speed at which the data (weights, keys, values, activations) is *transferred to the GPU from memory* dominates the latency, not how fast the computation actually happens. In other words, this is a *memory-bound operation*.
][
  // Splitwise: Efficient Generative LLM Inference Using Phase Splitting
  #figure(image("png/memory_usage.png", width: 80%))
]

= Optimization Techniques
== Overview
#slide[

  #let half-red = red.transparentize(50%)

    #set text(size: 10pt)
    #let pos = block(fill: green.transparentize(50%))[+]
    #let neg = block(fill: red.transparentize(50%))[-]
    #let que = block(fill: gray.transparentize(50%))[?]
    #let na = ""


    #tablex(
    columns: 8,
    align: center + horizon,
    auto-vlines: false,
    // repeat-header: true,

    /* --- header --- */
    [*Category*],
    [*Optimization*],
    [*Compute*],
    [*Memory*],
    [*Transmission*],
    [*Throughput*],
    [*TTFT*],
    [*TBT*],

    /* -------------- */ 
    rowspanx(3)[*Batch*], [Iteration-Level Batch], pos, pos, pos, pos, neg, neg,
    (), [Chunked Prefill], pos, pos, [], pos, pos, pos,
    (), [Prepack Prefill], pos, [], [], pos, neg, [],

    rowspanx(4)[*Parallelism*], [Pipeline Parallelism], pos, [], neg, pos, neg, que,
    (), [Tensor Parallelism], pos, [], neg, pos, neg, pos,
    (), [Sequence Parallelism], pos, [], neg, pos, pos, que,
    (), [Speculative Inference], pos, neg, neg, pos, pos, pos,


    rowspanx(5)[*Memory*], [Paging], [], pos, [], pos, [], [],
    (), [Disk Offloading], [], pos, neg, pos, [], [],
    (), [Prefix Caching], [], pos, [], pos, [], [],
    (), [Multi-Query Attention], pos, pos, [], pos, pos, pos,
    (), [Group-Query Attention], pos, pos, [], pos, pos, pos,

    rowspanx(4)[*Tranmission*], [Duplication], pos, neg, pos, pos, pos, pos,
    (), [Pulling], pos, neg, pos, pos, pos, pos,
    (), [Request Migration], pos, pos, neg, pos, pos, pos,
    (), [Disaggregated Arch], pos, pos, neg, pos, neg, neg,

    // rowspanx(5)[*Scheduling*], [Priority-Based], [], pos,
    // (), [Request-Level Prediction], [], pos,
    // (), [Machine-level Scheduler], [], pos,
    // (), [Instance Flip], [], pos,
    // (), [Global Profiling], [], pos,

    )
]

== Batch Processing

=== (Continous) Batch

#slide[
  #set text(size: 14pt)
  *Batch*: A group of requests that are processed together. 

  *Continous Batch*: A batch that is continuously processed, leveraging the opportunity by batching new requests once some old requests are finished
]


== Parallel Processing

#slide[
  #set text(size: 14pt)
  === Pipeline Parallelism

  PP involves sharding the model (vertically) into chunks, where each chunk comprises a subset of layers that is executed on a separate device.
][
  #set text(size: 14pt)
  === Tensor Parallelism

  TP involves sharding the model (horizontally) into chunks, where each chunk comprises a subset of the model's parameters.
][
  #set text(size: 14pt)
  === Sequence Parallelism

  SP involves sharding the input sequence into chunks, where each chunk is processed by a separate device.
]

=== Speculative Inference#footnote_link("Blockwise Parallel Decoding for Deep Autoregressive Models", "https://arxiv.org/abs/1811.03115")

#slide[
  #set text(size: 14pt)
  A draft model temporarily predicts multiple future steps that are verified or rejected in parallel.

  Benefits:
  1. *Parallel token generation*: Multiple candidate tokens are predicted simultaneously, speeding up inference.
  2. *Reduced sequential dependence*: Parallel generation reduces the need to wait for each token to be computed one at a time.
][
  #figure(
    image("png/speculative.png", width: 90%),
  )
]


== Memory Optimizations
=== KV Cache

#slide[
  #set text(size: 12pt)
  #figure(image("png/kvcache_final.gif", height: 50%))

  Transformers use attention mechanisms that compute attention scores between tokens. The KV Cache helps by storing previously computed key-value pairs, allowing the model to quickly access and reuse them for new tokens, avoiding redundant calculations.

][
  #set text(size: 12pt)
  #figure(image("png/memory_layout.png", height: 40%))

  \
  \

  Memory layout when serving an LLM with 13B parameters on NVIDIA A100. The parameters (gray) persist in GPU memory throughout serving. The memory for the KV cache (red) is (de)allocated per serving request. A small amount of memory (yellow) is used ephemerally for activation.
]

=== Paged Attention#footnote_link("Efficient Memory Management for Large Language Model Serving with PagedAttention", "https://arxiv.org/abs/2309.06180")

#slide[
  #set text(size: 14pt)
  Paged Attention is a technique that divides the attention matrix into smaller pages, which are processed sequentially. This allows the model to process large attention matrices that do not fit in GPU memory.

  #figure(
    image("png/paged_attention_waste.png", width: 90%),
  )
]

#slide[
  #figure(
    image("png/paged_attention.png", width: 90%),
  )
]


=== Group-query Attention#footnote_link("GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "https://arxiv.org/abs/2305.13245")
#slide[
  #set text(size: 14pt)

  - Standard Attention: Compute attention for each query separately. Complexity is $O(n^2)$.
  - Multi-Query Attention: Reuse the same attention matrix for multiple queries. Queries are similar enough to share the same attention distribution.
  - Group-Query Attention: Divide queries into groups and compute attention for each group separately.

  #figure(
    image("png/grouped-query.png", width: 60%),
  )

  // https://arxiv.org/pdf/2305.13245
]

=== Prefix Caching

#slide[
  #set text(size: 14pt)
  Prefix Caching is a technique that caches the intermediate states of the model during the prefill phase. These states are then reused during the decode phase to speed up inference.

  // #figure(
  //   image("png/prefix_caching.png", width: 60%),
  // )
]

=== Flash Attention#footnote_link("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "https://arxiv.org/abs/2205.14135")
#slide[
  // 主要是为了解决长序列的问题，把query分成tile，交给多个线程处理，然后再合并结果
  #set text(size: 16pt)
  *GPU*: One kind of computation done on the input data at a time in sequence

  *Fusing*: Fusing multiple layers together during the actual computation can enable minimizing the data access by GPUs.

  FlashAttention uses *tiling* to fully compute and write out a small part of the final matrix at once

][
  #figure(
    image("png/flashAttention.png"),
  )
]

== Transmission Optimizations

=== KV Cache Offloading

#slide[
  #set text(size: 14pt)
  The KV Cache Offloading technique moves the KV cache from the GPU to the CPU to free up GPU memory for other tasks.
]

== Miscellaneous

=== Compression
#slide[
  === Quantization
  Quantization is the process of reducing the precision of a model’s weights and activations.


][
  === Sparsity
  Sparsity is the process of setting a portion of the model’s weights to zero. Then the model can be expressed as a sparse matrix.

][
  === Distillation
  Distillation is the process of training a smaller model to mimic the behavior of a larger model.
]

=== Cross Region & Cloud

#slide[
  #set text(size: 14pt)
  *Cross Region*: Distributing the model across multiple regions to reduce latency and improve availability.

  *Cloud*: Using cloud services to offload the model computation to reduce the load on the on-premises servers.

  SkyPilot
]

= Threats

// == Violations

// #slide[
//   #set text(size: 14pt)
//   #tablex(
//   columns: 2,
//   align: center + horizon,
//   auto-vlines: false,
//   repeat-header: true,

//   /* --- header --- */
//   [*Requirement*], [*Violations*],
//   /* -------------- */

//   [Stateless computation], [Logging, prioritization, history metadata],

//   [Enforceable guarantees], [Data transfer/duplication, data offloading, access control],

//   [No privileged runtime access], [Monitoring, debugging, profiling],

//   [Non-targetability], [Biased scheduler, input/output leakage],

//   [Verifiable transparency], [Uninspected code],
//   )

//   Universal problems: Access control between worker nodes.
// ]

== Academic Systems
#slide[
  #[
    #set text(size: 7pt)
    *S*: Stateless computation
    *E*: Enforceable guarantees
    *P*: No privileged runtime access
    *T*: Non-targetability
    *V*: Verifiable transparency
  ]

  #[
    #let model_header(name, year) = {
      let size = 5pt
      set text(size: size)
      [*#name*\ ]
      [#year]
    }

    #set text(size: 6pt)
    #let cm = emoji.checkmark.heavy
    #let first = "Initial"
    #let na = ""

    #tablex(
    columns: 22,
    align: center + horizon,
    auto-vlines: false,
    // repeat-header: true,

    /* --- header --- */
    [*Category*],
    [*Optimization*],
    [*Threat*],
    model_header("FT", 22),
    model_header("Orca", 22),
    model_header("vLLM", 23),
    model_header("FlexGen", 23),
    model_header("FastServe", 23),
    model_header("Some", 23),
    model_header("Sarathi", 23),
    model_header("SGLang", 23),
    model_header("Preble", 24),
    model_header("Lookahead", 24),
    model_header("REST", 24),
    model_header("SpecInfer", 24),
    model_header("Medusa", 24),
    model_header("DistServe", 24),
    model_header("Splitwise", 24),
    model_header("LoongServe", 24),
     model_header("AttentionStore", 24),
    model_header("TetriInfer", 24),
    model_header("InfiniteLLM", 24),

    /* -------------- */

    rowspanx(3)[*Batch*], [Iteration-Level Batch], na, [], first, cm, [], cm, [], cm,  [], [], [], [], [], [], cm, [], [], [], cm, [],
    (), [Chunked Prefill], na, [], [], [], [], [], [], first, [], cm, [], [], [], [], [], [], [], [], cm, [],
    (), [Prepack Prefill], na, [], [], [], [], [], [], [], [], [], [], [], cm, [], cm, [], [], [], cm, [],

    rowspanx(6)[*Parallelism*], [Speculation], [*S*], [], [], [], [], [], [], [], cm, [], cm, cm, cm, cm, [], [], [], [], [], [],
    (),  [Prompt-Based Speculation], [*S*], [], [], [], [], [], [], [], [], [], cm, [], [], [], [], [], [], [], [], [],
    (),  [Context-Based Speculation], [*S*], [], [], [], [], [], [], [], [], [], [], cm, [], [], [], [], [], [], [], [],
    (), [Tensor Parallelism], [], cm, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [SafeTensors], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Sequence Parallelism], na, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cm, [], [], [],

    rowspanx(5)[*Memory*], [Paging], na, [], [], first, [], [], [], cm, cm, [], [], [], [], [], [], [], [], [], cm, [],
    (), [Disk Offloading], [*SE*], [], [], [], cm, cm, [], [], [], [], [], [], [], [], [], [], [], cm, [], [],
    (), [Prefix Caching], [*SE*], [], [], [], [], [], [], [], cm, cm, [], [], [], [], [], [], [], [], [], [],
    // (), [Radix Attention], na, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Multi-Query Attention], na, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Group-Query Attention], [*T*], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],


    rowspanx(4)[*Tranmission*], [Duplication], [*T*], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Pulling], [*SET*], [], [], [], [], [], [], [], [], [], [], [], [], [], cm, [], [], [], [], [],
    (), [Request Migration], na,  [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cm, [], [], [],
    (), [Disaggregated Arch], na, [], [], [], [], [], [], [], [], [], [], [], [], [], cm, cm, [], [], cm, [],

    rowspanx(5)[*Scheduling*], [Priority-Based], [*T*], [], [], [], [], cm, [], cm, cm, cm, [], [], [], [], [], [], [], [], cm, [],
    (), [Request-Level Prediction], [*T*], [], [], [], [], cm, [], [], [], [], [], [], [Small Model], [], [], [], [], [], cm, [],
    (), [Machine-level Scheduler], [*ET*], [], [], [], [], cm, [], [], [], cm, [], [], [], [], [], cm, cm, [], cm, cm,
    (), [Instance Flip], na, [], [], [], [], [], [], [], [], [], [], [], [], [], [], cm, [], [], cm, [],
    (), [Global Profiling], [*P*], [], [], [], cm, [], [], [], [], [], [], [], [], [], cm, cm, [], [], [], [],

    [*Verification*], [Open Source], [*V*], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cm, [],
    )
    TorchServe - PyTorch

    Preble - more read

    AlpaServe

    Pollux

    Attmemo:Accelerating transformers with memoization on big memory systems

    Ring Attention

    Virtual Token Counter

    FastGen

    *Refer to DistServe Related Work*
  ]
]

== Industrial Systems

#slide[
  #[
    #set text(size: 7pt)
    *S*: Stateless computation
    *E*: Enforceable guarantees
    *P*: No privileged runtime access
    *T*: Non-targetability
    *V*: Verifiable transparency
  ]

  #[
    #let model_header(name, year) = {
      let size = 4pt
      set text(size: size)
      [*#name*\ ]
      [#year]
    }

    #set text(size: 6pt)
    #let cm = emoji.checkmark.heavy
    #let first = "Initial"
    #let na = ""

    #tablex(
    columns: 15,
    align: center + horizon,
    auto-vlines: false,
    // repeat-header: true,

    /* --- header --- */
    [*Category*],
    [*Optimization*],
    [*Threat*],
    model_header("vLLM", "Open Source"),
    model_header("LightLLM", "Open Source"),
    model_header("FlexFlow", "Open Source"),
    model_header("Mooncake", "Moonshot"),
    model_header("DeepSpeed", "Microsoft"),
    model_header("TensorRT", "NVIDIA"),
    model_header("TGI", "Hugging Face"),
    model_header("Llama", "Intel"),
    model_header("LMDeploy", "Shanghai AI lab"),
    model_header("fastllm", "Open Source"),
    model_header("rtp-llm", "Alibaba"),
    model_header("MindIE", "Huawei"),
    /* -------------- */

    rowspanx(3)[*Batch*], [Iteration-Level Batch], na, cm, [], cm, cm, cm, cm, cm, cm, cm, cm, cm, [],
    (), [Chunked Prefill], na, cm, [], [], cm, cm, [], [], [], [], [], [], [],
    (), [Prepack Prefill], [], na, [], [], [], [], [], [], [], [], [], [], [],

    rowspanx(5)[*Parallelism*], [Speculation], [*S*], [], [], cm, [], [], cm, cm, [], [], [], cm, cm,
    (), [Medusa], na, [], [], [], [], [], [], cm, [], [], [], cm, [],
    (), [Tensor Parallelism], na, [], [], [], [], [], [], cm, [], [], [], [], cm,
    (), [SafeTensors], na, [], [], [], [], [], [], cm, [], [], [], [], [],
    (), [Sequence Parallelism], na, [], [], [], [], [], [], [], [], [], [], [], [],

    rowspanx(6)[*Memory*], [Paging], na, cm, [], [], cm, cm, [], cm, [], [], [], cm, [],
    (), [Token Attention], na, [], cm, [], [], [], [], [], [], [], [], [], [],
    (), [Storage Offloading], [*SE*], [], [], [], cm, [], cm, [], [], [], [], [], cm,
    (), [Multi-Query Attention], na, [], [], [], [], [], cm, [], [], [], [], [], [],
    (), [Group-Query Attention], [*T*], [], [], [], [], [], cm, [], [], [], [], [], [],
    (), [Prefix Caching], na, cm, [], [], [], [], [], [], [], [], [], [], [],

    rowspanx(4)[*Tranmission*], [Duplication], [*T*], [], [], [], cm, [], [], [], [], [], [], [], [],
    (), [Pulling], [*SET*], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Request Migration], na, [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Disaggregated Arch], na, cm, [], [], cm, [], [], [], [], [], [], [], cm,

    rowspanx(5)[*Scheduling*], [Priority-Based], [*T*], [], [], [], cm, cm, [], [], [], [], [], [], [],
    (), [Request-Level Prediction], [*T*], [], cm, [], [], [], [], [], [], [], [], [], [],
    (), [Machine-level Scheduler], [*ET*], [], [], [], cm, [], [], [], [], [], [], [], [],
    (), [Instance Flip], na, [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Global Profiling], [*P*], [], [], [], cm, [], [], [], [], [], [], [], [],

    [*Verification*], [Open Source], [*V*], [], [], [], cm, [], [], [], [], [], [], [], [],
    )
  ]
]

=== Other Optimizations

#slide[
  #set text(size: 14pt)
  *Prompt Cache*: [Prefill, Memory] Reuse attention states across different LLM prompts. Parse the prompt and use reusable text segments(snippet)

  *Layer-wise Transmission*: [Transmission] Transmit each layer's output to the next layer in the pipeline, instead of transmitting the entire model's output.

  *LightLLM*: Use http as the interface to the system.

  *SkyPilot*: [Cross Region & Cloud] Given a job and its resource requirements (CPU/GPU/TPU), SkyPilot automatically figures out which locations (zone/region/cloud) have the compute to run the job, then sends it to the cheapest one to execute.

  *MLC LLM*: enable efficient execution of large language models across a wide range of hardware platforms, including mobile devices, edge devices, and even web browsers.
]

== Threats

=== Paging & Offloading

#slide[
  #set text(size: 16pt)
  Definition:
  - Paging: 使用类似于虚拟内存的机制，将模型参数分页存储在磁盘上，根据需要加载到内存中。
  - Offloading: 将模型参数从GPU内存中移动到CPU内存或磁盘中，以释放GPU内存供其他模型使用。

  Threats:
  - 分页处理过程中可能会产生包含敏感信息的日志，这些日志如果没有妥善管理，可能会泄露隐私数据。
  - 分页数据可能会被意外持久化到不安全的存储介质中，从而暴露隐私数据。
]

=== Duplication & Pulling

#slide[
  #set text(size: 16pt)
  Definition:
  - Duplication: 在不同的节点之间复制模型参数，以便在多个节点上并行执行推理任务。
  - Pulling: 从远程节点拉取模型参数，以便在本地节点上执行推理任务。

  Threats:
  - 模型参数的复制和拉取过程中可能会泄露隐私数据。
  - 模型参数的复制和拉取过程中可能会定向到恶意节点，从而导致隐私数据泄露。如果其中任何一个节点被攻破，攻击者可能获得整个模型的敏感信息。
  - 拉取模型参数可能导致数据不同步，尤其在多次拉取操作之间，可能出现数据不一致的情况，影响模型的准确性和隐私保护。
]

=== Priority-based Scheduler & Local Scheduler & Instance Flip

#slide[
  #set text(size: 13pt)
  Definition:
  - Priority-based Scheduler: 根据任务的优先级调度任务，以确保高优先级任务能够及时完成。
  - Local Scheduler: 在本地节点上调度任务，以减少任务调度的延迟。

  Threats:

  （优先级调度）
  - 可能通过观察任务的优先级来推断任务的重要性和敏感性，从而有针对性地进行攻击。
  - 在任务调度过程中，任务的调度信息（如任务类型、数据类型等）可能被泄露，导致隐私数据暴露。'
  （本地调度）
  - 在本地节点上调度任务时，所有任务和数据都集中在本地节点，如果本地节点被攻破，所有数据和任务信息都可能被泄露。
  - 本地节点可能会缓存大量的任务数据，如果这些缓存数据未妥善处理，可能会导致隐私泄露。
  - 为了减少调度延迟，可能会牺牲一些数据同步和一致性机制，导致数据不一致。
  （节点翻转）
  - 攻击者可能修改恶意节点的数据，来让恶意节点被选中执行任务，从而获取敏感信息。
  - 攻击者可能通过控制节点翻转的时机，来获取敏感信息-。
]

=== Disaggregated Architecture & Online/Offline Profiling

#slide[
  #set text(size: 16pt)
  Definition:
  - Disaggregated Architecture: 将Prefill和Decode的过程通过实例（instance）分离，以提高资源利用率和灵活性。
  - Online/Offline Profiling: 在线/离线性能分析，以优化模型推理性能。

  Threats:
  - 在进行用户画像时，会收集和存储大量的用户数据，包括在线行为数据和离线数据，这些数据一旦被泄露，可能对用户隐私造成严重威胁。
]

=== Iteration-Level Batch & Chunked Prefill & Prepack Prefill

#slide[
  #set text(size: 16pt)
  Definition:
  - Iteration-Level Batch: 在迭代级别上进行批处理，以提高模型推理性能。
  - Chunked Prefill: 将Prefill过程分块，以减少Prefill的延迟。
  - Prepack Prefill: 预先打包Prefill数据，以减少Prefill的延迟。

  Threats:
  - N/A.
]






#focus-slide[
  Thanks
]

// appendix by freezing last-slide-number
#let s = (s.methods.appendix)(self: s)
#let (slide, empty-slide) = utils.slides(s)

= Appendix

== References

#slide[
  https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
]

=== Tools

https://github.com/Trusted-AI/adversarial-robustness-toolbox