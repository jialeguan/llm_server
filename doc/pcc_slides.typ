#import "@preview/touying:0.5.2": *
#import themes.dewdrop: *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge
#import "@preview/tablex:0.0.8": tablex, hlinex, vlinex, colspanx, rowspanx


#show: dewdrop-theme.with(
  aspect-ratio: "16-9",
  footer: "PCC",
  navigation: "mini-slides",
  config-info(
    title: [Private Cloud Compute],
    // subtitle: [Subtitle],
    author: [Jiale Guan],
    date: datetime.today(),
    // institution: [Institution],
  ),
)


// #set heading(numbering: numbly("{1}.", default: "1.1"))

// #set heading(numbering: "1.1")
// #set heading(numbering: (..nums) => {
//   nums = nums.pos()
//   if nums.len() == 1 {
//     return "Appendix " + numbering("A.", ..nums)
//   } else if nums.len() == 2 {
//     return numbering("A.1.", ..nums)
//   } else {
//     return "Step " + numbering("1.", nums.last())
//   }
// })
#set heading(numbering: (..nums) => {
  nums = nums.pos()
  if nums.len() <= 2 {
    return numbering("1.1", ..nums)
  } else {
    return none
  }
})

#title-slide()

#outline-slide()

#let footnote_link(content, url) = {
  footnote(numbering: "*")[
    #set text(size: 9pt, fill: gray)
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

#slide[
  = Stateless computation

  Private Cloud Compute must use the personal user data that it receives exclusively for the purpose of fulfilling the user's request. This data must never be available to anyone other than the user, not even to Apple staff, not even during active processing. And *this data must not be retained*, including via logging or for debugging, after the response is returned to the user. In other words, we want a strong form of stateless data processing where *personal data leaves no trace* in the PCC system.
]

#slide[
  = Enforceable guarantees
  Security and privacy guarantees are strongest when they are entirely technically enforceable, which means it must be possible to *constrain and analyze all the components* that critically contribute to the guarantees of the overall Private Cloud Compute system. To use our example from earlier, it's very difficult to reason about what a TLS-terminating load balancer may do with user data during a debugging session. Therefore, PCC must not depend on such external components for its core security and privacy guarantees. Similarly, operational requirements such as collecting server metrics and error logs must be supported with mechanisms that do not undermine privacy protections.
]

#slide[
  = No privileged runtime access
  Private Cloud Compute *must not contain privileged interfaces* that would enable Apple's site reliability staff to bypass PCC privacy guarantees, even when working to resolve an outage or other severe incident. This also means that PCC must not support a mechanism by which the privileged access envelope could be enlarged at runtime, such as by loading additional software.
]

#slide[
  = Non-targetability
  An attacker should not be able to attempt to compromise personal data that belongs to specific, targeted Private Cloud Compute users without attempting a broad compromise of the entire PCC system. This must hold true even for exceptionally sophisticated attackers who can attempt physical attacks on PCC nodes in the supply chain or attempt to obtain malicious access to PCC data centers. In other words, a limited PCC compromise must not allow the attacker to *steer requests from specific users to compromised nodes*; targeting users should require a wide attack that's likely to be detected. To understand this more intuitively, contrast it with a traditional cloud service design where every application server is provisioned with database credentials for the entire application database, so a compromise of a single application server is sufficient to access any user's data, even if that user doesn't have any active sessions with the compromised application server.
]

#slide[
  = Verifiable transparency
  Security researchers need to be able to verify, with a high degree of confidence, that our privacy and security guarantees for Private Cloud Compute match our public promises. We already have an earlier requirement for our guarantees to be enforceable. Hypothetically, then, if security researchers had sufficient access to the system, they would be able to verify the guarantees. But this last requirement, verifiable transparency, goes one step further and does away with the hypothetical: *security researchers must be able to verify the security and privacy guarantees of Private Cloud Compute*, and they must be able to verify that the software that's running in the PCC production environment is the same as the software they inspected when verifying the guarantees.
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

== LLM Inference

#slide[
  Most of the popular decoder-only LLMs (GPT-3, for example) are pretrained on the causal modeling objective, essentially as next-word predictors. These LLMs take a series of tokens as inputs, and generate subsequent tokens autoregressively until they meet a stopping criteria.
]

== Prefill Phase

#slide[
  = Processing the input
  #set text(size: 16pt)
  In the prefill phase, the LLM processes the input tokens to compute the intermediate states (keys and values), which are used to generate the “first” new token. Each new token depends on all the previous tokens, but because the full extent of the input is known, at a high level this is a matrix-matrix operation that's *highly parallelized*. It effectively *saturates GPU utilization*.
]

== Decode Phase

#slide[
  #set text(size: 12pt)
  = Generating the output
  In the decode phase, the LLM generates output tokens autoregressively one at a time, until a stopping criteria is met. Each sequential output token needs to know all the previous iterations' output states (keys and values). This is like a matrix-vector operation that underutilizes the GPU compute ability compared to the prefill phase. The speed at which the data (weights, keys, values, activations) is *transferred to the GPU from memory* dominates the latency, not how fast the computation actually happens. In other words, this is a *memory-bound operation*.
  #figure(image("png/memory_usage.png", width: 60%))
][
  // Splitwise: Efficient Generative LLM Inference Using Phase Splitting

  // 1.	Prompt phase（蓝色点，prefill阶段）：
  // •	内存稳定：从图中可以看到，在prefill阶段（模型开始接受输入直到生成第一个token），内存占用保持相对较低并且稳定（大约在300GB左右）。这表明，在这个阶段，大部分内存占用可能来自于模型参数的加载和初始化，输入token数量对内存占用影响不大。
  // •	少量增长：随着batch size增加，内存占用有少量增长，但整体增长较为平缓，表明输入tokens在这个阶段不会对内存需求产生太大的变化。
  // 2.	Token phase（橙色点，decode阶段）：
  // •	内存增长显著：与prefill阶段不同，在decode阶段内存占用随着batch size的增加而显著增长。这是因为decode阶段每次生成一个新的token时，需要持续保留前面生成的所有tokens的上下文，并依赖这些信息来推理下一个token，因此内存需求会随着生成的tokens数量累积。
  // •	指数增长：图中的曲线展示了decode阶段的内存占用在tokens数量较大时呈现出接近指数式增长，表明随着生成更多的token，内存需求急剧上升。


  #set text(size: 12pt)

  = Prefill (Prompt) Phase
  The graph indicates that most of the memory usage in this phase may come from loading and initializing the model parameters, and the number of input tokens has little impact on memory usage.

  = Decode (Token) Phase
  In contrast, memory usage during the decode phase increases, as the model needs to retain the context of all previously generated tokens to infer the next token. The memory requirement increases significantly with the number of generated tokens.
]

= Optimizations

== Batch Processing

#slide[

  #set text(size: 14pt)
  *Batch*: A group of requests that are processed together.

  *Continous Batch*: A batch that is continuously processed, leveraging the opportunity by batching new requests once some old requests are finished
][
  #figure(
    image("png/continuous_batch.png", width: 70%),
  )
  // Fairness in Serving Large Language Models
]

== Parallel Processing
#slide[
  #set text(size: 18pt)
  = Pipeline Parallelism

  PP involves sharding the model (vertically) into chunks, where each chunk comprises a subset of layers that is executed on a separate device.
][
  #set text(size: 18pt)
  = Tensor Parallelism

  TP involves sharding the model (horizontally) into chunks, where each chunk comprises a subset of the model's parameters.
][
  #set text(size: 18pt)
  = Sequence Parallelism

  SP involves sharding the input sequence into chunks, where each chunk is processed by a separate device.
]

== Speculative Inference

#slide[
  #set text(size: 12pt)
  = Standard inference
  Sequence generation is strictly sequential. Each token must be generated based on the previously generated token, which leads to high latency, especially for long-sequence tasks.


][
  #set text(size: 12pt)
  == Speculative inference#footnote_link("Blockwise Parallel Decoding for Deep Autoregressive Models", "https://arxiv.org/abs/1811.03115")
  - *Predict multiple tokens ahead*: When generating the first token, the model simultaneously makes speculative predictions about the next several tokens.
  - *Parallel processing*: These speculative predictions allow the model to process multiple possible outcomes in parallel, speeding up the inference.
  - *Validate predicted paths*: If the speculative predictions are correct, the model can continue with these results, avoiding the need to recalculate. If the predictions are incorrect, the model adjusts and corrects the path.
]

#slide[
  #set text(size: 12pt)
  Algorithm#footnote_link("Accelerating Large Language Model Decoding with Speculative Sampling, 2023", "https://arxiv.org/abs/2302.01318") is as follows:

  - $p$ is the smaller draft model, $q$ is the larger target model.
  #figure(
    image("png/speculative_sampling.png", width: 90%),
  )
][
  #set text(size: 14pt)
  #figure(
    image("png/speculative.png", width: 90%),
  )
]

#slide[
  = Medusa
  #set text(size: 14pt)
  Medusa is a system that uses *speculative inference* to generate multiple tokens in parallel. It uses a *speculative model* to predict multiple tokens ahead and then validates the predicted paths to avoid redundant calculations.
]


== Memory

#slide[
  = KV Cache
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

#slide[
  = Paged Attention
  #set text(size: 14pt)
  Paged Attention#footnote_link("Efficient Memory Management for Large Language Model Serving with PagedAttention", "https://arxiv.org/abs/2309.06180") is a technique that divides the attention matrix into smaller pages, which are processed sequentially. This allows the model to process large attention matrices that do not fit in GPU memory.

  #figure(
    image("png/paged_attention_waste.png", width: 90%),
  )
]

#slide[
  = Paged Attention (cont.)
  #figure(
    image("png/paged_attention.png", width: 90%),
  )
]

#slide[
  = Group-Query Attention
  #set text(size: 14pt)

  - *Standard Attention*: Compute attention for each query separately. Complexity is $O(n^2)$.
  - *Multi-Query Attention*: Reuse the same attention matrix for multiple queries. Queries are similar enough to share the same attention distribution.
  - *Group-Query Attention*#footnote_link("GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "https://arxiv.org/abs/2305.13245"): Divide queries into groups and compute attention for each group separately.

  #figure(
    image("png/grouped-query.png", width: 50%),
  )

  // https://arxiv.org/pdf/2305.13245
]

#slide[
  = Prefix Caching
  #set text(size: 14pt)
  Prefix Caching is a technique that caches the intermediate states of the model during the prefill phase. These states are then reused during the decode phase to speed up inference.

  // #figure(
  //   image("png/prefix_caching.png", width: 60%),
  // )
]

#slide[
  = Flash Attention
  // 主要是为了解决长序列的问题，把query分成tile，交给多个线程处理，然后再合并结果
  #set text(size: 16pt)
  *GPU*: One kind of computation done on the input data at a time in sequence

  *Fusing*: Fusing multiple layers together during the actual computation can enable minimizing the data access by GPUs.

  FlashAttention#footnote_link("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "https://arxiv.org/abs/2205.14135") uses *tiling* to fully compute and write out a small part of the final matrix at once

][
  #figure(
    image("png/flashAttention.png"),
  )
]

#slide[
  = KV Cache Offloading
  #set text(size: 14pt)
  The KV Cache Offloading technique moves the KV cache from the GPU to the CPU to free up GPU memory for other tasks.
]

== Miscellaneous
#slide[
  = Quantization
  Quantization is the process of reducing the precision of a model’s weights and activations.


][
  = Sparsity
  Sparsity is the process of setting a portion of the model’s weights to zero. Then the model can be expressed as a sparse matrix.

][
  = Distillation
  Distillation is the process of training a smaller model to mimic the behavior of a larger model.
]

#slide[
  = Cross Region & Cloud
  #set text(size: 14pt)
  *Cross Region*: Distributing the model across multiple regions to reduce latency and improve availability.

  *Cloud*: Using cloud services to offload the model computation to reduce the load on the on-premises servers.

  SkyPilot
]

== Summary
#slide[
  = Stateful Inference Systems
  #set text(size: 16pt)
  *Static state* States in traditional systems can be modified after creation and require various consistency and coherence
  mechanisms to support parallelism. In LLM inference,
  once KVs are computed for a sequence, their values do not change.

  *Regular computation patterns* LLMs' transformer computation is regular. Its computing and memory consumption is determined by the model size, the prompt
  length, and the output generation length. The model size and
  a request's prompt length are known before execution, and
  output is generated one token per iteration. Thus, we can
  estimate the computing and memory consumption for every
  iteration.
]

#slide[
  #set text(size: 10pt)

  #let half-red = red.transparentize(50%)
  #let pos = block(fill: green.transparentize(50%))[+]
  #let neg = block(fill: red.transparentize(50%))[-]
  #let que = block(fill: gray.transparentize(50%))[?]
  #let na = ""

  #tablex(
    columns: 8,
    align: center + horizon,
    auto-vlines: false,
    // repeat-header: true,
    vlinex(x: 5, stroke: gray),

    /* --- header --- */
    rowspanx(2)[*Category*], rowspanx(2)[*Optimization*], colspanx(3)[*GPU Resources*], colspanx(3)[*Optimization Goal*],

    [*Compute*],
    [*Memory*],
    [*Transmission*],
    [*Throughput*],
    [*TTFT*],
    [*TBT*],

    /* -------------- */
    // 例子
    rowspanx(3)[*Batch*], [Iteration-Level Batch], pos, [], pos, pos, neg, neg,
    (), [Chunked Prefill], pos, [], [], pos, pos, pos,
    (), [Prepack Prefill], pos, [], [], pos, neg, [],

    rowspanx(4)[*Parallelism*], [Pipeline Parallelism], pos, [], neg, pos, neg, que,
    (), [Tensor Parallelism], pos, [], neg, pos, neg, pos,
    (), [Sequence Parallelism], pos, [], neg, pos, pos, que,
    (), [Speculative Inference], pos, neg, [], pos, [], pos,


    rowspanx(5)[*Memory*], [Paging], [], pos, [], pos, [], [],
    (), [Prefix Caching], [], pos, [], pos, [], [],
    (), [Disk Offloading], [], pos, neg, pos, [], [],
    (), [Multi-Query Attention], [], pos, [], pos, pos, pos,
    (), [Group-Query Attention], [], pos, [], pos, pos, pos,

    rowspanx(4)[*Tranmission*], [Duplication], pos, neg, pos, pos, pos, pos,
    (), [Pulling], pos, neg, pos, pos, pos, pos,
    (), [Request Migration], pos, pos, neg, pos, pos, pos,
    (), [Disaggregated Arch], pos, pos, pos, pos, neg, neg,

    // rowspanx(5)[*Scheduling*], [Priority-Based], [], pos,
    // (), [Request-Level Prediction], [], pos,
    // (), [Machine-level Scheduler], [], pos,
    // (), [Instance Flip], [], pos,
    // (), [Global Profiling], [], pos,

    )
]



= Threats

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
    #let model_header(name, year, url) = {
      let size = 5pt
      set text(size: size)
      link(url)[*#name*\ ]
      [#year]
    }

    #set text(size: 5pt)
    #let cm = emoji.checkmark.heavy
    #let f1 = "Initial"
    #let na = ""

    #tablex(
    columns: 21,
    align: center + horizon,
    auto-vlines: false,
    // repeat-header: true,

    /* --- header --- */
    [*Category*],
    [*Optimization*],
    [*Threat*],
    model_header("Orca", 2206, "https://www.usenix.org/conference/osdi22/presentation/yu"),
    model_header("FlexGen", 2303, "https://arxiv.org/abs/2303.06865"),
    model_header("FastServe", 2305, "https://arxiv.org/abs/2305.05920"),
    model_header("SpecInfer", 2305, "https://arxiv.org/abs/2305.09781"),
    model_header("vLLM", 2309, "https://arxiv.org/abs/2309.06180"),
    model_header("REST", 2311, "https://arxiv.org/abs/2311.08252"),
    model_header("Splitwise", 2311, "https://arxiv.org/abs/2311.18677"),
    model_header("SGLang", 2312, "https://arxiv.org/abs/2312.07104"),
    model_header("Lookahead", 2312, "https://arxiv.org/abs/2312.12728"),
    model_header("Sarathi", "23-24", "https://arxiv.org/abs/2403.02310"),

    model_header("InfiniteLLM", 2401, "https://arxiv.org/abs/2401.02669"),
    model_header("DistServe", 2401, "https://arxiv.org/abs/2401.09670"),
    model_header("Medusa", 2401, "https://arxiv.org/abs/2401.10774"),
    model_header("TetriInfer", 2401, "https://arxiv.org/abs/2401.11181"),
    model_header("AttentionStore", 2403, "https://arxiv.org/abs/2403.19708v2"),
    model_header("LoongServe", 2404, "https://arxiv.org/abs/2404.09526"),
    model_header("vAttention", 2405, "https://arxiv.org/abs/2405.04437"),
    model_header("Preble", 2407, "https://arxiv.org/abs/2407.00023"),

    /* -------------- */

    rowspanx(3)[*Batch*], [Iteration-Level Batch], na,
    f1, [], cm, cm, cm, [], [], [], [], cm, [], cm, [], cm, [], [], [], [],
    (), [Chunked Prefill], na,
    [], [], [], [], [], [], [], [], [], f1, [], [], [], cm, [], [], [], cm,
    (), [Prepack Prefill], na,
    [], [], [], [], [], [], [], [], [], [], [], cm, [], cm, [], [], [], [],

    rowspanx(6)[*Parallelism*], [Speculation], [*S*],
    [], [], [], cm, [], cm, [], cm, cm, [], [], [], cm, [], [], [], [], [],
    (), [Context-Based Speculation], [*S*],
    [], [], [], [], [], cm, [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Prompt-Based Speculation], [*S*],
    [], [], [], [], [], [], [], [], cm, [], [], [], [], [], [], [], [], [],
    (), [Tensor Parallelism], na,
    [], [], [], cm, [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [SafeTensors], na,
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Sequence Parallelism], na,
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cm, [], [],

    rowspanx(3)[*Memory*], [Paging], na,
    [], [], [], [], f1, [], [], cm, [], cm, [], [], [], cm, [], [], [], [],
    (), [Prefix Caching], [*SE*],
    [], [], [], [], [], [], [], cm, [], [], [], [], [], [], [], [], [], cm,
    (), [Disk Offloading], [*SE*],
    [], cm, cm, [], [], [], [], [], [], [], [], [], [], [], cm, [], [], [],

    rowspanx(4)[*Tranmission*], [Duplication], [*T*],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Pulling], [*SET*],
    [], [], [], [], [], [], [], [], [], [], [], cm, [], [], [], [], [], [],
    (), [Request Migration], na,
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cm, [], [],
    (), [Disaggregated Arch], na,
    [], [], [], [], [], [], cm, [], [], [], [], cm, [], cm, [], [], [], [],

    rowspanx(5)[*Scheduling*], [Priority-Based], [*T*],
    [], [], cm, [], [], [], [], cm, [], cm, [], [], [], cm, [], [], [], cm,
    (), [Request-Level Prediction], [*T*],
    [], [], cm, cm, [], [], [], [], [], [], [], [], [], cm, [], [], [], [],
    (), [Machine-level Scheduler], [*ET*],
    [], [], cm, [], [], [], cm, [], [], [], cm, [], [], cm, [], cm, [], cm,
    (), [Instance Flip], na,
    [], [], [], [], [], [], cm, [], [], [], [], [], [], cm, [], [], [], [],
    (), [Global Profiling], [*P*],
    [], cm, [], [], [], [], cm, [], [], [], [], cm, [], [], [], [], [], [],

    [*Verification*], [Open Source], [*V*],
    [], [], [], [], [], [], [], [], [], [], [], [], [], cm, [], [], [], [],
    )
  ]

  = Miscellaneous
  #set text(size: 10pt)
  #tablex(
    columns: 3,
    align: center + horizon,
    auto-vlines: false,
    // repeat-header: true,

    /* --- header --- */
   [*Title*], [*Keywords*], [*Optimizations*],
    /* -------------- */

    [*Prompt Cache*], [Prefill, Memory], [Reuse attention states across different LLM prompts. Parse the prompt and use reusable text segments(snippet)],
    [*Layer-wise Transmission*], [Transmission], [Transmit each layer's output to the next layer in the pipeline, instead of transmitting the entire model's output],
    [*LightLLM*], [Interface], [Use http as the interface to the system],
    [*SkyPilot*], [Cross Region & Cloud], [Given a job and its resource requirements (CPU/GPU/TPU), SkyPilot automatically figures out which locations (zone/region/cloud) have the compute to run the job, then sends it to the cheapest one to execute],
    [*MLC LLM*], [Efficient Execution], [Enable efficient execution of large language models across a wide range of hardware platforms, including mobile devices, edge devices, and even web browsers],
    )

  MemServe
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
    #let f1 = "Initial"
    #let na = ""

    #tablex(
    columns: 16,
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
    model_header("SGLang", "Open Source"),
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

    rowspanx(3)[*Batch*], [Iteration-Level Batch], na,
    cm, [], cm, [], cm, cm, cm, cm, cm, cm, cm, cm, [],
    (), [Chunked Prefill], na,
    cm, [], [], [], cm, cm, [], [], [], [], [], [], [],
    (), [Prepack Prefill], na,
    [], [], [], [], [], [], [], [], [], [], [], [], [],

    rowspanx(5)[*Parallelism*], [Speculation], [*S*],
    cm, [], cm, cm, [], [], cm, cm, [], [], [], cm, cm,
    (), [Medusa], na,
    [], [], [], [], [], [], [], cm, [], [], [], cm, [],
    (), [Tensor Parallelism], na,
    [], [], [], [], [], [], [], cm, [], [], [], [], cm,
    (), [SafeTensors], na,
    [], [], [], [], [], [], [], cm, [], [], [], [], [],
    (), [Sequence Parallelism], na,
    [], [], [], [], [], [], [], [], [], [], [], [], [],

    rowspanx(6)[*Memory*], [Paging], na,
    cm, [], [], cm, cm, cm, [], cm, [], [], [], cm, [],
    (), [Token Attention], na,
    [], cm, [], [], [], [], [], [], [], [], [], [], [],
    (), [Prefix Caching], [*S*],
    cm, [], [], [], [], [], cm, [], [], [], [], [], [],
    (), [Disk Offloading], [*SE*],
    [], [], [], cm, cm, [], cm, [], [], [], [], [], cm,
    // (), [Radix Attention], [*S*],
    // cm, [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Multi-Query Attention], na,
    [], [], [], [], [], [], cm, [], [], [], [], [], [],
    (), [Group-Query Attention], [*T*],
    [], [], [], [], [], [], cm, [], [], [], [], [], [],


    rowspanx(4)[*Tranmission*], [Duplication], [*T*],
    [], [], [], [], cm, [], [], [], [], [], [], [], [],
    (), [Pulling], [*SET*],
    [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Request Migration], na,
    [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Disaggregated Arch], na,
    cm, [], [], [], cm, [], [], [], [], [], [], [], cm,

    rowspanx(5)[*Scheduling*], [Priority-Based], [*T*],
    [], [], [], cm, cm, cm, [], [], [], [], [], [], [],
    (), [Request-Level Prediction], [*T*],
    [], cm, [], cm, [], [], [], [], [], [], [], [], [],
    (), [Machine-level Scheduler], [*ET*],
    [], [], [], cm, cm, [], [], [], [], [], [], [], [],
    (), [Instance Flip], na,
    [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Global Profiling], [*P*],
    [], [], [], [], cm, [], [], [], [], [], [], [], [],

    [*Verification*], [Open Source], [*V*],
    [], [], [], [], cm, [], cm, [], [], [], [], [], [],
    )
  ]

  // https://github.com/vllm-project/vllm/issues/4104 相对于只缓存Prefix Cache，vLLM的Prefix Caching功能还缓存了Generated KV Cache
]




#slide[
  #set text(size: 14pt)
  DL Serving: AlpaServe, Pollux

  Attention Serving: AttMemo, Ring Attention

  LLM Serving Fairness: VTC

  GPU Communication Lantencies: Flux(TP)
]

== Reasoning

// === Paging & Offloading

// #slide[
//   #set text(size: 16pt)
//   Definition:
//   - Paging: 使用类似于虚拟内存的机制，将模型参数分页存储在磁盘上，根据需要加载到内存中。
//   - Offloading: 将模型参数从GPU内存中移动到CPU内存或磁盘中，以释放GPU内存供其他模型使用。

//   Threats:
//   - 分页处理过程中可能会产生包含敏感信息的日志，这些日志如果没有妥善管理，可能会泄露隐私数据。
//   - 分页数据可能会被意外持久化到不安全的存储介质中，从而暴露隐私数据。
// ]

// === Duplication & Pulling

// #slide[
//   #set text(size: 16pt)
//   Definition:
//   - Duplication: 在不同的节点之间复制模型参数，以便在多个节点上并行执行推理任务。
//   - Pulling: 从远程节点拉取模型参数，以便在本地节点上执行推理任务。

//   Threats:
//   - 模型参数的复制和拉取过程中可能会泄露隐私数据。
//   - 模型参数的复制和拉取过程中可能会定向到恶意节点，从而导致隐私数据泄露。如果其中任何一个节点被攻破，攻击者可能获得整个模型的敏感信息。
//   - 拉取模型参数可能导致数据不同步，尤其在多次拉取操作之间，可能出现数据不一致的情况，影响模型的准确性和隐私保护。
// ]

// === Priority-based Scheduler & Local Scheduler & Instance Flip

// #slide[
//   #set text(size: 13pt)
//   Definition:
//   - Priority-based Scheduler: 根据任务的优先级调度任务，以确保高优先级任务能够及时完成。
//   - Local Scheduler: 在本地节点上调度任务，以减少任务调度的延迟。

//   Threats:

//   （优先级调度）
//   - 可能通过观察任务的优先级来推断任务的重要性和敏感性，从而有针对性地进行攻击。
//   - 在任务调度过程中，任务的调度信息（如任务类型、数据类型等）可能被泄露，导致隐私数据暴露。'
//   （本地调度）
//   - 在本地节点上调度任务时，所有任务和数据都集中在本地节点，如果本地节点被攻破，所有数据和任务信息都可能被泄露。
//   - 本地节点可能会缓存大量的任务数据，如果这些缓存数据未妥善处理，可能会导致隐私泄露。
//   - 为了减少调度延迟，可能会牺牲一些数据同步和一致性机制，导致数据不一致。
//   （节点翻转）
//   - 攻击者可能修改恶意节点的数据，来让恶意节点被选中执行任务，从而获取敏感信息。
//   - 攻击者可能通过控制节点翻转的时机，来获取敏感信息-。
// ]

// === Disaggregated Architecture & Online/Offline Profiling

// #slide[
//   #set text(size: 16pt)
//   Definition:
//   - Disaggregated Architecture: 将Prefill和Decode的过程通过实例（instance）分离，以提高资源利用率和灵活性。
//   - Online/Offline Profiling: 在线/离线性能分析，以优化模型推理性能。

//   Threats:
//   - 在进行用户画像时，会收集和存储大量的用户数据，包括在线行为数据和离线数据，这些数据一旦被泄露，可能对用户隐私造成严重威胁。
// ]

// === Iteration-Level Batch & Chunked Prefill & Prepack Prefill

// #slide[
//   #set text(size: 16pt)
//   Definition:
//   - Iteration-Level Batch: 在迭代级别上进行批处理，以提高模型推理性能。
//   - Chunked Prefill: 将Prefill过程分块，以减少Prefill的延迟。
//   - Prepack Prefill: 预先打包Prefill数据，以减少Prefill的延迟。

//   Threats:
//   - N/A.
// ]

#focus-slide[
  Thanks
]

#show: appendix
= Appendix

=== References

#slide[
  https://github.com/DefTruth/Awesome-LLM-Inference

  https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
]

=== Tools

https://github.com/Trusted-AI/adversarial-robustness-toolbox

