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
  mini-slides: (height: 3em, x: 1em, display-section: false, display-subsection: true, short-heading: true),
)

#set text(font: "San Francisco", weight: "light", size: 20pt)

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

#let cm = emoji.checkmark.heavy
#let cg = block(fill: green.transparentize(50%))[#cm]
#let cr = block(fill: red.transparentize(50%))[#cm]
#let na = ""

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
  [*Requirements*], [*Threats*], [*Guarantees*],
  /* -------------- */

  [Stateless computation], [Trace of data after processing\ Example:  Logging, debugging], [(Purpose) Only use user data to perform requested operations\
  (Transient) Delete the data after fulfilling the request\
  (Scope) Not available to even Apple staff\
  ],

  [Enforceable guarantees], [Technical enforceability\ Example:  External TLS-terminating load balancer], [
    (Hardware) Secure Enclave, Secure Boot\
    (System) Signed System Volume, Swift on Server\
    (Software) Code Signing, Sandboxing
  ],

  [No privileged runtime access], [Privileged interfaces\ Example:  Shell access by SREs], [
    No remote shell. Only pre-specified, structured, and audited \
    logs/metrics can leave the node\
    User data is reviewed by multiple indepedent layers\
  ],

  [Non-targetability], [Targeted attack\ Example:  Steer request to compromised nodes], [
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
  #set text(size: 16pt)
  Most of the popular decoder-only LLMs (GPT-3, for example) are pretrained on the causal modeling objective, essentially as next-word predictors. These LLMs take a series of tokens as inputs, and generate subsequent tokens autoregressively until they meet a stopping criteria.

  #figure(image("png/mindmap.png", width: 70%), caption: "LLM Inference", numbering: none)
]

== Phases

#slide[
  #let text_size = 12pt
  = Prefill: Processing the input
  #[
    #set text(size: text_size)
    In the prefill phase, the LLM processes the input tokens to compute the intermediate states (keys and values), which are used to generate the “first” new token. Each new token depends on all the previous tokens, but because the full extent of the input is known, at a high level this is a matrix-matrix operation that's *highly parallelized*. It effectively *saturates GPU utilization*.
  ]
  = Decode: Generating the output
  #[
    #set text(size: text_size)
    In the decode phase, the LLM generates output tokens autoregressively one at a time, until a stopping criteria is met. Each sequential output token needs to know all the previous iterations' output states (keys and values). This is like a matrix-vector operation that underutilizes the GPU compute ability compared to the prefill phase. The speed at which the data (weights, keys, values, activations) is *transferred to the GPU from memory* dominates the latency, not how fast the computation actually happens. In other words, this is a *memory-bound operation*.
  ]
  // #figure(image("png/memory_usage.png", width: 60%))
][
  #figure(image("png/two_phase.png", width: 90%))
]
// [
// // Splitwise: Efficient Generative LLM Inference Using Phase Splitting

// // 1.	Prompt phase（蓝色点，prefill阶段）：
// // •	内存稳定：从图中可以看到，在prefill阶段（模型开始接受输入直到生成第一个token），内存占用保持相对较低并且稳定（大约在300GB左右）。这表明，在这个阶段，大部分内存占用可能来自于模型参数的加载和初始化，输入token数量对内存占用影响不大。
// // •	少量增长：随着batch size增加，内存占用有少量增长，但整体增长较为平缓，表明输入tokens在这个阶段不会对内存需求产生太大的变化。
// // 2.	Token phase（橙色点，decode阶段）：
// // •	内存增长显著：与prefill阶段不同，在decode阶段内存占用随着batch size的增加而显著增长。这是因为decode阶段每次生成一个新的token时，需要持续保留前面生成的所有tokens的上下文，并依赖这些信息来推理下一个token，因此内存需求会随着生成的tokens数量累积。
// // •	指数增长：图中的曲线展示了decode阶段的内存占用在tokens数量较大时呈现出接近指数式增长，表明随着生成更多的token，内存需求急剧上升。


// #set text(size: 12pt)

// = Prefill (Prompt) Phase
// The graph indicates that most of the memory usage in this phase may come from loading and initializing the model parameters, and the number of input tokens has little impact on memory usage.

// = Decode (Token) Phase
// In contrast, memory usage during the decode phase increases, as the model needs to retain the context of all previously generated tokens to infer the next token. The memory requirement increases significantly with the number of generated tokens.
// ]

== Challenges

#slide[
  = Workload Heterogeneity
  Universality and application diversity lead to heterogeneity of the inference requests, in terms of input lengths, output lengths, expected latencies, etc
  = Execution Unpredictability
  Unknown a priori how many tokens will be generated before the stopping criteria is met. As such, the execution time and the resource demand of a request are both unpredictable.
  = Multi-Tenant and Dynamic Environment
  The system must scale to support multiple users and adapt to the dynamic nature of the environment.
]

#slide[
  = Queuing Delays
  The system must handle queuing delays, which can be caused by the system being overloaded or by the system waiting for external resources.
  = Preemptions
  The system must handle preemption, which can be caused by the system being overloaded or by the system waiting for external resources.
  = Interference
  Interference between requests can lead to performance degradation.
]

= Optimizations

== Memory Management

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
  #set text(size: 16pt)
  LLM inference architecture primarily comprises multiple stacked decoder blocks, each consisting of a self-attention module and a Feed-Forward Neural Network (FFN) module.
  #figure(image("png/kvcache_detail.png", width: 80%))
  // InstInfer
]

#slide[
  = Paged Attention
  #set text(size: 14pt)
  Paged Attention#footnote_link("Efficient Memory Management for Large Language Model Serving with PagedAttention", "https://arxiv.org/abs/2309.06180") is a technique that divides the attention matrix into smaller pages. This approach provides a near-perfect solution for mitigating fragmentation and hence, PagedAttention has become the de facto standard for dynamic memory allocation in LLM serving systems.

  #figure(
    image("png/paged_attention_waste.png", width: 90%),
  )
]

#slide[
  = Paged Attention
  #set text(size: 12pt)
  #figure(
    image("png/paged_attention.png", width: 80%),
  )

  *Pitfalls*#footnote_link("vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention. 2024","https://arxiv.org/abs/2405.04437")
  - Requires re-writing the attention kernel.
  - Adds software complexity and redundancy (CPU code), can degrade throughput by 11%.
  - Introduces performance overhead. 20-26% slower than original FasterTransformer kernel.
]

#slide[
  = Group-Query Attention
  #set text(size: 14pt)

  - *Standard Attention*: Compute attention for each query separately. Complexity is $O(n^2)$.
  - *Multi-Query Attention*: Reuse the same attention matrix for multiple queries. Queries are similar enough to share the same attention distribution.
  - *Group-Query Attention*#footnote_link("GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "https://arxiv.org/abs/2305.13245"): Divide queries into groups and compute attention for each group separately.
][
  #figure(
    image("png/grouped-query.png", width: 90%),
  )

  // https://arxiv.org/pdf/2305.13245
]

#slide[
  = Prefix Caching
  #set text(size: 14pt)
  Prefix Caching#footnote_link("ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition. 2024", "https://arxiv.org/abs/2402.15220") is a technique that caches the intermediate states of the model during the prefill phase. These states are then reused during the decode phase to speed up inference.

  #figure(
    image("png/prefix_caching.png", width: 60%),
  )
]

#slide[
  = Flash Attention
  // 主要是为了解决长序列的问题，把query分成tile，交给多个线程处理，然后再合并结果
  #set text(size: 16pt)
  *GPU*: One kind of computation done on the input data at a time in sequence

  *Fusing*: Fusing multiple layers together during the actual computation can enable minimizing the data access by GPUs.

  FlashAttention#footnote_link("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "https://arxiv.org/abs/2205.14135") uses *tiling* to fully compute and write out a small part of the final matrix at once

][
  #figure(image("png/flashAttention.png"))
]

#slide[
  = KV Cache Offloading
  #set text(size: 14pt)
  The KV Cache Offloading technique moves the KV cache from the GPU to the CPU to free up GPU memory for other tasks.
  \
  \

  #figure(image("png/memory_architecture.png", width: 50%))
][
]

#slide[
  = Real-World System: Mooncake
  #figure(image("png/pd.png", width: 50%))
]



== Batch Processing

#slide(composer: 2)[#grid.cell(colspan: 2)[

    #set text(size: 12pt)
    // https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
    *Static batching*: Client packs multiple prompts into requests and a response is returned after all sequences in the batch have been completed. Our inference servers support this but do not require it.

    *Dynamic batching*: Prompts are batched together on the fly inside the server. Typically, this method performs worse than static batching but can get close to optimal if responses are short or of uniform length. Does not work well when requests have different parameters.

    *Continous Batching*: A batch that is continuously processed, leveraging the opportunity by batching new requests once some old requests are finished
  ]][
  #figure(image("png/batch.png", width: 70%))
][
  #figure(image("png/continuous_batch.png", width: 70%))
  // Fairness in Serving Large Language Models
]

== Parallel Processing
#slide[
  #set text(size: 14pt)
  = Pipeline Parallelism

  PP involves sharding the model (vertically) into chunks, where each chunk comprises a subset of layers that is executed on a separate device.
][
  #set text(size: 14pt)
  = Tensor Parallelism

  TP involves sharding the model (horizontally) into chunks, where each chunk comprises a subset of the model's parameters.
][
  #set text(size: 14pt)
  = Sequence Parallelism

  SP involves sharding the input sequence into chunks, where each chunk is processed by a separate device.
]

== Speculative Inference

#slide[
  #set text(size: 14pt)
  = Standard inference
  Sequence generation is strictly sequential. Each token must be generated based on the previously generated token, which leads to high latency, especially for long-sequence tasks.


][
  #set text(size: 14pt)
  == Speculative inference#footnote_link("Blockwise Parallel Decoding for Deep Autoregressive Models", "https://arxiv.org/abs/1811.03115")
  - *Predict multiple tokens ahead*: When generating the first token, the model simultaneously makes speculative predictions about the next several tokens.
  - *Parallel processing*: These speculative predictions allow the model to process multiple possible outcomes in parallel, speeding up the inference.
  - *Validate predicted paths*: If the speculative predictions are correct, the model can continue with these results, avoiding the need to recalculate. If the predictions are incorrect, the model adjusts and corrects the path.
]

#slide[
  #set text(size: 12pt)
  Algorithm#footnote_link("Accelerating Large Language Model Decoding with Speculative Sampling, 2023", "https://arxiv.org/abs/2302.01318")

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

// #slide[
//   = Medusa
//   #set text(size: 14pt)
//   Medusa is a system that uses *speculative inference* to generate multiple tokens in parallel. It uses a *speculative model* to predict multiple tokens ahead and then validates the predicted paths to avoid redundant calculations.
// ]

== Summary
#slide[
  = Real-World System: Mooncake
  #set text(size: 14pt)
  Gray: Control plane
  #figure(image("png/mooncake.png", width: 50%))
]


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

#slide(composer: (5fr, 3fr))[
  #set text(size: 10pt)

  #let half-red = red.transparentize(50%)
  #let pos = block(fill: green.transparentize(50%))[+]
  #let neg = block(fill: red.transparentize(50%))[-]
  #let que = block(fill: gray.transparentize(50%))[?]

  #tablex(
    columns: 8,
    align: center + horizon,
    auto-vlines: false,
    // repeat-header: true,
    vlinex(x: 5),

    /* --- header --- */
    rowspanx(2)[*Category*], rowspanx(2)[*Optimization*], colspanx(3)[*GPU Resources*], colspanx(3)[*Optimization Goal*],

    [*Compute*],
    [*Memory*],
    [*Transmission*],
    [*Throughput*],
    [*TTFT*],
    [*TBT*],

    /* -------------- */
      rowspanx(5)[*Memory*], [Paging], neg, pos, [], pos, [], [],
    (), [Prefix Caching], [], pos, [], pos, [], [],
    (), [Disk Offloading], [], pos, neg, pos, [], [],
    (), [Multi-Query Attention], [], pos, [], pos, pos, pos,
    (), [Group-Query Attention], [], pos, [], pos, pos, pos,

    rowspanx(4)[*Tranmission*], [Duplication], pos, neg, pos, pos, pos, pos,
    (), [Pulling], pos, neg, pos, pos, pos, pos,
    (), [Request Migration], pos, pos, neg, pos, pos, pos,
    (), [Disaggregated Arch], pos, pos, pos, pos, neg, neg,

    rowspanx(3)[*Batch*], [Iteration-Level Batch], pos, [], pos, pos, neg, neg,
    (), [Chunked Prefill], pos, [], [], pos, pos, pos,
    (), [Prepack Prefill], pos, [], [], pos, neg, [],

    rowspanx(4)[*Parallelism*], [Pipeline Parallelism], pos, [], neg, pos, neg, que,
    (), [Tensor Parallelism], pos, [], neg, pos, neg, pos,
    (), [Sequence Parallelism], pos, [], neg, pos, pos, que,
    (), [Speculative Inference], pos, neg, [], pos, [], pos,

    // rowspanx(5)[*Scheduling*], [Priority-Based], [], pos,
    // (), [Request-Level Prediction], [], pos,
    // (), [Machine-level Scheduler], [], pos,
    // (), [Instance Flip], [], pos,
    // (), [Global Profiling], [], pos,
    )
][

]

#slide[
  = Trends
  #set text(size: 14pt)

  #tablex(
    columns: 4,
    align: center + horizon,

    auto-vlines: false,
    //
    /* --- header --- */
    [*Category*],
    [*Trend*],
    [*Examples*],
    [*Conflict*],

    /* -------------- */

    [*Memory*], [Enhanced memory management with finer granularity\ Improve reusability of KV Cache], [Paging\ Token-Level Optimization], [*S*],
    [*Transmission*], [Minimizing transmission latency], [Data Duplication\ Prefetching\ PD Disaggregation], [*T*],
    [*Scheduling*], [Customized scheduling for specific scenarios\ Cache-aware scheduler], [Request-level Predictions\ Machine-Level Scheduling\ Global profiling], [*STP*],
    [*Parallelism*], [Optimizing parallelism for resource reuse and efficiency], [Pipeline Parallelism\ Tensor Parallelism\ Sequence Parallelism\ Speculative Inference], [*ST*],
  )
  #[
    #set text(fill: luma(30%), size: 12pt)
    S: Stateless computation
    E: Enforceable guarantees
    T: Non-targetability
    P: No privileged runtime access
    V: Verifiable transparency
  ]
]

= Threats

== Stateless Computation

#slide[
  #set text(size: 14pt)
  = Requirement
  The system does not maintain any state between requests. Each request is processed independently, and the system does not store any information about previous requests.

  = Attacker's Capabilities
  - *Weak*: An attacker gains access to the system's storage mechanism, potentially compromising databases or disk storage. They can also query the system to infer the presence of sensitive information.
  - *Strong*: An attacker gains control over specific nodes, allowing them to request or intercept data within the system. However, they are unable to directly access the original prompt due to model sharding or other security measures.

  = Overview of Threats
  #set text(size: 12pt)
  #tablex(

    columns: 7,
    align: center + horizon,

    auto-vlines: false,
    //
    /* --- header --- */
    [*Capabilities*],
    [*Goal*],
    [*Query*],
    [*Access to Storage*],
    [*Access to Specific Nodes*],
    [*Control over Node*],
    [*Access to Prompt*],

    /* -------------- */
    [Weak], [Reconstructing User Inputs and Contexts],
    cm, cm, na, na, na,

    [Strong], [Reconstructing User Inputs and Contexts],
    cm, cm, cm, cm, na,
  )
]

#slide(composer: (2fr, 1fr))[
  #set text(size: 16pt)
  = Prompt \& KV Cache
  While the KV cache is related to the input prompt, it is not possible to directly infer the original prompt from it due to the complex and non-reversible nature of the transformations involved in generating the cache.

  = Precompute KV Cache
  Precompute the KV cache for a set of known sensitive prefixes. For instance, if the system is used for medical queries, precompute the KV cache for common medical terms.

][
  #set text(size: 14pt)
  = Input

  $X=[x_1,x_2,dots,x_n].$

  = Embedded Input

  $E &= "Embed"(X)+"Positional Encoding"(X) \
    &=[E_1,E_2,dots,E_n].$

  = K Cache

  $K = [E_1 W^k, E_2 W^k, dots, E_n W^k].$

  = V Cache

  $V = [E_1 W^v, E_2 W^v, dots, E_n W^v].$

]

#slide(composer: 2)[#grid.cell(colspan: 2)[= Weak Attacker]][
  #set text(size: 14pt)
  = Inference Attack
  An attacker could analyze cached data to infer patterns or user behaviors. Even without full query access, understanding what prefixes are frequently cached might reveal the types of queries being made to the system.

  Example: If the cache contains *frequent prefixes related to medical inquiries*, the attacker could infer that the LLM is being used in a healthcare context.
][
  #set text(size: 14pt)
  = Replay Attack
  Attacker can craft new queries that match existing keys in the cache. This allows them to replay or trigger cached computations, potentially extracting sensitive information based on model completions.

  Example: If the key in the KV cache is “My bank account number is...”, the attacker can craft similar queries to attempt to elicit a continuation that reveals more about the original input.
]

#slide(composer: 3)[#grid.cell(colspan: 3)[= Medium Attacker]][
  #set text(size: 14pt)
  = Targeted Data Extraction
  An attacker controlling a node could craft specific inputs designed to trigger the retrieval of cached prefixes. By doing this repeatedly, they can extract sensitive information from the cache based on the model's responses.

  Example: By submitting inputs like “My Social Security Number is...” and monitoring responses, they could *infer whether such a prefix was previously cached and what context or continuation it triggers*.
][
  #set text(size: 14pt)
  = Cache Content Mapping

  The attacker maps key tensors to actual input tokens using controlled nodes, thereby obtaining the full sequence of user inputs. The attacker can reconstruct the complete input sequence for high-frequency queries or commonly used phrases, directly exposing user input such as personal queries or confidential data.
][
]

#slide(composer: 2)[#grid.cell(colspan: 2)[= Strong Attacker]][
  #set text(size: 14pt)
  = Cache Poisoning
  The attacker can inject or modify state data to poison the cache. This poisoned state could then be used in future inferences, leading to persistent generation of incorrect or biased outputs.

  Example: If the attacker injects state data implying that “product X has a defect,” future queries about this product could lead to responses based on this false information.
][
  #set text(size: 14pt)
  = Denial of Service
  By controlling multiple nodes and flooding the system with crafted requests that target caching mechanisms, the attacker could overwhelm the cache, evicting legitimate prefixes and slowing down or disrupting normal operations.
]



#slide[
  #set text(size: 14pt)
  I: Inference Attack, R: Replay Attack, E: Targeted Data Extraction, M: Cache Content Mapping
  #let half-red = red.transparentize(50%)

  #tablex(
    columns: 8,
    align: center + horizon,
    vlinex(x: 6),


    auto-vlines: false,
    // repeat-header: true,
    // vlinex(x: 5, stroke: gray),

    /* --- header --- */
    rowspanx(2)[*Optimization*],
    rowspanx(2)[*Stored States*],
    rowspanx(2)[*Location*],
    rowspanx(2)[*Mitigation*],
    colspanx(2)[*Weak\ Attacker*],
    colspanx(2)[*Middle\ Attacker*],

    [I], [R], [E], [M],
    /* -------------- */

    [Prefix Caching], [KV Cache], [GPU Memory\ CPU Memory], [Cache Expiry\ Isolation],
    [], [], [], [],

    [Disk Offloading], [KV Cache], [Disk Storage\ (SSD, Hard Drive)], [Encryption],
    [], [], [], [],

    [Pulling], [KV Cache], [GPU Memory\ CPU Memory], [Randomized Scheduler],
    [], [], [], [],

    [Database-based\ Speculative Inference], [Token], [GPU Memory\ CPU Memory], [Differential Priavacy],
    [], [], [], [],
  )
]


== Non-Targetability

#slide[
  #set text(size: 14pt)
  *Non-Targetability*: An attacker should not be able to attempt to compromise personal data that belongs to specific, targeted Private Cloud Compute users without attempting a broad compromise of the entire PCC system.

  *Definition*: Let $S={S_1, S_2, dots, S_n}$ denote the set of all servers in the system, with the capability of each server $S_i$ represented by $C(S_i)$.
  The set of requests handled by these servers is denoted as $R(S) = {R(S_1), R(S_2), dots, R(S_n)}$.
  The system is considered non-targetable if, for any subset $T = {T_1, T_2, dots, T_m} subset.eq S$ of servers, the probability of compromising the data of a specific user $u$ is given by:

  $ P(u in R(T)) = frac(sum_(i=1)^m C(T_i), sum_(i=1)^n C(S_i)) $

  *Violations*:
  Duplication

  Pulling

  Priority-Based Scheduling

  Request-Level Prediction

  Machine-level Scheduler
]

== No Privileged Runtime Access

#slide[
  #set text(size: 16pt)
  *No Privileged Runtime Access*: The system must not contain privileged interfaces that would enable Apple's site reliability staff to bypass PCC privacy

  *Violations*:
  Global Profiling
]

== Enforceable Guarantees

#slide[
  #set text(size: 16pt)
  *Enforceable Guarantees*: The system must provide guarantees that can be enforced by the system itself. These guarantees must be technically enforceable and not rely on external components or human intervention.

  *Violations*:
  Prefix Caching,

  Disk Offloading,

  Pulling,

  Machine-Level Scheduler
]

== Verifiable Transparency

#slide[
  #set text(size: 16pt)
  *Verifiable Transparency*: Security researchers must be able to verify the security and privacy guarantees of Private Cloud Compute, and they must be able to verify that the software that's running in the PCC production environment is the same as the software they inspected when verifying the guarantees.

  *Violations*:
  Non open-source systems
]

= Summary
== Academic Systems
#slide[
  // #[
  //   #set text(size: 7pt)
  //   *S*: Stateless computation
  //   *E*: Enforceable guarantees
  //   *P*: No privileged runtime access
  //   *T*: Non-targetability
  //   *V*: Verifiable transparency
  // ]

  #[
    #let model_header(name, year, url) = {
      let size = 5pt
      set text(size: size)
      link(url)[*#name*\ ]
      [#year]
    }

    #set text(size: 7pt)
    #let f1 = "Initial"

    #tablex(
    columns: 23,
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
    model_header("Andes", 2405, "https://arxiv.org/abs/2404.16283"),
    model_header("Llumnix", 2406, "https://arxiv.org/abs/2406.03243"),
    model_header("Preble", 2407, "https://arxiv.org/abs/2407.00023"),
    model_header("TokenRecycling", 2408, "https://www.arxiv.org/abs/2408.08696"),

    /* -------------- */
    rowspanx(3)[*Memory*], [Paging], na,
    [], [], [], [], f1, [], [], cg, [], cg, [], [], [], cg, [], [], [], [], [], [],
    (), [Prefix Caching], [*SE*],
    [], [], [], [], [], [], [], cr, [], [], [], [], [], [], [], [], [], [], cr, [],
    (), [Disk Offloading], [*SE*],
    [], cr, [], [], [], [], [], [], [], [], [], [], [], [], cr, [], [], [], [], [],

    rowspanx(4)[*Tranmission*], [Duplication], [*T*],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Pulling], [*SET*],
    [], [], [], [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [], [], [],
    (), [Request Migration], na,
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cg, [], cg, [], [],
    (), [Disaggregated Arch], na,
    [], [], [], [], [], [], cg, [], [], [], [], cg, [], cg, [], [], [], [], [], [],

    rowspanx(3)[*Batch*], [Iteration-Level Batch], na,
    f1, [], cg, cg, cg, [], [], [], [], cg, [], cg, [], cg, [], [], [], [], [], [],
    (), [Chunked Prefill], na,
    [], [], [], [], [], [], [], [], [], f1, [], [], [], cg, [], [], [], [], cg, [],
    (), [Prepack Prefill], na,
    [], [], [], [], [], [], [], [], [], [], [], cg, [], cg, [], [], [], [], [], [],

    rowspanx(5)[*Parallelism*], [Speculation], na,
    [], [], [], cg, [], cg, [], cg, cg, [], [], [], cg, [], [], [], [], [], [], [],
    (), [Context-Based Speculation], [*S*],
    [], [], [], [], [], cr, [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Database-Based Speculation], [*S*],
    [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [], [], [], [], [], cr,
    (), [Tensor Parallelism], na,
    [], [], [], cg, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    (), [Sequence Parallelism], na,
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cg, [], [], [], [],

    rowspanx(5)[*Scheduling*], [Priority-Based], [*T*],
    [], [], cr, [], [], [], [], cr, [], cr, [], [], [], cr, [], [], cr, cr, cr, [],
    (), [Request-Level Prediction], [*T*],
    [], [], cr, cr, [], [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [],
    (), [Machine-level Scheduler], [*ET*],
    [], [], cr, [], [], [], cr, [], [], [], cr, [], [], cr, [], cr, [], [], cr, [],
    (), [Instance Flip], na,
    [], [], [], [], [], [], cg, [], [], [], [], [], [], cg, [], [], [], [], [], [],
    (), [Global Profiling], [*P*],
    [], cr, [], [], [], [], cr, [], [], [], [], cr, [], [], [], [], [], [], [], [],

    [*Verification*], [Non Open-Source], [*V*],
    [], [], [], [], [], [], [], [], [], [], [], [], [], cr, [], [], [], [], [], cr,
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
   [*Title*], [*Keywords*], [*Contributions*],
    /* -------------- */

    [*Prompt Cache*], [Prefill, Memory], [Reuse attention states across different LLM prompts. Parse the prompt and use reusable text segments(snippet)],
    [*Layer-wise Transmission*], [Transmission], [Transmit each layer's output to the next layer in the pipeline, instead of transmitting the entire model's output],
    [*LightLLM*], [Interface], [Use http as the interface to the system],
    [*SkyPilot*], [Cross Region & Cloud], [Given a job and its resource requirements (CPU/GPU/TPU), SkyPilot automatically figures out which locations (zone/region/cloud) have the compute to run the job, then sends it to the cheapest one to execute],
    [*MLC LLM*], [Efficient Execution], [Enable efficient execution of large language models across a wide range of hardware platforms, including mobile devices, edge devices, and even web browsers],
    [*vAttention*], [Virtual Memory], [stores KV-cache in contiguous
virtual memory and leverages OS support for on-demand
allocation of physical memory],
    [*MemServe*], [API, Framework], [an elastic memory pool API managing distributed memory and KV caches across serving instances],
    [*CacheGen*], [Network, Streaming], [CacheGen uses a custom tensor encoder, leveraging KV cache's
distributional properties to encode a KV cache into more compact
bitstream representations],
    [*DynamoLLM*], [Energy], [It exploits heterogeneity in inference compute properties and fluctuations in inference workloads to save energy],
    )
]

// DL Serving: AlpaServe, Pollux

// Attention Serving: AttMemo, Ring Attention

// LLM Serving Fairness: VTC

// GPU Communication Lantencies: Flux(TP)
== Industrial Systems
#slide[
  // #[
  //   #set text(size: 7pt)
  //   *S*: Stateless computation
  //   *E*: Enforceable guarantees
  //   *P*: No privileged runtime access
  //   *T*: Non-targetability
  //   *V*: Verifiable transparency
  // ]

  #[
    #let model_header(name, year) = {
      let size = 5pt
      set text(size: size)
      [*#name*\ ]
      [#year]
    }

    #set text(size: 6pt)
    #let f1 = "Initial"

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

    rowspanx(3)[*Batch*], [Iteration-Level Batch], na,
    cm, [], cm, [], cm, cm, cm, cm, cm, cm, cm, cm, [],
    (), [Chunked Prefill], na,
    cm, [], [], [], cm, cm, [], [], [], [], [], [], [],
    (), [Prepack Prefill], na,
    [], [], [], [], [], [], [], [], [], [], [], [], [],

    rowspanx(3)[*Parallelism*], [Speculation], [*S*],
    cm, [], cm, cm, [], [], cm, cm, [], [], [], cm, cm,
    (), [Tensor Parallelism], na,
    [], [], [], [], [], [], [], cm, [], [], [], [], cm,
    (), [Sequence Parallelism], na,
    [], [], [], [], [], [], [], [], [], [], [], [], [],

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

    [*Verification*], [Non Open-Source], [*V*],
    [], [], [], [], cm, [], cm, [], [], [], [], [], [],
    )
  ]

  // https://github.com/vllm-project/vllm/issues/4104 相对于只缓存Prefix Cache，vLLM的Prefix Caching功能还缓存了Generated KV Cache
]

#slide[
  #set text(size: 10pt)
  The roadmap of the vLLM project includes the following features:

  #tablex(
    columns: 8,
    align: center + horizon,
    auto-vlines: false,
    // repeat-header: true,

    /* --- header --- */
    [*Version*],
    [*Date*],
    [*Memory*],
    [*Transmission*],
    [*Batch*],
    [*Parallelism*],
    [*Scheduling*],
    [*Model*],

    /* -------------- */
    [v0.1], 
    [2306], 
    [Paging], 
    na, 
    [Continuous Batching], 
    na, 
    na,
    [MQA, GQA],

    [v0.2],
    [2309],
    [],
    [],
    [],
    [Better TP \& EP Support],
    [],
    [AWQ],

    [v0.3],
    [2401],
    [Prefix Caching],
    [],
    [],
    [],
    [],
    [GPTQ],

    [v0.4],
    [2404],
    [],
    [Optimize Distributed Communication],
    [Chucked Prefill],
    [Speculative Inference],
    [],
    [],

    [v0.5],
    [2407],
    [CPU Offloading],
    [],
    [],
    [Support PP],
    [Schedule multiple GPU steps in advances],
    [FP8],

    [v0.6],
    [2409],
    [],
    [],
    [],
    [],
    [Asynchronous output processor],
    [],
  )
]


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
#slide[
  #set text(size: 12pt)
  = Intro
  https://github.com/DefTruth/Awesome-LLM-Inference

  https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

  = Parallelism

  https://developer.nvidia.com/blog/demystifying-ai-inference-deployments-for-trillion-parameter-large-language-models/

  = Utilities
  https://github.com/Trusted-AI/adversarial-robustness-toolbox
]

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
//
// 分布式的图
// 趋势，结论