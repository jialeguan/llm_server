#import "@preview/touying:0.5.2": *
#import themes.dewdrop: *
#import "tables/pcc_table.typ": *
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

// LLM Inference
= Private Cloud Compute
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

== Taxonomy

#slide[
  #set text(size: 12pt)
  #taxonmy_table
]

= LLM Serving Systems
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
  #set text(size: 16pt)
  = Workload Heterogeneity
  Universality and application diversity lead to heterogeneity of the inference requests, in terms of input lengths, output lengths, expected latencies, etc
  - Queuing Delays, Preemptions, Interference
  = Execution Unpredictability
  Unknown a priori how many tokens will be generated before the stopping criteria is met. As such, the execution time and the resource demand of a request are both unpredictable.
  = Multi-Tenant and Dynamic Environment
  The system must scale to support multiple users and adapt to the dynamic nature of the environment.
]

= Optimizations

== Memory Management

#slide[
  = KV Cache
  #set text(size: 12pt)
  #figure(image("png/kvcache_final.gif", height: 50%))

  Transformers use attention mechanisms that compute attention scores between tokens. The KV Cache helps by storing previously computed key-value pairs, allowing the model to quickly access and reuse them for new tokens, avoiding redundant calculations.

  // Q（查询）和 K（键）做矩阵乘法得到的是一个相关性得分矩阵，表示查询与键之间的相似度。这些得分反映了查询对每个键的关注程度，通常会通过 softmax 函数进行归一化，以生成权重分布，随后用于加权求和值（V）以生成最终的输出。这个过程使模型能够聚焦于与当前查询最相关的信息。还有什么具体的内容你想了解吗？
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
  = Prefix Caching
  #set text(size: 14pt)
  Prefix Caching#footnote_link("ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition. 2024", "https://arxiv.org/abs/2402.15220") is a technique that caches the intermediate states of the model during the prefill phase. These states are then reused during the decode phase to speed up inference.

  #figure(
    image("png/prefix_caching.png", width: 60%),
  )
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
  // #set text(size: 16pt)
  = Data Parallelism (DP)

  Each device processes a different batch of data. This allows multiple devices to work on independent data batches in parallel, improving throughput.

  = Sequence Parallelism (SP)

  The input sequence is divided into chunks, with each chunk processed by a separate device. This enables parallel processing of sequence segments across devices.
][
  = Pipeline Parallelism (PP)

  The model is split vertically into chunks, where each chunk contains a subset of layers. Each device handles a different set of layers, allowing different stages of the model to be executed in parallel across devices.

  = Tensor Parallelism (TP)

  The model's parameters are sharded horizontally into chunks, with each chunk distributed across devices. This allows the computation within each layer to be split and processed in parallel.
]

== Speculative Inference

#slide[
  #set text(size: 16pt)
  = Standard inference
  Sequence generation is strictly sequential. Each token must be generated based on the previously generated token, which leads to high latency, especially for long-sequence tasks.


][
  #set text(size: 16pt)
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

#slide(composer: (5fr, 3fr))[
  #set text(size: 10pt)
  #opt_goal_table
][

]

#slide[
  = Trends
  #set text(size: 14pt)
  #trend_table
  #pcc_req
]

= Threats
== Stateless Computation

#slide[
  #set text(size: 16pt)
  = Requirement
  The system does not maintain any state between requests and the system does not store any information about previous requests.

  = Attacker's Capabilities
  - *Weak*: An attacker gains access to the system's storage mechanism, potentially compromising databases or disk storage. They can also query the system to infer the presence of sensitive information.
  - *Strong*: An attacker gains control over specific nodes, allowing them to request or intercept data within the system. However, they are unable to directly access the original prompt due to model sharding or other security measures.

  = Overview of Attacker's Capabilities
  #set text(size: 12pt)
  #s_attacker_table
]

#slide(composer: (2fr, 1fr))[
  #set text(size: 16pt)
  = Prompt \& KV Cache
  While the KV cache is related to the input prompt, it is not possible to directly infer the original prompt from it due to the complex and non-reversible nature of the transformations involved in generating the cache.

  = Precompute KV Cache
  Precompute the KV cache for a set of known sensitive prefixes. For instance, if the system is used for medical queries, precompute the KV cache for common medical terms.

  = State
  In LLM inference, once KVs are computed for a sequence, their values do not change.
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
  #set text(size: 16pt)
  = Inference Attack
  An attacker could analyze cached data to infer patterns or user behaviors. Even without full query access, understanding what prefixes are frequently cached might reveal the types of queries being made to the system.

  Example: If the cache contains *frequent prefixes related to medical inquiries*, the attacker could infer that the LLM is being used in a healthcare context.
][
  #set text(size: 16pt)
  = Replay Attack
  Attacker can craft new queries that match existing keys in the cache. This allows them to replay or trigger cached computations, potentially extracting sensitive information based on model completions.

  Example: If the key in the KV cache is “My bank account number is...”, the attacker can craft similar queries to attempt to elicit a continuation that reveals more about the original input.
]

#slide(composer: 3)[#grid.cell(colspan: 3)[= Strong Attacker]][
  #set text(size: 16pt)
  = Targeted Data Extraction
  An attacker controlling a node could craft specific inputs designed to trigger the retrieval of cached prefixes. By doing this repeatedly, they can extract sensitive information from the cache based on the model's responses.

  Example: By submitting inputs like “My Social Security Number is...” and monitoring responses, they could *infer whether such a prefix was previously cached and what context or continuation it triggers*.
][
  #set text(size: 16pt)
  = Cache Content Mapping

  The attacker maps key tensors to actual input tokens using controlled nodes, thereby obtaining the full sequence of user inputs. The attacker can reconstruct the complete input sequence for high-frequency queries or commonly used phrases, directly exposing user input such as personal queries or confidential data.
][
]

// #slide(composer: 2)[#grid.cell(colspan: 2)[= Strong Attacker]][
//   #set text(size: 14pt)
//   = Cache Poisoning
//   The attacker can inject or modify state data to poison the cache. This poisoned state could then be used in future inferences, leading to persistent generation of incorrect or biased outputs.

//   Example: If the attacker injects state data implying that “product X has a defect,” future queries about this product could lead to responses based on this false information.
// ][
//   #set text(size: 14pt)
//   = Denial of Service
//   By controlling multiple nodes and flooding the system with crafted requests that target caching mechanisms, the attacker could overwhelm the cache, evicting legitimate prefixes and slowing down or disrupting normal operations.
// ]



#slide[
  #set text(size: 14pt)
  #s_attack_table
]


== Non-Targetability

#slide[
  #set text(size: 18pt)
  *Non-Targetability*: An attacker should not be able to attempt to compromise personal data that belongs to specific, targeted Private Cloud Compute users without attempting a broad compromise of the entire PCC system.

  *Definition*: Let $S={S_1, S_2, dots, S_n}$ denote the set of all servers in the system, with the capability of each server $S_i$ represented by $C(S_i)$.
  The set of requests handled by these servers is denoted as $R(S) = {R(S_1), R(S_2), dots, R(S_n)}$.
  The system is considered non-targetable if, for any subset $T = {T_1, T_2, dots, T_m} subset.eq S$ of servers, the probability of compromising the data of a specific user $u$ is given by:

  $ P(u in R(T)) = frac(sum_(i=1)^m C(T_i), sum_(i=1)^n C(S_i)) $

  // = Attacker's Capabilities

  // An attacker gains control over specific nodes, allowing them to request or intercept data within the system.

  // - *Weak*: Each node only has access to the metadata of the requests it processes. (length, time, etc.)
  // - *Strong*: Each node has access to the full prompt of the requests it processes.
]

#slide[
  #set text(size: 14pt)
  $ P(u in R(T)) = frac(sum_(i=1)^m C(T_i), sum_(i=1)^n C(S_i)) $

  = Goal
  - Increase Hit Count: $sum C(T_i)$
  - Decrease Miss Count: $sum C(S_i) - sum C(T_i)$

  #t_attack_table
]

== Ecosystem

#slide[
  #set text(size: 16pt)
  = No Privileged Runtime Access

  *Definition*

  The system must not contain privileged interfaces that would enable site reliability staff to bypass PCC privacy.

  *Violations*
  - Global Profiling
][

  #set text(size: 16pt)
  = Enforceable Guarantees
  *Definition*

  The system must provide guarantees that can be enforced by the system itself. These guarantees must be technically enforceable and not rely on external components or human intervention.

  *Violations*
  - Disk Offloading

  - Pulling

  - Machine-level Scheduler
][
  #set text(size: 16pt)
  = Verifiable Transparency
  *Definition*

  Security researchers must be able to verify the security and privacy guarantees of Private Cloud Compute, and they must be able to verify that the software that's running in the PCC production environment is the same as the software they inspected when verifying the guarantees.

  *Violations*
  - Non open-source systems
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
    #set text(size: 6pt)
    #academic_table
  ]

  = Miscellaneous
  #set text(size: 10pt)
  #misc_table
]

== Industrial Systems
#slide[
  #[
    #set text(size: 6pt)
    #industrial_table
  ]

  // https://github.com/vllm-project/vllm/issues/4104 相对于只缓存Prefix Cache，vLLM的Prefix Caching功能还缓存了Generated KV Cache
]

#slide[
  #set text(size: 14pt)
  The roadmap of the vLLM project includes the following features:
  #set text(size: 10pt)
  #vllm_table
]

#slide[
  = Trends
  #set text(size: 14pt)
  #trend_table

  #pcc_req
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

#focus-slide[
  Thanks
]

#show: appendix
= Appendix
#slide[
  #set text(size: 16pt)
  = Attention
  https://zh.d2l.ai/chapter_attention-mechanisms/attention-cues.html
  
  = Inference Optimization
  https://github.com/DefTruth/Awesome-LLM-Inference

  https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

  = Parallelism

  https://developer.nvidia.com/blog/demystifying-ai-inference-deployments-for-trillion-parameter-large-language-models/

  = Utilities
  https://github.com/Trusted-AI/adversarial-robustness-toolbox
]

#slide[
  = Confidential Computing on nVIDIA H100 GPU: A Performance Benchmark Study

  #figure(image("png/tee_throughput.png", width: 70%))
]

#slide[
  = Confidential Computing on nVIDIA H100 GPU: A Performance Benchmark Study

  #figure(image("png/tee_prefill.png", width: 60%))
  #figure(image("png/tee_decode.png", width: 60%))
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
