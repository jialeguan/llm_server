#import "@preview/touying:0.5.2": *
#import themes.dewdrop: *
#import "tables/report_table.typ": *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge
#import "@preview/tablex:0.0.8": tablex, hlinex, vlinex, colspanx, rowspanx

#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#show: dewdrop-theme.with(
  aspect-ratio: "16-9",
  footer: "PCC",
  navigation: "mini-slides",
  config-info(
    title: [隐私云计算视角下的大模型推理优化],
    subtitle: [],
    author: [管佳乐],
    date: datetime.today(),
    // institution: [Institution],
  ),
  mini-slides: (height: 3em, x: 1em, display-section: false, display-subsection: true, short-heading: true),
)

#set text(font: "San Francisco", weight: "light", size: 20pt)

// #set heading(numbering: numbly("{1}.", default: "1.1"))

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

= 大模型推理
== 服务系统
#let arch_arg(arg1, arg2, arg3, arg4) = fletcher-diagram(
  let user = (0, 0),
  let scheduler = (3, 0),
  let server = (6, 0),

  node(user, [*客户端*]),
  node(scheduler, [*服务端*]),
  node(server, [*工作节点*]),

  edge(user, scheduler, "-|>", [#arg1], shift: 3pt, label-side: left),
  edge(scheduler, user, "-|>", [#arg2], shift: 3pt, label-side: left),
  edge(scheduler, server, "-|>", [#arg3], shift: 3pt, label-side: left),
  edge(server, scheduler, "-|>", [#arg4], shift: 3pt, label-side: left),
)


#let arch = fletcher-diagram(
  let main_row = 1,
  let upper_row = 0,
  let lower_row = 2,
  let left_col = 0,
  let middle_col = 3,
  let right_col = 6,

  let user = (left_col, main_row),
  let scheduler = (middle_col, main_row),
  let worker_a = (right_col, upper_row),
  let worker_b = (right_col, lower_row),

  node-corner-radius: 4pt,
  node(user, [*客户端*]),
  node(scheduler, [*服务端*]),
  node(worker_a, [*预填充节点*]),
  node(worker_b, [*解码节点*]),

  edge(user, scheduler, "-|>", [`enc(input)`], shift: 3pt, label-side: left),
  edge(scheduler, user, "-|>", [`enc(output)`], shift: 3pt, label-side: left),
  edge(scheduler, worker_a, "-|>", [`input`], shift: 3pt, label-side: left),
  edge(worker_b, scheduler, "-|>", [`output`], shift: 3pt, label-side: left),
  edge(worker_a, worker_b, "-|>", [`KV Cache` (Intermediate Data)], shift: 3pt, label-side: left),
)

#slide[
  = 传统的服务器架构
  // 端到端加密
  #arch_arg(`enc(input)`, `enc(output)`, `enc(input)`, `enc(output)`)

  #pause
  // 匿名化处理
  #arch_arg(`san(input)`, `san(output)`, `san(input)`, `san(output)`)

  #pause
  = 大模型的服务器架构
  #arch_arg(`enc(input)`, `enc(output)`, `input`, `output`)
]

#slide[
  #arch_arg(`enc(input)`, `enc(output)`, `input`, `output`)

  #pause
  #arch
]


== 阶段
#let prefill_graph = fletcher-diagram(
  node-stroke: .1em,
  spacing: 4em,
  node((0, 0), `Is`, corner-radius: 2em),
  node((1, 0), `tomato`, corner-radius: 2em),
  node((2, 0), `a`, corner-radius: 2em),
  node((3, 0), `fruit`, corner-radius: 2em),
  node((4, 0), `?`, corner-radius: 2em),

  edge((0, 0), (1, 0), "<|-", bend: 45deg),
  edge((1, 0), (2, 0), "<|-", bend: 45deg),
  edge((2, 0), (3, 0), "<|-", bend: 45deg),
  edge((3, 0), (4, 0), "<|-", bend: 45deg),
  edge((0, 0), (2, 0), "<|-", bend: 45deg),
  edge((1, 0), (3, 0), "<|-", bend: 45deg),
  edge((2, 0), (4, 0), "<|-", bend: 45deg),
  edge((0, 0), (3, 0), "<|-", bend: 45deg),
  edge((1, 0), (4, 0), "<|-", bend: 45deg),
  edge((0, 0), (4, 0), "<|-", bend: 45deg),
)


#slide[
  = 预填充: Processing the input
  #set text(size: 14pt)
  在预填充阶段, LLM 处理输入的 token 以计算中间状态 (KV Cache), 这些状态用于生成“第一个”新 token。每个 token 都依赖于之前的所有 token, 但由于输入的完整内容是已知的，从宏观来看，这是一个 Matrix-Matrix 的操作，且高度并行化。这一过程有效地使 GPU 达到最大利用率。
  #figure(image("../png/two_phase.png", width: 90%))
][
  #set text(size: 14pt)

  #pause
  #prefill_graph

  #pause
  #figure(image("../png/qk.png", width: 100%)),
]

#let decode_graph = fletcher-diagram(
  node-stroke: .1em,
  spacing: 4em,
  node((0, 0), `Is`, corner-radius: 2em),
  node((1, 0), `tomato`, corner-radius: 2em),
  node((2, 0), `a`, corner-radius: 2em),
  node((3, 0), `fruit`, corner-radius: 2em),
  node((4, 0), `?`, corner-radius: 2em),
  node((5, 0), `Yes`, corner-radius: 2em),

  edge((0, 0), (5, 0), "<|-", bend: 45deg),
  edge((1, 0), (5, 0), "<|-", bend: 45deg),
  edge((2, 0), (5, 0), "<|-", bend: 45deg),
  edge((3, 0), (5, 0), "<|-", bend: 45deg),
  edge((4, 0), (5, 0), "<|-", bend: 45deg),
)
#slide[
  = 解码: Generating the output
  #set text(size: 14pt)
  在解码阶段，LLM 以自回归的方式逐个生成输出标记，直到满足停止条件。每一个连续的输出标记都需要了解之前所有迭代的输出状态（KV Cache）。这一过程类似于 Matrix-Vector 操作，相较于预填充阶段，GPU 计算能力未被充分利用。延迟主要受数据（权重、键、值、激活）从内存传输到 GPU 的速度影响，而不是计算的速度。换句话说，这是一个受限于内存的操作。
  #figure(image("../png/two_phase.png", width: 90%))
][
  #set text(size: 10pt)
  #pause
  #decode_graph

  #pause
  #figure(image("../png/qk_2.png", width: 100%)),
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

== 优化的挑战
// #slide(composer: 2)[#grid.cell(colspan: 2)[= Weak Attacker]]

#slide[
  = 任务的异质性
  #set text(size: 14pt)
  通用性和应用的多样性导致推理请求的异构性，表现为输入长度、输出长度、期望延迟等方面的差异。

  比如在代码生成任务中，输入长度可能是几十行代码，而在问答任务中，输入长度可能是几个句子。这种异构性使得难以设计一个通用的优化策略，需要根据具体的任务特性进行优化。
][
  = 执行的不可预测性
  #set text(size: 14pt)
  在输出结束之前，无法事先知道将生成多少个 token。因此，请求的执行时间和资源需求都是不可预测的。这种不确定性使得难以为每个请求分配合适的资源，需要动态调整资源分配策略。
][
  = 环境的动态性
  #set text(size: 14pt)
  系统必须能够同时支持多个用户的请求，这要求在资源调度、隔离和优先级控制方面有很强的弹性。推理环境经常变化，模型更新、负载变化等动态因素要求系统具备良好的扩展性和适应性。
]

= 隐私云计算
== 背景

#slide[
  #set text(size: 15pt)

  #arch

  挑战：
  - 很难在实际应用中大规模使用端到端的加密
  - 由于应用和场景的多样性，不能期待所有的数据都是匿名化的

  #pause

  - 在工作节点的处理过程中，可能会产生一些中间状态，这些状态可能包含用户的敏感信息
  - 部分节点的攻陷可能会导致大量用户数据的泄露
]


== 要求

#slide[
  #set text(size: 12pt)
  #taxonmy_table
]

= 无状态计算
== 定义
#slide[
  = 无状态计算

  "Private Cloud Compute must use the personal user data that it receives exclusively for the purpose of fulfilling the user's request. This data must never be available to anyone other than the user, not even to Apple staff, not even during active processing. And *this data must not be retained*, including via logging or for debugging, after the response is returned to the user. In other words, we want a strong form of stateless data processing where *personal data leaves no trace* in the PCC system."

  在请求完成之后，系统不应该保留任何用户数据。这意味着，系统不应该保留用户的输入、输出、中间状态等数据。这样可以避免用户数据的泄露和滥用。
]

== 数据管理

#slide[
  = KV Cache: 空间换时间
  #set text(size: 10pt)
  #decode_graph

  #pause
  #figure(image("../png/qk_2.png", width: 90%)),
][

]

#slide[
  = KV Cache: 空间换时间
  #set text(size: 12pt)
  #figure(image("../png/kvcache_final.gif", height: 50%))

  Transformers use attention mechanisms that compute attention scores between tokens. The KV Cache helps by storing previously computed key-value pairs, allowing the model to quickly access and reuse them for new tokens, avoiding redundant calculations.

  // Q（查询）和 K（键）做矩阵乘法得到的是一个相关性得分矩阵，表示查询与键之间的相似度。这些得分反映了查询对每个键的关注程度，通常会通过 softmax 函数进行归一化，以生成权重分布，随后用于加权求和值（V）以生成最终的输出。这个过程使模型能够聚焦于与当前查询最相关的信息。还有什么具体的内容你想了解吗？
][
  #set text(size: 12pt)
  #figure(image("../png/memory_layout.png", height: 40%))

  \
  \

  Memory layout when serving an LLM with 13B parameters on NVIDIA A100. The parameters (gray) persist in GPU memory throughout serving. The memory for the KV cache (red) is (de)allocated per serving request. A small amount of memory (yellow) is used ephemerally for activation.
]

// #slide[
//   #set text(size: 16pt)
//   LLM inference architecture primarily comprises multiple stacked decoder blocks, each consisting of a self-attention module and a Feed-Forward Neural Network (FFN) module.
//   #figure(image("../png/kvcache_detail.png", width: 80%))
//   // InstInfer
// ]

#slide[
  = PagedAttention：分页管理
  #set text(size: 14pt)
  PagedAttention#footnote_link("Efficient Memory Management for Large Language Model Serving with PagedAttention", "https://arxiv.org/abs/2309.06180") 是一种将注意力矩阵划分为较小页面的技术。这种方法一定程度上解决了内存碎片问题，因此，PagedAttention 已成为大模型推理系统中动态内存分配的事实标准。

  #figure(
    image("../png/paged_attention_waste.png", width: 90%),
  )
]

#slide[
  = Paged Attention
  #set text(size: 12pt)
  #figure(
    image("../png/paged_attention.png", width: 90%),
  )

  // *Pitfalls*#footnote_link("vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention. 2024","https://arxiv.org/abs/2405.04437")
  // - Requires re-writing the attention kernel.
  // - Adds software complexity and redundancy (CPU code), can degrade throughput by 11%.
  // - Introduces performance overhead. 20-26% slower than original FasterTransformer kernel.
  //
  注意到在请求完成后，这些分页数据和相关的中间结果会被释放或清除，不会留下任何可被后续请求利用的状态或痕迹。每个请求的输入和输出是独立的。PagedAttention 仅影响计算的内存管理，不会影响请求的无状态性。
]

#slide(composer: (4fr, 6fr))[
  = Prefix Caching
  #set text(size: 14pt)
  // Prefix Caching#footnote_link("ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition. 2024", "https://arxiv.org/abs/2402.15220") 是一种用于加速大模型推理（特别是语言模型推理）的技术。其主要目的是通过缓存和重用推理过程中已经计算过的部分，减少重复计算，从而提高效率。
  #set text(size: 12pt)
  ```json
  [Instructions]
  You are an AI chatbot. You are having a conversation with a human by following rules:
  - You do not have a name.
  - You are helpful, creative, clever, and friendy.
  [Examples]
  Human: Hello, who are you?
  Al: I am an AI chatbot. How can I help you?

  [Q1]
  Human: Tell me about the second world war.

  [Q2]
  Human: What is the capital of France?

  [Q3]
  Human: What is the weather like in Paris today?
  ```
][
  #pause
  #figure(
    image("../png/prefix_caching_1.png", width: 80%),
  )

  #pause
  #figure(
    image("../png/prefix_caching_2.png", width: 40%),
  )
]

#slide[
  = Prefix Caching & 无状态计算

  #set text(size: 14pt)
  Prefix Caching 通过存储之前生成的键（keys）和值（values）来提高自回归模型的推理效率，但在这一过程中，可能会引入一些状态，这些状态的存在可能会影响无状态计算的原则。

  = 数据泄露

  如果缓存中的数据没有适时更新或删除，过期的信息可能会在意外情况下被泄露。例如，某个用户的敏感查询结果未能及时清除，可能会被后续请求访问到。

  = 信息推断

  由于 Prefix Caching 将历史生成信息与当前查询关联。在推理过程中，如果前缀缓存被重复使用或共享给不同的推理任务，攻击者可能通过比较不同任务的输出，推断出前一个输入的信息。例如，在生成式任务中，攻击者可以通过观察缓存的相似性，来推测用户的部分输入。

  = 访问模式分析

  Prefix Caching 可能导致某些查询或生成过程频繁访问缓存，这种访问模式的可见性可能使攻击者能够分析并推测出缓存内容。通过访问模式，攻击者可能会获取到敏感信息的特征或上下文。
]

// #slide[
//   = Group-Query Attention
//   #set text(size: 14pt)

//   - *Standard Attention*: 分别计算每个查询。复杂度为 $O(n^2)$.
//   - *Multi-Query Attention*: 为多个查询复用相同的注意力矩阵。查询足够相似，可以共享相同的注意力分布。
//   - *Group-Query Attention*#footnote_link("GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "https://arxiv.org/abs/2305.13245"): 更泛用，将查询分成多个组，并分别为每个组计算注意力。
// ][
//   #figure(
//     image("../png/grouped-query.png", width: 90%),
//   )

//   // https://arxiv.org/pdf/2305.13245
// ]


== 传输
#slide[
  = KV Cache Offloading
  #set text(size: 14pt)
  KV cache 保存了每个生成步骤的键（Key）和值（Value），避免重复计算。然而，随着生成的标记数增加，KV cache 会不断变大，最终占用大量的 GPU 内存。

  KV Cache Offloading 通过将这些缓存数据从高性能但有限的 GPU 内存中转移到较为廉价但速度较慢的内存设备，以释放 GPU 内存，供其他计算操作使用。当模型需要访问这些转移出去的 KV 缓存时，再从 CPU 内存或存储中按需重新加载到 GPU 内存进行计算。

  #figure(image("../png/memory_architecture.png", height: 30%))
][
  #pause
  = KV Cache Offloading & 无状态计算
  #set text(size: 14pt)
  = 数据泄露
  在无状态计算中，理想情况下每个请求结束后所有状态都应被清除，而且状态不应该在请求之间进行任何形式的存储或复用。如果 KV Cache 被 offload 到 CPU 或硬盘，可能会导致状态数据被存储在较长期的存储介质上。尤其是当缓存的数据保留在硬盘上时，它在多个请求之间的清除可能会不如内存中的状态那么及时，从而违反无状态计算的要求。

  // TODO: 数据不同步
]

#slide[
  = Data Duplication
  #[
    #set text(size: 14pt)
    在多服务器环境中，KV Cache可能会被复制到不同的节点上，以便在多个节点上并行执行推理任务。

    Data Duplication 由 master 负责，它将 KV Cache 复制到多个节点上，以便在多个节点上并行执行推理任务。
  ]

  #pause
  = Data Duplication & 无状态计算
  #[
    #set text(size: 14pt)

    = 状态残留
    当多个节点拥有 KV Cache 的副本时，执行结束后，由于同步或者过期策略的问题，这些副本可能仍然存在于内存或存储中。如果 KV Cache 中存储了敏感信息（如用户输入、生成的内容等），数据重复可能导致多个节点持有这些敏感数据的副本。在执行结束后，这些副本仍可能被访问或分析，增加了隐私泄露的风险。

    = 差分分析
    如果 KV Cache 的状态在多个节点上存在多份副本，攻击者可能会通过分析这些副本之间的差异，推测出某些用户的输入或敏感信息。
  ]
]

== 推测执行
#slide[
  = Speculative Inference
  #set text(size: 14pt)
  在这种方法中，系统会基于当前的输入和模型状态，提前生成多个可能的输出或结果。模型会对这些结果评估，选择最有可能的结果返回给用户。这种方法可以提高系统的响应速度。

  #figure(
    image("../png/speculative.png", width: 60%),
  )
]

#slide[
  = Speculative Inference & 无状态计算
  #set text(size: 14pt)
  预测性推理本身并不违反无状态计算的原则，因为它只是提前生成可能的结果，而不会保留这些结果。然而，预测性推理可能会导致一些状态残留，例如生成的结果可能会被缓存，这可能会增加隐私泄露的风险。

  = 缓存泄露
  如果生成的结果被缓存，这些结果可能会在执行结束后仍然存在于缓存中，增加了隐私泄露的风险。

  = 信息推断
  预测用的数据库可能会包含敏感信息，攻击者可能会通过分析这些数据库，推测出某些用户的输入或敏感信息。
]

== 状态残留
#slide[
  #set text(size: 14pt)
  #s_attack_table
]

= 不可针对性
== 定义
#slide[
  = 不可针对性
  #set text(size: 16pt)
  An attacker should not be able to attempt to compromise personal data that belongs to specific, targeted Private Cloud Compute users without attempting a broad compromise of the entire PCC system. This must hold true even for exceptionally sophisticated attackers who can attempt physical attacks on PCC nodes in the supply chain or attempt to obtain malicious access to PCC data centers. In other words, a limited PCC compromise must not allow the attacker to *steer requests from specific users to compromised nodes*; targeting users should require a wide attack that's likely to be detected. To understand this more intuitively, contrast it with a traditional cloud service design where every application server is provisioned with database credentials for the entire application database, so a compromise of a single application server is sufficient to access any user's data, even if that user doesn't have any active sessions with the compromised application server.

  #pause
  PCC 要求不能像传统设计中，攻陷一个服务器就可以访问所有用户的数据。如果仅仅攻陷部分节点，攻击者不应该能够访问到特定用户的数据。
]


#slide[
  #set text(size: 16pt)
  // *定义*: Let $S={S_1, S_2, dots, S_n}$ denote the set of all servers in the system, with the capability of each server $S_i$ represented by $C(S_i)$.
  // The set of requests handled by these servers is denoted as $R(S) = {R(S_1), R(S_2), dots, R(S_n)}$.
  // The system is considered non-targetable if, for any subset $T = {T_1, T_2, dots, T_m} subset.eq S$ of servers, the probability of compromising the data of a specific user $u$ is given by:

  设 $S={S_1, S_2, dots, S_n}$ 为系统中所有服务器的集合，每个服务器 $S_i$ 的能力用 $C(S_i)$ 表示。这些服务器处理的请求集合表示为 $R(S) = {R(S_1), R(S_2), dots, R(S_n)}$。如果对于任何服务器子集 $T = {T_1, T_2, dots, T_m} subset.eq S$，攻陷特定用户 $u$ 的数据被攻陷的概率如下所示，则该系统被认为是*不可针对的*:

  $ P(u in R(T)) = frac(sum_(i=1)^m C(T_i), sum_(i=1)^n C(S_i)) $

  #pause
  = 目标

  - 增加 Hit: $sum C(T_i)$

  - 减少 Miss: $sum C(S_i) - sum C(T_i)$
  // = Attacker's Capabilities

  // An attacker gains control over specific nodes, allowing them to request or intercept data within the system.

  // - *Weak*: Each node only has access to the metadata of the requests it processes. (length, time, etc.)
  // - *Strong*: Each node has access to the full prompt of the requests it processes.
]

== 提高针对性
#slide[
  #set text(size: 12pt)
  #t_attack_table
]

= 总结

// == 优化的趋势
// #slide[
//   #set text(size: 14pt)
//   #trend_table
//   #pcc_req
// ]
== 大模型推理优化的发展方向
#slide(composer: (1fr, 1fr))[#grid.cell(colspan: 2)[
    #set text(size: 10pt)
    vLLM 版本的路线图包括以下功能更新：
    #vllm_table
  ]][
  #set text(size: 14pt)
  = 发展方向
  - 支持大规模的并行化和负载均衡
  - 细致的调度和资源复用
  - 自动化和自适应的优化策略
  - 针对不同场景进行定制化的优化策略
][
  #set text(size: 14pt)
  #pause
  = 合规的挑战
  - 需要大量的状态管理来实现并行化和负载均衡
  - 精致的调度可能会引入定向攻击和隐私泄露
  - 自适应的优化策略可能会导致优化用数据被持续泄露
  - 攻击者可以通过推测用户的使用场景来推断用户的隐私信息
]

== 相关的论文
#slide[
  // #[
  //   #set text(size: 7pt)
  //   *S*: Stateless computation
  //   *E*: Enforceable guarantees
  //   *P*: No privileged runtime access
  //   *T*: Non-targetability
  //   *V*: Verifiable transparency
  // ]

  #set text(size: 2pt)
  #pcc_req
  #set text(size: 5pt)
  #academic_table
]

#slide[
  = Misc
  #set text(size: 10pt)
  #misc_table
]

== 工业界的推理系统
#slide[
  #[
    #set text(size: 6pt)
    #industrial_table
  ]

  // https://github.com/vllm-project/vllm/issues/4104 相对于只缓存Prefix Cache，vLLM的Prefix Caching功能还缓存了Generated KV Cache
]



== 大模型推理与可信计算

#slide[
  #set text(size: 14pt)
  = NVIDIA H100 GPU

  #figure(image("../png/h100.png", width: 90%))
][
  #set text(size: 12pt)

  *硬件隔离*：H100 使用硬件级的安全隔离机制，例如通过虚拟化技术或受信执行环境（Trusted Execution Environment, TEE），来确保多个工作负载可以在同一台机器上运行时相互隔离，防止数据被未授权的实体访问。它利用专门的硬件模块，如安全内存加密（SME）和安全加密虚拟化（SEV），来保护内存中的数据。

  *数据加密*：在数据的存储和传输过程中，H100 通过硬件支持的加密机制（例如 AES 加密）来确保数据在 GPU 内部、内存中以及网络上传输时都是加密状态，防止数据在使用过程中被泄露。H100 的加密引擎可以对内存中的数据进行实时加密和解密。

  *硬件根信任（Root of Trust）*：H100 采用硬件根信任机制，它通过引导安全启动链，确保设备从硬件层开始处于可信状态。这意味着从设备启动到运行机密工作负载的整个过程中，只有经过验证的、未篡改的软件和固件才可以执行。

  *远程证明（Remote Attestation）*：H100 支持远程证明功能，这可以让外部用户验证 GPU 是否处于可信的状态，从而确保数据仅在安全的环境中运行。通过远程证明机制，用户可以验证系统的配置、固件版本以及运行的工作负载的完整性。
]

#slide[
  = Confidential Computing on nVIDIA H100 GPU: A Performance Benchmark Study

  由专注于Web3可验证计算的初创公司 Phala Network 于2024年9月6日发布在ArXiv上

  = 观察

  - 对于吞吐量指标，平均的性能损失小于 7%
    - 随着 token 数量的增加，性能损失逐渐减小
    - 对于70B参数的模型，性能损失可以忽略不计
  - 预填充阶段受到的影响相对更大
    - 可能是由于初始加载和加密开销
    - 对于小模型，第一个token的生成平均延迟增加了接近20%
]

#show: appendix

= 附录

#focus-slide[
  Thanks
]
#slide[
  = 吞吐量损失 -7%
  #set text(size: 12pt)
  = Confidential Computing on nVIDIA H100 GPU: A Performance Benchmark Study
  #figure(image("../png/tee_throughput.png", width: 100%))
][
  #set text(size: 12pt)
  计算资源的利用效率：大模型通常需要更强的计算资源，如更多的内存和更复杂的计算。这意味着当启用TEE时，虽然加密和保护数据的开销增加，但相较于整体的计算量，这些开销占比可能较小。而小模型本身计算量较少，因而加密和保护数据的开销在总计算量中的占比相对较大，导致表现出的性能损耗较明显。

  TEE的处理机制：TEE会对模型的推理过程进行加密和数据隔离，这些操作对小模型的相对简单的计算任务来说，可能产生更多干扰。而大模型本身就具有大量的并发处理、计算层次复杂等特性，因此TEE的影响在复杂的计算流程中被“稀释”了。

  I/O密集度：小模型可能对I/O更敏感（如输入输出数据流），而TEE的机制通常会增加I/O开销，导致小模型的性能损失比大模型更明显。
]

#slide[
  #set text(size: 12pt)
  = Confidential Computing on nVIDIA H100 GPU: A Performance Benchmark Study

  #figure(image("../png/tee_prefill.png", width: 100%))

  Prefill 阶段受到的影响相对更大

  初始加载和加密开销：在prefill阶段，模型需要加载大量的初始上下文、参数等数据。而TEE开启时，会有数据加密和保护的额外开销，这会导致在prefill阶段的额外延迟。这部分工作在TTFT上表现得更明显，而在ITL阶段（生成后续tokens）相对稳定，因为后续tokens的生成依赖于已经缓存或预处理的结果。
][
  #set text(size: 12pt)
  初始加载和加密开销：在prefill阶段，模型需要加载大量的初始上下文、参数等数据。而TEE开启时，会有数据加密和保护的额外开销，这会导致在prefill阶段的额外延迟。这部分工作在TTFT上表现得更明显，而在ITL阶段（生成后续tokens）相对稳定，因为后续tokens的生成依赖于已经缓存或预处理的结果。

  上下文窗口的处理：在prefill阶段，模型需要处理输入的完整上下文（或是大部分上下文），对于每个token的生成，TEE的安全处理会增加上下文的开销，这在TTFT上表现为显著的影响。而ITL通常只涉及每次单个token的生成，计算量较小，影响较弱。

  并行与串行处理的差异：prefill阶段可能涉及更多的并行计算和预加载数据，在此过程中，TEE可能会限制部分并行化的效率，使得整体prefill时间增加。而在ITL阶段，由于生成每个token的过程是逐步串行的，受TEE影响较小。

  缓存与推理优化：prefill过程中可能无法充分利用缓存机制，尤其是对于较小的模型，整个序列的推理需要在初始阶段完成较多的计算。而在生成后的每个token推理时，缓存的KV机制可以提高效率，抵消一部分TEE带来的开销。因此，ITL阶段受影响较小。
]

#slide[
  #set text(size: 12pt)
  = Confidential Computing on nVIDIA H100 GPU: A Performance Benchmark Study
  #figure(image("../png/tee_decode.png", width: 100%))
  固定开销的摊薄效应：TEE引入的加密、数据保护等固定开销对于较短的序列（如100 tokens以内）来说，影响较大，因为短序列的计算时间本身较短，额外的保护开销占比更高。然而，随着token数量增加，模型的整体推理时间增加，固定的加密和保护开销在总计算中的占比减少，表现为overhead逐渐降低。
][
  #set text(size: 12pt)

  并行化处理效率提升：随着token数量的增加，模型在生成过程中可能会更多依赖并行处理和GPU的并发能力，而TEE对并行处理的影响较小，因此在处理更长的序列时，整体的计算效率提升，加密的开销变得更为可忽略。sv

  缓存和推理优化的效果：长序列的推理过程中，模型更容易利用KV cache等缓存机制来减少重复计算。尤其是在生成后续tokens时，缓存能显著提高效率，进一步抵消了TEE带来的额外开销。而短序列则在生成第一个token时受缓存的作用较小，overhead显得更高。

  I/O操作的相对减少：对于短序列，I/O操作（如加载输入、预处理等）所占比例较大，而TEE的加密机制会显著增加I/O的时间成本。随着序列长度增加，计算的主要瓶颈转向模型推理本身，而I/O的比例相对降低，TEE对I/O的影响也逐渐变得不那么显著。

  计算复杂度的稀释：较长的序列意味着更多的推理步骤和计算，而每一步的加密和保护开销被这些额外的计算步骤稀释掉了。换句话说，随着推理步骤的增加，TEE的影响不再占据推理时间的主要部分，模型的计算复杂度逐渐主导整个过程。
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
