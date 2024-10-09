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
    title: [大模型推理中的隐私云计算],
    subtitle: [Private Cloud Compute 中的无状态计算],
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
#let arch_simple = fletcher-diagram(
  let user = (0, 0),
  let scheduler = (3, 0),
  let server = (6, 0),

  node(user, [*客户端*]),
  node(scheduler, [*服务器*]),
  node(server, [*工作节点*]),

  edge(user, scheduler, "-|>", [`enc(input)`], shift: 3pt, label-side: left),
  edge(scheduler, user, "-|>", [`enc(output)`], shift: 3pt, label-side: left),
  edge(scheduler, server, "-|>", [`input`], shift: 3pt, label-side: left),
  edge(server, scheduler, "-|>", [`output`], shift: 3pt, label-side: left),
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
  node(scheduler, [*服务器*]),
  node(worker_a, [*预填充节点*]),
  node(worker_b, [*解码节点*]),

  edge(user, scheduler, "-|>", [`enc(input)`], shift: 3pt, label-side: left),
  edge(scheduler, user, "-|>", [`enc(output)`], shift: 3pt, label-side: left),
  edge(scheduler, worker_a, "-|>", [`input`], shift: 3pt, label-side: left),
  edge(worker_b, scheduler, "-|>", [`output`], shift: 3pt, label-side: left),
  edge(worker_a, worker_b, "-|>", [`KV Cache` (中间数据)], shift: 3pt, label-side: left),
)

#slide[
  #arch_simple

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
  通用性和应用的多样性导致推理请求的异构性，表现为输入长度、输出长度、期望延迟等方面的差异。比如在代码生成任务中，输入长度可能是几十行代码，而在问答任务中，输入长度可能是几个句子。这种异构性使得难以设计一个通用的优化策略，需要根据具体的任务特性进行优化。
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
  - 无法使用端到端加密
  - 无法在输入中使用匿名化

]

// == 无状态计算
// #slide[
//   = Stateless computation
//   Private Cloud Compute must use the personal user data that it receives exclusively for the purpose of fulfilling the user's request. This data must never be available to anyone other than the user, not even to Apple staff, not even during active processing. And *this data must not be retained*, including via logging or for debugging, after the response is returned to the user. In other words, we want a strong form of stateless data processing where *personal data leaves no trace* in the PCC system.
// ]


== 要求

#slide[
  #set text(size: 12pt)
  #taxonmy_table
]

= 大模型推理中的状态

== 数据管理

#slide[
  = KV Cache
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
  = PagedAttention
  #set text(size: 14pt)
  PagedAttention#footnote_link("Efficient Memory Management for Large Language Model Serving with PagedAttention", "https://arxiv.org/abs/2309.06180") 是一种将注意力矩阵划分为较小页面的技术。这种方法几乎完美地解决了内存碎片问题，因此，PagedAttention 已成为大模型推理系统中动态内存分配的事实标准。

  #figure(
    image("../png/paged_attention_waste.png", width: 90%),
  )
]

// #slide[
//   = Paged Attention
//   #set text(size: 12pt)
//   #figure(
//     image("../png/paged_attention.png", width: 80%),
//   )

//   *Pitfalls*#footnote_link("vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention. 2024","https://arxiv.org/abs/2405.04437")
//   - Requires re-writing the attention kernel.
//   - Adds software complexity and redundancy (CPU code), can degrade throughput by 11%.
//   - Introduces performance overhead. 20-26% slower than original FasterTransformer kernel.
// ]

#slide(composer: (3fr, 7fr))[
  = Prefix Caching
  #set text(size: 14pt)
  Prefix Caching#footnote_link("ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition. 2024", "https://arxiv.org/abs/2402.15220") 是一种用于加速大模型推理（特别是语言模型推理）的技术。其主要目的是通过缓存和重用推理过程中已经计算过的部分，减少重复计算，从而提高效率。
][
  #figure(
    image("../png/prefix_caching.png", width: 100%),
  )
]

#slide[
  = Prefix Caching & 无状态计算

  #set text(size: 14pt)
  Prefix Caching 通过存储之前生成的键（keys）和值（values）来提高自回归模型的推理效率，但在这一过程中，可能会引入一些状态，这些状态的存在可能会影响无状态计算的原则。

  = 信息推断

  由于 Prefix Caching 将历史生成信息与当前查询关联。在推理过程中，如果前缀缓存被重复使用或共享给不同的推理任务，攻击者可能通过比较不同任务的输出，推断出前一个输入的信息。例如，在生成式任务中，攻击者可以通过观察缓存的相似性，来推测用户的部分输入。

  = 访问模式分析

  Prefix Caching 可能导致某些查询或生成过程频繁访问缓存，这种访问模式的可见性可能使攻击者能够分析并推测出缓存内容。通过访问模式，攻击者可能会获取到敏感信息的特征或上下文。

  = 缓存泄露

  如果缓存中的数据没有适时更新或删除，过期的信息可能会在意外情况下被泄露。例如，某个用户的敏感查询结果未能及时清除，可能会被后续请求访问到。

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
  如果 KV Cache 中存储了敏感信息（如用户输入、生成的内容等），KV Cache Offloading 可能会导致这些敏感信息被转移到不安全的存储介质中，增加了隐私泄露的风险。
]

#slide[
  = Pull, Data Duplication
  #[
    #set text(size: 14pt)
    在多服务器环境中，KV Cache可能会被复制到不同的节点上，以便在多个节点上并行执行推理任务。

    Pull 由执行的节点负责，它将 KV Cache 从其他节点拉取到本地节点上，以便在本地节点上执行推理任务。

    Data Duplication 由调度器负责，它将 KV Cache 复制到多个节点上，以便在多个节点上并行执行推理任务。
  ]

  #pause
  = Data Duplication & 无状态计算
  #[
    #set text(size: 14pt)

    = 状态残留
    当多个节点拥有 KV Cache 的副本时，执行结束后，这些副本可能仍然存在于内存或存储中。如果 KV Cache 中存储了敏感信息（如用户输入、生成的内容等），数据重复可能导致多个节点持有这些敏感数据的副本。在执行结束后，这些副本仍可能被访问或分析，增加了隐私泄露的风险。

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

= 合规的挑战

== 优化的趋势
#slide[
  #set text(size: 14pt)
  #trend_table
  #pcc_req
]

== 论文
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

== 工业界的推理系统
#slide[
  #[
    #set text(size: 6pt)
    #industrial_table
  ]

  // https://github.com/vllm-project/vllm/issues/4104 相对于只缓存Prefix Cache，vLLM的Prefix Caching功能还缓存了Generated KV Cache
]

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
][
  #set text(size: 14pt)
  = 合规的挑战
  - 需要大量的状态管理来实现并行化和负载均衡
  - 精致的调度可能会引入定向攻击和隐私泄露
  - 自动化和自适应的优化策略可能会导致优化用数据被泄露
]

// #slide[
//   = Trends
//   #set text(size: 14pt)
//   #trend_table

//   #pcc_req
// ]



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
= 附录
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

  #figure(image("../png/tee_throughput.png", width: 70%))
]

#slide[
  = Confidential Computing on nVIDIA H100 GPU: A Performance Benchmark Study

  #figure(image("../png/tee_prefill.png", width: 60%))
  #figure(image("../png/tee_decode.png", width: 60%))
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
