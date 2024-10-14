#import "@preview/tablex:0.0.8": tablex, hlinex, vlinex, colspanx, rowspanx

#let cm = emoji.checkmark.heavy
#let cg = block(fill: green.transparentize(50%))[#cm]
#let cr = block(fill: red.transparentize(50%))[#cm]
#let na = ""
#let half-red = red.transparentize(50%)
#let pos = block(fill: green.transparentize(50%))[+]
#let neg = block(fill: red.transparentize(50%))[-]
#let que = block(fill: gray.transparentize(50%))[?]
#let f1 = "Initial"

#let pcc_req = [
  #set text(fill: luma(30%), size: 12pt)
  S: Stateless computation
  E: Enforceable guarantees
  T: Non-targetability
  P: No privileged runtime access
  V: Verifiable transparency
]

#let trend_table = tablex(
  columns: 4,
  align: center + horizon,

  auto-vlines: false,
  //
  /* --- header --- */
  [*领域*],
  [*趋势*],
  [*示例*],
  [*PCC的要求*],

  /* -------------- */

  [*内存管理*], [更精细的内存管理，更高的复用度], [Paging\ Token-Level Optimization], [*无状态计算*],
  [*传输*], [更精细的传输，使得资源能够按需到达], [Data Duplication\ Prefetching/Pulling\ PD Disaggregation], [*T*],
  [*调度*], [根据场景进行优化], [Request-level Predictions\ Machine-Level Scheduling\ Global profiling], [*STP*],
  [*并行*], [更高的并行度], [Pipeline Parallelism\ Tensor Parallelism\ Sequence Parallelism\ Speculative Inference], [*S*],
)

#let taxonmy_table = tablex(
  columns: 4,
  align: center + horizon,
  auto-vlines: false,
  repeat-header: true,

  /* --- header --- */
  [*类别*], [*要求*], [*可能的威胁*], [*Apple 提出的一些方案*],
  /* -------------- */

  rowspanx(2)[技术], [无状态计算\ Stateless computation], [数据处理的痕迹\ Example:  Logging, debugging], [(目的) Only use user data to perform requested operations\
  (时间) Delete the data after fulfilling the request\
  (范围) Not available to even Apple staff\
  ],

  (), [不可针对性\ Non-targetability], [定向攻击，定向引导请求\ Example:  Steer request to compromised nodes], [
  (硬件) Strengthened supply chain\
  // Revalidation before being provisioned for PCC\
  (调度) Requests cannot be user/content-specific routed\
  (加密匿名) OHTTP Relay, RSA Blind Signature\
  (访问控制) No system-wide encryption
  ],

  rowspanx(3)[生态], [可强制执行的保证\ Enforceable guarantees], [外部组件引入的威胁\ Example:  External TLS-terminating load balancer], [
  (硬件) Secure Enclave, Secure Boot\
  (系统) Signed System Volume, Swift on Server\
  (软件) Code Signing, Sandboxing
  ],

  (), [无特权运行时访问\ No privileged runtime access], [后门和运维 Shell\ Example:  Shell access by SREs], [
  No remote shell. Only pre-specified, structured, and audited \
  logs/metrics can leave the node\
  User data is reviewed by multiple indepedent layers\
  ],

  (), [可验证的透明性\ Verifiable transparency], [没有经过安全专家验证的代码], [
  Every production build of PCC publicly available
  ],
)

#let opt_goal_table = tablex(
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
  rowspanx(4)[*Memory*], [Paging], neg, pos, [], pos, [], [],
  (), [Prefix Caching], neg, pos, [], pos, [], [],
  (), [Disk Offloading], [], pos, neg, pos, [], [],
  (), [Group-Query Attention], [], pos, [], pos, pos, pos,

  rowspanx(4)[*Tranmission*], [Duplication], pos, neg, pos, pos, pos, pos,
  (), [Pulling], pos, neg, pos, pos, pos, pos,
  (), [Request Migration], pos, pos, neg, pos, pos, pos,
  (), [Disaggregated Arch], pos, pos, pos, pos, neg, neg,

  rowspanx(3)[*Batch*], [Continuous Batch], pos, [], pos, pos, neg, neg,
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

#let s_attacker_table = tablex(
  columns: 7,
  align: center + horizon,

  auto-vlines: false,
  //
  /* --- header --- */
  [*Capabilities*],
  [*Goal*],
  [*Query*],
  [*Access to Storage*],
  [*Knowledge of Model*],
  [*Access to Specific Nodes*],
  [*Control over Node*],

  /* -------------- */
  [Weak], [Reconstructing User Inputs and Contexts],
  cm, cm, cm, na, na,

  [Strong], [Reconstructing User Inputs and Contexts],
  cm, cm, cm, cm, cm,
)

#let s_attack_table = tablex(
  columns: 8,
  align: center + horizon,

  auto-vlines: false,
  // repeat-header: true,
  // vlinex(x: 5, stroke: gray),

  /* --- header --- */
  rowspanx(2)[*优化*],
  rowspanx(2)[*残留的状态*],
  rowspanx(2)[*储存位置*],
  rowspanx(2)[*缓解策略*],
  colspanx(4)[*攻击*],

  [数据泄露],
  [信息推断],
  [访问模式分析],
  [差分分析],
  /* -------------- */

  [Prefix Caching], [Token\ KV Cache], [GPU 内存\ CPU 内存], [过期机制 \ 隔离],
  cm, cm, cm, na,

  [Disk Offloading], [KV Cache], [外置存储器\ (SSD, Hard Drive)], [加密],
  cm, na, na, na,

  [Data Duplication], [KV Cache], [GPU 内存\ CPU 内存], [过期同步机制],
  cm, [], [], cm,

  [Database-based\ Speculative Inference], [Token], [GPU 内存\ CPU 内存], [差分隐私\ 使用小模型预测],
  cm, cm, na, [],
)

#let t_attack_table = tablex(
  columns: 3,
  align: center + horizon,
  // vlinex(x: 6),

  auto-vlines: false,
  // repeat-header: true,
  // vlinex(x: 5, stroke: gray),

  /* --- header --- */
  [*优化*],
  [*增加 Hit*],
  [*减少 Miss*],

  /* -------------- */
  [*Duplication*],
  [#cm\ 复制目标的缓存到其他受害者],
  [#cm\ 复制非目标的缓存到非受害者],

  [*Pulling*],
  [#cm\ 从目标的缓存中拉取数据],
  na,

  [*Request Migration*],
  [#cm\ 迁入目标的请求],
  [#cm\ 迁出非目标的请求],

  [*Priority-Based Scheduling*],
  [#cm\ 提高目标的请求的优先级],
  [#cm\ 降低非目标的请求的优先级],

  // [*Request-Level Prediction*],
  // [#cm\ 提高目标的请求的优先级],
  // [#cm\ 降低非目标的请求的优先级],

  [*Instance Flip*],
  [#cm\ 让受害者节点转变成预填充节点],
  [],
)

#let model_header_url(name, year, url) = {
  let size = 5pt
  set text(size: size)
  link(url)[*#name*\ ]
  [#year]
}

#let academic_table = tablex(
  columns: 24,
  align: center + horizon,
  auto-vlines: false,
  // repeat-header: true,

  /* --- header --- */
  [*类别*],
  [*优化*],
  [*威胁*],
  model_header_url("Orca", 2206, "https://www.usenix.org/conference/osdi22/presentation/yu"),
  model_header_url("FlexGen", 2303, "https://arxiv.org/abs/2303.06865"),
  model_header_url("FastServe", 2305, "https://arxiv.org/abs/2305.05920"),
  model_header_url("SpecInfer", 2305, "https://arxiv.org/abs/2305.09781"),
  model_header_url("vLLM", 2309, "https://arxiv.org/abs/2309.06180"),
  model_header_url("REST", 2311, "https://arxiv.org/abs/2311.08252"),
  model_header_url("Splitwise", 2311, "https://arxiv.org/abs/2311.18677"),
  model_header_url("SGLang", 2312, "https://arxiv.org/abs/2312.07104"),
  model_header_url("Lookahead", 2312, "https://arxiv.org/abs/2312.12728"),
  model_header_url("Sarathi", "23-24", "https://arxiv.org/abs/2403.02310"),

  model_header_url("InfiniteLLM", 2401, "https://arxiv.org/abs/2401.02669"),
  model_header_url("DistServe", 2401, "https://arxiv.org/abs/2401.09670"),
  model_header_url("Medusa", 2401, "https://arxiv.org/abs/2401.10774"),
  model_header_url("TetriInfer", 2401, "https://arxiv.org/abs/2401.11181"),
  model_header_url("AttentionStore", 2403, "https://arxiv.org/abs/2403.19708v2"),
  model_header_url("LoongServe", 2404, "https://arxiv.org/abs/2404.09526"),
  model_header_url("Andes", 2405, "https://arxiv.org/abs/2404.16283"),
  model_header_url("Llumnix", 2406, "https://arxiv.org/abs/2406.03243"),
  model_header_url("Preble", 2407, "https://arxiv.org/abs/2407.00023"),
  model_header_url("MInference", 2407, "https://arxiv.org/pdf/2407.02490"),
  model_header_url("TR", 2408, "https://www.arxiv.org/abs/2408.08696"),

  /* -------------- */
  rowspanx(3)[*内存*], [Paging], na,
  [], [], [], [], f1, [], [], cg, [], cg, [], [], [], cg, [], [], [], [], [], [], [],
  (), [Prefix Caching], [*SE*],
  [], [], [], [], [], [], [], cr, [], [], [], [], [], [], [], [], [], [], cr, [], [],
  (), [Disk Offloading], [*SE*],
  [], cr, [], [], [], [], [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [],

  rowspanx(4)[*传输*], [Duplication], [*ST*],
  [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
  (), [Pulling], [*T*],
  [], [], [], [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [], [], [], [],
  (), [Request Migration], na,
  [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cg, [], cg, [], [], [],
  (), [Disaggregated Arch], na,
  [], [], [], [], [], [], cg, [], [], [], [], cg, [], cg, [], [], [], [], [], [], [],

  rowspanx(3)[*批处理*], [Continuous Batch], na,
  f1, [], cg, cg, cg, [], [], [], [], cg, [], cg, [], cg, [], [], [], [], [], [], [],
  (), [Chunked Prefill], na,
  [], [], [], [], [], [], [], [], [], f1, [], [], [], cg, [], [], [], [], cg, [], [],
  (), [Prepack Prefill], na,
  [], [], [], [], [], [], [], [], [], [], [], cg, [], cg, [], [], [], [], [], [], [],

  rowspanx(5)[*并行*], [Speculation], na,
  [], [], [], cg, [], cg, [], cg, cg, [], [], [], cg, [], [], [], [], [], [], [], [],
  (), [Context-Based Speculation], [*S*],
  [], [], [], [], [], cr, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
  (), [Database-Based Speculation], [*S*],
  [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [], [], [], [], [], [], cr,
  (), [Tensor Parallelism], na,
  [], [], [], cg, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
  (), [Sequence Parallelism], na,
  [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cg, [], [], [], [], [],

  rowspanx(5)[*调度*], [Priority-Based], [*T*],
  [], [], cr, [], [], [], [], cr, [], cr, [], [], [], cr, [], [], cr, cr, cr, [], [],
  (), [Request-Level Prediction], [*T*],
  [], [], cr, cr, [], [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [], [],
  (), [Machine-level Scheduler], [*E*],
  [], [], cr, [], [], [], cr, [], [], [], cr, [], [], cr, [], cr, [], [], cr, [], [],
  (), [Instance Flip], [*T*],
  [], [], [], [], [], [], cr, [], [], [], [], [], [], cr, [], [], [], [], [], [], [],
  (), [Global Profiling], [*P*],
  [], cr, [], [], [], [], cr, [], [], [], [], cr, [], [], [], [], [], [], [], cr, [],

  [*审计*], [Non Open-Source], [*V*],
  [], [], [], [], [], [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [], cr,
)

#let model_header(name, year) = {
  let size = 5pt
  set text(size: size)
  [*#name*\ ]
  [#year]
}

#let industrial_table = tablex(
  columns: 16,
  align: center + horizon,
  auto-vlines: false,
  // repeat-header: true,

  /* --- header --- */
  [*类别*],
  [*优化*],
  [*威胁*],
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

  rowspanx(5)[*内存*], [Paging], na,
  cg, [], [], cg, cg, cg, [], cg, [], [], [], cg, [],
  (), [Token Attention], na,
  [], cg, [], [], [], [], [], [], [], [], [], [], [],
  (), [Prefix Caching], [*S*],
  cr, [], [], [], [], [], cr, [], [], [], [], [], [],
  (), [Disk Offloading], [*SE*],
  [], [], [], cr, cr, [], cr, [], [], [], [], [], cr,
  // (), [Radix Attention], [*S*],
  // cg, [], [], [], [], [], [], [], [], [], [], [], [],
  (), [Multi-Query Attention], na,
  [], [], [], [], [], [], cr, [], [], [], [], [], [],
  // (), [Group-Query Attention], [*T*],
  // [], [], [], [], [], [], cg, [], [], [], [], [], [],

  rowspanx(4)[*传输*], [Duplication], [*ST*],
  [], [], [], [], cr, [], [], [], [], [], [], [], [],
  (), [Pulling], [*T*],
  [], [], [], [], [], [], [], [], [], [], [], [], [],
  (), [Request Migration], na,
  [], [], [], [], [], [], [], [], [], [], [], [], [],
  (), [Disaggregated Arch], na,
  cg, [], [], [], cg, [], [], [], [], [], [], [], cg,

  rowspanx(3)[*批处理*], [Continuous Batch], na,
  cg, [], cg, [], cg, cg, cg, cg, cg, cg, cg, cg, [],
  (), [Chunked Prefill], na,
  cg, [], [], [], cg, cg, [], [], [], [], [], [], [],
  (), [Prepack Prefill], na,
  [], [], [], [], [], [], [], [], [], [], [], [], [],

  rowspanx(3)[*并行*], [Speculation], na,
  cg, [], cg, cg, [], [], cg, cg, [], [], [], cg, cg,
  (), [Tensor Parallelism], na,
  [], [], [], [], [], [], [], cg, [], [], [], [], cg,
  (), [Sequence Parallelism], na,
  [], [], [], [], [], [], [], [], [], [], [], [], [],

  rowspanx(5)[*调度*], [Priority-Based], [*T*],
  [], [], [], cr, cr, cr, [], [], [], [], [], [], [],
  (), [Request-Level Prediction], [*T*],
  [], cr, [], cr, [], [], [], [], [], [], [], [], [],
  (), [Machine-level Scheduler], [*E*],
  [], [], [], cr, cr, [], [], [], [], [], [], [], [],
  (), [Instance Flip], [*T*],
  [], [], [], [], [], [], [], [], [], [], [], [], [],
  (), [Global Profiling], [*P*],
  [], [], [], [], cr, [], [], [], [], [], [], [], [],

  [*审计*], [Non Open-Source], [*V*],
  [], [], [], [], cr, [], cr, [], [], [], [], [], [],
)

#let vllm_table = tablex(
  columns: 8,
  align: center + horizon,
  auto-vlines: false,
  // repeat-header: true,

  /* --- header --- */
  [*版本*],
  [*日期*],
  [*内存*],
  [*传输*],
  [*批处理*],
  [*并行*],
  [*调度*],
  [*模型*],
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
  [Chunked Prefill],
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

#let misc_table = tablex(
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
  [*MInference*], [Prefill, Long Context], [Addresses the expensive computational cost and the unacceptable latency of the attention calculations in the pre-filling stage of long-context LLMs by leveraging dynamic sparse attention with spatial aggregation patterns],
  [*Shared Attention*], [Attention], [directly sharing pre-computed attention weights across multiple layers in LLMs],
  [*SnapKV*], [Compression], [Observing that specific tokens within prompts gain consistent attention from each head during generation, our methodology not only retrieve crucial information but also enhances processing efficiency],
)