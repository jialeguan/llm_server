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
  [*Category*],
  [*Trend*],
  [*Examples*],
  [*Conflict*],

  /* -------------- */

  [*Memory*], [Enhanced memory management with finer granularity\ Improve reusability of KV Cache], [Paging\ Token-Level Optimization], [*S*],
  [*Transmission*], [Minimizing transmission latency], [Data Duplication\ Prefetching\ PD Disaggregation], [*T*],
  [*Scheduling*], [Customized scheduling for specific scenarios\ Cache-aware scheduler], [Request-level Predictions\ Machine-Level Scheduling\ Global profiling], [*STP*],
  [*Parallelism*], [Optimizing parallelism for resource reuse and efficiency], [Pipeline Parallelism\ Tensor Parallelism\ Sequence Parallelism\ Speculative Inference], [*S*],
)

#let taxonmy_table = tablex(
  columns: 4,
  align: center + horizon,
  auto-vlines: false,
  repeat-header: true,

  /* --- header --- */
  [*Category*], [*Requirements*], [*Threats*], [*Guarantees*],
  /* -------------- */

  rowspanx(2)[Technical], [Stateless computation], [Trace of data after processing\ Example:  Logging, debugging], [(Purpose) Only use user data to perform requested operations\
  (Transient) Delete the data after fulfilling the request\
  (Scope) Not available to even Apple staff\
  ],

  (), [Non-targetability], [Targeted attack\ Example:  Steer request to compromised nodes], [
  (Hardware) Hardened supply chain\
  // Revalidation before being provisioned for PCC\
  (Scheduler) Requests cannot be user/content-specific routed\
  (Anonymity) OHTTP Relay, RSA Blind Signature\
  (Scope) No system-wide encryption
  ],

  rowspanx(3)[Ecosystem], [Enforceable guarantees], [Technical enforceability\ Example:  External TLS-terminating load balancer], [
  (Hardware) Secure Enclave, Secure Boot\
  (System) Signed System Volume, Swift on Server\
  (Software) Code Signing, Sandboxing
  ],

  (), [No privileged runtime access], [Privileged interfaces\ Example:  Shell access by SREs], [
  No remote shell. Only pre-specified, structured, and audited \
  logs/metrics can leave the node\
  User data is reviewed by multiple indepedent layers\
  ],

  (), [Verifiable transparency], [Uninspected code], [
  Every production build of PCC publicly available
  ],
)

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
//

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
  vlinex(x: 6),

  auto-vlines: false,
  // repeat-header: true,
  // vlinex(x: 5, stroke: gray),

  /* --- header --- */
  rowspanx(2)[*Optimization*],
  rowspanx(2)[*Stored States*],
  rowspanx(2)[*Location*],
  rowspanx(2)[*Mitigation*],
  colspanx(2)[*Weak Attacker*],
  colspanx(2)[*Strong Attacker*],

  [Inference],
  [Replay],
  [Extraction],
  [Mapping],
  /* -------------- */

  [Prefix Caching], [KV Cache], [GPU Memory\ CPU Memory], [Cache Expiry\ Isolation],
  [], [], cm, cm,

  [Disk Offloading], [KV Cache], [Disk Storage\ (SSD, Hard Drive)], [Encryption],
  cm, cm, cm, na,

  [Pulling], [KV Cache], [GPU Memory\ CPU Memory], [Randomized Scheduler],
  [], [], cm, cm,

  [Database-based\ Speculative Inference], [Token], [GPU Memory\ CPU Memory], [Differential Priavacy],
  [], [], cm, [],
)

#let t_attack_table = tablex(
  columns: 3,
  align: center + horizon,
  // vlinex(x: 6),

  auto-vlines: false,
  // repeat-header: true,
  // vlinex(x: 5, stroke: gray),

  /* --- header --- */
  [*Optimization*],
  [*Increase Hit Rate*],
  [*Decrease Miss Rate*],

  /* -------------- */
  [*Duplication*],
  [#cm\ Duplicate targeted cache to other victims],
  [#cm\ Duplicate untargeted cache to non-victims],

  [*Pulling*],
  [#cm\ Pull targeted cache from other victims],
  na,

  [*Priority-Based Scheduling*],
  [#cm\ Prioritize targeted requests],
  [#cm\ Deprioritize untargeted requests],

  [*Request-Level Prediction*],
  [#cm\ Prioritize targeted requests],
  [#cm\ Deprioritize untargeted requests],

  [*Instance Flip*],
  [#cm\ Flip to prefill instance],
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
  [*Category*],
  [*Optimization*],
  [*Threat*],
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
  rowspanx(3)[*Memory*], [Paging], na,
  [], [], [], [], f1, [], [], cg, [], cg, [], [], [], cg, [], [], [], [], [], [], [],
  (), [Prefix Caching], [*SE*],
  [], [], [], [], [], [], [], cr, [], [], [], [], [], [], [], [], [], [], cr, [], [],
  (), [Disk Offloading], [*SE*],
  [], cr, [], [], [], [], [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [],

  rowspanx(4)[*Tranmission*], [Duplication], [*T*],
  [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
  (), [Pulling], [*SET*],
  [], [], [], [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [], [], [], [],
  (), [Request Migration], na,
  [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cg, [], cg, [], [], [],
  (), [Disaggregated Arch], na,
  [], [], [], [], [], [], cg, [], [], [], [], cg, [], cg, [], [], [], [], [], [], [],

  rowspanx(3)[*Batch*], [Iteration-Level Batch], na,
  f1, [], cg, cg, cg, [], [], [], [], cg, [], cg, [], cg, [], [], [], [], [], [], [],
  (), [Chunked Prefill], na,
  [], [], [], [], [], [], [], [], [], f1, [], [], [], cg, [], [], [], [], cg, [], [],
  (), [Prepack Prefill], na,
  [], [], [], [], [], [], [], [], [], [], [], cg, [], cg, [], [], [], [], [], [], [],

  rowspanx(5)[*Parallelism*], [Speculation], na,
  [], [], [], cg, [], cg, [], cg, cg, [], [], [], cg, [], [], [], [], [], [], [], [],
  (), [Context-Based Speculation], [*S*],
  [], [], [], [], [], cr, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
  (), [Database-Based Speculation], [*S*],
  [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [], [], [], [], [], [], cr,
  (), [Tensor Parallelism], na,
  [], [], [], cg, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
  (), [Sequence Parallelism], na,
  [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cg, [], [], [], [], [],

  rowspanx(5)[*Scheduling*], [Priority-Based], [*T*],
  [], [], cr, [], [], [], [], cr, [], cr, [], [], [], cr, [], [], cr, cr, cr, [], [],
  (), [Request-Level Prediction], [*T*],
  [], [], cr, cr, [], [], [], [], [], [], [], [], [], cr, [], [], [], [], [], [], [],
  (), [Machine-level Scheduler], [*E*],
  [], [], cr, [], [], [], cr, [], [], [], cr, [], [], cr, [], cr, [], [], cr, [], [],
  (), [Instance Flip], [*T*],
  [], [], [], [], [], [], cr, [], [], [], [], [], [], cr, [], [], [], [], [], [], [],
  (), [Global Profiling], [*P*],
  [], cr, [], [], [], [], cr, [], [], [], [], cr, [], [], [], [], [], [], [], cr, [],

  [*Verification*], [Non Open-Source], [*V*],
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

  rowspanx(4)[*Memory*], [Paging], na,
  cm, [], [], cm, cm, cm, [], cm, [], [], [], cm, [],
  (), [Token Attention], na,
  [], cm, [], [], [], [], [], [], [], [], [], [], [],
  (), [Prefix Caching], [*S*],
  cm, [], [], [], [], [], cm, [], [], [], [], [], [],
  (), [Disk Offloading], [*SE*],
  [], [], [], cm, cm, [], cm, [], [], [], [], [], cm,
  // (), [Radix Attention], [*S*],
  // cm, [], [], [], [], [], [], [], [], [], [], [], [],
  // (), [Multi-Query Attention], na,
  // [], [], [], [], [], [], cm, [], [], [], [], [], [],
  // (), [Group-Query Attention], [*T*],
  // [], [], [], [], [], [], cm, [], [], [], [], [], [],

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
  (), [Machine-level Scheduler], [*E*],
  [], [], [], cm, cm, [], [], [], [], [], [], [], [],
  (), [Instance Flip], [*T*],
  [], [], [], [], [], [], [], [], [], [], [], [], [],
  (), [Global Profiling], [*P*],
  [], [], [], [], cm, [], [], [], [], [], [], [], [],

  [*Verification*], [Non Open-Source], [*V*],
  [], [], [], [], cm, [], cm, [], [], [], [], [], [],
)

#let vllm_table = tablex(
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