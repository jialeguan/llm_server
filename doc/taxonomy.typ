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
#let (init, slides, touying-outline, alert, speaker-note) = utils.methods(s)
#show: init

#show strong: alert

#let (slide, empty-slide, title-slide, new-section-slide, focus-slide) = utils.slides(s)
#show: slides

= Privacy Leakage Surface

== Architecture

#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#let architecture_graph = fletcher-diagram(
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
  node(user, [*User*]),
  node(scheduler, [*Scheduler*]),
  node(worker_a, [*Worker A*]),
  node(worker_b, [*Worker B*]),

  edge(user, scheduler, "-|>", [Input], shift: 3pt, label-side: left),
  edge(scheduler, user, "-|>", [Output], shift: 3pt, label-side: left),
  edge(scheduler, worker_a, "-|>", [Task], shift: 3pt, label-side: left),
  edge(worker_a, scheduler, "-|>", [Result], shift: 3pt, label-side: left),
  edge(scheduler, worker_b, "-|>", [Task], shift: 3pt, label-side: left),
  edge(worker_b, scheduler, "-|>", [Result], shift: 3pt, label-side: left),
  edge(worker_a, worker_b, "-|>", [Data], shift: 3pt, label-side: left),
  edge(worker_b, worker_a, "-|>", [Data], shift: 3pt, label-side: left),
)
#slide[
  #architecture_graph
]

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

== Optimization Goals

#slide[
  - Lower resource requirements
    - Model compression
    - Collaborative inference
    - Offloading to CPU and disk
  - Higher throughput
    - Batch
    - Tensor parallelism
    - Sequence parallelism (Input sequence)
    - Pipeline parallelism (Stage placement)
  - Lower latency
    -
]

== Violations

#slide[
  #set text(size: 14pt)
  #tablex(
  columns: 2,
  align: center + horizon,
  auto-vlines: false,
  repeat-header: true,

  /* --- header --- */
  [*Requirement*], [*Violations*],
  /* -------------- */

  [Stateless computation], [Logging, prioritization, history metadata],

  [Enforceable guarantees], [Data transfer/duplication, data offloading, access control],

  [No privileged runtime access], [Monitoring, debugging, profiling],

  [Non-targetability], [Biased scheduler, input/output leakage],

  [Verifiable transparency], [Uninspected code],
  )

  Universal problems: Access control between worker nodes.
]


#slide[
  #let cm = emoji.checkmark.heavy

  #set text(size: 14pt)
  #tablex(
  columns: 6,
  align: center + horizon,
  auto-vlines: false,
  // repeat-header: true,

  /* --- header --- */
  [*Requirement*], [*Violations*], [*FT\ 22*], [*FlexGen\ 23*], [*Sarathi\ 23*],[*Mooncake\ 24*],
  /* -------------- */

  rowspanx(2)[Stateless computation], [Logging], [], [], [],  [],
  (), [History metadata], [], [], [], [],

  rowspanx(2)[Enforceable guarantees], [Data transfer/duplication], [], [],  [],cm,
  (), [Data offloading], [], cm, [], cm,

  rowspanx(3)[No privileged runtime access], [Monitoring], [], [], [], [],
  (), [Debugging], [], [], [], [],
  (), [Offline Profiling], [], [], [], cm,

  rowspanx(2)[Non-targetability], [Priority-based scheduler], [], [], [], [#cm\ length],
  (), [Input/output leakage], [], [], [], [],

  [Verifiable transparency], [Uninspected code], [], [], [], cm,
  )
]

== Optimizations
#slide[
  #[
    #set text(size: 12pt)
    *S*: Stateless computation
    *E*: Enforceable guarantees
    *P*: No privileged runtime access
    *T*: Non-targetability
    *V*: Verifiable transparency
   
  ]

  #[
    #set text(size: 11pt)
    #let cm = emoji.checkmark.heavy
    #let first = emoji.star
    #let na = ""

    #tablex(
    columns: 12,
    align: center + horizon,
    auto-vlines: false,
    // repeat-header: true,

    /* --- header --- */
    [*Type*],
    [*Optimization*],
    [*Threat*],
    [*FT*\ 22],
    [*Orca*\ 22],
    [*vLLM*\ 23],
    [*FlexGen*\ 23],
    [*Sarathi*\ 23],
    [*DistServe*\ 24],
    [*Mooncake*\ Moonshot],
    link("https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen")[*DeepSpeed*\ Microsoft],
    [*TensorRT*\ NVIDIA],
    /* -------------- */

    rowspanx(5)[*Storage*\ (KVCache)], [Paging], na, [], [], first, [], cm, [], cm, cm, [],
    (), [Duplication], na, [], [], [], [], [], [], cm, [], [],
    (), [Offloading], [*SE*], [], [], [], cm, [], [], cm, [], [],
    (), [Pulling], [*SET*], [], [], [], [], [], cm, [], [], [],
    (), [Model Compression], na, [], [], [], [], [], [], [], [], [],

    rowspanx(4)[*Scheduler*], [Priority-Based], [*T*], [], [], [], [], [], [], cm, cm, [],
    (), [Online/Offline Profiling], [*P*], [], [], [], [], [], cm, cm, [], [],
    (), [Local Scheduler], [*ET*], [], [], [], [], [], [], cm, [], [],
    (), [Disaggregated Arch], na, [], [], [], [], [], cm, cm, [], [],

    rowspanx(3)[*PP*], [Iteration-Level Batch], na, [], first, cm, [], cm, cm, cm, cm, [],
    (), [Chunked Prefill], na, [], [], [], [], first, [], cm, cm, [],
    (), [Prepack Prefill], na, [], [], [], [], [], cm, [], [], [],

    [*Verification*], [Uninspected Code], [*V*], [], [], [], [], [], [], cm, [], [],
    )
    // #[SP: Sequence parallelism\
    // TP: Tensor parallelism\
    // PP: Pipeline parallelism\
    // ]
  ]
]

=== Production LLMs

#slide[
  #set text(size: 14pt)

  #tablex(
    columns: 3,
    align: center + horizon,
    auto-vlines: false,
    // repeat-header: true,

    /* --- header --- */
    [*Server*], [*Vendor*], [*Website*],
    /* -------------- */
    [-LLM], [NVIDIA], [`https://github.com/NVIDIA/TensorRT-LLM?tab=readme-ov-file`],

  )
  TorchServe
  Triton
  HuggingFace TGI
  FastServe

  DistServe
  TetriInfer

  Prompt Cache
  SGLang
  AttentionStore
  Preble

  Pollux

  *Refer to DistServe Related Work*
]


#focus-slide[
  Thanks
]

// appendix by freezing last-slide-number
#let s = (s.methods.appendix)(self: s)
#let (slide, empty-slide) = utils.slides(s)

= Appendix

=== Appendix

#slide[
  https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen
]