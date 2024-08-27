#import "@preview/tablex:0.0.8": tablex, hlinex, vlinex, colspanx, rowspanx
  
#set page(flipped: true)

#let model_header(name, year) = {
  let size = 6pt
  set text(size: size)
  [*#name*\ ]
  [#year]
}


#set text(size: 6pt)
#let cm = emoji.checkmark.heavy
#let first = "Initial"
#let na = ""

#tablex(
columns: 19,
align: center + horizon,
auto-vlines: false,
// repeat-header: true,

/* --- header --- */
[*Type*],
[*Optimization*],
[*Threat*],
model_header("FT", 22),
model_header("Orca", 22),
model_header("vLLM", 23),
model_header("FlexGen", 23),
model_header("FastServe", 23),
model_header("FastServe2", 23),
model_header("Sarathi", 23),
model_header("Lookahead", 24),
model_header("REST", 24),
model_header("SpecInfer", 24),
model_header("Medusa", 24),
model_header("DistServe", 24),
model_header("Splitwise", 24),
model_header("LoongServe", 24),
model_header("TetriInfer", 24),
model_header("InfiniteLLM", 24),

/* -------------- */

rowspanx(3)[*Batch*], [Iteration-Level Batch], na, [], first, cm, [], cm, [], cm, [], [], [], [], cm, [], [], cm, [], 
(), [Chunked Prefill], na, [], [], [], [], [], [], first, [], [], [], [], [], [], [], cm, [], 
(), [Prepack Prefill], na, [], [], [], [], [],  [],[], [], [], cm, [], cm, [], [], cm, [], 

rowspanx(6)[*Parallel*], [Speculation], [*S*], [], [], [], [], [], [], [], cm, cm, cm, cm, [], [], [], [], [],
(),  [Prompt-Based Speculation], [*S*], [], [], [], [], [], [], [], cm, [], [], [], [], [], [], [], [],
(),  [Context-Based Speculation], [*S*], [], [], [], [], [], [], [], [], cm, [], [], [], [], [], [], [],
(), [Tensor Parallelism], [], cm, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
(), [SafeTensors], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 
(), [Sequence Parallelism], na, [], [], [], [], [], [], [], [], [], [], [], [], [], cm, [], [], 

rowspanx(3)[*Memory*], [Paging], na, [], [], first, [], [], [], cm, [], [], [], [], [], [], [], cm, [], 
(), [Multi-Query Attention], na, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
(), [Grouped-Query Attention], [*T*], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 

rowspanx(5)[*Tranmission*], [Offloading], [*SE*], [], [], [], cm, cm, [], [], [], [], [], [], [], [], [], [], [], 
(), [Duplication], [*T*], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
(), [Pulling], [*SET*], [], [], [], [], [], [], [], [], [], [], [], cm, [], [], [], [], 
(), [Request Migration], na,  [], [], [], [], [], [], [], [], [], [], [], [], [], cm, [], [], 
(), [Disaggregated Arch], na, [], [], [], [], [], [], [], [], [], [], [], cm, cm, [], cm, [], 

rowspanx(5)[*Scheduling*], [Priority-Based], [*T*], [], [], [], [], cm, [], cm, [], [], [], [], [], [], [], cm, [], 
(), [Request-Level Prediction], [*T*], [], [], [], [], cm, [], [], [], [], [Small Model], [], [], [], [], cm, [],
(), [Machine-level Scheduler], [*ET*], [], [], [], [], cm, [], [], [], [], [], [], [], cm, cm, cm, cm,
(), [Instance Flip], na, [], [], [], [], [], [],[], [], [], [], [], [], cm, [], cm, [],
(), [Global Profiling], [*P*], [], [], [], cm, [], [], [], [], [], [], [], cm, cm, [], [], [], 

[*Verification*], [Open Source], [*V*], [], [], [], [], [], [], [], [], [], [], [], [], [], [], cm, [], 
)