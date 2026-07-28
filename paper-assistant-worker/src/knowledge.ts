export interface KnowledgeChunk {
  id: string;
  keywords: string[];
  text: string;
}

export const knowledgeChunks: KnowledgeChunk[] = [
  {
    id: "Abstract",
    keywords: [
      "abstract",
      "summary",
      "omniqec",
      "ai scientist",
      "automated",
      "paper",
      "overview",
    ],
    text:
      "OmniQEC is an AI-scientist framework for discovering practical quantum error-correcting codes. It turns natural-language requirements into an end-to-end search over code construction, verification, circuit synthesis, noise simulation, and decoding. The paper argues that practical quality must be judged at the circuit level rather than from algebraic code parameters alone.",
  },
  {
    id: "Motivation",
    keywords: [
      "motivation",
      "practical",
      "proxy",
      "mislead",
      "distance",
      "code-centric",
      "why",
      "implementation",
    ],
    text:
      "Conventional code-centric discovery optimizes proxies such as rate, distance, and check weight. Those quantities matter, but they omit ancilla overhead, syndrome-extraction scheduling, error propagation, decoder behavior, and the physical noise model. A code that looks superior algebraically can therefore rank worse after the full implementation is compiled and simulated.",
  },
  {
    id: "Architecture",
    keywords: [
      "architecture",
      "workflow",
      "orchestrator",
      "pipeline",
      "agent",
      "feedback",
      "system",
    ],
    text:
      "The workflow is User Query → Orchestrator → Fast Loop → Slow Loop → Feedback. The orchestrator translates constraints into a structured campaign, calls scientific tools, maintains evidence from previous iterations, and routes promising candidates to progressively more expensive evaluation. Slow-loop results are fed back into subsequent generations.",
  },
  {
    id: "Fast Loop",
    keywords: [
      "fast loop",
      "screening",
      "construction",
      "commutation",
      "logical qubit",
      "distance",
      "sparsity",
      "fast",
      "exploration",
    ],
    text:
      "The fast loop explores broadly at low cost. A generative search agent proposes executable construction programs, and code-characterization tools check CSS commutation, logical-qubit count, distance, sparsity, and related algebraic properties. Only candidates that survive this screening proceed to costly circuit-level evaluation.",
  },
  {
    id: "Slow Loop",
    keywords: [
      "slow loop",
      "circuit",
      "syndrome",
      "ancilla",
      "stim",
      "detector",
      "decoder",
      "bp osd",
      "bp lsd",
      "relay",
      "slow",
    ],
    text:
      "The slow loop evaluates the complete implementation. Selected codes are compiled into syndrome-extraction circuits with ancilla allocation and scheduling, converted to Stim circuits and detector error models, sampled under circuit-level noise, and decoded with configurable methods including BP+OSD, BP+LSD, and Relay-BP. This loop provides the physical evidence used for final ranking.",
  },
  {
    id: "Code Families",
    keywords: [
      "family",
      "families",
      "bivariate bicycle",
      "bb",
      "trivariate bicycle",
      "tricycle",
      "tb",
      "tt",
      "lifted product",
      "lp",
      "generalized lifted product",
      "glp",
    ],
    text:
      "The study searches four qLDPC construction families: bivariate bicycle (BB), trivariate bicycle or tricycle (TB/TT), lifted product (LP), and generalized lifted product (GLP). These families are explored under complete physical-qubit budgets, including implementation overhead rather than data qubits alone.",
  },
  {
    id: "Performance Frontier",
    keywords: [
      "performance",
      "frontier",
      "budget",
      "n=98",
      "n=240",
      "98",
      "240",
      "72 12 6",
      "144 12 12",
      "llm",
      "claude",
      "gpt",
      "deepseek",
      "surpass",
      "outperform",
    ],
    text:
      "Across three LLM backends and fourteen total-physical-qubit budgets per backend, OmniQEC finds the first candidate that surpasses the BB [[72,12,6]] baseline at N=98. The strongest Claude- and GPT-based campaigns also surpass BB [[144,12,12]] at N=240. These comparisons are made with circuit-level logical-error performance under the stated budget.",
  },
  {
    id: "Physical-error Sweep",
    keywords: [
      "physical error",
      "error rate",
      "sweep",
      "logical error",
      "ler",
      "29.23",
      "1.66",
      "0.002",
      "0.01",
      "gamma",
      "exponent",
      "5.60",
      "5.57",
      "5.20",
      "3.79",
    ],
    text:
      "In the physical-error-rate sweep, the best OmniQEC design has 29.23× lower logical error rate than its baseline at p=0.002 and 1.66× lower at p=0.01. Representative fitted suppression exponents are Γ=5.20 for GPT [[48,4,8]], Γ=5.60 for Claude [[58,2,7]], and Γ=5.57 for DeepSeek [[48,4,8]], compared with Γ=3.79 for the BB baseline.",
  },
  {
    id: "Ablations",
    keywords: [
      "ablation",
      "feedback",
      "full workflow",
      "fast only",
      "slow only",
      "loop",
      "remove",
    ],
    text:
      "The ablation study compares reduced workflows with the full slow-fast feedback system. The complete workflow is consistently superior: inexpensive code-level exploration supplies diversity, while circuit-level evidence corrects misleading proxies and improves subsequent search. Removing either part weakens the discovery process.",
  },
  {
    id: "Cost",
    keywords: [
      "cost",
      "token",
      "compute",
      "scaling",
      "budget",
      "n",
      "expense",
      "price",
    ],
    text:
      "The reported LLM cost depends only weakly on the total physical-qubit budget N. This suggests that the language-model orchestration overhead does not grow rapidly with code size over the tested range, although circuit synthesis and simulation remain the expensive scientific evaluation steps.",
  },
  {
    id: "Discussion",
    keywords: [
      "discussion",
      "limitation",
      "future",
      "preview",
      "release",
      "code",
      "citation",
      "conclusion",
    ],
    text:
      "The current manuscript is a preliminary research preview under active revision. It demonstrates that tool-using language-model agents can coordinate practical QEC discovery when their proposals are grounded by executable verification and circuit-level feedback. Source code and formal citation details are reserved for the formal release.",
  },
];
