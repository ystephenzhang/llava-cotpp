
## Inference Paradigms (LLaVA-CoT)

**Single-pass generation (baseline)**
- Path: predict.py
- Flow: build chat with image + prompt, single `model.generate` with streaming; no re-ranking or multi-stage control.

**Self-judged best-of-N sampling**
- Paths: inference/demo/simple_inference.py (`generate_inner_best_of_N`), inference/VLMEvalKit/inference_demo.py (same idea, larger N).
- Flow: sample N full answers (stop on `</CONCLUSION>`), run pairwise tournaments where the same model judges two candidates with image + prompt context, keep winners until one remains.
- Paradigm: self-consistency plus self-evaluation; whole-answer sampling and re-ranking.

**Sentence-level beam with self-judge**
- Path: inference/demo/simple_inference.py (`generate_inner_sentence_beam`) and VLMEvalKit variant.
- Flow: generate sentence-by-sentence (stop on period or `</CONCLUSION>`); at each step sample beam of next sentences, judge picks best continuation, carry forward and repeat.
- Paradigm: incremental constrained decoding with model-in-the-loop selection.

**Stage-wise beam with self-judge (default)**
- Path: inference/demo/simple_inference.py (`generate_inner_stage_beam`, default `type=stage`) and VLMEvalKit.
- Flow: enforce four tagged blocks in order `<SUMMARY>`, `<CAPTION>`, `<REASONING>`, `<CONCLUSION>`; for each block, sample beam completions to its end tag, judge compares candidates with stage-specific rubric (summary quality, caption accuracy, reasoning correctness, conclusion alignment), keep winner and proceed.
- Paradigm: structured multi-stage generation with per-stage self-critique.

**VLMEvalKit integration notes**
- Uses same paradigms with dataset-specific prompts and slightly different beam sizes; default path routes to stage-wise beam.
- All paths rely on patched `inference/processing_mllama.py` dropped into the Transformers install for correct image handling.

## MCTS Inference Goal + Plan

**Goal**
- Add an MCTS-based inference mode starting from the current stage-wise beam: each node holds a partial completion up to a stage boundary; rollouts get a scalar value from the existing judge (0/1 or float) to guide search.

**Implementation Plan**
- State/Node: track (stage_index in [`SUMMARY`,`CAPTION`,`REASONING`,`CONCLUSION`], accumulated `input_ids`, generated text segments). Terminal when final stage closed.
- Actions/Expansion: from a leaf, generate K candidates for the next stage segment (sample to end tag) using current tokenizer/model; one child per candidate with its text + ids appended.
- Selection: UCT over children (e.g., `Q + c*sqrt(log(N_parent)/N_child)`) using judge score as value; maintain visit counts and mean value.
- Simulation/Rollout: for a selected leaf, either (a) run a one-shot completion for remaining stages, or (b) run judge on the newly produced segment against a baseline (e.g., empty or best-so-far) to return a value; keep rollout inexpensive.
- Backprop: propagate rollout value up ancestors, updating visit counts and average rewards.
- Stopping/Output: until all stages closed; return the child of root with highest value, reconstruct full text from its path.
- Integration: add `type="mcts"` option in demo/VLMEvalKit generators; reuse existing judge prompts per stage for value computation; keep sampling kwargs consistent with stage beam; guard with small defaults for K, simulations, and exploration constant to avoid long latency.
