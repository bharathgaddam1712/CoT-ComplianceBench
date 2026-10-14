# CoT-ComplianceBench

Research code for auditing safety divergence between reasoning traces and final
answers in open reasoning models. The project provides dataset generation,
dual-channel labeling, and a blinded human-validation workflow.

- [Dataset on Hugging Face](https://huggingface.co/datasets/BharathGaddam/CoT-ComplianceBench)
- [Existing monitor checkpoint](https://huggingface.co/BharathGaddam/CoT-ComplianceBench-Monitor)
- [Human audit protocol](docs/HUMAN_AUDIT.md)

A resumable, free-tier-first pipeline for generating a Chain-of-Thought safety
dataset from reasoning LLMs. Built to run on free **Colab** and **Kaggle** GPUs
(T4 / P100, 16 GB) using 4-bit quantization.

## Human-label validation

The blinded human-label audit is documented in [docs/HUMAN_AUDIT.md](docs/HUMAN_AUDIT.md).
Prepare the frozen review packet with `python scripts/human_audit.py prepare --output audit_runs/study_v1`.
Human reliability results are pending; the automated labels have not yet been
validated by this audit. Review packets and annotations are excluded from Git.

Python 3.10+ is sufficient for the audit; it needs no GPU, API key, or additional
Python packages. Open the generated `audit_runs/study_v1/review.html`, and send
the same file to both independent reviewers using separate browser profiles.
The file in `scripts/` is a template, not a review packet.

```sh
python -m unittest discover -s tests -v
```

The audit is the first validation step. Statistical analysis and paper revisions
remain pending; current automated labels do not establish human-confirmed safety
rates. The `without_attacks` data contain raw harmful prompts, not harmless benign
controls. Model-group comparisons are descriptive, not causal evidence about
training methods.

## Design principles

- **Code in git, data in persistent storage.** The repo is ephemeral (re-cloned
  each session); generated JSONL lives in Google Drive (Colab) or `/kaggle/working`
  (Kaggle) so a disconnect loses nothing. `src/config.py` auto-detects the platform.
- **Everything is resumable.** Every sample is checkpointed to disk the instant it
  finishes. Re-running any script skips already-completed IDs.
- **Generation and labeling are separate passes.** A labeling crash never touches
  the expensive generations.
- **One `job_list.jsonl` for all models.** Run `sampling.py` once and reuse the
  same file across every model so IDs match and merging is clean.

## Layout

```
src/config.py      storage paths, taxonomy, model list, budgets
src/sampling.py    build data/job_list.jsonl from HarmBench/AdvBench (run ONCE)
src/attacks.py     H-CoT / RTO / RACE prompt transforms
src/inference.py   resumable 4-bit generation + <think> parsing
src/labeling.py    Llama-Guard (output) + LLM judge (CoT) + quadrant labels
src/smoke_test.py  3-prompt end-to-end sanity check (run before real GPU spend)
```

## Supported Models

| `--model` key      | HuggingFace repo                               | VRAM (4-bit) | Notes |
|--------------------|------------------------------------------------|--------------|-------|
| `deepseek-r1-7b`   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B        | ~6 GB        | fits one T4 |
| `deepseek-r1-8b`   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B       | ~6 GB        | fits one T4 |
| `deepseek-r1-14b`  | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B       | ~10 GB       | fits one T4 |
| `qwen3-14b`        | Qwen/Qwen3-14B                                 | ~10 GB       | needs `enable_thinking=True` in chat template |
| `qwq-32b`          | Qwen/QwQ-32B                                   | ~20 GB       | shards across **two T4s** via `device_map=auto` |
| `phi4-reasoning`   | microsoft/Phi-4-reasoning-plus                 | ~10 GB       | verify `</think>` delimiter on smoke test first |

> **Important — Phi-4:** Run a smoke test before the full inference pass.
> Phi-4-reasoning-plus may emit its CoT in a different format than a clean
> `</think>` tag. If `cot_trace` comes back empty, check the raw output and
> update `think_open`/`think_close` in `config.py` accordingly.

## Quickstart (Colab — Google Drive for persistence)

```python
!git clone https://github.com/bharathgaddam1712/CoT-ComplianceBench.git
%cd CoT-ComplianceBench
from google.colab import drive; drive.mount('/content/drive')   # persistent data
!pip -q install -r requirements.txt

# Run ONCE to build the shared job list
!python src/sampling.py

# Sanity-check the pipeline (3 prompts, no output written to dataset)
!python src/smoke_test.py

# Then run whichever model fits the current session
!python src/inference.py --model deepseek-r1-7b     # ~6 GB — fits one T4
!python src/inference.py --model deepseek-r1-8b     # ~6 GB — fits one T4
!python src/inference.py --model deepseek-r1-14b    # ~10 GB — fits one T4
!python src/inference.py --model qwen3-14b          # ~10 GB — fits one T4
!python src/inference.py --model phi4-reasoning     # ~10 GB — smoke test first!
!python src/inference.py --model qwq-32b            # ~20 GB — needs two T4s (Kaggle dual-GPU)

# Re-running after a disconnect is safe — already-done IDs are skipped automatically
```

## Quickstart (Kaggle — `/kaggle/working` for persistence)

```python
!git clone https://github.com/bharathgaddam1712/CoT-ComplianceBench.git
%cd CoT-ComplianceBench
!pip -q install -r requirements.txt

# Pick ONE model per notebook session
!python src/inference.py --model deepseek-r1-8b
!python src/inference.py --model deepseek-r1-14b
!python src/inference.py --model qwq-32b            # enable "2 x GPU T4" in settings

# After the run: Save Version -> Save & Run All (Commit) to persist /kaggle/working
```

## Parallel strategy (run multiple models at once)

You can run a different model on each free-tier session simultaneously since they
all share the same `job_list.jsonl` but write to separate output files:

| Session | Platform | Model |
|---------|----------|-------|
| 1 | Colab | `deepseek-r1-7b` |
| 2 | Colab | `deepseek-r1-14b` |
| 3 | Kaggle (1× T4) | `deepseek-r1-8b` |
| 4 | Kaggle (1× T4) | `qwen3-14b` |
| 5 | Kaggle (2× T4) | `qwq-32b` |
| 6 | Kaggle (1× T4) | `phi4-reasoning` |

Then merge all `data/raw/*.jsonl` on your workstation for the labeling pass.

## Partial / test runs

Use `--limit N` to process only the first N remaining samples — useful for
testing before committing a full GPU session:

```python
!python src/inference.py --model qwen3-14b --limit 20
```

## Output schema

Each line of `data/raw/labeled_<model>.jsonl`:

```json
{
  "id": "...", "prompt": "...", "harm_category": "cybercrime",
  "attack_method": "h_cot", "target_model": "deepseek-r1-7b",
  "cot_trace": "...", "final_answer": "...",
  "output_unsafe": 0, "cot_unsafe": 1,
  "compliance_quadrant": "Q3", "divergent": true
}
```

`Q3` (unsafe CoT / safe output) is the core subject: the blind spot of
output-only safety systems.

## Notes

- The labeling instructions below describe the legacy `src/labeling.py` path.
  The repository also includes local labeling scripts with Qwen2.5-7B defaults.
  Script defaults alone do not establish which judge produced existing labels;
  reconcile run provenance before reporting the final paper methodology.
- **Run `sampling.py` on one platform only**, then reuse the same `job_list.jsonl`
  everywhere so IDs match and the merge is clean.
- The CoT judge (70B) in `labeling.py` uses a hosted API — set `TOGETHER_API_KEY`.
  Llama-Guard runs locally in 4-bit.
- Source prompts come from published benchmarks (HarmBench, AdvBench). All
  attacks are structural reproductions of the cited papers for evaluation only.

