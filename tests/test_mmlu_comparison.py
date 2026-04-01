"""MMLU benchmark: f16 reference vs 4+4bit per-layer rotation + factored_int8 norm.

Lightweight MMLU evaluator — computes log-likelihood of each answer choice
and picks the most likely one. Reports per-subject and overall accuracy.
"""

import gc
import logging
import time
from collections import defaultdict

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from turboquant_model.model import TurboQuantConfig, quantize_model
from turboquant_model.module import TurboQuantLinear

CHOICE_LABELS = ["A", "B", "C", "D"]


def format_mmlu_prompt(question: str, choices: list[str], subject: str) -> str:
    """Format MMLU question as a completion prompt."""
    subject_str = subject.replace("_", " ")
    prompt = f"The following is a multiple choice question about {subject_str}.\n\n"
    prompt += f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{CHOICE_LABELS[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt


def eval_mmlu(model, tokenizer, device, max_subjects=None, max_per_subject=None):
    """Evaluate MMLU accuracy via log-likelihood scoring.

    Returns (overall_acc, per_subject_acc dict, total_questions).
    """
    ds = load_dataset("cais/mmlu", "all", split="test")

    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)

    # Group by subject for optional limiting
    subject_questions = defaultdict(list)
    for item in ds:
        subject_questions[item["subject"]].append(item)

    subjects = sorted(subject_questions.keys())
    if max_subjects:
        subjects = subjects[:max_subjects]

    total_correct = 0
    total_questions = 0

    model.eval()
    with torch.no_grad():
        for si, subject in enumerate(subjects):
            questions = subject_questions[subject]
            if max_per_subject:
                questions = questions[:max_per_subject]

            for item in questions:
                prompt = format_mmlu_prompt(
                    item["question"], item["choices"], item["subject"]
                )
                gold = item["answer"]  # 0-3

                # Score each choice by log-likelihood of the answer token
                choice_scores = []
                for ci, label in enumerate(CHOICE_LABELS):
                    full_text = prompt + " " + label
                    tokens = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
                    logits = model(tokens).logits
                    # Log-prob of the last token (the answer letter)
                    log_probs = nn.functional.log_softmax(logits[0, -2], dim=-1)
                    answer_token_id = tokenizer.encode(" " + label, add_special_tokens=False)[-1]
                    choice_scores.append(log_probs[answer_token_id].item())

                pred = max(range(4), key=lambda i: choice_scores[i])
                correct = pred == gold
                subject_correct[subject] += int(correct)
                subject_total[subject] += 1
                total_correct += int(correct)
                total_questions += 1

            acc = subject_correct[subject] / subject_total[subject] * 100
            if (si + 1) % 10 == 0 or si == len(subjects) - 1:
                logger.info(
                    f"  [{si+1}/{len(subjects)}] {subject}: "
                    f"{acc:.1f}% ({subject_correct[subject]}/{subject_total[subject]})"
                )

    overall_acc = total_correct / total_questions * 100 if total_questions else 0
    per_subject_acc = {
        s: subject_correct[s] / subject_total[s] * 100 for s in subjects
    }
    return overall_acc, per_subject_acc, total_questions


# ---------- Load tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# =====================================================
# 1. f16 reference
# =====================================================
logger.info("=" * 60)
logger.info("Evaluating: f16 reference")
logger.info("=" * 60)

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=dtype, trust_remote_code=True
).to(device).eval()

t0 = time.time()
ref_acc, ref_subjects, ref_total = eval_mmlu(
    ref_model, tokenizer, device, max_subjects=57, max_per_subject=10
)
ref_time = time.time() - t0
logger.info(f"f16 MMLU: {ref_acc:.2f}% ({ref_total} questions) in {ref_time:.0f}s")

del ref_model
gc.collect()

# =====================================================
# 2. 4+4bit per-layer rotation + factored_int8 norm
# =====================================================
logger.info("=" * 60)
logger.info("Evaluating: 4+4bit per-layer rotation + factored_int8")
logger.info("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=dtype, trust_remote_code=True
).to(device).eval()

config = TurboQuantConfig(
    bit_width=4,
    group_size=128,
    residual_bit_width=4,
    rotation="qr",
    rotation_strategy="per_layer",
    norm_codec="factored_int8",
    entropy_coding=True,
)

model = quantize_model(model, config)

# Apply norm codec + disable fused kernels
for m in model.modules():
    if isinstance(m, TurboQuantLinear):
        m.use_cutile = False
        m.use_triton = False
        m.use_metal = False
        if m.weight_norms.dim() >= 2:
            m.apply_norm_codec("factored_int8")

t0 = time.time()
quant_acc, quant_subjects, quant_total = eval_mmlu(
    model, tokenizer, device, max_subjects=57, max_per_subject=10
)
quant_time = time.time() - t0
logger.info(f"4+4bit MMLU: {quant_acc:.2f}% ({quant_total} questions) in {quant_time:.0f}s")

# =====================================================
# Summary
# =====================================================
print(f"\n{'='*80}")
print("MMLU COMPARISON SUMMARY")
print(f"{'='*80}")
print(f"{'Model':<45} {'Accuracy':>10} {'Questions':>10}")
print("-" * 70)
print(f"{'f16 reference':<45} {ref_acc:>9.2f}% {ref_total:>10}")
print(f"{'4+4bit per-layer + factored_int8':<45} {quant_acc:>9.2f}% {quant_total:>10}")
print(f"\nDelta: {quant_acc - ref_acc:+.2f}%")
print(f"{'='*80}")

# Per-subject comparison for subjects with biggest gaps
gaps = []
for s in ref_subjects:
    if s in quant_subjects:
        gaps.append((s, quant_subjects[s] - ref_subjects[s]))
gaps.sort(key=lambda x: x[1])

print(f"\nTop 5 degradations:")
for s, delta in gaps[:5]:
    print(f"  {s:<40} {ref_subjects[s]:>6.1f}% → {quant_subjects[s]:>6.1f}% ({delta:+.1f}%)")

print(f"\nTop 5 improvements:")
for s, delta in gaps[-5:]:
    print(f"  {s:<40} {ref_subjects[s]:>6.1f}% → {quant_subjects[s]:>6.1f}% ({delta:+.1f}%)")
