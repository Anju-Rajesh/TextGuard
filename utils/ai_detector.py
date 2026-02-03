"""
Perplexity + DetectGPT-style curvature (perturbation) detector.

What it does:
1) Computes perplexity of input under a causal language model (LM).
2) Creates N "natural" perturbations using a T5 span-infilling model.
3) Computes a curvature score:  ll(original) - mean(ll(perturbations))
   (AI-generated text tends to sit near a probability peak → higher curvature)

Notes:
- This is STILL NOT a perfect detector. Use as a signal, not a verdict.
- Works best on English and reasonably long text (>= ~150–200 words).
"""

import math
import random
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer as HFTokenizer,
    AutoModelForSeq2SeqLM,
)

# -----------------------------
# Config
# -----------------------------
@dataclass
class DetectConfig:
    # Base LM for likelihood/perplexity (choose an open causal LM that fits your machine)
    # Good small default: "gpt2" (fast, but weaker). Better: "EleutherAI/gpt-neo-1.3B" (heavier).
    base_lm_name: str = "gpt2"

    # Mask-and-fill model for perturbations (T5 infilling style)
    # "t5-base" is a decent balance; "t5-small" is faster.
    masker_name: str = "t5-small"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Sliding window settings for long texts
    max_length: int = 512
    stride: int = 256

    # Perturbation settings
    num_perturbations: int = 10
    # Approx span masking rate: fraction of words to mask (T5 infilling)
    mask_fraction: float = 0.15
    # Maximum number of spans to mask
    max_spans: int = 3

    # Output thresholds (tune on your dataset!)
    # These are rough defaults; you should calibrate.
    curvature_ai_threshold: float = 1.0   # Lowered slightly for sensitivity (higher => more AI-like)
    ppl_ai_threshold: float = 40.0        # Perplexity threshold for supporting signal


# -----------------------------
# Utility: log-likelihood & perplexity with sliding window
# -----------------------------
@torch.no_grad()
def compute_log_likelihood(
    text: str,
    tokenizer: HFTokenizer,
    model: AutoModelForCausalLM,
    max_length: int = 512,
    stride: int = 256,
    device: str = "cpu",
) -> float:
    """
    Returns average log-likelihood per token (higher is more probable).
    Uses sliding window for long text, similar to common perplexity scripts.
    """
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0].to(device)

    nlls = []
    total_tokens = 0

    # sliding window
    for start in range(0, input_ids.size(0), stride):
        end = min(start + max_length, input_ids.size(0))
        window = input_ids[start:end]

        if window.numel() < 2:
            continue

        # labels = inputs; but ignore tokens that overlap previous window to avoid double-counting
        labels = window.clone()
        if start > 0:
            overlap = max_length - stride
            # In this window, ignore the first `overlap` tokens for loss
            labels[:overlap] = -100

        out = model(window.unsqueeze(0), labels=labels.unsqueeze(0))
        # out.loss is mean over non-ignored tokens in this window
        # Convert to total NLL by multiplying by token count included
        # Count included tokens:
        included = (labels != -100).sum().item()
        if included > 0:
            nlls.append(out.loss.item() * included)
            total_tokens += included

        if end == input_ids.size(0):
            break

    if total_tokens == 0:
        return float("-inf")

    total_nll = sum(nlls)  # negative log-likelihood total
    avg_nll = total_nll / total_tokens
    avg_ll = -avg_nll
    return avg_ll


def compute_perplexity_from_avg_ll(avg_ll: float) -> float:
    # Perplexity = exp(-avg_ll)
    if avg_ll == float("-inf"):
        return float("inf")
    return float(math.exp(-avg_ll))


# -----------------------------
# Utility: T5 span-infilling perturbations
# -----------------------------
def _mask_spans(words: List[str], mask_fraction: float, max_spans: int) -> List[str]:
    """
    Replace a few spans with a single T5 sentinel <extra_id_0>.
    T5 expects special tokens like <extra_id_0>.
    """
    if not words:
        return words

    n_words = len(words)
    n_to_mask = max(1, int(n_words * mask_fraction))

    # choose number of spans (1..max_spans) and split n_to_mask across them
    n_spans = min(max_spans, max(1, n_to_mask // 3))
    n_spans = max(1, n_spans)

    # choose span start indices
    starts = sorted(random.sample(range(n_words), k=min(n_spans, n_words)))
    spans = []
    remaining = n_to_mask
    for s in starts:
        if remaining <= 0:
            break
        divisor = (len(starts) - len(spans))
        if divisor <= 0: divisor = 1
        span_len = min(max(1, remaining // divisor), n_words - s)
        spans.append((s, s + span_len))
        remaining -= span_len

    # merge overlaps
    spans = sorted(spans, key=lambda x: x[0])
    merged = []
    for a, b in spans:
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    # apply masking: replace each span with <extra_id_0>
    out = []
    i = 0
    for a, b in merged:
        out.extend(words[i:a])
        out.append("<extra_id_0>")
        i = b
    out.extend(words[i:])
    return out


@torch.no_grad()
def t5_infill(
    masked_text: str,
    t5_tokenizer: HFTokenizer,
    t5_model: AutoModelForSeq2SeqLM,
    device: str,
    max_new_tokens: int = 64,
) -> str:
    """
    Runs T5 to fill <extra_id_0> span.
    Output from T5 often includes sentinels; we strip them.
    """
    inputs = t5_tokenizer(masked_text, return_tensors="pt", truncation=True).to(device)
    out_ids = t5_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=1.0,
    )
    decoded = t5_tokenizer.decode(out_ids[0], skip_special_tokens=False)

    # Remove any sentinel artifacts from decoded.
    # T5 might produce text like: "filled text <extra_id_1> ..."
    # We'll stop at next sentinel if present.
    cut = decoded.find("<extra_id_1>")
    if cut != -1:
        decoded = decoded[:cut]
    decoded = decoded.replace("<extra_id_0>", "").strip()
    return decoded


def make_perturbations(
    text: str,
    t5_tokenizer: HFTokenizer,
    t5_model: AutoModelForSeq2SeqLM,
    cfg: DetectConfig,
) -> List[str]:
    """
    Create N perturbations: mask some spans and infill with T5, then reconstruct.
    """
    words = text.split()
    if len(words) < 30:
        # Too short → perturbations are unstable
        return []

    perturbations = []
    for _ in range(cfg.num_perturbations):
        masked_words = _mask_spans(words, cfg.mask_fraction, cfg.max_spans)
        masked_text = " ".join(masked_words)

        fill = t5_infill(masked_text, t5_tokenizer, t5_model, cfg.device)
        # Replace the sentinel with the generated fill
        perturbed = masked_text.replace("<extra_id_0>", fill)
        perturbations.append(perturbed)

    return perturbations


# -----------------------------
# Main detector
# -----------------------------
class PerplexityCurvatureDetector:
    def __init__(self, cfg: Optional[DetectConfig] = None):
        self.cfg = cfg or DetectConfig()
        self.device = self.cfg.device

        # Check for local models first
        import os
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        local_base_lm = os.path.join(base_path, "models", self.cfg.base_lm_name)
        if os.path.exists(local_base_lm):
            print(f"Loading local base model from: {local_base_lm}")
            model_name_or_path = local_base_lm
        else:
            print(f"Local model not found at {local_base_lm}, downloading from Hub: {self.cfg.base_lm_name}")
            model_name_or_path = self.cfg.base_lm_name

        local_masker = os.path.join(base_path, "models", self.cfg.masker_name)
        if os.path.exists(local_masker):
            print(f"Loading local masker model from: {local_masker}")
            masker_name_or_path = local_masker
        else:
             print(f"Local masker not found at {local_masker}, downloading from Hub: {self.cfg.masker_name}")
             masker_name_or_path = self.cfg.masker_name

        # Base LM
        self.base_tok = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.base_tok.pad_token is None:
            self.base_tok.pad_token = self.base_tok.eos_token

        self.base_lm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.cfg.dtype,
        ).to(self.device)
        self.base_lm.eval()

        # T5 masker
        self.t5_tok = AutoTokenizer.from_pretrained(masker_name_or_path)
        # Handle potential missing special tokens map in local T5
        # (AutoTokenizer usually handles this, but good to be safe)
        
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(
            masker_name_or_path,
            torch_dtype=self.cfg.dtype,
        ).to(self.device)
        self.t5.eval()

    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 50:
            return {
                "error": "Text too short for reliable analysis. Provide at least ~50+ chars (preferably 150+ words)."
            }

        # 1) Base log-likelihood + perplexity
        ll_orig = compute_log_likelihood(
            text, self.base_tok, self.base_lm,
            max_length=self.cfg.max_length,
            stride=self.cfg.stride,
            device=self.device
        )
        ppl = compute_perplexity_from_avg_ll(ll_orig)

        # 2) Perturbations + curvature
        perts = make_perturbations(text, self.t5_tok, self.t5, self.cfg)
        ll_perts = []
        for p in perts:
            llp = compute_log_likelihood(
                p, self.base_tok, self.base_lm,
                max_length=self.cfg.max_length,
                stride=self.cfg.stride,
                device=self.device
            )
            if llp != float("-inf"):
                ll_perts.append(llp)

        if ll_perts:
            curvature = ll_orig - (sum(ll_perts) / len(ll_perts))
        else:
            curvature = None

        # 3) Enhanced scoring logic
        # Curvature is our most reliable signal (DetectGPT principle)
        # We use a logistic-like scaling for curvature to handle outliers
        # curvature ~0.0 => 0 score
        # curvature ~2.0 => high score
        if curvature is not None:
            # Shifted curvature score: 0 to 100
            # A curvature of 1.0 is often the "tipping point" for AI
            curv_score = 100 / (1 + math.exp(-2.5 * (curvature - 0.8)))
        else:
            curv_score = 0.0

        # Perplexity Signal (Model-Dependent)
        # Low perplexity (e.g. < 30) is suspicious
        # High perplexity (e.g. > 100) is very human-like
        if ppl < 10: ppl_signal = 100.0  # Extremely repetitive/predictable
        elif ppl < 40: ppl_signal = 80.0 # Likely AI
        elif ppl < 80: ppl_signal = 40.0 # Uncertain
        else: ppl_signal = 10.0          # likely Human

        # Combine: 70% Curvature, 30% Perplexity
        score = (curv_score * 0.70) + (ppl_signal * 0.30)
        
        # Clamp it
        score = max(0.0, min(100.0, score))
        score = round(score, 2)

        # 4) Conclusion bands
        if score >= 80:
            conclusion = "Highly Likely AI-generated"
        elif score >= 60:
            conclusion = "Likely AI-generated"
        elif score >= 40:
            conclusion = "Uncertain / Mixed"
        else:
            conclusion = "Likely Human-written"

        return {
            "base_lm": self.cfg.base_lm_name,
            "masker_model": self.cfg.masker_name,
            "avg_log_likelihood": round(ll_orig, 6),
            "perplexity": round(ppl, 3),
            "num_perturbations_used": len(ll_perts),
            "curvature": None if curvature is None else round(curvature, 6),
            "ai_score_0_100": score,
            "conclusion": conclusion,
            "notes": [
                "Curvature is the main DetectGPT-style signal (higher => more AI-like).",
                "Perplexity is model-dependent; use it as a supporting signal.",
                "Calibrate thresholds on your own dataset for real accuracy."
            ],
        }

# --- GLOBAL DETECTOR INSTANCE (Lazy Loaded) ---
_detector = None

def analyze_text_ai(text: str):
    """
    Main entry point for the AI detection logic.
    Used by app.py. Returns (probability, conclusion).
    """
    global _detector
    if _detector is None:
        # We load models on first call to avoid startup overhead if not used.
        cfg = DetectConfig(
            base_lm_name="distilgpt2",       # Fast and relatively light
            masker_name="t5-small",    # Good balance
            num_perturbations=10,      # Increased for better stability/accuracy
        )
        print("Initializing Advanced AI Detector models... (This may take a moment)")
        _detector = PerplexityCurvatureDetector(cfg)
        print("Models loaded successfully.")
    
    result = _detector.analyze(text)
    
    if "error" in result:
        return 0.0, result["error"]
        
    return result["ai_score_0_100"], result["conclusion"]


if __name__ == "__main__":
    # Test execution
    test_text = "This is a simple test string to verify the detector works."
    score, conclusion = analyze_text_ai(test_text)
    print(f"Score: {score}, Conclusion: {conclusion}")




