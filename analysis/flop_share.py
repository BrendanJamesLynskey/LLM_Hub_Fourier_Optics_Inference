#!/usr/bin/env python3
"""FLOP-share (Amdahl) analysis: which part of LLM inference could a Fourier-optical
transform engine take over?

Writes ``results.md`` and ``results.json`` beside this file. Every number in the
FOptInf decks and in the phase-B design note comes from here.

Model shape: Llama-3-8B (32 layers, d = 4096, 32 heads, 8 KV heads, SwiGLU d_ff = 14,336,
vocab 128,256), taken from Disaggregated_Inference_Sim's ``hardware.LLAMA3_8B`` so the two
cannot drift. Variants keep that shape and swap the token mixer or the weight structure:

  transformer   Llama-3-8B as it is (GQA attention). No transforms.
  fnet          FNet-style: each attention block (projections and attention) replaced by an
                unparameterised 2-D DFT over (sequence, hidden), real part kept. NOT causal:
                shape-only, an upper bound for "a model made of Fourier transforms".
  hyena         Hyena order 2 replacing each attention block: in-projection to 3 streams,
                short depthwise conv (3 taps), two causal long convolutions by FFT, gating,
                output projection.
  hybrid        1 attention layer in 4, Hyena elsewhere (the ratio is illustrative).
  hyena_circ    hyena plus block-circulant weights (block 256) in every projection and the
                MLP: the case where matmuls themselves become transforms (speculative).

FLOP conventions (stated in results.md): matmul 2mnk; real FFT of length n 2.5 n log2 n and
complex 5 n log2 n (FFTW's convention); complex pointwise multiply 6, complex add 2.
Causal long convolutions zero-pad to the next power of two >= 2L (Hyena paper, sec. 3.3).
Softmax, norms and activations are not counted, as in Disaggregated_Inference_Sim.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from disagg_sim.hardware import H100_SXM, LLAMA3_8B, CostModel

HERE = Path(__file__).parent
M = LLAMA3_8B
D, L_LAYERS, FF, V = M.d_model, M.n_layers, M.d_ff, M.vocab
KV = M.n_kv_heads * M.head_dim
ORDER = 2                    # Hyena order N (the paper's language models use order 2)
SHORT = 3                    # short depthwise conv taps (Hyena paper, appendix)
CIRC_BLOCK = 256             # block-circulant block size (illustrative)
PROMPTS = [512, 2048, 8192, 32768, 131072]


def rfft(n: int) -> float:
    return 2.5 * n * math.log2(n)


def cfft(n: int) -> float:
    return 5.0 * n * math.log2(n)


def pow2_at_least(n: int) -> int:
    return 1 << (n - 1).bit_length()


# ─────────────────────────────────────────────────────────── op ledger ──
@dataclass
class Ops:
    """FLOPs by class. ``transform`` = FFT/IFFT work; ``spectral`` = pointwise work in the
    Fourier domain (what a 4f system's mask does for free); ``dense`` = ordinary matmuls;
    ``attention`` = QK^T and AV; ``other`` = short convs, gating, accumulation."""

    dense: float = 0.0
    attention: float = 0.0
    transform: float = 0.0
    spectral: float = 0.0
    other: float = 0.0

    def __iadd__(self, o: "Ops") -> "Ops":
        for k in self.__dict__:
            setattr(self, k, getattr(self, k) + getattr(o, k))
        return self

    def scaled(self, f: float) -> "Ops":
        return Ops(**{k: v * f for k, v in self.__dict__.items()})

    @property
    def total(self) -> float:
        # left to right, not sum(): Python 3.12's sum() of floats is compensated, which would break
        # bit-exact parity with the JavaScript port (js/flop_model.js)
        return self.dense + self.attention + self.transform + self.spectral + self.other

    @property
    def optical(self) -> float:
        """What a transform engine could take: transforms plus Fourier-plane multiplies."""
        return self.transform + self.spectral


def dense_matrix(m: int, n: int, tokens: int, circulant: bool) -> Ops:
    """y = W x for W of shape (m, n), applied to ``tokens`` vectors."""
    if not circulant:
        return Ops(dense=2.0 * m * n * tokens)
    k = CIRC_BLOCK
    bins = k // 2 + 1
    # FFT each of the n/k input blocks, multiply by (m/k)(n/k) block spectra, accumulate
    # over n/k blocks, IFFT each of the m/k output blocks (CirCNN's FFT -> multiply -> IFFT).
    return Ops(transform=tokens * ((n // k) * rfft(k) + (m // k) * rfft(k)),
               spectral=tokens * (m // k) * (n // k) * bins * 6.0,
               other=tokens * (m // k) * (n // k - 1) * bins * 2.0)


def mlp(tokens: int, circulant: bool) -> Ops:
    o = Ops()
    for m, n in ((FF, D), (FF, D), (D, FF)):          # gate, up, down
        o += dense_matrix(m, n, tokens, circulant)
    return o                                            # the SwiGLU product is not counted (as the sim)


def attention_block(lens: list[int]) -> Ops:
    tokens = sum(lens)
    o = Ops(dense=2.0 * tokens * (2 * D * D + 2 * D * KV))
    # causal attention: QK^T and AV, 2*d*c each at position c (diagonal included), as the sim
    for s in lens:                     # a loop, not sum(): see Ops.total
        o.attention += 2.0 * D * s * (s + 1)
    return o


def hyena_block(lens: list[int], circulant: bool = False) -> Ops:
    tokens = sum(lens)
    o = Ops()
    o += dense_matrix((ORDER + 1) * D, D, tokens, circulant)       # in-projection
    o += dense_matrix(D, D, tokens, circulant)                     # out-projection
    o.other += tokens * 2.0 * SHORT * (ORDER + 1) * D              # short depthwise conv
    o.other += tokens * ORDER * D                                  # gating multiplies
    for s in lens:
        n = pow2_at_least(2 * s)
        # per channel, per long conv: FFT(input), multiply by the cached filter spectrum, IFFT
        o.transform += ORDER * D * 2 * rfft(n)
        o.spectral += ORDER * D * (n // 2 + 1) * 6.0
    return o


def fnet_block(lens: list[int]) -> Ops:
    o = Ops()
    for s in lens:
        # 2-D DFT of a real (s, d) block: real FFTs along d, complex FFTs along the sequence
        o.transform += s * rfft(D) + D * cfft(pow2_at_least(s))
    return o


def lm_head(tokens: int) -> Ops:
    return Ops(dense=2.0 * V * D * tokens)


def prefill(variant: str, lens: list[int], lm_all_tokens: bool = True) -> Ops:
    tokens = sum(lens)
    o = Ops()
    for layer in range(L_LAYERS):
        circ = variant == "hyena_circ"
        if variant == "transformer" or (variant == "hybrid" and layer % 4 == 0):
            o += attention_block(lens)
        elif variant in ("hyena", "hybrid", "hyena_circ"):
            o += hyena_block(lens, circ)
        elif variant == "fnet":
            o += fnet_block(lens)
        o += mlp(tokens, circ)
    o += lm_head(tokens if lm_all_tokens else len(lens))
    return o


VARIANTS = ["transformer", "fnet", "hyena", "hybrid", "hyena_circ"]
LABEL = {"transformer": "Transformer (Llama-3-8B)", "fnet": "FNet-shaped (non-causal)",
         "hyena": "Hyena-2", "hybrid": "Hybrid 1:3 attention:Hyena",
         "hyena_circ": "Hyena-2 + block-circulant weights"}


def hyena_paper_count(s: int) -> float:
    """The Hyena paper's own FLOP count for one Hyena layer (appendix A.2), for comparison.
    'order' there counts the N+1 projections; log is taken as log2 here."""
    order = ORDER + 1
    return 2 * (order * D * D * s + order * D * s * SHORT
                + 5 * (order - 1) * D * math.log2(s) * s + D * D * s)


# ───────────────────────────────────────────────────────────── decode ──
def decode_per_token(variant: str, ctx: int, ds: int = 16) -> dict:
    """FLOPs for one new token at context ``ctx`` (batch 1), per mixer style."""
    per_layer_dense = 2.0 * (2 * D * D + 2 * D * KV) + 2.0 * 3 * D * FF
    hy_dense = 2.0 * ((ORDER + 1) * D * D + D * D) + 2.0 * 3 * D * FF
    head = 2.0 * V * D
    if variant == "transformer":
        return dict(dense=L_LAYERS * per_layer_dense + head, mixer=L_LAYERS * 4.0 * D * (ctx + 1),
                    transform=0.0)
    chans = ORDER * D
    if variant == "hyena_direct":      # cache past projections; direct causal dot product
        mixer = chans * 2.0 * (ctx + 1)
        return dict(dense=L_LAYERS * hy_dense + head, mixer=L_LAYERS * mixer, transform=0.0)
    if variant == "hyena_distilled":   # Laughing Hyena: diagonal complex SSM of order ds
        mixer = chans * 8.0 * ds
        return dict(dense=L_LAYERS * hy_dense + head, mixer=L_LAYERS * mixer, transform=0.0)
    if variant == "hyena_recompute":   # recompute the whole FFT convolution every token
        n = pow2_at_least(2 * (ctx + 1))
        t = chans * 2 * rfft(n)
        return dict(dense=L_LAYERS * hy_dense + head, mixer=L_LAYERS * chans * (n // 2 + 1) * 6.0,
                    transform=L_LAYERS * t)
    if variant == "hyena_tiled":       # Flash-Inference-style relaxed tiling, amortised
        t = sp = 0.0
        lvl = 0
        while (1 << lvl) <= ctx:
            b = 1 << lvl
            n = 4 * b                  # linear conv of b inputs with 2b-1 lags fits in 4b
            t += (2 * rfft(n)) / (2 * b)
            sp += ((n // 2 + 1) * 6.0) / (2 * b)
            lvl += 1
        mixer_direct = 2.0             # the lag-0 term, h0 * u_t
        return dict(dense=L_LAYERS * hy_dense + head, mixer=L_LAYERS * chans * (sp + mixer_direct),
                    transform=L_LAYERS * chans * t, levels=lvl)
    raise ValueError(variant)


# ────────────────────────────────────────────────── numerical checks ──
def causal_fft_conv(u: np.ndarray, h: np.ndarray, pad: bool = True) -> np.ndarray:
    s = len(u)
    n = pow2_at_least(2 * s) if pad else s
    return np.fft.irfft(np.fft.rfft(u, n) * np.fft.rfft(h, n), n)[:s]


def direct_causal_conv(u: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.convolve(u, h)[: len(u)]


def tiled_online_conv(u: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Online causal convolution, one input at a time, in the relaxed tiled style.

    Every pair (j < i) is charged exactly once: at the highest bit l where j and i differ,
    j lies in block [a, a+B) and i in [a+B, a+2B) with B = 2^l and a a multiple of 2B. When
    input a+B-1 arrives, that block is convolved (by FFT) with lags 1..2B-1 and added to the
    outputs a+B..a+2B-1, all still in the future. The lag-0 term is direct."""
    s = len(u)
    y = np.zeros(s)
    for t in range(s):
        y[t] += h[0] * u[t]                    # output t is final now: emit it
        b = 1
        while b <= t + 1:
            if (t + 1) % (2 * b) == b:         # block [t+1-b, t] has just completed
                a = t + 1 - b
                blk = u[a:t + 1]
                lags = np.zeros(2 * b)
                lags[1:2 * b] = h[1:2 * b] if len(h) >= 2 * b else np.pad(h[1:], (0, 2 * b - len(h)))
                n = 4 * b
                c = np.fft.irfft(np.fft.rfft(blk, n) * np.fft.rfft(lags, n), n)
                for i in range(a + b, min(a + 2 * b, s)):
                    y[i] += c[i - a]
            b *= 2
    return y


def checks(rng: np.random.Generator) -> dict:
    s = 1024
    u = rng.standard_normal(s)
    h = rng.standard_normal(s) * np.exp(-np.arange(s) / 200.0)
    ref = direct_causal_conv(u, h)
    scale = np.max(np.abs(ref))
    out = {}
    out["fft_padded_max_rel"] = float(np.max(np.abs(causal_fft_conv(u, h) - ref)) / scale)
    out["fft_unpadded_max_rel"] = float(np.max(np.abs(causal_fft_conv(u, h, pad=False) - ref)) / scale)
    out["tiled_online_max_rel"] = float(np.max(np.abs(tiled_online_conv(u, h) - ref)) / scale)
    # Causality probe: perturb the last input and see which earlier outputs move.
    u2 = u.copy()
    u2[-1] += 1.0
    moved_pad = np.abs(causal_fft_conv(u2, h) - causal_fft_conv(u, h))[:-1].max()
    moved_circ = np.abs(causal_fft_conv(u2, h, pad=False) - causal_fft_conv(u, h, pad=False))[:-1].max()
    out["future_leak_padded"] = float(moved_pad)
    out["future_leak_unpadded"] = float(moved_circ)
    # FNet mixing: Re(FFT2) of a (s, d) block; perturb the last token, look at the first.
    x = rng.standard_normal((256, 64))
    x2 = x.copy()
    x2[-1, :] += 1.0
    f1, f2 = np.real(np.fft.fft2(x)), np.real(np.fft.fft2(x2))
    out["fnet_first_token_change"] = float(np.abs(f2[0] - f1[0]).max())
    # Prefill/decode agreement: token-by-token direct outputs equal the prefill FFT outputs.
    dec = np.array([np.dot(u[: t + 1][::-1], h[: t + 1]) for t in range(s)])
    out["decode_vs_prefill_max_rel"] = float(np.max(np.abs(dec - causal_fft_conv(u, h))) / scale)
    out["n"] = s
    return out


# ───────────────────────────────────────────────── analogue precision ──
def bf16(x: np.ndarray) -> np.ndarray:
    b = x.astype(np.float32).view(np.uint32)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000            # round to nearest even
    return b.astype(np.uint32).view(np.float32).astype(np.float64)


def fp8_e4m3(x: np.ndarray) -> np.ndarray:
    """Round to FP8 E4M3 (3 mantissa bits; max 448; subnormals below 2^-6), per-tensor scaled
    so the largest magnitude maps to 448, as inference engines do."""
    scale = 448.0 / np.max(np.abs(x))
    y = x * scale
    e = np.floor(np.log2(np.maximum(np.abs(y), 2.0 ** -6)))
    q = 2.0 ** (e - 3)
    return np.round(y / q) * q / scale


def int8(x: np.ndarray) -> np.ndarray:
    scale = 127.0 / np.max(np.abs(x))
    return np.round(x * scale) / scale


def rel_rms(a: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - ref) ** 2) / np.mean(ref ** 2)))


def analogue_conv(u, h, enob: float, rng, passes: int = 1, slices: int = 1, slice_bits: int = 4):
    """A 4f-style causal convolution whose output is read by an ADC of ``enob`` effective
    bits. The ADC full scale is set per pass to the pass's own peak (ideal gain control);
    its error is modelled as noise of rms q/sqrt(12), q = 2*FS / 2^enob (the definition of
    ENOB). ``passes`` > 1 repeats and averages (Garg et al.); ``slices`` > 1 splits the
    input into digit planes and recombines them digitally (the FHE route)."""
    def one(x):
        y = causal_fft_conv(x, h)
        fs = np.max(np.abs(y))
        q = 2 * fs / 2 ** enob
        return np.mean([y + rng.standard_normal(len(y)) * q / math.sqrt(12) for _ in range(passes)],
                       axis=0)
    if slices == 1:
        return one(u)
    # signed-magnitude digit planes of a block-floating-point input
    sc = np.max(np.abs(u))
    total_bits = slices * slice_bits
    iu = np.round(u / sc * (2 ** (total_bits - 1) - 1))
    sign, mag = np.sign(iu), np.abs(iu).astype(np.int64)
    acc = np.zeros(len(u))
    for k in range(slices):
        plane = sign * ((mag >> (slice_bits * k)) & ((1 << slice_bits) - 1))
        acc += one(plane) * 2 ** (slice_bits * k)
    return acc * sc / (2 ** (total_bits - 1) - 1)


def precision(rng) -> dict:
    out = {"lengths": {}, "targets": {}}
    for s in (256, 4096, 65536):
        u = rng.standard_normal(s)
        h = rng.standard_normal(s) * np.exp(-np.arange(s) / (s / 8))
        y = causal_fft_conv(u, h)
        crest = float(np.max(np.abs(y)) / np.sqrt(np.mean(y ** 2)))
        rows = {}
        for enob in (4, 6, 8, 10, 12, 14):
            rows[enob] = rel_rms(analogue_conv(u, h, enob, rng), y)
        avg4 = rel_rms(analogue_conv(u, h, 8, rng, passes=4), y)
        avg16 = rel_rms(analogue_conv(u, h, 8, rng, passes=16), y)
        sliced = rel_rms(analogue_conv(u, h, 8, rng, slices=2, slice_bits=8), y)
        predicted = {e: crest * 2 ** -e * 2 / math.sqrt(12) for e in rows}
        out["lengths"][s] = dict(crest=crest, err=rows, predicted=predicted,
                                 avg4_at8=avg4, avg16_at8=avg16, sliced2x8_at8=sliced)
        if s == 4096:
            out["targets"] = {"BF16": rel_rms(bf16(y), y), "FP8 E4M3": rel_rms(fp8_e4m3(y), y),
                              "INT8": rel_rms(int8(y), y)}
            out["crest_4096"] = crest
    # ENOB needed to match each format's own rounding error (same metric), from the rule
    # rel = crest * 2^-ENOB * 2/sqrt(12)  =>  ENOB = log2(crest * 2 / (sqrt(12) * rel))
    c = out["crest_4096"]
    out["enob_needed"] = {k: math.log2(c * 2 / (math.sqrt(12) * v)) for k, v in out["targets"].items()}
    return out


def fhe_rule_bits(n: int, b: int, d: int) -> int:
    """FHESim 04's exactness rule: smallest ENOB with 2^(ENOB-1) > d * n * (2^b - 1)^2."""
    fs = d * n * ((1 << b) - 1) ** 2
    return math.floor(math.log2(fs)) + 2 if fs > 0 else 1


# ───────────────────────────────────────────────── conversion energy ──
def conversions_per_token(variant: str, s: int) -> float:
    """DAC+ADC sample pairs per prompt token for the transform work, per forward pass.

    A 4f pass takes the input in (one DAC sample per real input) and returns the convolved
    output (one ADC sample per real output, coherent detection): zero padding is optical, not
    converted. Hyena: ORDER*D long convs per layer. FNet: the whole (s, d) block per layer.
    Circulant: each block-circulant matrix needs n inputs in and m outputs out per token."""
    if variant == "hyena":
        return L_LAYERS * ORDER * D
    if variant == "hybrid":
        return sum(ORDER * D for layer in range(L_LAYERS) if layer % 4)
    if variant == "fnet":
        return L_LAYERS * D
    if variant == "hyena_circ":
        mats = [((ORDER + 1) * D, D), (D, D), (FF, D), (FF, D), (D, FF)]
        # a circulant block row is a sum over input blocks: the sum can be done optically
        # (incoherent addition of the m/k outputs) only per pass, so count n in + m out
        return L_LAYERS * (ORDER * D + sum(max(m, n) for m, n in mats))
    return 0.0


def walden_pj(fom_fj: float, enob: float) -> float:
    return fom_fj * 2 ** enob * 1e-3


# ─────────────────────────────────────────────────────────── report ──
def fmt(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f}%"


def main() -> None:
    rng = np.random.default_rng(20261004)
    res: dict = {"machine": f"{platform.processor() or platform.machine()}, Python {sys.version.split()[0]}, "
                            f"NumPy {np.__version__}"}
    lines: list[str] = []
    w = lines.append
    w("# FLOP-share (Amdahl) analysis: Fourier optics in LLM inference\n")
    w(f"Generated by `analysis/flop_share.py` on 2026-10-04 ({res['machine']}). Every number in the "
      "FOptInf decks comes from this file.\n")
    w("Conventions: matmul 2mnk; real FFT of length n = 2.5 n log2 n and complex = 5 n log2 n "
      "(FFTW's convention); complex pointwise multiply 6, complex add 2; causal long "
      "convolutions zero-pad to the next power of two at least 2L. Softmax, norms and "
      "activations are not counted (as in Disaggregated_Inference_Sim). Shape: Llama-3-8B "
      f"(L={L_LAYERS}, d={D}, d_ff={FF}, GQA {M.n_kv_heads} KV heads, vocab {V}). The LM head is "
      "charged for every prompt token, as the simulator does, unless a row says otherwise.\n")

    # ── 0. cross-check the transformer row against the simulator ──
    cm = CostModel(LLAMA3_8B, H100_SXM, 1)
    sim_ok = []
    for s in PROMPTS[:3]:
        mine = prefill("transformer", [s]).total
        sims = cm.prefill([s]).flops
        sim_ok.append((s, mine, sims, mine == sims))
    dec_mine = decode_per_token("transformer", 2048)
    dec_sim = cm.decode([2048]).flops
    res["crosscheck"] = dict(prefill=[dict(s=s, mine=a, sim=b, equal=e) for s, a, b, e in sim_ok],
                             decode_2048=dict(mine=dec_mine["dense"] + dec_mine["mixer"], sim=dec_sim))
    w("## 1. Cross-check against the simulator\n")
    w("The transformer variant must reproduce `disagg_sim.hardware.CostModel` exactly (same model shape, "
      "same counting rules):\n")
    w("| Step | This script (FLOPs) | Simulator (FLOPs) | Equal |")
    w("|---|---|---|---|")
    for s, a, b, e in sim_ok:
        w(f"| prefill {s} | {a:,.0f} | {b:,.0f} | {e} |")
    dm = dec_mine["dense"] + dec_mine["mixer"]
    w(f"| decode b=1 ctx 2048 | {dm:,.0f} | {dec_sim:,.0f} | {dm == dec_sim} |\n")

    # ── 2. prefill shares ──
    w("## 2. Prefill: what share of the FLOPs is transform work?\n")
    w("*Optical share* = FFT/IFFT FLOPs plus the pointwise multiplies in the Fourier domain (what a "
      "4f system's mask does). *Amdahl bound* = 1 / (1 - optical share): the prefill speed-up if "
      "that work became free and nothing else changed.\n")
    w("| Variant | Prompt | GFLOP | Dense matmul | Attention | Optical share | Amdahl bound |")
    w("|---|---|---|---|---|---|---|")
    shares: dict = {}
    for v in VARIANTS:
        shares[v] = {}
        for s in PROMPTS:
            o = prefill(v, [s])
            f = o.optical / o.total
            shares[v][s] = dict(total=o.total, dense=o.dense / o.total, attention=o.attention / o.total,
                                transform=o.transform / o.total, spectral=o.spectral / o.total,
                                other=o.other / o.total, optical=f, amdahl=1 / (1 - f))
            w(f"| {LABEL[v]} | {s:,} | {fmt(o.total / 1e9)} | {pct(o.dense / o.total, 1)} | "
              f"{pct(o.attention / o.total, 1)} | {pct(f)} | {1 / (1 - f):.3f}x |")
    res["prefill"] = shares
    w("")
    w("The optical share at a glance (the same numbers, one row per variant):\n")
    w("| Variant | " + " | ".join(f"{s:,} tokens" for s in PROMPTS) + " |")
    w("|---" * (len(PROMPTS) + 1) + "|")
    for v in VARIANTS:
        w(f"| {LABEL[v]} | " + " | ".join(pct(shares[v][s]["optical"]) for s in PROMPTS) + " |")
    w("")
    # last-token LM head
    w("The same with the LM head applied to the last prompt token only (what serving engines do):\n")
    w("| Variant | Prompt | GFLOP | Optical share | Amdahl bound |")
    w("|---|---|---|---|---|")
    res["prefill_lm_last"] = {}
    for v in ("hyena", "hyena_circ"):
        res["prefill_lm_last"][v] = {}
        for s in (2048, 32768):
            o = prefill(v, [s], lm_all_tokens=False)
            f = o.optical / o.total
            res["prefill_lm_last"][v][s] = dict(total=o.total, optical=f, amdahl=1 / (1 - f))
            w(f"| {LABEL[v]} | {s:,} | {fmt(o.total / 1e9)} | {pct(f)} | {1 / (1 - f):.3f}x |")
    w("")

    # Hyena paper count, for comparison
    s = 2048
    mine_layer = hyena_block([s])
    paper = hyena_paper_count(s)
    paper_fft = 2 * 5 * ORDER * D * math.log2(s) * s
    res["hyena_paper_compare"] = dict(s=s, mine_layer=mine_layer.total, paper_layer=paper,
                                      mine_fft_share_layer=mine_layer.optical / mine_layer.total,
                                      paper_fft_share_layer=paper_fft / paper)
    w(f"* The Hyena operator (one layer's mixer, without the MLP) counted here against the Hyena paper's own "
      f"formula (appendix A.2, s = {s:,}): this script {mine_layer.total / 1e9:.2f} GFLOP, of which {pct(mine_layer.optical / mine_layer.total)} "
      f"is FFT convolution; the paper's formula {paper / 1e9:.2f} GFLOP, of which "
      f"{pct(paper_fft / paper)}. The difference is padding (2L here) and the FFT constant; the "
      "conclusion does not depend on it.\n")

    # ── 3. time-weighted share ──
    w("## 3. Time-weighted share (illustrative)\n")
    w("GPUs run FFTs far below their matmul rate (FlashFFTConv, arXiv:2311.05908, reports poor "
      "utilisation of the FFT on matmul units). If transform FLOPs run at a fraction r of the dense "
      "rate, their share of *time* is larger. r is an illustrative assumption, not a measurement.\n")
    w("| Variant | Prompt | r = 1 | r = 1/4 | r = 1/16 | Amdahl bound at r = 1/16 |")
    w("|---|---|---|---|---|---|")
    res["time_weighted"] = {}
    for v in ("fnet", "hyena", "hybrid", "hyena_circ"):
        res["time_weighted"][v] = {}
        for s in (2048, 32768):
            o = prefill(v, [s])
            row = {}
            for r in (1, 4, 16):
                t_opt = o.optical * r
                row[r] = t_opt / (o.total - o.optical + t_opt)
            res["time_weighted"][v][s] = row
            w(f"| {LABEL[v]} | {s:,} | {pct(row[1])} | {pct(row[4])} | {pct(row[16])} | "
              f"{1 / (1 - row[16]):.3f}x |")
    w("")

    # ── 4. decode ──
    w("## 4. Decode: one new token, batch 1\n")
    w("Decode generates one token at a time. A long convolution can be served by caching past "
      "inputs and taking a direct dot product (O(c) per token), by a distilled recurrence "
      "(Laughing Hyena, arXiv:2310.18780; state size 16 here, illustrative), or by relaxed "
      "tiling in which blocks of 2^l inputs are convolved by FFT once they complete (Flash "
      "Inference, arXiv:2410.12982). Recomputing the whole FFT every token is shown as the "
      "naive bound.\n")
    w("| Mixer | Context | GFLOP / token | Mixer FLOPs / token | Transform share | Tile levels |")
    w("|---|---|---|---|---|---|")
    res["decode"] = {}
    for v in ("transformer", "hyena_direct", "hyena_distilled", "hyena_tiled", "hyena_recompute"):
        res["decode"][v] = {}
        for c in (2048, 32768):
            d = decode_per_token(v, c)
            tot = d["dense"] + d["mixer"] + d["transform"]
            res["decode"][v][c] = dict(total=tot, mixer=d["mixer"] + d["transform"],
                                       transform_share=d["transform"] / tot, levels=d.get("levels"))
            w(f"| {v} | {c:,} | {tot / 1e9:,.3f} | {(d['mixer'] + d['transform']) / 1e6:,.1f} M | "
              f"{pct(d['transform'] / tot)} | {d.get('levels', '')} |")
    w("")
    lv = {c: res["decode"]["hyena_tiled"][c]["levels"] for c in (2048, 32768)}
    res["decode_conversions"] = {c: 0.5 * n for c, n in lv.items()}
    w("* Conversions per token per channel (DAC+ADC pairs): a prefill long convolution converts each "
      "input once and each output once, **1 pair per token**. Relaxed tiling at decode converts, at "
      "each of its tile levels, a block of B inputs and B outputs every 2B steps, 0.5 pairs per token "
      "per level: " + "; ".join(f"**{0.5 * n:g} pairs** at context {c:,} ({n} levels)" for c, n in lv.items())
      + ", in small passes on the latency-critical path.\n")

    # ── 5. numerical checks ──
    ck = checks(rng)
    res["checks"] = ck
    w("## 5. Numerical checks (NumPy, n = 1,024, random input, decaying random filter)\n")
    w("| Check | Result |")
    w("|---|---|")
    w(f"| Causal FFT convolution (zero-padded) vs direct, max error / peak | {ck['fft_padded_max_rel']:.1e} |")
    w(f"| Unpadded (circular) FFT convolution vs direct, max error / peak | {ck['fft_unpadded_max_rel']:.2f} |")
    w(f"| Perturb the last input: largest change in any earlier output, padded | {ck['future_leak_padded']:.1e} |")
    w(f"| ... unpadded (circular wrap leaks the future) | {ck['future_leak_unpadded']:.2f} |")
    w(f"| FNet mixing: perturb the last token, change in the first token's output | {ck['fnet_first_token_change']:.1f} |")
    w(f"| Token-by-token decode (direct) vs prefill FFT output, max error / peak | {ck['decode_vs_prefill_max_rel']:.1e} |")
    w(f"| Relaxed tiled online convolution vs direct, max error / peak | {ck['tiled_online_max_rel']:.1e} |\n")

    # ── 6. precision ──
    pr = precision(rng)
    res["precision"] = pr
    w("## 6. Analogue precision for floating-point workloads\n")
    w("A causal FFT convolution read out through an ADC of a given ENOB, with the full scale set "
      "to each pass's own peak (ideal gain control). Error = relative RMS error against the exact "
      "float64 result. The rule rel = crest x 2^-ENOB x 2/sqrt(12) follows from the definition of "
      "ENOB, where crest is the output's peak-to-RMS ratio.\n")
    w("| Length | Crest | ENOB 4 | ENOB 6 | ENOB 8 | ENOB 10 | ENOB 12 | ENOB 14 | Rule at ENOB 8 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s, r in pr["lengths"].items():
        e = r["err"]
        w(f"| {s:,} | {r['crest']:.2f} | " + " | ".join(f"{e[k]:.1e}" for k in (4, 6, 8, 10, 12, 14))
          + f" | {r['predicted'][8]:.1e} |")
    w("")
    w("Ways to buy precision at ENOB 8 (same metric):\n")
    w("| Length | 1 pass | 4 passes averaged | 16 passes averaged | 2 digit planes of 8 bits |")
    w("|---|---|---|---|---|")
    for s, r in pr["lengths"].items():
        w(f"| {s:,} | {r['err'][8]:.1e} | {r['avg4_at8']:.1e} | {r['avg16_at8']:.1e} | {r['sliced2x8_at8']:.1e} |")
    w("")
    w("Averaging k passes cuts noise by sqrt(k): one extra bit costs 4x the passes (Garg et al., "
      "arXiv:2102.06365). Digit planes do *not* help here: the top plane carries almost all of the "
      "signal and is read at the same ENOB relative to its own peak.\n")
    w("What each number format's own rounding costs (same metric, length 4,096) and the ENOB the rule "
      "says matches it:\n")
    w("| Format | Rounding error (relative RMS) | ENOB to match | Averaged passes at ENOB 8 |")
    w("|---|---|---|---|")
    pr["passes_at8"] = {}
    for k, v in pr["targets"].items():
        need = pr["enob_needed"][k]
        p8 = math.ceil(4 ** max(0.0, need - 8))
        pr["passes_at8"][k] = p8
        w(f"| {k} | {v:.1e} | {need:.1f} | {p8} |")
    w("")
    rule = {}
    for n in (16, 4096, 65536):
        rule[n] = fhe_rule_bits(n, 1, 8)
    res["fhe_rule_long"] = rule
    w("* Exact rounding, by FHESim 04's rule 2^(ENOB-1) > d n (2^b - 1)^2 with 8-bit inputs and filters in "
      "1-bit digits (d = 8, the most forgiving split; n = products summed per output), needs "
      + "; ".join(f"ENOB {b} for L = {n:,}" for n, b in rule.items())
      + ": feasible for FHE's 16-point blocks, not for sequence-length convolutions, so float workloads "
      "use the noise rule above instead.\n")

    # ── 7. conversion energy ──
    w("## 7. Conversion energy against the digital work it replaces\n")
    w("Converter energy per sample = Walden FoM x 2^ENOB, with FHESim 04's illustrative FoMs (DAC 10 "
      "fJ/step, ADC 20 fJ/step). Digital energy = the replaced transform FLOPs x 1.0 pJ/FLOP "
      "(Disaggregated_Inference_Sim's illustrative H100 coefficient). Prompt 2,048, per prompt token.\n")
    w("Energy per conversion at each ENOB (the Walden rule, illustrative FoMs):\n")
    w("| ENOB | DAC pJ | ADC pJ | Pair pJ |")
    w("|---|---|---|---|")
    res["walden"] = {}
    for e in (4, 6, 8, 10, 12, 14, 16):
        dac, adc = walden_pj(10, e), walden_pj(20, e)
        res["walden"][e] = dict(dac=dac, adc=adc)
        w(f"| {e} | {dac:,.2f} | {adc:,.2f} | {dac + adc:,.2f} |")
    w("")
    w("Per prompt token, for each variant's transform work:\n")
    w("| Variant | Pairs / token | Replaced FLOPs / token | Digital uJ | ENOB 6 uJ | ENOB 8 uJ | ENOB 10 uJ | ENOB 12 uJ | Break-even ENOB |")
    w("|---|---|---|---|---|---|---|---|---|")
    res["energy"] = {}
    s = 2048
    for v in ("fnet", "hyena", "hybrid", "hyena_circ"):
        o = prefill(v, [s])
        conv = conversions_per_token(v, s)
        rep = o.optical / s
        dig_uj = rep * 1.0e-12 * 1e6
        row = {}
        for e in (6, 8, 10, 12):
            row[e] = conv * (walden_pj(10, e) + walden_pj(20, e)) * 1e-12 * 1e6
        # 30 fJ * 2^E * conv = rep * 1 pJ  =>  E = log2(rep / (conv * 0.03))
        be = math.log2(rep * 1.0 / (conv * 0.030)) if conv else float("nan")
        res["energy"][v] = dict(conversions=conv, replaced_flops=rep, digital_uj=dig_uj, optical_uj=row,
                                breakeven_enob=be, total_uj_digital=o.total / s * 1e-12 * 1e6)
        w(f"| {LABEL[v]} | {conv:,.0f} | {rep / 1e6:,.1f} M | {dig_uj:,.1f} | " +
          " | ".join(f"{row[e]:,.1f}" for e in (6, 8, 10, 12)) + f" | {be:.1f} |")
    w("")
    tot = res["energy"]["hyena"]["total_uj_digital"]
    w(f"* For scale: the whole Hyena-2 prefill costs {tot:,.0f} uJ per prompt token of dynamic FLOP energy "
      "at 1 pJ/FLOP; the transform work is a small slice of it.\n")

    # ── 8. mask capacity ──
    w("## 8. Holding the filters: Fourier-plane mask capacity (illustrative)\n")
    w("A 4f system multiplies by one mask per pass. Every long-convolution filter (or circulant block) "
      "needs its spectrum on the mask while the matching inputs stream through, so the mask values a "
      "forward pass needs, against what a spatial light modulator holds, decide how often the mask "
      "must be rewritten. Device figures from Miscuglio et al. (arXiv:2008.05853, Optica 2020): a "
      "2-megapixel digital micromirror device runs at 1,031 Hz with 8-bit depth and about 20 kHz with "
      "1-bit depth; liquid-crystal (LC) SLMs of the same resolution settle in tens of Hz (30 Hz used here). "
      "Rewrites = mask values / 2 million, rounded up. "
      "One complex mask value per pixel is an optimistic simplification.\n")
    w("| Variant | Prompt | Complex mask values per forward pass | Rewrites | DMD 20 kHz | DMD 1.03 kHz | LC 30 Hz |")
    w("|---|---|---|---|---|---|---|")
    res["mask"] = {}
    for v in ("hyena", "hyena_circ"):
        res["mask"][v] = {}
        for s in (2048, 32768):
            n = pow2_at_least(2 * s)
            vals = L_LAYERS * ORDER * D * (n // 2 + 1)
            if v == "hyena_circ":
                k = CIRC_BLOCK
                mats = [((ORDER + 1) * D, D), (D, D), (FF, D), (FF, D), (D, FF)]
                vals = L_LAYERS * (ORDER * D * (n // 2 + 1) + sum((m // k) * (nn // k) * (k // 2 + 1) for m, nn in mats))
            rew = math.ceil(vals / 2e6)
            res["mask"][v][s] = dict(values=vals, rewrites=rew, t20k=rew / 2e4, t1031=rew / 1031, t30=rew / 30)
            w(f"| {LABEL[v]} | {s:,} | {vals:,.0f} | {rew:,} | {rew / 2e4 * 1e3:,.1f} ms | {rew / 1031 * 1e3:,.0f} ms | {rew / 30:,.1f} s |")
    w("")
    w("A rewrite is amortised over every sequence in the prefill batch that shares the mask (all of "
      "them: the filters are weights). The prefill of one 2,048-token Llama-3-8B prompt on one H100 "
      "takes 59.0 ms in Disaggregated_Inference_Sim (results.md section 1), for comparison.\n")

    # ── 10. what prefill hands to decode ──
    w("## 9. What prefill hands to decode: cache and state bytes\n")
    w("Disaggregation moves this over the KV link once per request. BF16 (2 bytes) per real value; "
      "the distilled state is complex (4 bytes per value), state size 16 (illustrative).\n")
    w("| Decode style | Bytes per prompt token | Bytes for a 2,048-token prompt | Grows with the prompt |")
    w("|---|---|---|---|")
    kv_tok = M.kv_bytes_per_token
    conv_tok = L_LAYERS * ORDER * D * 2.0
    dist = L_LAYERS * ORDER * D * 16 * 4.0
    res["handoff"] = dict(kv_per_token=kv_tok, conv_cache_per_token=conv_tok, distilled_state=dist,
                          kv_2048=kv_tok * 2048, conv_2048=conv_tok * 2048)
    w(f"| Transformer KV cache (GQA) | {kv_tok:,.0f} | {kv_tok * 2048 / 1e6:,.1f} MB | yes |")
    w(f"| Hyena direct: cached projection inputs | {conv_tok:,.0f} | {conv_tok * 2048 / 1e6:,.1f} MB | yes |")
    w(f"| Hyena distilled recurrence state | n/a | {dist / 1e6:,.1f} MB | no |\n")
    w("Hyena-2's cached inputs are 2 x d per layer per token against GQA's 2 x 8 heads x 128: four "
      "times the bytes, so direct-convolution decode is *more* memory-bound than attention decode, "
      "and its hand-off four times larger. A distilled recurrence makes the hand-off a constant.\n")

    # ── 9. verdict ──
    hy = shares["hyena"][2048]
    hc = shares["hyena_circ"][2048]
    res["headline"] = dict(hyena_optical_2048=hy["optical"], hyena_amdahl_2048=hy["amdahl"],
                           circ_optical_2048=hc["optical"], circ_amdahl_2048=hc["amdahl"],
                           fnet_optical_2048=shares["fnet"][2048]["optical"],
                           transformer_attention_32768=shares["transformer"][32768]["attention"])
    w("## 10. Headline\n")
    w(f"* A Llama-3-8B-shaped **Hyena-2** model spends {pct(hy['optical'])} of its prefill FLOPs on "
      f"transform work at 2,048 tokens: if the optics made it free, prefill would be at most "
      f"{hy['amdahl']:.3f}x faster (Amdahl). The FFT is what makes Hyena cheap; the MLP and "
      "projections stay digital and dominate.")
    w(f"* **FNet-shaped** mixing is {pct(shares['fnet'][2048]['optical'])} of prefill FLOPs, and it is "
      "not causal, so it does not apply to a decoder at all.")
    hl = res["prefill_lm_last"]["hyena_circ"][2048]
    res["headline"].update(circ_lm_last_optical_2048=hl["optical"], circ_lm_last_amdahl_2048=hl["amdahl"])
    w(f"* Only when the **weights themselves are structured** (block-circulant, speculative at LLM "
      f"scale) does transform work matter: {pct(hc['optical'])} of a far smaller total (Amdahl bound "
      f"{hc['amdahl']:.2f}x) with the LM head charged for every prompt token, because the dense LM "
      f"head is then the bulk of the work; with the LM head on the last token only, "
      f"{pct(hl['optical'])} and {hl['amdahl']:.1f}x.")
    w(f"* **Decode** is token by token: a direct cached dot product or a distilled recurrence needs no "
      "transform; relaxed tiling does use FFTs, but in small, frequent, latency-critical passes with "
      f"{res['decode_conversions'][2048]:g}x prefill's conversions per token per channel at context "
      "2,048. Prefill, with long transforms and a "
      "mask held across the whole batch, is the natural target, which is why disaggregation fits.")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
