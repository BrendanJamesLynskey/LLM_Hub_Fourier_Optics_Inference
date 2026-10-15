#!/usr/bin/env python3
"""Inside a photonic link: first-order models of a modulator, a coherent receiver and the signal chain of a
four-channel photonic link module (arXiv:2608.21536), for FOptInf 04.

It writes results.md and results.json beside this file. Every number on the deck's slides comes from them (through
build.py's TABLE/CELL directives) or from js/fop04_models.js, the port that draws the animations; test_link_hw.py
checks that the port reproduces these functions (exact for arithmetic, within 1e-12 relative where sin, cos, exp
or pow enter, because the C library and the JavaScript engine may round those differently in the last place).

Every model is a first-order textbook model. Parameters are either from a cited source or ILLUSTRATIVE, and the
deck says which. The code is plain Python with explicit loops in a fixed order (no NumPy), so the port can follow
the same floating-point operations.

Run:  python3 link_hw.py
"""
import json
import math
import platform
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q_E = 1.602176634e-19            # electron charge, C (exact, SI 2019)

# ── parameters: (value, source) ───────────────────────────────────────────────────────────────────────────────
P = dict(
    R=0.65,             # PD responsivity A/W: foundry C-band Ge PD, TE, 1530 nm (Hu et al., Photonics Research 11, 961, 2023)
    baud=14e9,          # PIC device rate stated in the paper (14 GBaud)
    i_n=10e-12,         # TIA input-referred noise density A/rtHz: ILLUSTRATIVE, typical of published 50G-class linear TIAs
    rin_db=-145.0,      # laser RIN dB/Hz: ILLUSTRATIVE (DFB class)
    cmrr_db=25.0,       # balanced-pair common-mode rejection dB: ILLUSTRATIVE
    p_sig_uW=10.0,      # signal power per detector for output 3's full swing, uW: ILLUSTRATIVE
    swing=0.22,         # drive +/- swing in units of V_pi: ILLUSTRATIVE
    vpi=6.8,            # V_pi in volts: ILLUSTRATIVE (silicon depletion MZMs are several volts; Felix Rosa et al. 2017)
    p_lo_dbm=-5.0,      # LO power per detector pair, dBm: ILLUSTRATIVE (a 7-way then 4-way split of a ~13-16 dBm laser)
    r_t=1000.0,         # TIA transimpedance, ohms: ILLUSTRATIVE
    q_req=4.9,          # Q (half a level step over rms noise) for about 1e-6 per decision: Gaussian noise
    gbw=1.0e12,         # TIA amplifier gain-bandwidth product A*f_A, Hz: ILLUSTRATIVE (SiGe-class)
)
LO_PHASE_DEG = (0.0, 0.0, 0.0, 90.0)           # detectors 1-3 read Re, detector 4 reads Im (a patent cited by the paper)
ERR_IN_DEG = (0.0, 25.0, -15.0, 10.0)          # uncalibrated input-path phases: ILLUSTRATIVE
ERR_OUT_DEG = (20.0, -30.0, 15.0, 35.0)        # uncalibrated output/LO-path phases: ILLUSTRATIVE
EX_BITS = (1, 0, 1, 1)                         # the worked example
VECS = [tuple((n >> k) & 1 for k in range(4)) for n in range(16)]   # input n: lane k carries bit k of n
W4 = ((1, 0), (0, -1), (-1, 0), (0, 1))        # (-j)^m for m = 0..3: the exact 4-point DFT twiddles


def rad(d):
    return d * math.pi / 180.0


# ── the 4-point DFT (exact twiddles, fixed summation order) ──────────────────────────────────────────────────
def dft4(xr, xi):
    """X_k = sum_n x_n (-j)^(k n): what the optical Fourier transform computes, before its 1/2 scale."""
    yr, yi = [0.0] * 4, [0.0] * 4
    for k in range(4):
        sr, si = 0.0, 0.0
        for n in range(4):
            wr, wi = W4[(k * n) % 4]
            sr = sr + (xr[n] * wr - xi[n] * wi)
            si = si + (xr[n] * wi + xi[n] * wr)
        yr[k], yi[k] = sr, si
    return yr, yi


def dft4_int(bits):
    """The integer DFT of a 0/1 vector: the alphabet the module's outputs carry."""
    re, im = [0] * 4, [0] * 4
    for k in range(4):
        for n in range(4):
            wr, wi = W4[(k * n) % 4]
            re[k] += bits[n] * wr
            im[k] += bits[n] * wi
    return re, im


def rotate(xr, xi, deg):
    """Multiply each element by exp(j * deg)."""
    outr, outi = [0.0] * len(xr), [0.0] * len(xr)
    for n in range(len(xr)):
        c, s = math.cos(rad(deg[n])), math.sin(rad(deg[n]))
        outr[n] = xr[n] * c - xi[n] * s
        outi[n] = xr[n] * s + xi[n] * c
    return outr, outi


# ── the modulator ──────────────────────────────────────────────────────────────────────────────────────────────
def mzm_arms(dphi, push_pull=True):
    """Two arms of half the input field each, recombined. Push-pull: +dphi/2 and -dphi/2, so the sum is real
    (cos(dphi/2)); single-arm drive puts all of dphi on one arm and the sum picks up a phase (chirp)."""
    p1, p2 = (dphi / 2.0, -dphi / 2.0) if push_pull else (dphi, 0.0)
    a1 = (0.5 * math.cos(p1), 0.5 * math.sin(p1))
    a2 = (0.5 * math.cos(p2), 0.5 * math.sin(p2))
    return dict(a1=a1, a2=a2, sum=(a1[0] + a2[0], a1[1] + a2[1]))


def bias_point(bias, swing):
    """Binary drive of +/- swing (units of V_pi, so +/- pi*swing of phase) about a bias phase. Returns the field and
    power for '1' (phase lowered, towards the peak) and '0', with field = cos(dphi/2), power = field^2."""
    d1, d0 = bias - math.pi * swing, bias + math.pi * swing
    e1, e0 = math.cos(d1 / 2.0), math.cos(d0 / 2.0)
    return dict(e1=e1, e0=e0, p1=e1 * e1, p0=e0 * e0, de=e1 - e0, dp=e1 * e1 - e0 * e0)


def dither(phi, a=0.1, n=64):
    """A bias controller's error signals at bias phase phi: a dither a*sin(theta) on the bias, the monitor power
    P = (1 + cos(phase)) / 2 sampled n times per dither period, and synchronous (lock-in) detection at the 1st and
    2nd harmonics. Analytically avg = 1/2 + cos(phi) J0(a) / 2, h1 = -sin(phi) J1(a), h2 = cos(phi) J2(a)."""
    s0 = s1 = s2 = 0.0
    for m in range(n):
        th = 2.0 * math.pi * m / n
        p = 0.5 * (1.0 + math.cos(phi + a * math.sin(th)))
        s0 = s0 + p
        s1 = s1 + p * math.sin(th)
        s2 = s2 + p * math.cos(2.0 * th)
    return dict(avg=s0 / n, h1=2.0 * s1 / n, h2=2.0 * s2 / n)


def bessel_j(order, x, terms=20):
    s = 0.0
    for k in range(terms):
        s += (-1) ** k * (x / 2) ** (2 * k + order) / (math.factorial(k) * math.factorial(k + order))
    return s


# ── the chain ──────────────────────────────────────────────────────────────────────────────────────────────────
def lane_levels():
    """Quadrature bias, +/- swing: the field of a '1' and a '0' as a fraction of the maximum field."""
    s = math.sin(math.pi * P["swing"])
    return math.sqrt(0.5 * (1.0 + s)), math.sqrt(0.5 * (1.0 - s))


def chain_model(bits, err_in=None, err_out=None):
    """Field and current at each stage for one input vector. Fields in sqrt(uW), currents in uA.

    Lane fields x_n = sqrt(p_lane) * (f1 or f0), rotated by the input-path phase errors; the passive optical
    Fourier transform is unitary, Y = DFT(x) / 2; then the output-path phase errors; each balanced homodyne
    detector reads i = 2 R E_LO Re(Y exp(-j lo_phase))."""
    f1, f0 = lane_levels()
    d = f1 - f0
    p_lane = P["p_sig_uW"] / (d * d)               # uW per lane at full field: output 3's full swing = p_sig
    a = math.sqrt(p_lane)
    xr = [a * (f1 if b else f0) for b in bits]
    xi = [0.0] * 4
    if err_in is not None:
        xr, xi = rotate(xr, xi, err_in)
    yr, yi = dft4(xr, xi)
    yr, yi = [v * 0.5 for v in yr], [v * 0.5 for v in yi]
    if err_out is not None:
        yr, yi = rotate(yr, yi, err_out)
    e_lo = math.sqrt(10.0 ** (P["p_lo_dbm"] / 10.0) * 1e3)    # sqrt(uW)
    g = 2.0 * P["R"] * e_lo                                      # uA per sqrt(uW)
    proj, cur = [0.0] * 4, [0.0] * 4
    for k in range(4):
        th = rad(LO_PHASE_DEG[k])
        proj[k] = yr[k] * math.cos(th) + yi[k] * math.sin(th)
        cur[k] = g * proj[k]
    return dict(f1=f1, f0=f0, p_lane=p_lane, a=a, xr=xr, xi=xi, yr=yr, yi=yi, e_lo=e_lo, g=g, proj=proj, i=cur)


def chain_scale():
    """The offset (output 1's carrier current, from the all-zero input) and the current of one level step."""
    c = chain_model((0, 0, 0, 0))
    step = c["g"] * (c["a"] * (c["f1"] - c["f0"])) / 2.0
    return c["i"][0], step


def ideal_levels(bits):
    """What each detector should read, in level steps: Re X0, Re X1, Re X2, Im X3."""
    re, im = dft4_int(bits)
    return [re[0], re[1], re[2], im[3]]


def detected_levels(bits, err_in=None, err_out=None):
    off, step = chain_scale()
    c = chain_model(bits, err_in, err_out)
    return [(c["i"][k] - (off if k == 0 else 0.0)) / step for k in range(4)]


def decide(levels):
    """Nearest level (round half away from zero, the same in Python and JavaScript for these values)."""
    return [int(math.floor(v + 0.5)) for v in levels]


def pam4_symbols(bits):
    """The symbol each detector sends to the FPGA after offset removal and saturation (an illustrative mapping):
    output 1 (0..4) and output 3 (-2..2, shifted to 0..4) clip their top level to 3; outputs 2 and 4 use 0..2."""
    l = ideal_levels(bits)
    return [min(l[0], 3), l[1] + 1, min(l[2] + 2, 3), l[3] + 1]


def idft4_int(X0, X1r, X1i, X2):
    """4 x_n from the four outputs, using X3 = conj(X1) (real inputs): integer inverse DFT times 4."""
    Xr = (X0, X1r, X2, X1r)
    Xi = (0, X1i, 0, -X1i)
    out = []
    for n in range(4):
        s = 0
        for k in range(4):
            wr, wi = W4[(k * n) % 4]          # inverse uses (+j)^(kn) = conj of (-j)^(kn)
            s += Xr[k] * wr + Xi[k] * wi       # real part of X_k * conj(w)
        out.append(s)
    return out


def decode(sym):
    """Undo the saturation: candidates for a clipped level, kept if X0 and X2 have the same parity (they differ by
    2(x1 + x3)) and the inverse DFT gives 0/1 bits. Returns (bits or None, candidates after parity, after validity)."""
    c0 = [sym[0]] if sym[0] < 3 else [3, 4]
    c2 = [sym[2] - 2] if sym[2] < 3 else [1, 2]
    x1r, x1i = sym[1] - 1, -(sym[3] - 1)
    par = [(a, b) for a in c0 for b in c2 if (a - b) % 2 == 0]
    ok = []
    for a, b in par:
        x4 = idft4_int(a, x1r, x1i, b)
        if all(v in (0, 4) for v in x4):
            ok.append(tuple(v // 4 for v in x4))
    return (ok[0] if len(ok) == 1 else None), len(par), len(ok)


# ── the receiver ───────────────────────────────────────────────────────────────────────────────────────────────
def homodyne_terms(p_lo_W, p_sig_W=None):
    """Balanced homodyne: peak signal current and the three noise currents (rms) in a Nyquist noise bandwidth."""
    if p_sig_W is None:
        p_sig_W = P["p_sig_uW"] * 1e-6
    B = P["baud"] / 2.0
    R = P["R"]
    i_sig = 2.0 * R * math.sqrt(p_sig_W * p_lo_W)
    th = P["i_n"] * math.sqrt(B)
    shot = math.sqrt(2.0 * Q_E * R * (p_lo_W + p_sig_W) * B)
    rin = R * p_lo_W * math.sqrt(10.0 ** (P["rin_db"] / 10.0) * B) * 10.0 ** (-P["cmrr_db"] / 20.0)
    return dict(sig=i_sig, th=th, shot=shot, rin=rin, tot=math.sqrt(th * th + shot * shot + rin * rin))


def q_per_step(p_lo_W, p_sig_W):
    """Half a level step over the rms noise. Output 3's full swing (two steps) is the 10 uW reference signal, so a
    step is i_sig(10 uW)/2 at any LO power; the shot noise uses the signal power the output actually carries."""
    step = homodyne_terms(p_lo_W)["sig"] / 2.0
    return (step / 2.0) / homodyne_terms(p_lo_W, p_sig_W)["tot"]


def tia_limit(f3db, c_t):
    """Shunt-feedback TIA transimpedance limit (Sackinger): R_T <= A f_A / (2 pi C_T f_3dB^2)."""
    return P["gbw"] / (2.0 * math.pi * c_t * f3db * f3db)


# ── skew: four lanes summed before any decision ───────────────────────────────────────────────────────────────
def prbs15(n, seed=1):
    """PRBS15 (x^15 + x^14 + 1), n bits."""
    s, out = seed, []
    for _ in range(n):
        b = ((s >> 14) ^ (s >> 13)) & 1
        s = ((s << 1) | b) & 0x7FFF
        out.append(b)
    return out


SKEW = dict(sps=32, nsym=600, edge=20)


def skew_lanes():
    b = prbs15(4 * SKEW["nsym"])
    return [b[k * SKEW["nsym"]:(k + 1) * SKEW["nsym"]] for k in range(4)]


def gauss_kernel(bw):
    """Gaussian channel of -3 dB bandwidth bw x baud, sampled at sps per symbol over +/- 4 symbols, unit sum."""
    sps = SKEW["sps"]
    sigma = math.sqrt(math.log(2.0)) / (2.0 * math.pi * bw)
    h = []
    for i in range(8 * sps + 1):
        t = (i - 4 * sps) / sps
        h.append(math.exp(-t * t / (2.0 * sigma * sigma)))
    s = 0.0
    for v in h:
        s = s + v
    return [v / s for v in h]


def filtered(bits, h):
    sps, half = SKEW["sps"], (len(h) - 1) // 2
    n = len(bits) * sps
    out = [0.0] * n
    for m in range(n):
        acc = 0.0
        for i in range(len(h)):
            j = m + i - half
            if 0 <= j < n:
                acc = acc + h[i] * bits[j // sps]
        out[m] = acc
    return out


def eye_opening(lanes_f, lanes, shift):
    """Lane 1 arrives `shift` samples late; the four filtered lanes are summed (one OFT output, unit weights).
    Returns the worst inner-eye opening, in level steps, at the best sampling phase. The label of each sample is
    the correct same-symbol sum, so a late lane's leftover bit counts against the eye."""
    sps, nsym, edge = SKEW["sps"], SKEW["nsym"], SKEW["edge"]
    best = -1e9
    for ph in range(sps):
        lo = [1e9] * 5           # min of samples labelled l
        hi = [-1e9] * 5          # max of samples labelled l
        for s in range(edge, nsym - edge):
            m = s * sps + ph
            v = lanes_f[0][m - shift] + lanes_f[1][m] + lanes_f[2][m] + lanes_f[3][m]
            lab = lanes[0][s] + lanes[1][s] + lanes[2][s] + lanes[3][s]
            if v < lo[lab]:
                lo[lab] = v
            if v > hi[lab]:
                hi[lab] = v
        gap = 1e9
        for l in range(4):
            g = lo[l + 1] - hi[l]
            if g < gap:
                gap = g
        if gap > best:
            best = gap
    return best


# ── run everything and write the results ─────────────────────────────────────────────────────────────────────
def fmt(x, d=2):
    return f"{x:,.{d}f}"


def bits_s(b):
    return "[" + ", ".join(str(v) for v in b) + "]"


def cplx(r, i, d=2):
    if abs(i) < 5e-13:
        return fmt(r, d).replace("-", "−")
    if abs(r) < 5e-13:
        if d == 0 and abs(i) == 1:
            return "j" if i > 0 else "−j"
        return f"{fmt(i, d)}j".replace("-", "−")
    im = "j" if d == 0 and abs(i) == 1 else f"{fmt(abs(i), d)}j"
    return f"{fmt(r, d)} {'+' if i >= 0 else '−'} {im}".replace("-", "−")


def main():
    J = dict(params=dict(P, lo_phase_deg=LO_PHASE_DEG, err_in_deg=ERR_IN_DEG, err_out_deg=ERR_OUT_DEG,
                         ex_bits=EX_BITS, skew=SKEW))
    md = []
    w = md.append
    w("# Inside a photonic link: modulator, receiver and signal-chain numbers")
    w("")
    w(f"Generated by `analysis/link_hw/link_hw.py` on {date.today().isoformat()} ({platform.machine()}, Python "
      f"{sys.version.split()[0]}). Every number in FOptInf 04 comes from this file or from `js/fop04_models.js`, "
      "its tested port. First-order textbook models; parameters marked *illustrative* are not from a source.")
    w("")
    w("## 1. Parameters")
    w("")
    w("| Parameter | Value | Source |")
    w("|---|---|---|")
    rows = [("Photodiode responsivity", f"{P['R']} A/W", "Hu et al., Photonics Research 11, 961 (2023): foundry Ge PD, TE, 1530 nm"),
            ("Symbol rate", "14 GBaud (noise bandwidth 7 GHz)", "the paper (arXiv:2608.21536)"),
            ("TIA input noise density", "10 pA/√Hz", "illustrative: typical of published 50G-class linear TIAs"),
            ("Laser RIN", f"{P['rin_db']:.0f} dB/Hz", "illustrative"),
            ("Balanced-pair CMRR", f"{P['cmrr_db']:.0f} dB", "illustrative"),
            ("Signal power for output 3's full swing", f"{P['p_sig_uW']:.0f} µW", "illustrative"),
            ("Drive swing", f"±{P['swing']} Vπ (±{P['swing'] * P['vpi']:.2f} V on Vπ = {P['vpi']} V)", "illustrative"),
            ("LO power per detector pair", f"{P['p_lo_dbm']:.0f} dBm", "illustrative"),
            ("TIA transimpedance", f"{P['r_t']:,.0f} Ω", "illustrative"),
            ("LO phases (detectors 1-4)", "0°, 0°, 0°, 90°", "three real, one imaginary: a patent cited by the paper (WO2025168827A1)"),
            ("Uncalibrated input-path phases", "0°, 25°, −15°, 10°", "illustrative"),
            ("Uncalibrated output/LO-path phases", "20°, −30°, 15°, 35°", "illustrative"),
            ("TIA gain-bandwidth product", "1 THz", "illustrative")]
    for r in rows:
        w("| " + " | ".join(r) + " |")
    w("")

    # bias points
    w("## 2. Bias points")
    w("")
    w("Push-pull MZM: field = cos(Δφ/2), power = cos²(Δφ/2). Binary drive of ±swing about each bias; '1' is the drive "
      "towards the peak.")
    w("")
    J["bias"] = {}
    for swing in (0.15, 0.22):
        w(f"### Drive ±{swing} Vπ")
        w("")
        w("| Bias | Δφ | Field '1' | Field '0' | Field step | Power '1' | Power '0' | Power step |")
        w("|---|---|---|---|---|---|---|---|")
        J["bias"][str(swing)] = {}
        for name, b, lab in (("peak", 0.0, "0"), ("quadrature", math.pi / 2, "π/2"), ("null", math.pi, "π")):
            r = bias_point(b, swing)
            J["bias"][str(swing)][name] = r
            w(f"| {name} | {lab} | {r['e1']:.3f} | {r['e0']:.3f} | {r['de']:.3f} | {r['p1']:.3f} | {r['p0']:.3f} | {r['dp']:.3f} |")
        w("")
    # dither
    w("## 3. Holding the bias: error signals")
    w("")
    w("Dither 0.1 rad on the bias, 64 samples per dither period, synchronous detection. Numerical lock-in against the "
      "Bessel-function result (avg = ½ + cos φ·J0(a)/2, h1 = −sin φ·J1(a), h2 = cos φ·J2(a)):")
    w("")
    w("| Bias phase | Avg power − ½ | 1st harmonic | 2nd harmonic | Largest difference from analytic |")
    w("|---|---|---|---|---|")
    J["dither"] = []
    a = 0.1
    j0, j1, j2 = bessel_j(0, a), bessel_j(1, a), bessel_j(2, a)
    worst = 0.0
    for k in range(17):
        phi = k * math.pi / 8
        r = dither(phi, a)
        an = dict(avg=0.5 + math.cos(phi) * j0 / 2, h1=-math.sin(phi) * j1, h2=math.cos(phi) * j2)
        diff = max(abs(r[x] - an[x]) for x in r)
        worst = max(worst, diff)
        J["dither"].append(dict(phi=phi, **r))
        if k % 2 == 0:
            z = lambda v: 0.0 if abs(v) < 5e-9 else v      # no "-0.0000"
            w(f"| {k / 8:g}π | {z(r['avg'] - 0.5):+.4f} | {z(r['h1']):+.4f} | {z(r['h2']):+.5f} | {diff:.1e} |")
    w("")
    w(f"* At quadrature (π/2) the average power error is zero and changes sign through it, so it alone steers the loop; "
      f"at null (π) the average power is at its minimum and only the 1st harmonic (zero there, with a sign) says which way to move. Worst lock-in vs analytic: {worst:.1e}.")
    w("")

    # chain
    off, step = chain_scale()
    ex, exe = chain_model(EX_BITS), chain_model(EX_BITS, ERR_IN_DEG, ERR_OUT_DEG)
    f1, f0 = ex["f1"], ex["f0"]
    e0, dd = ex["a"] * f0, ex["a"] * (f1 - f0)
    lev, levu = detected_levels(EX_BITS), detected_levels(EX_BITS, ERR_IN_DEG, ERR_OUT_DEG)
    sym = pam4_symbols(EX_BITS)
    xr_int, xi_int = dft4_int(EX_BITS)
    w("## 4. Constellations through the chain: the worked example")
    w("")
    w(f"Input bits {bits_s(EX_BITS)} (lanes 1 to 4). Calibrated unless the row says otherwise.")
    w("")
    w("| Stage | What the signal is | Worked example |")
    w("|---|---|---|")
    w(f"| 0 · FPGA transmitter | Four binary NRZ lanes at 14 GBaud | bits {', '.join(map(str, EX_BITS))} |")
    v = P["swing"] * P["vpi"]
    w(f"| 1 · Driver | Linear amplification to ±{v:.2f} V (illustrative Vπ {P['vpi']} V, so ±{P['swing']} Vπ) | "
      + ", ".join(("+" if b else "−") + f"{v:.2f} V" for b in EX_BITS) + " |")
    w(f"| 2 · MZM at quadrature | Field {f1:.3f} ('1') or {f0:.3f} ('0') of the maximum; power {f1 * f1 * 100:.0f}% or {f0 * f0 * 100:.0f}% | "
      f"lane maximum {ex['p_lane']:.1f} µW; fields " + ", ".join(fmt(x) for x in ex["xr"]) + " √µW; carrier E₀ = "
      f"{e0:.2f}, step d = {dd:.2f} |")
    pr = chain_model(EX_BITS, ERR_IN_DEG, None)
    w(f"| 3 · Per-lane phase shifters | Set by calibration so the lanes arrive in phase | uncalibrated: lanes turned by "
      + ", ".join(f"{d:g}°" for d in ERR_IN_DEG) + " (illustrative) |")
    w(f"| 4 · Optical Fourier transform | Y = DFT(x) / 2, x = E₀ + d·bits | DFT(bits) = ["
      + ", ".join(cplx(xr_int[k], xi_int[k], 0) for k in range(4)) + "]; Y = ["
      + ", ".join(cplx(ex["yr"][k], ex["yi"][k]) for k in range(4)) + "] √µW |")
    w(f"| 5 · Balanced homodyne | i = 2·R·E_LO × projection; E_LO = {ex['e_lo']:.2f} √µW, so {ex['g']:.1f} µA per √µW | "
      "currents " + ", ".join(fmt(x, 1) for x in ex["i"]) + " µA |")
    mv = [x * P["r_t"] * 1e-3 for x in ex["i"]]
    w(f"| 6 · TIA and offset removal | {P['r_t']:,.0f} Ω; output 1's carrier offset {off:.1f} µA removed | "
      + ", ".join(fmt(x, 1) for x in mv) + f" mV; output 1 after offset {(ex['i'][0] - off) * P['r_t'] * 1e-3:.1f} mV; levels "
      + ", ".join(f"{x:.0f}" for x in lev) + " |")
    w(f"| 7 · PAM4 into the FPGA | Outputs 1 and 3 clip their top level | symbols {', '.join(map(str, sym))} |")
    dec = decode(sym)[0]
    w(f"| 8 · FPGA decode | Rebuild X, inverse DFT | X = [" + ", ".join(cplx(xr_int[k], xi_int[k], 0) for k in range(4))
      + f"]; bits {bits_s(dec)} |")
    w("")
    w("### Scale")
    w("")
    w("| Quantity | Value |")
    w("|---|---|")
    pw = [ex["yr"][k] ** 2 + ex["yi"][k] ** 2 for k in range(4)]
    for r in (("One level step (current)", f"{step:.1f} µA"),
              ("One level step (field)", f"{dd / 2:.2f} √µW"),
              ("Output 1 carrier offset", f"{off:.1f} µA"),
              ("Output 1 carrier field (2·E₀)", f"{2 * e0:.2f} √µW"),
              ("Output powers, worked example", ", ".join(fmt(x, 1) for x in pw) + " µW"),
              ("Total power in, worked example", f"{sum(x * x for x in ex['xr']):.1f} µW"),
              ("Total power out, worked example", f"{sum(pw):.1f} µW"),
              ("Uncalibrated, worked example: levels read", ", ".join(f"{x:.2f}" for x in levu)),
              ("Uncalibrated, worked example: decisions", ", ".join(str(x) for x in decide(levu)) + " (correct: "
               + ", ".join(str(x) for x in ideal_levels(EX_BITS)) + ")")):
        w("| " + " | ".join(r) + " |")
    w("")
    J["chain"] = dict(offset=off, step=step, example=ex, example_uncal=exe, levels=lev, levels_uncal=levu, symbols=sym,
                      all=[chain_model(b) for b in VECS],
                      all_uncal=[chain_model(b, ERR_IN_DEG, ERR_OUT_DEG) for b in VECS],
                      all_in=[chain_model(b, ERR_IN_DEG, None) for b in VECS],
                      all_out=[chain_model(b, None, ERR_OUT_DEG) for b in VECS])
    J["chain_brief"] = dict(offset=off, step=step, example_i=ex["i"], example_y=[ex["yr"], ex["yi"]], levels=lev,
                            levels_uncal=levu, decisions_uncal=decide(levu), symbols=sym, carrier=e0, d=dd,
                            p_lane=ex["p_lane"], e_lo=ex["e_lo"], g=ex["g"])

    # alphabet
    w("## 5. The DFT alphabet")
    w("")
    w("All 16 inputs. Outputs 1 and 3 are real; outputs 2 and 4 are complex and mirror each other (X3 = conj X1).")
    w("")
    w("| Output | Values taken | Distinct |")
    w("|---|---|---|")
    J["alphabet"] = []
    for k in range(4):
        vals = sorted({dft4_int(b)[0][k] + 1j * dft4_int(b)[1][k] for b in VECS}, key=lambda z: (z.real, z.imag))
        J["alphabet"].append([[z.real, z.imag] for z in vals])
        desc = ("real, " if all(z.imag == 0 for z in vals) else "complex, ") + ", ".join(
            cplx(z.real, z.imag, 0) for z in vals)
        w(f"| {k + 1} | {desc} | {len(vals)} |")
    conj_ok = all(dft4_int(b)[0][3] == dft4_int(b)[0][1] and dft4_int(b)[1][3] == -dft4_int(b)[1][1] for b in VECS)
    w("")
    w(f"* Output 4 is the complex conjugate of output 2 for every input: {conj_ok}. So reading Re of outputs 1, 2 and 3 "
      "and Im of output 4 recovers all four complex outputs.")
    w("")

    # PAM4
    w("## 6. Packing into PAM4 by saturation")
    w("")
    w("| Input | X (DFT of the bits) | PAM4 symbols | Candidates after parity | After 0/1 check | Decoded |")
    w("|---|---|---|---|---|---|")
    J["pam4"] = []
    codes = set()
    all_ok = True
    for b in VECS:
        s = pam4_symbols(b)
        codes.add(tuple(s))
        d, npar, nok = decode(s)
        all_ok &= d == tuple(b)
        re_, im_ = dft4_int(b)
        J["pam4"].append(dict(bits=b, symbols=s, decoded=d, npar=npar, nok=nok))
        w(f"| {''.join(map(str, b))} | " + ", ".join(cplx(re_[k], im_[k], 0) for k in range(4))
          + f" | {''.join(map(str, s))} | {npar} | {nok} | {''.join(map(str, d)) if d else '—'} |")
    n_clip = sum(1 for r in J["pam4"] if r["npar"] > 1)
    w("")
    w(f"* {len(codes)} distinct symbol patterns for 16 inputs; " + ("every input decodes back to itself" if all_ok else "SOME INPUTS DO NOT DECODE") + ". Parity alone "
      f"settles every clipped case but {n_clip}; there the inverse DFT rules out the impossible candidate.")
    w("")
    J["pam4_summary"] = dict(distinct=len(codes), all_ok=all_ok, parity_ambiguous=n_clip)

    # calibration
    w("## 7. Calibration: what each phase error does")
    w("")
    w("Detected levels for all 16 inputs on all 4 outputs (64 decisions). *Pass* = every level within half a step of "
      "the truth table, the criterion of a patent cited by the paper (WO2025168828A1).")
    w("")
    w("| Phase errors present | Wrong decisions (of 64) | Largest error (steps) | Passes |")
    w("|---|---|---|---|")
    J["calib"] = {}
    for name, ei, eo in (("both (uncalibrated)", ERR_IN_DEG, ERR_OUT_DEG), ("input path only", ERR_IN_DEG, None),
                         ("output/LO path only", None, ERR_OUT_DEG), ("none (calibrated)", None, None)):
        wrong, worst_e = 0, 0.0
        for b in VECS:
            L, I = detected_levels(b, ei, eo), ideal_levels(b)
            wrong += sum(1 for k in range(4) if decide(L)[k] != I[k])
            worst_e = max(worst_e, max(abs(L[k] - I[k]) for k in range(4)))
        J["calib"][name] = dict(wrong=wrong, worst=worst_e)
        w(f"| {name} | {wrong} | {worst_e:.2f} | {'yes' if worst_e < 0.5 else 'no'} |")
    w("")
    # rotation vs distortion, on output 2
    rot = []
    for b in VECS:
        c, ci = chain_model(b), chain_model(b, None, ERR_OUT_DEG)
        r0, r1 = math.hypot(c["yr"][1], c["yi"][1]), math.hypot(ci["yr"][1], ci["yi"][1])
        rot.append(abs(r0 - r1))
    dis = []
    for b in VECS:
        c, ci = chain_model(b), chain_model(b, ERR_IN_DEG, None)
        dis.append(abs(math.hypot(c["yr"][1], c["yi"][1]) - math.hypot(ci["yr"][1], ci["yi"][1])))
    w(f"* Output-path errors only rotate: every output-2 point keeps its distance from the origin (largest change "
      f"{max(rot):.1e} √µW). Input-path errors distort: distances change by up to {max(dis):.2f} √µW (a level step is "
      f"{dd / 2:.2f} √µW).")
    w("")
    J["calib_shape"] = dict(rotation_max=max(rot), distortion_max=max(dis))

    # noise
    w("## 8. Noise budget")
    w("")
    p_lo = 10 ** (P["p_lo_dbm"] / 10) * 1e-3
    t = homodyne_terms(p_lo)
    w(f"Balanced homodyne at {P['p_lo_dbm']:.0f} dBm LO, {P['p_sig_uW']:.0f} µW signal, 7 GHz noise bandwidth:")
    w("")
    w("| Term | Current |")
    w("|---|---|")
    for lab, k in (("Peak signal (output 3's full swing, two steps)", "sig"), ("TIA noise (rms)", "th"),
                   ("Shot noise (rms)", "shot"), ("LO RIN after CMRR (rms)", "rin"), ("Total noise (rms)", "tot")):
        w(f"| {lab} | {t[k] * 1e6:.2f} µA |")
    w("")
    snr = 20 * math.log10(t["sig"] / t["tot"])
    q_th = 0.5 * math.erfc(P["q_req"] / math.sqrt(2))
    w(f"* Peak-to-rms SNR {snr:.1f} dB. A decision fails when noise crosses half a level step: Q = (step/2)/σ = "
      f"{P['q_req']} gives {q_th:.1e} per threshold for Gaussian noise.")
    w("")
    w("### Margins")
    w("")
    w("Each output's worst case: the largest signal power it carries (more shot noise). The LO sets the level step, so "
      "the step is the same on every output; output 1 differs in its range (carrier offset plus four steps), not its step.")
    w("")
    w("| Output | Largest signal power | Noise (rms) | Q per step | Margin over Q = 4.9 | Range to handle |")
    w("|---|---|---|---|---|---|")
    J["noise"] = dict(p_lo_dbm=P["p_lo_dbm"], terms=t, snr_db=snr, q_threshold_rate=q_th, outputs={})
    for k, rng in ((0, f"{off:.0f} to {off + 4 * step:.0f} µA ({(off + 4 * step) / step:.1f} steps)"),
                   (2, f"{-2 * step:.0f} to {2 * step:.0f} µA (±2 steps)")):
        pmax = max(c["yr"][k] ** 2 + c["yi"][k] ** 2 for c in J["chain"]["all"])
        q = q_per_step(p_lo, pmax * 1e-6)
        noise = homodyne_terms(p_lo, pmax * 1e-6)["tot"]
        m = 20 * math.log10(q / P["q_req"])
        J["noise"]["outputs"][str(k + 1)] = dict(pmax_uW=pmax, noise=noise, q=q, margin_db=m)
        w(f"| {k + 1} | {pmax:.1f} µW | {noise * 1e6:.2f} µA | {q:.1f} | {m:.1f} dB | {rng} |")
    w("")
    w("### Against LO power")
    w("")
    w("| LO (dBm) | Signal (µA) | TIA (µA) | Shot (µA) | RIN (µA) | SNR (dB) | Q, output 3 | Q, output 1 |")
    w("|---|---|---|---|---|---|---|---|")
    sweep = []
    p1max = J["noise"]["outputs"]["1"]["pmax_uW"] * 1e-6
    p3max = J["noise"]["outputs"]["3"]["pmax_uW"] * 1e-6
    for k in range(75):
        dbm = -20.0 + 0.5 * k
        pl = 10.0 ** (dbm / 10.0) * 1e-3
        tt = homodyne_terms(pl)
        row = dict(dbm=dbm, sig=tt["sig"], th=tt["th"], shot=tt["shot"], rin=tt["rin"],
                   snr=20 * math.log10(tt["sig"] / tt["tot"]), q3=q_per_step(pl, p3max), q1=q_per_step(pl, p1max))
        sweep.append(row)
        if k % 10 == 0:
            w(f"| {dbm:.0f} | {tt['sig'] * 1e6:.1f} | {tt['th'] * 1e6:.2f} | {tt['shot'] * 1e6:.2f} | {tt['rin'] * 1e6:.3f} | "
              f"{row['snr']:.1f} | {row['q3']:.1f} | {row['q1']:.1f} |")
    best = max(sweep, key=lambda r: r["snr"])
    cross = next(r for r in sweep if r["shot"] > r["th"])
    w("")
    w(f"* Best SNR {best['snr']:.1f} dB at {best['dbm']:.1f} dBm LO; shot noise overtakes TIA noise above about "
      f"{cross['dbm']:.1f} dBm.".replace("-", "−") + " Below that, more LO buys SNR almost for free.")
    w("")
    J["noise"]["sweep"] = sweep
    J["noise"]["best"] = best
    J["noise"]["shot_over_tia_dbm"] = cross["dbm"]
    w("### Noise bandwidth")
    w("")
    w("White noise (the TIA's flat part, shot noise, RIN) grows as √B. If nothing limits the bandwidth to the signal's 7 GHz, a wider receiver integrates more "
      "(the amplifier-voltage-noise part, which rises with frequency, grows faster still and is not included here).")
    w("")
    w("| Noise bandwidth | TIA noise (rms) | Total noise (rms) | Q, output 3 | Margin over Q = 4.9 |")
    w("|---|---|---|---|---|")
    J["noise"]["bandwidth"] = []
    keep = P["baud"]
    for bwg in (7, 20, 35):
        P["baud"] = 2 * bwg * 1e9          # homodyne_terms uses baud/2 as the noise bandwidth
        tt = homodyne_terms(p_lo)
        step_i = homodyne_terms(p_lo)["sig"] / 2.0
        q = (step_i / 2.0) / tt["tot"]
        J["noise"]["bandwidth"].append(dict(ghz=bwg, th=tt["th"], tot=tt["tot"], q=q))
        w(f"| {bwg} GHz | {tt['th'] * 1e6:.2f} µA | {tt['tot'] * 1e6:.2f} µA | {q:.1f} | {20 * math.log10(q / P['q_req']):.1f} dB |")
    P["baud"] = keep
    w("")

    # TIA
    w("## 9. The transimpedance limit")
    w("")
    w("R_T ≤ A·f_A / (2π·C_T·f_3dB²), A·f_A = 1 THz (illustrative). Bandwidths 0.6 × the symbol rate (a rule of thumb).")
    w("")
    w("| C_T | R_T max at 8.4 GHz (14 GBaud) | R_T max at 33.6 GHz (56 GBaud) | Ratio |")
    w("|---|---|---|---|")
    J["tia"] = []
    for c in (80e-15, 150e-15, 300e-15):
        r14, r56 = tia_limit(0.6 * 14e9, c), tia_limit(0.6 * 56e9, c)
        J["tia"].append(dict(c_t=c, r14=r14, r56=r56))
        w(f"| {c * 1e15:.0f} fF | {r14:,.0f} Ω | {r56:,.0f} Ω | {r14 / r56:.0f}× |")
    w("")

    # skew
    w("## 10. Skew: four lanes summed before any decision")
    w("")
    w(f"Four PRBS15 NRZ lanes (0/1), {SKEW['nsym']} symbols, {SKEW['sps']} samples per symbol, a Gaussian channel, "
      "summed with unit weights (output 1 of the transform); lane 1 arrives late. Inner-eye opening in level steps at the "
      "best sampling phase.")
    w("")
    lanes = skew_lanes()
    J["skew"] = {}
    w("| Skew (UI) | ps at 14 GBaud | ps at 56 GBaud | Opening (BW 0.75 × baud) | Relative (0.75) | Opening (BW 0.5 × baud) | Relative (0.5) |")
    w("|---|---|---|---|---|---|---|")
    res = {}
    for bw in (0.75, 0.5):
        h = gauss_kernel(bw)
        lf = [filtered(l, h) for l in lanes]
        res[bw] = [eye_opening(lf, lanes, s) for s in range(SKEW["sps"] // 2 + 1)]
        J["skew"][str(bw)] = res[bw]
    for s in range(0, SKEW["sps"] // 2 + 1, 2):
        ui = s / SKEW["sps"]
        w(f"| {ui:.4g} | {ui / 14e9 * 1e12:.1f} | {ui / 56e9 * 1e12:.1f} | {res[0.75][s]:.3f} | {res[0.75][s] / res[0.75][0]:.2f} | "
          f"{res[0.5][s]:.3f} | {res[0.5][s] / res[0.5][0]:.2f} |")
    w("")

    J["generated"] = date.today().isoformat()
    (HERE / "results.md").write_text("\n".join(md) + "\n")
    (HERE / "results.json").write_text(json.dumps(J, indent=1, default=list) + "\n")
    print("wrote results.md and results.json")


if __name__ == "__main__":
    main()
