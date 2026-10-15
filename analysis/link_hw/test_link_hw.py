"""Parity and sanity: js/fop04_models.js must reproduce link_hw.py, and the models must obey the physics.

Arithmetic-only results (integer DFTs, PAM4 symbols, decoding, PRBS) must match exactly. Results that pass through
sin, cos, exp or pow must match within 1e-12 relative (the C library and the JavaScript engine may round those
differently in the last place); the test also reports how many came out bit-identical.

Run: ../../.venv/bin/python -m pytest -q test_link_hw.py   (needs node on PATH)
"""
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

import link_hw as L

JS = Path(__file__).resolve().parents[2] / "js" / "fop04_models.js"
TOL = 1e-12
ERRS = [(None, None), (L.ERR_IN_DEG, None), (None, L.ERR_OUT_DEG), (L.ERR_IN_DEG, L.ERR_OUT_DEG)]
STATS = {"exact": 0, "close": 0}


def node_run(body):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    script = f"const M = require({json.dumps(str(JS))}); const out = (() => {{ {body} }})(); console.log(JSON.stringify(out));"
    return json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout)


def same(a, b, what):
    """Compare nested numbers: exact, or within TOL relative (counted)."""
    if isinstance(a, dict):
        for k in a:
            same(a[k], b[k], f"{what}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), what
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{what}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if a == b:
            STATS["exact"] += 1
        else:
            assert abs(a - b) <= TOL * max(1.0, abs(a)), f"{what}: {a!r} vs {b!r}"
            STATS["close"] += 1
    else:
        assert a == b, f"{what}: {a!r} vs {b!r}"


def test_chain_matches_js():
    js = node_run("const E = [[null, null], [M.ERR_IN_DEG, null], [null, M.ERR_OUT_DEG], [M.ERR_IN_DEG, M.ERR_OUT_DEG]];"
                  "return E.map(([a, b]) => M.VECS.map(v => { const c = M.chainModel(v, a, b);"
                  " return [c.xr, c.xi, c.yr, c.yi, c.proj, c.i, M.detectedLevels(v, a, b), M.decide(M.detectedLevels(v, a, b))]; }));")
    for e, (ei, eo) in enumerate(ERRS):
        for n, v in enumerate(L.VECS):
            c = L.chain_model(v, ei, eo)
            lv = L.detected_levels(v, ei, eo)
            same([c["xr"], c["xi"], c["yr"], c["yi"], c["proj"], c["i"], lv, L.decide(lv)], js[e][n], f"chain e{e} v{n}")


def test_symbols_and_decode_exact():
    js = node_run("return M.VECS.map(v => [M.dft4Int(v), M.idealLevels(v), M.pam4Symbols(v), M.decode(M.pam4Symbols(v))]);")
    for n, v in enumerate(L.VECS):
        d = L.decode(L.pam4_symbols(v))
        assert js[n][0] == [list(x) for x in L.dft4_int(v)]
        assert js[n][1] == L.ideal_levels(v) and js[n][2] == L.pam4_symbols(v)
        assert js[n][3] == {"bits": list(d[0]), "npar": d[1], "nok": d[2]}
        assert d[0] == tuple(v)
    assert len({tuple(L.pam4_symbols(v)) for v in L.VECS}) == 16


def test_bias_dither_mzm_match_js():
    grid = [k * math.pi / 8 for k in range(17)]
    js = node_run(f"const g = {json.dumps(grid)};"
                  "return [g.map(b => [0.05, 0.15, 0.22, 0.4].map(s => M.biasPoint(b, s))), g.map(p => M.dither(p)),"
                  " g.map(p => [M.mzmArms(p, true), M.mzmArms(p, false)])];")
    same([[L.bias_point(b, s) for s in (0.05, 0.15, 0.22, 0.4)] for b in grid], js[0], "bias")
    same([L.dither(p) for p in grid], js[1], "dither")
    py = [[L.mzm_arms(p, True), L.mzm_arms(p, False)] for p in grid]
    same([[{k: list(v) for k, v in d.items()} for d in row] for row in py], js[2], "mzm")


def test_receiver_matches_js():
    dbm = [-20 + 0.5 * k for k in range(75)]
    js = node_run(f"const d = {json.dumps(dbm)};"
                  "return [d.map(x => M.homodyneTerms(Math.pow(10, x / 10) * 1e-3)), d.map(x => M.qPerStep(Math.pow(10, x / 10) * 1e-3, 142.7e-6)),"
                  " [80e-15, 150e-15, 300e-15].map(c => [8.4e9, 20e9, 33.6e9].map(f => M.tiaLimit(f, c)))];")
    same([L.homodyne_terms(10 ** (x / 10) * 1e-3) for x in dbm], js[0], "homodyne")
    same([L.q_per_step(10 ** (x / 10) * 1e-3, 142.7e-6) for x in dbm], js[1], "q")
    same([[L.tia_limit(f, c) for f in (8.4e9, 20e9, 33.6e9)] for c in (80e-15, 150e-15, 300e-15)], js[2], "tia")


def test_skew_matches_js_and_results():
    js = node_run("const lanes = M.skewLanes(); const out = {lanes};"
                  "for (const bw of [0.75, 0.5]) { const h = M.gaussKernel(bw); const lf = lanes.map(l => M.filtered(l, h));"
                  " out[bw] = Array.from({length: 17}, (_, s) => M.eyeOpening(lf, lanes, s).opening); } return out;")
    assert js["lanes"] == L.skew_lanes()
    res = json.loads((Path(__file__).parent / "results.json").read_text())
    for bw in ("0.75", "0.5"):
        same(res["skew"][bw], js[bw], f"skew {bw}")


def test_physics():
    # power conserved by the unitary transform; output 4 = conj(output 2); rotation keeps |Y|
    for v in L.VECS:
        c = L.chain_model(v)
        pin = sum(x * x for x in c["xr"])
        pout = sum(c["yr"][k] ** 2 + c["yi"][k] ** 2 for k in range(4))
        assert abs(pin - pout) < 1e-9 * pin
        assert c["yr"][3] == c["yr"][1] and c["yi"][3] == -c["yi"][1]
    # calibrated decisions are all right; the worked example reads 3, 0, 1, -1
    for v in L.VECS:
        assert L.decide(L.detected_levels(v)) == L.ideal_levels(v)
    assert L.decide(L.detected_levels(L.EX_BITS)) == [3, 0, 1, -1]
    # bias: null gives zero power step, quadrature zero ... field symmetry
    q, n = L.bias_point(math.pi / 2, 0.15), L.bias_point(math.pi, 0.15)
    assert abs(n["dp"]) < 1e-15 and abs(n["e1"] + n["e0"]) < 1e-15 and n["de"] > q["de"]
    # dither lock-in = Bessel result
    for k in range(17):
        phi = k * math.pi / 8
        r = L.dither(phi)
        assert abs(r["h1"] + math.sin(phi) * L.bessel_j(1, 0.1)) < 1e-14
        assert abs(r["h2"] - math.cos(phi) * L.bessel_j(2, 0.1)) < 1e-14
    # balanced homodyne: an LO 3 dB stronger gives sqrt(2) more signal
    a, b = L.homodyne_terms(1e-3), L.homodyne_terms(2e-3)
    assert abs(b["sig"] / a["sig"] - math.sqrt(2)) < 1e-12


def test_report_stats():
    print(f"\nparity: {STATS['exact']} floats bit-identical, {STATS['close']} within {TOL:g} relative")
