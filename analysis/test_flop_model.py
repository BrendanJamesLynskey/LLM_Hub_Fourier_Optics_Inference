"""Parity: js/flop_model.js must reproduce flop_share.py's prefill ledger exactly.

Run: ../.venv/bin/python -m pytest -q test_flop_model.py   (needs node on PATH)
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import flop_share as F

JS = Path(__file__).resolve().parent.parent / "js" / "flop_model.js"
CASES = [(v, lens, lm) for v in F.VARIANTS
         for lens in ([512], [2048], [8192], [32768], [131072], [1000, 3000, 77], [1])
         for lm in (True, False)]


def js_results():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    script = (f"const M = require({json.dumps(str(JS))});"
              "const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "console.log(JSON.stringify(cases.map(([v, l, lm]) => M.prefill(v, l, lm))));")
    out = subprocess.run([node, "-e", script], input=json.dumps(CASES), capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_js_matches_python_exactly():
    for (v, lens, lm), js in zip(CASES, js_results()):
        o = F.prefill(v, lens, lm_all_tokens=lm)
        for k in ("dense", "attention", "transform", "spectral", "other"):
            assert js[k] == getattr(o, k), (v, lens, lm, k)
        assert js["total"] == o.total and js["optical"] == o.optical, (v, lens, lm)
        assert js["share"] == o.optical / o.total


def test_transformer_matches_the_simulator():
    from disagg_sim.hardware import H100_SXM, LLAMA3_8B, CostModel
    cm = CostModel(LLAMA3_8B, H100_SXM, 1)
    for lens in ([512], [2048], [1000, 3000, 77]):
        assert F.prefill("transformer", lens).total == cm.prefill(lens).flops
