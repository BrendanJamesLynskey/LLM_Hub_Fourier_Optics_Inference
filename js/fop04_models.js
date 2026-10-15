// FOptInf 04: the model functions behind every figure and animation in "Inside a Photonic Link".
// A port of analysis/link_hw/link_hw.py, with the same floating-point operations in the same order, so the slides
// draw exactly what results.md reports. analysis/link_hw/test_link_hw.py checks the port against the Python
// (exact for arithmetic; within 1e-12 relative where sin, cos, exp or pow enter).
(function (root) {
    'use strict';
    const Q_E = 1.602176634e-19;
    const P = {
        R: 0.65, baud: 14e9, i_n: 10e-12, rin_db: -145.0, cmrr_db: 25.0, p_sig_uW: 10.0, swing: 0.22, vpi: 6.8,
        p_lo_dbm: -5.0, r_t: 1000.0, q_req: 4.9, gbw: 1.0e12,
    };
    const LO_PHASE_DEG = [0.0, 0.0, 0.0, 90.0];
    const ERR_IN_DEG = [0.0, 25.0, -15.0, 10.0];
    const ERR_OUT_DEG = [20.0, -30.0, 15.0, 35.0];
    const EX_BITS = [1, 0, 1, 1];
    const VECS = Array.from({ length: 16 }, (_, n) => [0, 1, 2, 3].map(k => (n >> k) & 1));
    const W4 = [[1, 0], [0, -1], [-1, 0], [0, 1]];
    const rad = d => d * Math.PI / 180.0;

    function dft4(xr, xi) {
        const yr = [0, 0, 0, 0], yi = [0, 0, 0, 0];
        for (let k = 0; k < 4; k++) {
            let sr = 0.0, si = 0.0;
            for (let n = 0; n < 4; n++) {
                const [wr, wi] = W4[(k * n) % 4];
                sr = sr + (xr[n] * wr - xi[n] * wi);
                si = si + (xr[n] * wi + xi[n] * wr);
            }
            yr[k] = sr; yi[k] = si;
        }
        return [yr, yi];
    }
    function dft4Int(bits) {
        const re = [0, 0, 0, 0], im = [0, 0, 0, 0];
        for (let k = 0; k < 4; k++) for (let n = 0; n < 4; n++) {
            const [wr, wi] = W4[(k * n) % 4];
            re[k] += bits[n] * wr; im[k] += bits[n] * wi;
        }
        return [re, im];
    }
    function rotate(xr, xi, deg) {
        const outr = new Array(xr.length), outi = new Array(xr.length);
        for (let n = 0; n < xr.length; n++) {
            const c = Math.cos(rad(deg[n])), s = Math.sin(rad(deg[n]));
            outr[n] = xr[n] * c - xi[n] * s;
            outi[n] = xr[n] * s + xi[n] * c;
        }
        return [outr, outi];
    }

    // ── the modulator ──
    function mzmArms(dphi, pushPull = true) {
        const [p1, p2] = pushPull ? [dphi / 2.0, -dphi / 2.0] : [dphi, 0.0];
        const a1 = [0.5 * Math.cos(p1), 0.5 * Math.sin(p1)], a2 = [0.5 * Math.cos(p2), 0.5 * Math.sin(p2)];
        return { a1, a2, sum: [a1[0] + a2[0], a1[1] + a2[1]] };
    }
    function biasPoint(bias, swing) {
        const d1 = bias - Math.PI * swing, d0 = bias + Math.PI * swing;
        const e1 = Math.cos(d1 / 2.0), e0 = Math.cos(d0 / 2.0);
        return { e1, e0, p1: e1 * e1, p0: e0 * e0, de: e1 - e0, dp: e1 * e1 - e0 * e0 };
    }
    function dither(phi, a = 0.1, n = 64) {
        let s0 = 0.0, s1 = 0.0, s2 = 0.0;
        for (let m = 0; m < n; m++) {
            const th = 2.0 * Math.PI * m / n;
            const p = 0.5 * (1.0 + Math.cos(phi + a * Math.sin(th)));
            s0 = s0 + p; s1 = s1 + p * Math.sin(th); s2 = s2 + p * Math.cos(2.0 * th);
        }
        return { avg: s0 / n, h1: 2.0 * s1 / n, h2: 2.0 * s2 / n };
    }

    // ── the chain ──
    function laneLevels() {
        const s = Math.sin(Math.PI * P.swing);
        return [Math.sqrt(0.5 * (1.0 + s)), Math.sqrt(0.5 * (1.0 - s))];
    }
    function chainModel(bits, errIn = null, errOut = null) {
        const [f1, f0] = laneLevels(), d = f1 - f0;
        const p_lane = P.p_sig_uW / (d * d), a = Math.sqrt(p_lane);
        let xr = bits.map(b => a * (b ? f1 : f0)), xi = [0.0, 0.0, 0.0, 0.0];
        if (errIn) [xr, xi] = rotate(xr, xi, errIn);
        let [yr, yi] = dft4(xr, xi);
        yr = yr.map(v => v * 0.5); yi = yi.map(v => v * 0.5);
        if (errOut) [yr, yi] = rotate(yr, yi, errOut);
        const e_lo = Math.sqrt(Math.pow(10.0, P.p_lo_dbm / 10.0) * 1e3), g = 2.0 * P.R * e_lo;
        const proj = [0, 0, 0, 0], i = [0, 0, 0, 0];
        for (let k = 0; k < 4; k++) {
            const th = rad(LO_PHASE_DEG[k]);
            proj[k] = yr[k] * Math.cos(th) + yi[k] * Math.sin(th);
            i[k] = g * proj[k];
        }
        return { f1, f0, p_lane, a, xr, xi, yr, yi, e_lo, g, proj, i };
    }
    function chainScale() {
        const c = chainModel([0, 0, 0, 0]);
        return [c.i[0], c.g * (c.a * (c.f1 - c.f0)) / 2.0];
    }
    function idealLevels(bits) { const [re, im] = dft4Int(bits); return [re[0], re[1], re[2], im[3]]; }
    function detectedLevels(bits, errIn = null, errOut = null) {
        const [off, step] = chainScale(), c = chainModel(bits, errIn, errOut);
        return [0, 1, 2, 3].map(k => (c.i[k] - (k === 0 ? off : 0.0)) / step);
    }
    const decide = levels => levels.map(v => Math.floor(v + 0.5));
    function pam4Symbols(bits) {
        const l = idealLevels(bits);
        return [Math.min(l[0], 3), l[1] + 1, Math.min(l[2] + 2, 3), l[3] + 1];
    }
    function idft4Int(X0, X1r, X1i, X2) {
        const Xr = [X0, X1r, X2, X1r], Xi = [0, X1i, 0, -X1i], out = [];
        for (let n = 0; n < 4; n++) {
            let s = 0;
            for (let k = 0; k < 4; k++) { const [wr, wi] = W4[(k * n) % 4]; s += Xr[k] * wr + Xi[k] * wi; }
            out.push(s);
        }
        return out;
    }
    function decode(sym) {
        const c0 = sym[0] < 3 ? [sym[0]] : [3, 4], c2 = sym[2] < 3 ? [sym[2] - 2] : [1, 2];
        const x1r = sym[1] - 1, x1i = -(sym[3] - 1);
        const par = [];
        for (const a of c0) for (const b of c2) if (((a - b) % 2 + 2) % 2 === 0) par.push([a, b]);
        const ok = [];
        for (const [a, b] of par) {
            const x4 = idft4Int(a, x1r, x1i, b);
            if (x4.every(v => v === 0 || v === 4)) ok.push(x4.map(v => v / 4));
        }
        return { bits: ok.length === 1 ? ok[0] : null, npar: par.length, nok: ok.length };
    }

    // ── the receiver ──
    function homodyneTerms(p_lo_W, p_sig_W = null) {
        if (p_sig_W === null) p_sig_W = P.p_sig_uW * 1e-6;
        const B = P.baud / 2.0, R = P.R;
        const sig = 2.0 * R * Math.sqrt(p_sig_W * p_lo_W);
        const th = P.i_n * Math.sqrt(B);
        const shot = Math.sqrt(2.0 * Q_E * R * (p_lo_W + p_sig_W) * B);
        const rin = R * p_lo_W * Math.sqrt(Math.pow(10.0, P.rin_db / 10.0) * B) * Math.pow(10.0, -P.cmrr_db / 20.0);
        return { sig, th, shot, rin, tot: Math.sqrt(th * th + shot * shot + rin * rin) };
    }
    function qPerStep(p_lo_W, p_sig_W) {
        const step = homodyneTerms(p_lo_W).sig / 2.0;
        return (step / 2.0) / homodyneTerms(p_lo_W, p_sig_W).tot;
    }
    const tiaLimit = (f3db, c_t) => P.gbw / (2.0 * Math.PI * c_t * f3db * f3db);

    // ── skew ──
    const SKEW = { sps: 32, nsym: 600, edge: 20 };
    function prbs15(n, seed = 1) {
        let s = seed; const out = [];
        for (let i = 0; i < n; i++) { const b = ((s >> 14) ^ (s >> 13)) & 1; s = ((s << 1) | b) & 0x7FFF; out.push(b); }
        return out;
    }
    function skewLanes() {
        const b = prbs15(4 * SKEW.nsym);
        return [0, 1, 2, 3].map(k => b.slice(k * SKEW.nsym, (k + 1) * SKEW.nsym));
    }
    function gaussKernel(bw) {
        const sps = SKEW.sps, sigma = Math.sqrt(Math.log(2.0)) / (2.0 * Math.PI * bw), h = [];
        for (let i = 0; i < 8 * sps + 1; i++) { const t = (i - 4 * sps) / sps; h.push(Math.exp(-t * t / (2.0 * sigma * sigma))); }
        let s = 0.0; for (const v of h) s = s + v;
        return h.map(v => v / s);
    }
    function filtered(bits, h) {
        const sps = SKEW.sps, half = (h.length - 1) / 2, n = bits.length * sps, out = new Float64Array(n);
        for (let m = 0; m < n; m++) {
            let acc = 0.0;
            for (let i = 0; i < h.length; i++) { const j = m + i - half; if (j >= 0 && j < n) acc = acc + h[i] * bits[Math.floor(j / sps)]; }
            out[m] = acc;
        }
        return out;
    }
    function eyeOpening(lf, lanes, shift) {
        const { sps, nsym, edge } = SKEW;
        let best = -1e9, bestPh = 0;
        for (let ph = 0; ph < sps; ph++) {
            const lo = [1e9, 1e9, 1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9, -1e9, -1e9];
            for (let s = edge; s < nsym - edge; s++) {
                const m = s * sps + ph;
                const v = lf[0][m - shift] + lf[1][m] + lf[2][m] + lf[3][m];
                const lab = lanes[0][s] + lanes[1][s] + lanes[2][s] + lanes[3][s];
                if (v < lo[lab]) lo[lab] = v;
                if (v > hi[lab]) hi[lab] = v;
            }
            let gap = 1e9;
            for (let l = 0; l < 4; l++) { const g = lo[l + 1] - hi[l]; if (g < gap) gap = g; }
            if (gap > best) { best = gap; bestPh = ph; }
        }
        return { opening: best, phase: bestPh };
    }
    // the summed waveform (for drawing the eye)
    function summed(lf, shift, from, to) {
        const out = new Float64Array(to - from);
        for (let m = from; m < to; m++) out[m - from] = lf[0][m - shift] + lf[1][m] + lf[2][m] + lf[3][m];
        return out;
    }

    // a balanced pair: each photodiode reads |E +/- L|^2 / 2; the difference is 2 Re(E L*)
    function balancedPair(er, ei, lr, li) {
        const p1 = ((er + lr) * (er + lr) + (ei + li) * (ei + li)) / 2.0;
        const p2 = ((er - lr) * (er - lr) + (ei - li) * (ei - li)) / 2.0;
        return { p1, p2, diff: p1 - p2, expect: 2.0 * (er * lr + ei * li) };
    }

    const API = { P, LO_PHASE_DEG, ERR_IN_DEG, ERR_OUT_DEG, EX_BITS, VECS, SKEW, dft4, dft4Int, rotate, mzmArms, biasPoint, dither,
        laneLevels, chainModel, chainScale, idealLevels, detectedLevels, decide, pam4Symbols, idft4Int, decode, homodyneTerms,
        qPerStep, tiaLimit, prbs15, skewLanes, gaussKernel, filtered, eyeOpening, summed, balancedPair };
    if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.FOP04 = API;
})(typeof window !== 'undefined' ? window : globalThis);
