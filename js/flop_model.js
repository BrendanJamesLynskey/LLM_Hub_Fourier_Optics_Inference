// flop_model.js: JavaScript port of analysis/flop_share.py's prefill FLOP ledger, used live in
// FOptInf deck 02. Same shape (Llama-3-8B), same variants, same counting conventions and the same
// floating-point operation order, so the numbers are identical to results.md (tested by
// analysis/test_flop_model.py, which runs this file in node).
(function (root) {
    const D = 4096, L = 32, FF = 14336, V = 128256, KV = 8 * 128;   // Llama-3-8B
    const ORDER = 2, SHORT = 3, BLOCK = 256;
    const VARIANTS = ['transformer', 'fnet', 'hyena', 'hybrid', 'hyena_circ'];
    const LABEL = { transformer: 'Transformer (Llama-3-8B)', fnet: 'FNet-shaped (non-causal)', hyena: 'Hyena-2',
                    hybrid: 'Hybrid 1:3 attention:Hyena', hyena_circ: 'Hyena-2 + block-circulant weights' };

    const rfft = n => 2.5 * n * Math.log2(n);           // FFTW's convention; n is a power of two
    const cfft = n => 5.0 * n * Math.log2(n);
    const pow2AtLeast = n => 2 ** Math.ceil(Math.log2(n)); // exact for the integer n used here
    const ops = () => ({ dense: 0, attention: 0, transform: 0, spectral: 0, other: 0 });
    const KEYS = ['dense', 'attention', 'transform', 'spectral', 'other'];
    const add = (a, b) => { for (const k of KEYS) a[k] = a[k] + b[k]; return a; };
    const total = o => o.dense + o.attention + o.transform + o.spectral + o.other;
    const div = (a, b) => Math.floor(a / b);

    function denseMatrix(m, n, tokens, circ) {
        const o = ops();
        if (!circ) { o.dense = 2.0 * m * n * tokens; return o; }
        const k = BLOCK, bins = div(k, 2) + 1;
        o.transform = tokens * (div(n, k) * rfft(k) + div(m, k) * rfft(k));
        o.spectral = tokens * div(m, k) * div(n, k) * bins * 6.0;
        o.other = tokens * div(m, k) * (div(n, k) - 1) * bins * 2.0;
        return o;
    }
    function mlp(tokens, circ) {
        const o = ops();
        for (const [m, n] of [[FF, D], [FF, D], [D, FF]]) add(o, denseMatrix(m, n, tokens, circ));
        return o;
    }
    function attentionBlock(lens, tokens) {
        const o = ops();
        o.dense = 2.0 * tokens * (2 * D * D + 2 * D * KV);
        let a = 0; for (const s of lens) a = a + 2.0 * D * s * (s + 1);
        o.attention = a;
        return o;
    }
    function hyenaBlock(lens, tokens, circ) {
        const o = ops();
        add(o, denseMatrix((ORDER + 1) * D, D, tokens, circ));
        add(o, denseMatrix(D, D, tokens, circ));
        o.other = o.other + tokens * 2.0 * SHORT * (ORDER + 1) * D;
        o.other = o.other + tokens * ORDER * D;
        for (const s of lens) {
            const n = pow2AtLeast(2 * s);
            o.transform = o.transform + ORDER * D * 2 * rfft(n);
            o.spectral = o.spectral + ORDER * D * (div(n, 2) + 1) * 6.0;
        }
        return o;
    }
    function fnetBlock(lens) {
        const o = ops();
        for (const s of lens) o.transform = o.transform + (s * rfft(D) + D * cfft(pow2AtLeast(s)));
        return o;
    }
    function prefill(variant, lens, lmAll = true) {
        let tokens = 0; for (const s of lens) tokens += s;
        const o = ops(), circ = variant === 'hyena_circ';
        for (let layer = 0; layer < L; layer++) {
            if (variant === 'transformer' || (variant === 'hybrid' && layer % 4 === 0)) add(o, attentionBlock(lens, tokens));
            else if (variant === 'hyena' || variant === 'hybrid' || variant === 'hyena_circ') add(o, hyenaBlock(lens, tokens, circ));
            else if (variant === 'fnet') add(o, fnetBlock(lens));
            add(o, mlp(tokens, circ));
        }
        const head = ops(); head.dense = 2.0 * V * D * (lmAll ? tokens : lens.length);
        add(o, head);
        const t = total(o), opt = o.transform + o.spectral;
        return { ...o, total: t, optical: opt, share: opt / t, amdahl: 1 / (1 - opt / t) };
    }
    // Share of *time* if transform work runs at 1/r of the dense rate (illustrative).
    function timeShare(o, r) { const t = o.optical * r; return t / (o.total - o.optical + t); }

    const api = { VARIANTS, LABEL, prefill, timeShare };
    if (typeof module !== 'undefined') module.exports = api; else root.FlopModel = api;
})(this);
