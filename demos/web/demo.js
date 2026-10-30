// Nearest words and analogies over the trimmed tables make_tables.py writes
// into out/web/. Serve the repository root (python3 -m http.server) and open
// /demos/web/: the tables are fetched from /out/web/, one language the first
// time a form needs it.
//
// normalize() must stay what tokenize() in scripts/common.py does, since the
// tables are keyed by it: compatibility-decompose, fold the Hebrew geresh and
// gershayim and the typographic apostrophe, drop every combining mark
// (niqqud, accents) and lowercase.
(function () {
    'use strict';

    const DATA = '../../out/web/';
    const tables = {};
    const status = document.getElementById('status');

    function normalize(word) {
        return word.trim()
            .replace(/׳|’/g, '\'')
            .replace(/״/g, '"')
            .normalize('NFKD')
            .replace(/\p{Mn}/gu, '')
            .toLowerCase();
    }

    async function fetchOk(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(url + ': ' + response.status + ' (run make_tables.py first)');
        return response;
    }

    // One language: its words, an index from word to row, and the rows.
    function table(lang) {
        if (!tables[lang]) {
            status.textContent = 'loading ' + lang + '…';
            tables[lang] = Promise.all([
                fetchOk(DATA + lang + '.words.json').then(function (r) { return r.json(); }),
                fetchOk(DATA + lang + '.int8').then(function (r) { return r.arrayBuffer(); }),
                fetchOk(DATA + lang + '.scale.f32').then(function (r) { return r.arrayBuffer(); }),
            ]).then(function (parts) {
                const words = parts[0];
                const rows = new Int8Array(parts[1]);
                const index = new Map();
                words.forEach(function (word, i) { index.set(word, i); });
                status.textContent = '';
                return {
                    words: words,
                    index: index,
                    rows: rows,
                    scale: new Float32Array(parts[2]),
                    dim: rows.length / words.length,
                };
            });
        }
        return tables[lang];
    }

    function vector(t, word) {
        const row = t.index.get(normalize(word));
        if (row === undefined) return null;
        const v = new Float32Array(t.dim);
        const base = row * t.dim;
        for (let j = 0; j < t.dim; j++) v[j] = t.rows[base + j] * t.scale[row];
        return v;
    }

    // The k rows with the highest cosine to query, skipping the words in exclude.
    function nearest(t, query, k, exclude) {
        let norm = 0;
        for (let j = 0; j < t.dim; j++) norm += query[j] * query[j];
        norm = Math.sqrt(norm);
        const best = [];
        for (let i = 0; i < t.words.length; i++) {
            const base = i * t.dim;
            let dot = 0;
            for (let j = 0; j < t.dim; j++) dot += t.rows[base + j] * query[j];
            const score = dot * t.scale[i] / norm;
            if (best.length < k || score > best[best.length - 1].score) {
                if (exclude.has(t.words[i])) continue;
                best.push({ word: t.words[i], score: score });
                best.sort(function (x, y) { return y.score - x.score; });
                if (best.length > k) best.pop();
            }
        }
        return best;
    }

    function show(list, found) {
        list.replaceChildren.apply(list, found.map(function (item) {
            const li = document.createElement('li');
            li.dir = 'auto';
            li.textContent = item.word + ' ';
            const score = document.createElement('span');
            score.textContent = item.score.toFixed(2);
            li.appendChild(score);
            return li;
        }));
    }

    function fail(list, message) {
        list.replaceChildren();
        status.textContent = message;
    }

    async function similar(form) {
        const list = document.getElementById('similar-results');
        const from = await table(form.from.value);
        const to = await table(form.to.value);
        const v = vector(from, form.word.value);
        if (!v) return fail(list, form.word.value + ': not in the table');
        status.textContent = '';
        const exclude = from === to ? new Set([normalize(form.word.value)]) : new Set();
        show(list, nearest(to, v, 12, exclude));
    }

    async function analogy(form) {
        const list = document.getElementById('analogy-results');
        const from = await table(form.from.value);
        const to = await table(form.to.value);
        const given = [form.a.value, form.b.value, form.c.value];
        const vs = given.map(function (w) { return vector(from, w); });
        const missing = given.filter(function (w, i) { return !vs[i]; });
        if (missing.length) return fail(list, 'not in the table: ' + missing.join(', '));
        status.textContent = '';
        // b - a + c: Mikolov's 3CosAdd.
        const q = new Float32Array(from.dim);
        for (let j = 0; j < from.dim; j++) q[j] = vs[1][j] - vs[0][j] + vs[2][j];
        const exclude = from === to ? new Set(given.map(normalize)) : new Set();
        show(list, nearest(to, q, 8, exclude));
    }

    [['similar', similar], ['analogy', analogy]].forEach(function (pair) {
        const form = document.getElementById(pair[0]);
        form.addEventListener('submit', function (event) {
            event.preventDefault();
            pair[1](form).catch(function (error) { status.textContent = error.message; });
        });
    });
})();
