"""A stdlib-only mathematical demo, NOT a SGLang kernel or GPU test.

Run from the repository root: python3 test/manual/kernels/paged_attention_reference.py

Compare dense softmax, logical-token gathers from permuted physical pages,
online (m, l, u) softmax, and split output + log-sum-exp (LSE) merging.
All calculations use Python floats and natural logarithms. This deliberately
does not model GPU layouts, rounding, dispatch, performance, or backend support.
An empty/all-masked row returns zero output and LSE = -inf by *this demo's*
convention; that is not a claim about any production attention backend.
"""

import math
import random

NEG_INF = -math.inf


def dense(rows, width):
    """One direct stable softmax over (score, value_vector) pairs."""
    active = [(score, value) for score, value in rows if score != NEG_INF]
    if not active:
        return [0.0] * width, NEG_INF
    maximum = max(score for score, _ in active)
    weights = [math.exp(score - maximum) for score, _ in active]
    total = math.fsum(weights)
    output = [
        math.fsum(w * value[d] for w, (_, value) in zip(weights, active)) / total
        for d in range(width)
    ]
    return output, maximum + math.log(total)


def online(rows, width, tile_size):
    """Keep m=max(score), l=sum(exp(score-m)), u=sum(exp(score-m)*V)."""
    assert tile_size > 0
    m, l, u = NEG_INF, 0.0, [0.0] * width
    for start in range(0, len(rows), tile_size):
        tile = [(s, v) for s, v in rows[start : start + tile_size] if s != NEG_INF]
        if not tile:  # Never evaluate -inf - -inf for an empty tile.
            continue
        new_m = max(m, max(s for s, _ in tile))
        correction = 0.0 if l == 0.0 else math.exp(m - new_m)
        weights = [math.exp(s - new_m) for s, _ in tile]
        u = [
            correction * u[d]
            + math.fsum(w * v[d] for w, (_, v) in zip(weights, tile))
            for d in range(width)
        ]
        l = correction * l + math.fsum(weights)
        m = new_m
    if l == 0.0:
        return [0.0] * width, NEG_INF
    return [value / l for value in u], m + math.log(l)


def merge(parts, width):
    """Weight each normalized partial output by exp(LSE), not equally."""
    active = [(out, lse) for out, lse in parts if lse != NEG_INF]
    if not active:
        return [0.0] * width, NEG_INF
    maximum = max(lse for _, lse in active)
    weights = [math.exp(lse - maximum) for _, lse in active]
    total = math.fsum(weights)
    output = [
        math.fsum(w * out[d] for w, (out, _) in zip(weights, active)) / total
        for d in range(width)
    ]
    return output, maximum + math.log(total)


def split_attention(rows, width, split_size, tile_size):
    assert split_size > 0
    parts = [online([], width, tile_size)]  # An empty split contributes no mass.
    parts += [
        online(rows[i : i + split_size], width, tile_size)
        for i in range(0, len(rows), split_size)
    ]
    return merge(parts, width)


def page_cache(keys, values, page_size, rng):
    """Synthetic [physical_page][offset] storage, with unused pages/padding."""
    count = (len(keys) + page_size - 1) // page_size
    page_ids = list(range(count + 2))
    rng.shuffle(page_ids)
    table = page_ids[:count]  # logical_page -> physical_page; not token order.
    pages = [[None] * page_size for _ in page_ids]
    for token, record in enumerate(zip(keys, values)):
        pages[table[token // page_size]][token % page_size] = record
    return pages, table


def gather(pages, table, page_size, length):
    # Length, not page capacity, excludes the non-full final page's padding.
    records = [pages[table[t // page_size]][t % page_size] for t in range(length)]
    return [record[0] for record in records], [record[1] for record in records]


def attention_rows(query, keys, values, head, query_heads, position, window=None):
    """Causal GQA: KV head = query head // group size; window includes self."""
    kv_heads = len(keys[0])
    assert query_heads % kv_heads == 0
    kv_head = head // (query_heads // kv_heads)
    scale = 1.0 / math.sqrt(len(query))
    rows = []
    for token, (key, value) in enumerate(zip(keys, values)):
        visible = token <= position and (window is None or token > position - window)
        score = (
            math.fsum(q * k for q, k in zip(query, key[kv_head])) * scale
            if visible
            else NEG_INF
        )
        rows.append((score, value[kv_head]))
    return rows


def check(actual, expected):
    out, lse = actual
    expected_out, expected_lse = expected
    assert len(out) == len(expected_out)
    for a, b in zip(out, expected_out):
        assert math.isclose(a, b, rel_tol=2e-11, abs_tol=2e-11), (actual, expected)
    assert lse == expected_lse or math.isclose(
        lse, expected_lse, rel_tol=2e-13, abs_tol=2e-11
    ), (actual, expected)


def check_easy_example():
    rows = [(math.log(w), [v]) for w, v in zip((1, 2, 4, 8), (10, 20, 30, 40))]
    expected = ([490.0 / 15.0], math.log(15.0))
    check(dense(rows, 1), expected)
    left, right = dense(rows[:2], 1), dense(rows[2:], 1)
    check(left, ([50.0 / 3.0], math.log(3.0)))
    check(right, ([110.0 / 3.0], math.log(12.0)))
    wrong_mean = (left[0][0] + right[0][0]) / 2.0
    assert math.isclose(wrong_mean, 80.0 / 3.0)
    assert not math.isclose(wrong_mean, expected[0][0])
    # Exhaust every contiguous partition of these four tokens, in both orders.
    for cuts in range(1 << (len(rows) - 1)):
        ends = [i for i in range(1, len(rows)) if cuts & (1 << (i - 1))] + [4]
        for tile_size in range(1, 6):
            parts, start = [], 0
            for end in ends:
                parts.append(online(rows[start:end], 1, tile_size))
                start = end
            check(merge(parts, 1), expected)
            check(merge(list(reversed(parts)), 1), expected)
            check(online(rows, 1, tile_size), expected)
    print(f"Exact example: output = 490/15 = {expected[0][0]:.10f}")
    print(f"Incorrect unweighted partial mean = {wrong_mean:.10f}")


def check_edge_cases():
    # Naive exp(1000) overflows; subtracting a maximum remains finite.
    for scores in ((1000, 1000, -1000), (-1000, -1000, -2000)):
        rows = list(zip(scores, ([1.0], [3.0], [9.0])))
        expected = ([2.0], scores[0] + math.log(2.0))
        check(dense(rows, 1), expected)
        for tile in range(1, 5):
            check(online(rows, 1, tile), expected)
            for split in range(1, 5):
                check(split_attention(rows, 1, split, tile), expected)
    empty = ([0.0, 0.0], NEG_INF)
    for rows in ([], [(NEG_INF, [99.0, -99.0])] * 5):
        check(dense(rows, 2), empty)
        for tile in (1, 2, 8):
            check(online(rows, 2, tile), empty)
            check(split_attention(rows, 2, 2, tile), empty)
    check(merge([], 2), empty)
    check(merge([empty, empty], 2), empty)
    live = ([2.0, -3.0], 7.0)
    check(merge([empty, live, empty], 2), live)
    # A valid tile after and before masked tiles exercises initialization/skip.
    masked = [(NEG_INF, [100.0, 100.0])] * 2
    rows = masked + [(0.0, live[0])] + masked
    check(online(rows, 2, 2), (live[0], 0.0))
    check(split_attention(rows, 2, 2, 1), (live[0], 0.0))


def check_mask_and_gqa():
    keys = [[[0.0], [0.0]] for _ in range(3)]
    values = [[[10.0], [100.0]], [[20.0], [200.0]], [[30.0], [300.0]]]
    # Four query heads share two KV heads in contiguous groups of two.
    for head, expected in enumerate((15.0, 15.0, 150.0, 150.0)):
        rows = attention_rows([1.0], keys, values, head, 4, position=1)
        check(dense(rows, 1), ([expected], math.log(2.0)))
        rows = attention_rows([1.0], keys, values, head, 4, 1, window=1)
        check(dense(rows, 1), ([20.0 if head < 2 else 200.0], 0.0))
        rows = attention_rows([1.0], keys, values, head, 4, -1)
        check(dense(rows, 1), ([0.0], NEG_INF))
    pages, table = page_cache(keys, values, 2, random.Random(7))
    assert table != list(range(len(table)))  # Explicit physical permutation.
    assert pages[table[-1]][1] is None  # Final logical page has only one token.
    assert gather(pages, table, 2, 3) == (keys, values)


def check_randomized():
    comparisons = 0
    for seed in (7, 19, 2026):
        rng = random.Random(seed)
        for length in (1, 2, 7, 13):
            keys = [
                [[rng.uniform(-3, 3) for _ in range(3)] for _ in range(2)]
                for _ in range(length)
            ]
            values = [
                [[rng.uniform(-5, 5) for _ in range(2)] for _ in range(2)]
                for _ in range(length)
            ]
            for query_heads in (2, 4, 8):  # MHA, GQA groups of two and four.
                queries = [
                    [rng.uniform(-2, 2) for _ in range(3)] for _ in range(query_heads)
                ]
                for page_size in (1, 2, 3, 4, 8, 16):
                    pages, table = page_cache(keys, values, page_size, rng)
                    paged_k, paged_v = gather(pages, table, page_size, length)
                    assert (paged_k, paged_v) == (keys, values)
                    for position, window in (
                        (length - 1, None),
                        (length // 2, 3),
                        (0, 1),
                        (-1, None),
                    ):
                        for head, query in enumerate(queries):
                            args = (head, query_heads, position, window)
                            rows = attention_rows(query, keys, values, *args)
                            paged_rows = attention_rows(query, paged_k, paged_v, *args)
                            expected = dense(rows, 2)
                            check(dense(paged_rows, 2), expected)
                            for tile in (1, 2, 3, 5, 16):
                                check(online(paged_rows, 2, tile), expected)
                                for split in (1, 2, 3, 4, 7, 16):
                                    actual = split_attention(paged_rows, 2, split, tile)
                                    check(actual, expected)
                                    comparisons += 1
    return comparisons


def main():
    if not __debug__:
        raise SystemExit("Run without -O: this mathematical demo uses assertions.")
    check_easy_example()
    check_edge_cases()
    check_mask_and_gqa()
    count = check_randomized()
    print(f"PASS: {count:,} split comparisons; seeds 7, 19, 2026.")
    print("Dense/paged/online/split, masks, GQA, padding, extremes, empty rows: PASS")
    print("Mathematical CPU reference only; no SGLang or GPU kernels were executed.")


if __name__ == "__main__":
    main()
