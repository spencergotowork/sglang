"""Small CPU models for the interview's mechanisms, not production kernels."""

import json
import math
import random


def acceptance():
    rng = random.Random(27)
    gamma, alpha, rounds = 3, 0.5, 100_000
    accepted = 0
    for _ in range(rounds):
        for _ in range(gamma):
            if rng.random() >= alpha:
                break
            accepted += 1
    predicted = sum(alpha**j for j in range(gamma + 1))
    observed = 1 + accepted / rounds
    assert abs(observed - predicted) < 0.02
    # This is a different definition: observed accepted-prefix count / all drafts.
    global_accept_rate = accepted / (rounds * gamma)
    assert abs(observed - (1 + gamma * global_accept_rate)) < 1e-12
    return dict(
        gamma=gamma,
        conditional_alpha=alpha,
        expected_output=predicted,
        simulated_output=observed,
        observed_global_rate=global_accept_rate,
        output_if_global_rate_is_half=1 + gamma * 0.5,
        speedup_if_round_cost_is_1_2=predicted / 1.2,
    )


def dcp_commit():
    # Simplified target KV only. Prefix [0, 1, 2, 3, 4], two DCP ranks.
    # At positions 5, 6, 7, verify generates temporary KV for three proposals.
    # Only positions 5 and 6 survive; a correction token at 7 has NO KV yet.
    prefix = {p: f"prefix:{p}" for p in range(5)}
    temporary = {5: "draft:A", 6: "draft:B", 7: "draft:C"}
    n_ranks = 2
    variants = []
    for accepted_count in range(4):
        valid = {
            **prefix,
            **{p: v for p, v in temporary.items() if p < 5 + accepted_count},
        }
        shards = [
            {p // n_ranks: v for p, v in valid.items() if p % n_ranks == rank}
            for rank in range(n_ranks)
        ]
        recovered = {
            row * n_ranks + rank: value
            for rank, shard in enumerate(shards)
            for row, value in shard.items()
        }
        assert recovered == valid
        assert len(recovered) == 5 + accepted_count
        variants.append(
            dict(
                accepted_drafts=accepted_count,
                committed_kv_length=len(valid),
                shards=shards,
                correction_or_bonus_kv_ready=False,
            )
        )
    return variants


def softmax_merge():
    scores, values = [0.0, 1.0, -2.0, 3.0, 0.5], [1.0, 4.0, 2.0, -1.0, 5.0]

    def partial(ids):
        m = max(scores[i] for i in ids)
        exps = [math.exp(scores[i] - m) for i in ids]
        z = sum(exps)
        return sum(w * values[i] for w, i in zip(exps, ids)) / z, m + math.log(z)

    parts = [partial(list(range(r, len(scores), 2))) for r in range(2)]
    m = max(lse for _, lse in parts)
    weights = [math.exp(lse - m) for _, lse in parts]
    merged = sum(o * w for (o, _), w in zip(parts, weights)) / sum(weights)
    dense, _ = partial(list(range(len(scores))))
    wrong = sum(o for o, _ in parts) / len(parts)
    assert math.isclose(merged, dense, abs_tol=1e-12)
    return dict(dense=dense, merged_with_lse=merged, wrong_plain_average=wrong)


def global_topk():
    # Equal scores use one global token-ID tie break on every rank.
    scores = [0.8, 0.8, -0.2, 0.9, 0.1, 0.8]
    k = 2

    def select(ids):
        return sorted(ids, key=lambda i: (-scores[i], i))[:k]

    local = [select(range(rank, len(scores), 2)) for rank in range(2)]
    merged = select([i for candidates in local for i in candidates])
    dense = select(range(len(scores)))
    assert merged == dense
    assert sum(map(len, local)) > k
    return dict(local_candidates=local, global_topk=merged, dense_topk=dense)


def cache_axes():
    # Same layer, next time step: select the NEW top-k, then fetch only misses.
    previous, current = {1, 2, 3, 4}, {2, 3, 4, 9}
    resident = previous
    misses = current - resident
    assert misses == {9}
    # Reusing old selection would silently attend to 1 instead of 9.
    assert previous != current
    changed_hotspot = {8, 9, 10, 11}
    cold_misses = changed_hotspot - resident
    assert len(cold_misses) == 4
    return dict(
        selection_changed=True,
        required_topk=sorted(current),
        host_fetch=sorted(misses),
        changed_hotspot_misses=sorted(cold_misses),
        hit_ratio=len(current & resident) / len(current),
        note="Cross-layer index sharing does not share each layer's KV values.",
    )


def expert_union():
    same = [{0, 1}] * 4
    disjoint = [{2 * i, 2 * i + 1} for i in range(4)]

    # In EP with 4 ranks and expert e on rank e % 4, equal-sized experts can run
    # concurrently. Unique bytes across the cluster != critical-path bytes.
    def describe(routes):
        union = set().union(*routes)
        return dict(
            unique_experts=len(union),
            experts_by_rank=[len({e for e in union if e % 4 == r}) for r in range(4)],
            routed_token_expert_pairs=sum(map(len, routes)),
        )

    return dict(same_routes=describe(same), disjoint_routes=describe(disjoint))


def exact_rejection():
    # Exact identity for one proposal step on a common, finite token universe.
    p, q = [0.1, 0.6, 0.3], [0.6, 0.4, 0.0]
    accepted_mass = [min(a, b) for a, b in zip(p, q)]
    residual_mass = [max(a - b, 0) for a, b in zip(p, q)]
    rejection_probability = 1 - sum(accepted_mass)
    residual_total = sum(residual_mass)
    recovered = [
        a + rejection_probability * r / residual_total
        for a, r in zip(accepted_mass, residual_mass)
    ]
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(p, recovered))
    return dict(
        target=p,
        draft=q,
        recovered_distribution=recovered,
        target_token_outside_draft_vocab_is_preserved=True,
    )


if __name__ == "__main__":
    print(
        json.dumps(
            {
                fn.__name__: fn()
                for fn in (
                    acceptance,
                    dcp_commit,
                    softmax_merge,
                    global_topk,
                    cache_axes,
                    expert_union,
                    exact_rejection,
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
