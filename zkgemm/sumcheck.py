"""
Sumcheck protocol for verifying inner-product claims.

Proves: H = sum_{k in {0,1}^m} T_A[k] * T_B[k]

where T_A and T_B are bookkeeping tables derived from the multilinear
extensions of the input matrices.

Each round, the prover sends a degree-2 univariate polynomial
g_ell(X) represented by three evaluations (g(0), g(1), g(2)).
The verifier checks consistency and derives a random challenge
via Fiat-Shamir.
"""

from __future__ import annotations

from dataclasses import dataclass

from zkgemm.field import P, add, sub, mul, div, inv
from zkgemm.mle import update_table
from zkgemm.native_import import import_zkgemm_cuda
from zkgemm.transcript import Transcript

try:
    _zk = import_zkgemm_cuda()
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


@dataclass
class SumcheckRound:
    """One round of sumcheck: evaluations of g_ell at 0, 1, 2."""

    evals: tuple[int, int, int]


@dataclass
class SumcheckProof:
    """Complete sumcheck proof."""

    claimed_sum: int
    rounds: list[SumcheckRound]


# Precompute inverse of 2 for Lagrange interpolation
_INV2 = inv(2)


def interpolate_degree2(e0: int, e1: int, e2: int, t: int) -> int:
    """Evaluate the degree-2 polynomial defined by g(0)=e0, g(1)=e1, g(2)=e2 at t.

    Uses Lagrange interpolation over {0, 1, 2}:
      L_0(t) = (t-1)(t-2)/2
      L_1(t) = t(2-t)
      L_2(t) = t(t-1)/2

    All arithmetic in F_p.
    """
    # L_0(t) = (t-1)*(t-2) / 2
    l0 = mul(mul(sub(t, 1), sub(t, 2)), _INV2)
    # L_1(t) = t * (2-t)
    l1 = mul(t, sub(2, t))
    # L_2(t) = t * (t-1) / 2
    l2 = mul(mul(t, sub(t, 1)), _INV2)

    return add(add(mul(e0, l0), mul(e1, l1)), mul(e2, l2))


def _compute_round_evals(
    T_A: list[int], T_B: list[int]
) -> tuple[int, int, int]:
    """Compute g(0), g(1), g(2) for the current sumcheck round.

    g(alpha) = sum_{b} T_A[alpha || b] * T_B[alpha || b]

    For multilinear T_A, T_B, the value at alpha=2 is obtained by
    linear extrapolation: f(2) = 2*f(1) - f(0).
    """
    if _HAS_NATIVE:
        return _zk.native_compute_round_evals(T_A, T_B)

    half = len(T_A) // 2
    e0 = 0
    e1 = 0
    e2 = 0

    for b in range(half):
        a0 = T_A[2 * b]
        a1 = T_A[2 * b + 1]
        b0 = T_B[2 * b]
        b1 = T_B[2 * b + 1]

        e0 = add(e0, mul(a0, b0))
        e1 = add(e1, mul(a1, b1))

        # Linear extrapolation to alpha = 2
        a2 = sub(add(a1, a1), a0)  # 2*a1 - a0
        b2 = sub(add(b1, b1), b0)  # 2*b1 - b0
        e2 = add(e2, mul(a2, b2))

    return e0, e1, e2


def sumcheck_prove(
    T_A: list[int],
    T_B: list[int],
    claimed_sum: int,
    transcript: Transcript,
) -> tuple[SumcheckProof, list[int]]:
    """Run the sumcheck prover.

    Args:
        T_A: table of A_tilde(r_i, k) for k in {0,1}^m, length n = 2^m.
        T_B: table of B_tilde(k, r_j) for k in {0,1}^m, length n.
        claimed_sum: the value C_tilde(r_i, r_j) being proved.
        transcript: Fiat-Shamir transcript (mutated in place).

    Returns:
        proof: SumcheckProof with m rounds.
        challenges: list of m field elements (r_0, ..., r_{m-1}).
    """
    n = len(T_A)
    assert n == len(T_B) and (n & (n - 1)) == 0, "Table length must be a power of 2"
    m = n.bit_length() - 1

    transcript.absorb(b"claimed_sum", claimed_sum)

    rounds: list[SumcheckRound] = []
    challenges: list[int] = []

    # Work on mutable copies
    cur_A = list(T_A)
    cur_B = list(T_B)

    for _round in range(m):
        e0, e1, e2 = _compute_round_evals(cur_A, cur_B)

        # Absorb round polynomial into transcript
        transcript.absorb(b"e0", e0)
        transcript.absorb(b"e1", e1)
        transcript.absorb(b"e2", e2)

        # Derive challenge
        r = transcript.squeeze(b"challenge")
        challenges.append(r)

        rounds.append(SumcheckRound(evals=(e0, e1, e2)))

        # Update tables for next round
        if _HAS_NATIVE:
            cur_A = _zk.native_update_table(cur_A, r)
            cur_B = _zk.native_update_table(cur_B, r)
        else:
            cur_A = update_table(cur_A, r)
            cur_B = update_table(cur_B, r)

    proof = SumcheckProof(claimed_sum=claimed_sum, rounds=rounds)
    return proof, challenges


def sumcheck_verify(
    proof: SumcheckProof,
    transcript: Transcript,
) -> tuple[bool, int, list[int]]:
    """Run the sumcheck verifier.

    Args:
        proof: SumcheckProof from the prover.
        transcript: Fiat-Shamir transcript (must start in same state as prover's).

    Returns:
        ok: True if all round checks pass.
        final_claim: value the last round polynomial evaluates to at the last challenge.
        challenges: list of m challenges for the caller to verify the final evaluation.
    """
    transcript.absorb(b"claimed_sum", proof.claimed_sum)

    current_claim = proof.claimed_sum
    challenges: list[int] = []

    for rnd in proof.rounds:
        e0, e1, e2 = rnd.evals

        # Check: g(0) + g(1) must equal the current claim
        if add(e0, e1) != current_claim:
            return False, 0, challenges

        # Absorb round polynomial
        transcript.absorb(b"e0", e0)
        transcript.absorb(b"e1", e1)
        transcript.absorb(b"e2", e2)

        # Derive challenge (must match prover)
        r = transcript.squeeze(b"challenge")
        challenges.append(r)

        # Next claim = g(r)
        current_claim = interpolate_degree2(e0, e1, e2, r)

    return True, current_claim, challenges
