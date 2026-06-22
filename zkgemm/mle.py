"""
Multilinear extension (MLE) operations.

Core primitives for the GKR protocol:
- eq polynomial evaluation and vectorized construction
- MLE evaluation via Thaler's bookkeeping table
- Matrix MLE evaluation
- Sumcheck table building and updating
"""

from __future__ import annotations

from zkgemm.field import P, add, sub, mul


def eq_eval(x: list[int], y: list[int]) -> int:
    """Evaluate eq(x, y) = prod_i (x_i * y_i + (1 - x_i)(1 - y_i)).

    For binary y_i: factor is x_i if y_i=1, else (1 - x_i).
    Works for arbitrary field elements in both x and y.
    """
    assert len(x) == len(y)
    result = 1
    for xi, yi in zip(x, y):
        # factor = xi*yi + (1-xi)*(1-yi)
        t0 = mul(xi, yi)
        t1 = mul(sub(1, xi), sub(1, yi))
        factor = add(t0, t1)
        result = mul(result, factor)
    return result


def eq_vector(r: list[int]) -> list[int]:
    """Build table: eq_vec[b] = eq(r, b) for all b in {0,1}^m.

    Uses LSB-first ordering: variable r[i] corresponds to bit i of the
    index.  This matches the convention used by mle_eval and update_table,
    where the even/odd (LSB) split processes the first variable.

    Construction doubles the table at each step by placing bit i = 0
    entries in the lower half and bit i = 1 entries in the upper half:
        new[j]        = old[j] * (1 - r_i)    (bit i = 0)
        new[j + half] = old[j] * r_i           (bit i = 1)

    Time: O(2^m).  Space: O(2^m).
    """
    table = [1]
    for ri in r:
        one_minus_ri = sub(1, ri)
        half = len(table)
        new_table = [0] * (2 * half)
        for j in range(half):
            new_table[j] = mul(table[j], one_minus_ri)
            new_table[j + half] = mul(table[j], ri)
        table = new_table
    return table


def mle_eval(values: list[int], point: list[int]) -> int:
    """Evaluate the MLE of `values` at `point`.

    values: 2^m field elements (function on {0,1}^m).
    point:  m field elements.

    Uses Thaler's bookkeeping table technique:
      For each variable, fold the table in half by interpolating
      between pairs using the corresponding point coordinate.

    Time: O(2^m).
    """
    m = len(point)
    assert len(values) == (1 << m), f"Expected {1 << m} values, got {len(values)}"

    # Work on a copy
    buf = list(values)
    size = len(buf)

    for i in range(m):
        half = size >> 1
        ri = point[i]
        one_minus_ri = sub(1, ri)
        for b in range(half):
            # buf[b] = buf[2b]*(1-ri) + buf[2b+1]*ri
            buf[b] = add(mul(buf[2 * b], one_minus_ri), mul(buf[2 * b + 1], ri))
        size = half

    return buf[0]


def matrix_mle_eval(
    matrix_flat: list[int], n: int, r_row: list[int], r_col: list[int]
) -> int:
    """Evaluate MLE of an n x n matrix at (r_row, r_col).

    matrix_flat: n*n elements in row-major order, A[i*n + j] = A[i][j].
    n: matrix dimension (power of 2).
    r_row, r_col: each m = log2(n) field elements.

    Algorithm:
      result = sum_{i,j} A[i][j] * eq(r_row, i) * eq(r_col, j)
             = sum_i eq(r_row, i) * (sum_j A[i][j] * eq(r_col, j))

    Time: O(n^2).
    """
    eq_row = eq_vector(r_row)
    eq_col = eq_vector(r_col)

    result = 0
    for i in range(n):
        row_sum = 0
        for j in range(n):
            row_sum = add(row_sum, mul(matrix_flat[i * n + j], eq_col[j]))
        result = add(result, mul(eq_row[i], row_sum))
    return result


def build_sumcheck_tables(
    A_flat: list[int],
    B_flat: list[int],
    n: int,
    r_i: list[int],
    r_j: list[int],
) -> tuple[list[int], list[int]]:
    """Build initial tables for the sumcheck prover.

    T_A[k] = A_tilde(r_i, k) = sum_i A[i][k] * eq(r_i, i)
    T_B[k] = B_tilde(k, r_j) = sum_j B[k][j] * eq(r_j, j)

    for all k in {0, 1, ..., n-1} (binary vectors of length m).

    Time: O(n^2).
    """
    eq_row = eq_vector(r_i)
    eq_col = eq_vector(r_j)

    # T_A[k] = sum_i A[i][k] * eq_row[i]   (column k of A dotted with eq_row)
    T_A = [0] * n
    for k in range(n):
        s = 0
        for i in range(n):
            s = add(s, mul(A_flat[i * n + k], eq_row[i]))
        T_A[k] = s

    # T_B[k] = sum_j B[k][j] * eq_col[j]   (row k of B dotted with eq_col)
    T_B = [0] * n
    for k in range(n):
        s = 0
        for j in range(n):
            s = add(s, mul(B_flat[k * n + j], eq_col[j]))
        T_B[k] = s

    return T_A, T_B


def rect_matrix_mle_eval(
    matrix_flat: list[int], n_rows: int, n_cols: int,
    r_row: list[int], r_col: list[int]
) -> int:
    """Evaluate MLE of an n_rows × n_cols matrix at (r_row, r_col).

    matrix_flat: n_rows*n_cols elements in row-major order.
    r_row: log2(n_rows) field elements.
    r_col: log2(n_cols) field elements.

    Time: O(n_rows * n_cols).
    """
    eq_row = eq_vector(r_row)
    eq_col = eq_vector(r_col)

    result = 0
    for i in range(n_rows):
        row_sum = 0
        for j in range(n_cols):
            row_sum = add(row_sum, mul(matrix_flat[i * n_cols + j], eq_col[j]))
        result = add(result, mul(eq_row[i], row_sum))
    return result


def build_block_sumcheck_tables(
    A_flat: list[int],
    B_flat: list[int],
    bs: int,
    n_inner: int,
    r_a: list[int],
    r_b: list[int],
) -> tuple[list[int], list[int]]:
    """Build sumcheck tables for a block proof.

    A_sub is bs × n_inner, B_sub is n_inner × bs.
    The sumcheck proves: C_block(r_a, r_b) = Σ_k A_tilde(r_a, k) × B_tilde(k, r_b)

    T_A[k] = Σ_a A_sub[a][k] × eq(r_a, a)   for k in {0,...,n_inner-1}
    T_B[k] = Σ_b B_sub[k][b] × eq(r_b, b)   for k in {0,...,n_inner-1}

    Time: O(bs × n_inner).
    """
    eq_row = eq_vector(r_a)   # length bs
    eq_col = eq_vector(r_b)   # length bs

    # T_A[k] = sum_a A_sub[a][k] * eq_row[a]
    T_A = [0] * n_inner
    for k in range(n_inner):
        s = 0
        for a in range(bs):
            s = add(s, mul(A_flat[a * n_inner + k], eq_row[a]))
        T_A[k] = s

    # T_B[k] = sum_b B_sub[k][b] * eq_col[b]
    T_B = [0] * n_inner
    for k in range(n_inner):
        s = 0
        for b in range(bs):
            s = add(s, mul(B_flat[k * bs + b], eq_col[b]))
        T_B[k] = s

    return T_A, T_B


def update_table(table: list[int], challenge: int) -> list[int]:
    """Halve the table by fixing the first variable to `challenge`.

    new[b] = table[2b] * (1 - challenge) + table[2b+1] * challenge

    Time: O(len(table)).
    """
    half = len(table) // 2
    one_minus_r = sub(1, challenge)
    new_table = [0] * half
    for b in range(half):
        new_table[b] = add(
            mul(table[2 * b], one_minus_r), mul(table[2 * b + 1], challenge)
        )
    return new_table
