"""Principal-eigenvector weights and the consistency ratio."""

import math

import numpy as np


RANDOM_INDEX = (0.0, 0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32,
                1.41, 1.45, 1.49, 1.51, 1.54, 1.56, 1.57, 1.59)


def solve_ahp_weights(
    matrix: np.ndarray, rubric_ids: list[str]
) -> tuple[dict[str, float], float]:
    """Return normalized positive weights and CR for a reciprocal matrix."""
    size = len(rubric_ids)
    if size == 0 or len(set(rubric_ids)) != size:
        raise ValueError("rubric_ids must be nonempty and unique")
    matrix = np.array(matrix, dtype=float, copy=True)
    if matrix.shape != (size, size):
        raise ValueError("matrix dimensions must match rubric_ids")
    if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0):
        raise ValueError("matrix entries must be positive and finite")
    if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("matrix diagonal must equal one")
    if not np.allclose(matrix * matrix.T, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("matrix must be reciprocal")
    if size == 1:
        return {rubric_ids[0]: 1.0}, 0.0

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    principal_index = int(np.argmax(eigenvalues.real))
    eigenvalue = eigenvalues[principal_index]
    eigenvector = eigenvectors[:, principal_index]
    if abs(float(eigenvalue.imag)) > 1e-8 or np.max(np.abs(eigenvector.imag)) > 1e-8:
        raise ValueError("principal eigenpair must be real")
    vector = eigenvector.real.astype(float)
    if float(np.sum(vector)) < 0:
        vector = -vector
    total = float(np.sum(vector))
    if np.any(vector <= 0) or not math.isfinite(total) or total <= 0:
        raise ValueError("principal eigenvector must be positive and finite")
    normalized = vector / total
    weights = {rubric_id: float(normalized[i]) for i, rubric_id in enumerate(rubric_ids)}
    cr = 0.0
    if size > 2:
        ci = max(0.0, (float(eigenvalue.real) - size) / (size - 1))
        cr = ci / RANDOM_INDEX[min(size, 15)]
    return weights, cr
