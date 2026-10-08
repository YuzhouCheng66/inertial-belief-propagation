"""Supported eta injection and precision diagnostics."""
import numpy as np

def build_lift(p, dst, U, P, rtol=1e-13):
    S = np.zeros_like(P)
    np.add.at(S, dst, p)
    S = .5 * (S + S.transpose(0, 2, 1))
    vals, V = np.linalg.eigh(S)
    scales = np.maximum(np.max(np.abs(vals), axis=1, keepdims=True), 1e-300)
    if np.any(vals < -1e-11 * scales):
        raise ArithmeticError('Indefinite incoming precision; supported lift invalid')
    mask = vals > rtol * scales
    iv = np.zeros_like(vals)
    iv[mask] = 1. / vals[mask]
    invS = (V * iv[:, None, :]) @ V.transpose(0, 2, 1)
    lift = np.ascontiguousarray(p @ (invS @ P)[dst])
    total = np.zeros_like(P)
    np.add.at(total, dst, lift)
    mean_map = np.linalg.solve(P, total)
    ranks = mask.sum(axis=1)
    d = P.shape[1]
    diagnostics = dict(
        incoming_rank_deficient=int(np.sum(ranks < d)),
        rank_deficient=int(np.sum(ranks < d)),
        unreachable_dimensions=int(np.sum(d - ranks)),
        min_incoming_eigenvalue=float(vals.min()),
        lift_identity_error=float(np.max(np.abs(mean_map - np.eye(d)))),
        lift_full_rank=bool(np.all(ranks == d)),
        lift_policy='factor-supported; only range(P^-1 S) is reachable',
    )
    return lift, mean_map, ranks, diagnostics


def supported_injection(P, delta, tolerance=1e-9):
    """Return a supported eta increment, or reject an unreachable mean target."""
    target = np.asarray(delta)
    achieved = np.einsum('nij,nj->ni', P['lift_mean_map'], target)
    error = np.linalg.norm(achieved - target)
    if error > tolerance * max(1., np.linalg.norm(target)):
        raise ValueError(f'Mean correction outside incoming-message range: error={error:g}')
    return np.einsum('eij,ej->ei', P['lift'], target[P['dst']])


def relative_residual(L, eta, dst, invP):
    total = L.u.copy()
    np.add.at(total, dst, eta)
    x = np.einsum('nij,nj->ni', invP, total)
    h = L.b - np.einsum('nij,nj->ni', L.D, x)
    np.add.at(h, L.i, -np.einsum('eij,ej->ei', L.Hij, x[L.j]))
    np.add.at(h, L.j, -np.einsum('eji,ej->ei', L.Hij, x[L.i]))
    invD = np.linalg.inv(L.D)
    norm = lambda z: np.sqrt(max(0., np.einsum('ni,nij,nj->', z, invD, z)))
    return float(norm(h) / max(norm(L.b), 1e-300))


def precision_diagnostics(L, p, P, G):
    src, dst, rev, As, B, Ad, bs, bd = L.directed()
    # Direct Schur defect measures the equation encoded by the actual stored
    # normal-equation blocks. Roundoff can differ from PSD factor evaluation.
    defect = p - Ad - G @ B
    vals = np.linalg.eigvalsh(P)
    if not np.isfinite(P).all() or vals.min() <= 0:
        raise ArithmeticError('Non-SPD/nonfinite frozen belief precision')
    return dict(
        schur_defect_relative=float(np.linalg.norm(defect) / max(np.linalg.norm(p), 1e-300)),
        schur_defect_absolute=float(np.max(np.abs(defect))),
        min_belief_eigenvalue=float(vals.min()),
        max_belief_condition=float(np.max(vals[:, -1] / vals[:, 0])),
        precision_is_exact=False,
    )
