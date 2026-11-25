"""Plant construction for the wall-anchored mass-spring-damper chain.

Topology (wall on the left, free right end), N masses::

    |\\
    |--[k1,c1]--(m1)--[k2,c2]--(m2)-- ... --[kN,cN]--(mN)   free
    |/

Spring/damper ``i`` (1-indexed) couples mass ``i-1`` to mass ``i``, with mass 0
being the fixed wall. Hence there are N springs and N dampers.

Continuous-time second-order dynamics::

    M q'' + C q' + K q = B_f u + w

with M = diag(masses), and K, C tridiagonal. In first-order state-space form with
state x = [q (n); q' (n)]::

    x' = A x + B u + G w,   A = [[0, I], [-M^{-1} K, -M^{-1} C]]
                                 B = [[0], [M^{-1} B_f]]

The measurement selects sensed masses' (position, velocity) pairs.

All builders are pure functions of numpy arrays; they do not depend on the env class
or the codesign base API.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import cont2discrete


def tridiagonal_from_links(link_values: np.ndarray) -> np.ndarray:
    """Assemble the tridiagonal coupling matrix for a wall-anchored chain.

    ``link_values[i]`` is the coefficient of link ``i`` (0-indexed) connecting mass
    ``i-1`` (or the wall, for ``i==0``) to mass ``i``. Returns the ``n x n`` matrix
    ``K`` such that the elastic/damping force on the chain is ``K q``.

    Diagonal: ``K[i,i] = link[i] + link[i+1]`` for interior masses; the free right end
    has ``K[n-1,n-1] = link[n-1]``. Off-diagonals: ``K[i,i+1] = K[i+1,i] = -link[i+1]``.
    """
    v = np.asarray(link_values, dtype=float)
    n = v.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        diag = v[i]
        if i + 1 < n:
            diag += v[i + 1]
        M[i, i] = diag
        if i + 1 < n:
            M[i, i + 1] = -v[i + 1]
            M[i + 1, i] = -v[i + 1]
    return M


def actuator_matrix(masses: np.ndarray, actuators: np.ndarray) -> np.ndarray:
    """Input matrix ``B`` (state-space) for the given actuator placement.

    Column ``k`` corresponds to the k-th actuated mass (in ascending mass index).
    A force on mass ``i`` enters the velocity block scaled by ``1/mass_i``.
    Returns an array of shape ``(2n, n_act)``; if no actuators, shape ``(2n, 0)``.
    """
    masses = np.asarray(masses, dtype=float)
    n = masses.shape[0]
    act_idx = np.flatnonzero(np.asarray(actuators).astype(bool))
    B = np.zeros((2 * n, act_idx.size))
    for col, i in enumerate(act_idx):
        B[n + i, col] = 1.0 / masses[i]
    return B


def sensor_matrix(sensors: np.ndarray) -> np.ndarray:
    """Measurement matrix ``Cs`` selecting (position, velocity) of sensed masses.

    Returns shape ``(2 * n_sensed, 2n)``. Rows come in (position, velocity) pairs per
    sensed mass, in ascending mass index.
    """
    sensors = np.asarray(sensors).astype(bool)
    n = sensors.shape[0]
    sen_idx = np.flatnonzero(sensors)
    C = np.zeros((2 * sen_idx.size, 2 * n))
    for row, i in enumerate(sen_idx):
        C[2 * row, i] = 1.0        # position of mass i
        C[2 * row + 1, n + i] = 1.0  # velocity of mass i
    return C


def continuous_state_space(
    masses: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    actuators: np.ndarray,
):
    """Build continuous-time (A, B, G) from the physical embodiment.

    ``G`` maps a process-noise vector acting on the velocities (shape ``(2n, n)``).
    Returns ``(A, B, G)``.
    """
    masses = np.asarray(masses, dtype=float)
    n = masses.shape[0]
    Minv = np.diag(1.0 / masses)
    K = tridiagonal_from_links(stiffness)
    C = tridiagonal_from_links(damping)

    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Minv @ K
    A[n:, n:] = -Minv @ C

    B = actuator_matrix(masses, actuators)

    G = np.zeros((2 * n, n))
    G[n:, :] = Minv  # noise enters as a force on each mass
    return A, B, G


def discretize(A: np.ndarray, B: np.ndarray, dt: float):
    """Zero-order-hold discretization of (A, B) at step ``dt``.

    Returns ``(Ad, Bd)``. Handles the no-input case (B with zero columns) by
    discretizing A alone.
    """
    n = A.shape[0]
    if B.shape[1] == 0:
        Ad, _, _, _, _ = cont2discrete((A, np.zeros((n, 1)), np.eye(n), 0.0), dt)
        return Ad, np.zeros((n, 0))
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(n), np.zeros((n, B.shape[1]))), dt)
    return Ad, Bd


class Plant:
    """Discrete-time plant assembled from an embodiment, ready for stepping.

    Holds the discretized dynamics plus the measurement and noise-injection maps.
    Construction is the only expensive step; :meth:`step` is a couple of matmuls.
    """

    def __init__(
        self,
        masses: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
        actuators: np.ndarray,
        sensors: np.ndarray,
        dt: float,
    ) -> None:
        self.n = int(np.asarray(masses).shape[0])
        self.dt = float(dt)
        self.actuators = np.asarray(actuators).astype(bool)
        self.sensors = np.asarray(sensors).astype(bool)

        A, B, G = continuous_state_space(masses, stiffness, damping, self.actuators)
        self.A_cont = A
        self.Ad, self.Bd = discretize(A, B, dt)
        # Discretize the noise input with the same ZoH map as a generic input.
        _, self.Gd = discretize(A, G, dt)
        self.Cs = sensor_matrix(self.sensors)

    @property
    def n_actuators(self) -> int:
        return int(self.actuators.sum())

    @property
    def n_sensors(self) -> int:
        return int(self.sensors.sum())

    def step_state(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Advance the state one discrete step: x+ = Ad x + Bd u + Gd w."""
        x = np.asarray(x, dtype=float)
        xp = self.Ad @ x
        if self.Bd.shape[1] > 0 and u.size > 0:
            xp = xp + self.Bd @ u[: self.Bd.shape[1]]
        if w is not None and w.size > 0:
            xp = xp + self.Gd @ w
        return xp

    def measure(self, x: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
        """Measurement y = Cs x (+ noise v). Empty array if no sensors."""
        if self.Cs.shape[0] == 0:
            return np.zeros(0)
        y = self.Cs @ np.asarray(x, dtype=float)
        if v is not None and v.size > 0:
            y = y + v
        return y
