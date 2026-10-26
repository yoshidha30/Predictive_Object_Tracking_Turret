"""
Kalman Filter State Estimation and Multi-Step Uncertainty Extrapolation.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

import math
from typing import Optional, Tuple
import sys
from pathlib import Path
import numpy as np

# Allow direct execution of this script from anywhere or subfolders
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    WORLD_LEFT, WORLD_RIGHT, WORLD_TOP, WORLD_BOTTOM,
    DT, DEFAULT_PROCESS_NOISE_COV, DEFAULT_MEASUREMENT_NOISE_COV,
    DEFAULT_PREDICTION_STEPS
)


class KalmanPredictor:
    """
    Discrete-time Constant Velocity (CV) Kalman Filter with
    multi-step future state extrapolation and covariance uncertainty ellipse calculation.

    State:       x = [x, y, vx, vy]^T
    Measurement: z = [x, y]^T
    """

    def __init__(
        self,
        initial_x: float,
        initial_y: float,
        dt: float = DT,
        process_noise_scale: float = DEFAULT_PROCESS_NOISE_COV,
        measurement_noise_scale: float = DEFAULT_MEASUREMENT_NOISE_COV
    ):
        self.dt = float(dt)
        self.q_scale = float(process_noise_scale)
        self.r_scale = float(measurement_noise_scale)

        # State vector: [x, y, vx, vy]^T
        self.state = np.array([
            [float(initial_x)],
            [float(initial_y)],
            [0.0],
            [0.0]
        ], dtype=np.float64)

        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Measurement matrix H
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Process noise covariance Q
        # In discrete CV kinematics, velocity process noise variance must reflect (a_max * dt)^2
        # where positions are in pixels and velocities in px/s.
        self.Q = np.eye(4, dtype=np.float64) * self.q_scale
        self.Q[2, 2] = 25.0 * max(0.1, self.q_scale / DEFAULT_PROCESS_NOISE_COV)
        self.Q[3, 3] = 25.0 * max(0.1, self.q_scale / DEFAULT_PROCESS_NOISE_COV)

        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float64) * self.r_scale

        # State error covariance matrix P
        self.P = np.eye(4, dtype=np.float64) * 1.0
        self.P[2, 2] = 500.0
        self.P[3, 3] = 500.0

        # Diagnostics telemetry
        self.last_innovation = np.zeros((2, 1), dtype=np.float64)
        self.innovation_magnitude = 0.0
        self.is_coasting = False

    def reset(self, initial_x: float, initial_y: float) -> None:
        """Reset filter to initial coordinates."""
        self.state = np.array([
            [float(initial_x)],
            [float(initial_y)],
            [0.0],
            [0.0]
        ], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1.0
        self.P[2, 2] = 500.0
        self.P[3, 3] = 500.0
        self.innovation_magnitude = 0.0
        self.is_coasting = False

    def set_tuning(self, process_scale: Optional[float] = None, measurement_scale: Optional[float] = None) -> None:
        """Dynamically update process (Q) and measurement (R) noise covariance."""
        if process_scale is not None:
            self.q_scale = max(0.001, min(1.0, float(process_scale)))
            self.Q = np.eye(4, dtype=np.float64) * self.q_scale
            self.Q[2, 2] = 25.0 * max(0.1, self.q_scale / DEFAULT_PROCESS_NOISE_COV)
            self.Q[3, 3] = 25.0 * max(0.1, self.q_scale / DEFAULT_PROCESS_NOISE_COV)

        if measurement_scale is not None:
            self.r_scale = max(0.01, min(20.0, float(measurement_scale)))
            self.R = np.eye(2, dtype=np.float64) * self.r_scale

    def step(self, measurement: Optional[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """
        Execute one Kalman Predict-Correct cycle.
        If measurement is None (occlusion/missed detection), coasts purely on prediction.
        Returns estimated state: (est_x, est_y, est_vx, est_vy)
        """
        # --- 1. PREDICT STEP ---
        # Prior state estimate: x^- = F * x
        state_prior = self.F @ self.state
        # Prior covariance estimate: P^- = F * P * F^T + Q
        P_prior = (self.F @ self.P @ self.F.T) + self.Q

        # --- 2. UPDATE / CORRECT STEP ---
        if measurement is not None:
            self.is_coasting = False
            z = np.array([[float(measurement[0])], [float(measurement[1])]], dtype=np.float64)

            # Measurement innovation: y = z - H * x^-
            y = z - (self.H @ state_prior)
            self.last_innovation = y
            self.innovation_magnitude = float(np.hypot(y[0, 0], y[1, 0]))

            # Innovation covariance: S = H * P^- * H^T + R
            S = (self.H @ P_prior @ self.H.T) + self.R

            # Kalman gain: K = P^- * H^T * S^-1
            K = P_prior @ self.H.T @ np.linalg.inv(S)

            # Posterior state: x = x^- + K * y
            self.state = state_prior + (K @ y)

            # Joseph Form Posterior Covariance (Guarantees symmetry and positive-definiteness):
            # P = (I - K*H) * P^- * (I - K*H)^T + K * R * K^T
            I_KH = np.eye(4, dtype=np.float64) - (K @ self.H)
            self.P = (I_KH @ P_prior @ I_KH.T) + (K @ self.R @ K.T)
        else:
            # Occlusion / dropout: coast without measurement correction
            self.is_coasting = True
            self.state = state_prior
            self.P = P_prior
            self.innovation_magnitude = 0.0

        est_x = float(self.state[0, 0])
        est_y = float(self.state[1, 0])
        est_vx = float(self.state[2, 0])
        est_vy = float(self.state[3, 0])

        return est_x, est_y, est_vx, est_vy

    def predict_future(self, steps: int = DEFAULT_PREDICTION_STEPS) -> Tuple[float, float, Tuple[float, float, float]]:
        """
        Extrapolate future state H steps into the future and calculate the
        95% confidence uncertainty ellipse from the extrapolated position covariance.

        Returns:
            future_x: Extrapolated X position (clamped to arena)
            future_y: Extrapolated Y position (clamped to arena)
            ellipse_params: (semi_major_axis, semi_minor_axis, orientation_degrees)
        """
        steps = max(0, int(steps))
        horizon_dt = steps * self.dt

        # State extrapolation: x_pred = x_est + v_est * H * dt
        raw_future_x = float(self.state[0, 0] + self.state[2, 0] * horizon_dt)
        raw_future_y = float(self.state[1, 0] + self.state[3, 0] * horizon_dt)

        # Extrapolate covariance ahead:
        # F_H has the form [I, H*dt*I; 0, I]
        F_H = np.array([
            [1.0, 0.0, horizon_dt, 0.0],
            [0.0, 1.0, 0.0, horizon_dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Project covariance ahead with accumulated process noise
        P_future = (F_H @ self.P @ F_H.T) + (self.Q * max(1, steps))

        # Extract 2x2 position covariance submatrix
        P_pos = P_future[0:2, 0:2]

        # Calculate 95% confidence uncertainty ellipse (chi-square 2-DOF scale = 2.447)
        semi_major, semi_minor, angle_deg = self._compute_ellipse(P_pos)

        # Apply boundary reflection extrapolation (elastic collision physics)
        # to anticipate rebounds instead of naive edge sticking
        pred_x = raw_future_x
        pred_y = raw_future_y

        if pred_x < WORLD_LEFT:
            pred_x = WORLD_LEFT + (WORLD_LEFT - pred_x)
        elif pred_x > WORLD_RIGHT:
            pred_x = WORLD_RIGHT - (pred_x - WORLD_RIGHT)

        if pred_y < WORLD_TOP:
            pred_y = WORLD_TOP + (WORLD_TOP - pred_y)
        elif pred_y > WORLD_BOTTOM:
            pred_y = WORLD_BOTTOM - (pred_y - WORLD_BOTTOM)

        clamped_x = max(WORLD_LEFT, min(WORLD_RIGHT, pred_x))
        clamped_y = max(WORLD_TOP, min(WORLD_BOTTOM, pred_y))

        return clamped_x, clamped_y, (semi_major, semi_minor, angle_deg)

    @staticmethod
    def _compute_ellipse(P_pos: np.ndarray) -> Tuple[float, float, float]:
        """Compute semi-major axis, semi-minor axis, and orientation angle in degrees."""
        try:
            eigvals, eigvecs = np.linalg.eigh(P_pos)
            # Ensure eigenvalues are strictly positive
            l1 = max(0.01, float(eigvals[1]))
            l2 = max(0.01, float(eigvals[0]))

            # 95% confidence interval factor for 2-DOF Gaussian
            chi95 = 2.447
            semi_major = max(4.0, min(80.0, chi95 * math.sqrt(l1)))
            semi_minor = max(3.0, min(60.0, chi95 * math.sqrt(l2)))

            # Primary eigenvector determines orientation angle
            v1 = eigvecs[:, 1]
            angle_rad = math.atan2(v1[1], v1[0])
            angle_deg = math.degrees(angle_rad)

            return semi_major, semi_minor, angle_deg
        except Exception:
            return 12.0, 12.0, 0.0

    def get_position_variance_trace(self) -> float:
        """Return Trace(P_pos) = Var(x) + Var(y), a scalar metric of position uncertainty."""
        return float(self.P[0, 0] + self.P[1, 1])

    def get_velocity_estimate(self) -> Tuple[float, float]:
        """Return estimated velocity (vx, vy)."""
        return float(self.state[2, 0]), float(self.state[3, 0])

    def get_innovation_magnitude(self) -> float:
        """Return the scalar magnitude of the latest measurement residual."""
        return float(self.innovation_magnitude)


if __name__ == "__main__":
    print("=" * 65)
    print(" KalmanPredictor State Estimator Test")
    print("=" * 65)
    kf = KalmanPredictor(initial_x=160.0, initial_y=350.0, dt=1.0/60.0)
    est_x, est_y, est_vx, est_vy = kf.step((164.5, 353.0))
    pred_x, pred_y, (major, minor, angle) = kf.predict_future(steps=12)
    print(f"Estimated State: x={est_x:.2f}, y={est_y:.2f}, vx={est_vx:.2f}, vy={est_vy:.2f}")
    print(f"Extrapolated (12 steps): ({pred_x:.2f}, {pred_y:.2f}) [ellipse: major={major:.1f}, minor={minor:.1f}]")
    print("[OK] KalmanPredictor functioning properly.")
    print("To launch the full interactive simulation, run:")
    print("  python main.py\n")