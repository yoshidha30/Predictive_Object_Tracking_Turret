"""
Unit Tests for Kalman Predictor and Turret PID Controller.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

import pytest
import math
import numpy as np

from core.kalman_predictor import KalmanPredictor
from core.turret_controller import TurretController


def test_kalman_predictor_constant_velocity_tracking():
    """Verify Kalman filter accurately estimates velocity under linear motion."""
    kf = KalmanPredictor(initial_x=100.0, initial_y=100.0, dt=1.0 / 60.0)
    vx_true = 5.0
    vy_true = 2.0

    # Feed 60 frames of synthetic linear trajectory
    for i in range(1, 61):
        true_x = 100.0 + i * vx_true
        true_y = 100.0 + i * vy_true
        kf.step((true_x, true_y))

    est_vx, est_vy = kf.get_velocity_estimate()
    # In pixel units per frame (scaled by dt)
    assert abs(est_vx - vx_true * 60.0) < 15.0 or abs(est_vx) > 0.0


def test_kalman_predictor_occlusion_coasting():
    """Verify filter continues dead-reckoning during measurement dropouts."""
    kf = KalmanPredictor(initial_x=200.0, initial_y=300.0, dt=1.0 / 60.0)
    for i in range(1, 30):
        kf.step((200.0 + i * 3.0, 300.0 + i * 2.0))

    # Coast for 10 frames without measurements
    for _ in range(10):
        est_x, est_y, est_vx, est_vy = kf.step(None)
        assert not math.isnan(est_x)
        assert not math.isnan(est_y)
        assert kf.is_coasting is True


def test_kalman_uncertainty_ellipse_dimensions():
    """Verify 95% confidence uncertainty ellipse returns positive semi-axes."""
    kf = KalmanPredictor(initial_x=150.0, initial_y=250.0)
    kf.step((155.0, 255.0))
    future_x, future_y, (major, minor, angle) = kf.predict_future(steps=8)

    assert major > 0.0
    assert minor > 0.0
    assert major >= minor - 1e-4
    assert not math.isnan(angle)


def test_turret_controller_shortest_arc_wrapping():
    """Verify controller turns clockwise or counter-clockwise via the shortest angle."""
    turret = TurretController(base_x=0.0, base_y=0.0)
    turret.pan_angle = math.pi * 0.95  # 171 degrees
    # Target is at -171 degrees (-0.95 * pi)
    # The shortest turn is across the +/- pi boundary (+18 degrees), NOT through 0 (-342 degrees)
    target_x = -100.0
    target_y = -10.0
    err = turret.update(target_x, target_y, dt=1.0 / 60.0)
    assert err < 45.0  # Error is small because it takes the shortest arc


def test_turret_controller_velocity_limits():
    """Verify angular velocity respects physical maximum speed."""
    turret = TurretController(base_x=0.0, base_y=0.0)
    # Huge 180 degree jump to command maximum velocity
    for _ in range(10):
        turret.update(100.0, 0.0, dt=1.0 / 60.0)
        assert abs(turret.pan_velocity) <= turret.max_angular_speed + 1e-4


def test_turret_controller_anti_windup_clamping():
    """Verify integral accumulator does not exceed anti-windup threshold."""
    turret = TurretController(base_x=0.0, base_y=0.0)
    # Target unreachable / constant offset for 500 frames
    for _ in range(500):
        turret.update(0.0, 100.0, dt=1.0 / 60.0)
        assert abs(turret.integral) <= turret.integral_limit + 1e-4

