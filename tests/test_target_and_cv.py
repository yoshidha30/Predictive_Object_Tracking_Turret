"""
Unit Tests for Target Engine and Computer Vision Subsystem.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

import pytest
import math
from core.config import (
    WORLD_LEFT, WORLD_RIGHT, WORLD_TOP, WORLD_BOTTOM, BALL_RADIUS
)
from core.target_engine import TargetEngine
from core.cv_tracker import CVTracker


def test_target_engine_initialization():
    """Verify target engine initializes at expected coordinates and bounds."""
    engine = TargetEngine(mode=1)
    x, y, vx, vy = engine.get_state()
    assert x >= WORLD_LEFT + BALL_RADIUS
    assert x <= WORLD_RIGHT - BALL_RADIUS
    assert y >= WORLD_TOP + BALL_RADIUS
    assert y <= WORLD_BOTTOM - BALL_RADIUS
    assert math.hypot(vx, vy) > 0.0


def test_target_engine_bounds_clamping():
    """Verify target remains strictly inside bounds across 300 simulation steps."""
    engine = TargetEngine(mode=1)
    engine.set_speed_multiplier(2.5)
    for _ in range(300):
        x, y, _, _ = engine.update(1.0 / 60.0)
        assert WORLD_LEFT + BALL_RADIUS <= x <= WORLD_RIGHT - BALL_RADIUS
        assert WORLD_TOP + BALL_RADIUS <= y <= WORLD_BOTTOM - BALL_RADIUS


def test_target_engine_modes_execution():
    """Verify all 5 trajectory modes update kinematics without errors."""
    engine = TargetEngine()
    for mode in [1, 2, 3, 4, 5]:
        engine.reset(mode=mode)
        for _ in range(30):
            x, y, vx, vy = engine.update(1.0 / 60.0, mouse_pos=(400.0, 300.0))
            assert not math.isnan(x)
            assert not math.isnan(y)
            assert not math.isnan(vx)
            assert not math.isnan(vy)


def test_cv_tracker_delay_exactness():
    """Verify CV tracker accurately buffers and retrieves delayed positions."""
    tracker = CVTracker(initial_x=100.0, initial_y=200.0, latency_frames=5)
    tracker.set_noise(0.0)
    tracker.set_occlusion(0.0)

    # Feed sequence of positions
    positions = [(100.0 + i * 2.0, 200.0 + i * 3.0) for i in range(20)]
    for i, (tx, ty) in enumerate(positions):
        det = tracker.process_frame(tx, ty)
        if i >= 6:
            # Expected position from 5 frames ago: positions[i - 5]
            expected_x, expected_y = positions[i - 5]
            assert abs(det.delayed_true_x - expected_x) < 1e-3
            assert abs(det.delayed_true_y - expected_y) < 1e-3
            assert det.detected is True
            assert det.meas_x is not None
            assert abs(det.meas_x - expected_x) < 2.0  # OpenCV centroid matches within 2 pixels


def test_cv_tracker_occlusion_dropout():
    """Verify setting occlusion probability to 1.0 reliably triggers dropouts."""
    tracker = CVTracker(initial_x=150.0, initial_y=250.0, latency_frames=4)
    tracker.set_occlusion(1.0)
    det = tracker.process_frame(160.0, 260.0)
    assert det.detected is False
    assert det.meas_x is None
    assert det.is_occluded is True


def test_cv_tracker_pip_surface_generation():
    """Verify CV diagnostic PiP surface is generated with valid dimensions."""
    tracker = CVTracker(initial_x=200.0, initial_y=300.0)
    det = tracker.process_frame(210.0, 310.0)
    assert det.pip_surface is not None
    assert det.pip_surface.get_width() == 170
    assert det.pip_surface.get_height() == 115
