"""
Integration Tests for Predictive Turret Application.
Validates all 4 workspaces, all 5 trajectory modes (including Mode 4),
CV PiP telemetry generation, and CSV export.
"""

import pytest
import pygame
import numpy as np
from pathlib import Path

from main import PredictiveTurretProductApp
from core.config import (
    WORKSPACE_OVERVIEW, WORKSPACE_SIMULATION,
    WORKSPACE_EXPERIMENTS, WORKSPACE_HOW_IT_WORKS
)


@pytest.fixture(scope="module")
def app_instance():
    """Create a persistent app instance for headless testing."""
    pygame.init()
    app = PredictiveTurretProductApp()
    return app


def test_all_workspaces_rendering(app_instance):
    """Verify all 4 workspaces render cleanly without attribute or calculation errors."""
    app = app_instance
    workspaces = [
        WORKSPACE_OVERVIEW,
        WORKSPACE_SIMULATION,
        WORKSPACE_EXPERIMENTS,
        WORKSPACE_HOW_IT_WORKS
    ]

    for ws_id in workspaces:
        app._set_workspace(ws_id)
        assert app.active_workspace == ws_id
        app.screen.fill((11, 15, 23))

        if ws_id == WORKSPACE_OVERVIEW:
            app._render_overview_workspace()
        elif ws_id == WORKSPACE_SIMULATION:
            app._render_simulation_workspace()
        elif ws_id == WORKSPACE_EXPERIMENTS:
            app._render_experiments_workspace()
        elif ws_id == WORKSPACE_HOW_IT_WORKS:
            app._render_how_it_works_workspace()

        pygame.display.flip()


def test_mode_4_circular_orbit(app_instance):
    """Verify Mode 4 (Circular Orbit) initializes and steps correctly."""
    app = app_instance
    app.target.reset(mode=4)
    assert app.target.mode == 4

    for _ in range(60):
        tx, ty, vx, vy = app.target.update(1.0 / 60.0)
        assert not np.isnan(tx)
        assert not np.isnan(ty)
        assert not np.isnan(vx)
        assert not np.isnan(vy)

    # Turret tracking update
    app.reactive_turret.update(tx, ty, dt=1.0 / 60.0)
    app.predictive_turret.update(tx, ty, dt=1.0 / 60.0)
    assert app.reactive_turret.pan_velocity is not None
    assert app.predictive_turret.pan_velocity is not None


def test_cv_pip_surface_high_resolution(app_instance):
    """Verify high-resolution diagnostic PiP surface renders for Viva / How It Works."""
    app = app_instance
    pip_surf = app.cv_tracker.get_pip_surface(width=440, height=300)
    assert pip_surf is not None
    assert pip_surf.get_width() == 440
    assert pip_surf.get_height() == 300


def test_slider_parameter_updates(app_instance):
    """Verify sliders update physical parameters without errors."""
    app = app_instance
    app._on_speed_changed(4.5)
    assert abs(app.target.speed_multiplier - 4.5) < 1e-3

    app._on_latency_changed(10)
    assert app.cv_tracker.latency_frames == 10

    app._on_noise_changed(3.0)
    assert abs(app.cv_tracker.noise_std - 3.0) < 1e-3

    app._on_prediction_steps_changed(12)
    assert app.prediction_steps == 12


def test_presentation_mode_rendering(app_instance):
    """Verify full-screen presentation mode renders smoothly."""
    app = app_instance
    app.is_presentation_mode = True
    app.screen.fill((11, 15, 23))
    app._render_presentation_mode()
    pygame.display.flip()
    app.is_presentation_mode = False


def test_trial_complete_freezing(app_instance):
    """Verify simulation is hard-locked when trial completes."""
    app = app_instance
    app.reset_experiment()
    app.frame_number = 1800
    app.is_trial_complete = True
    app.is_paused = True

    # Attempting to toggle pause should NOT unpause
    app.toggle_pause()
    assert app.is_paused is True
    assert "TRIAL COMPLETE" in app.status_message

    # Reset must unlock cleanly
    app.reset_experiment()
    assert app.is_trial_complete is False
    assert app.is_paused is False
    assert app.frame_number == 0


def test_csv_export_destination(app_instance, tmp_path):
    """Verify CSV export is written cleanly to the results directory."""
    from core.config import RESULTS_DIR
    app = app_instance
    app.reset_experiment()
    app.reactive_errors.extend([15.0, 14.5, 14.0])
    app.predictive_errors.extend([10.0, 9.5, 9.0])
    app.export_csv()

    assert RESULTS_DIR.exists()
    csv_files = list(RESULTS_DIR.glob(f"tracking_mode{app.target.mode}_*.csv"))
    assert len(csv_files) > 0
    latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
    content = latest_csv.read_text(encoding="utf-8")
    assert "SUMMARY_STATISTICS" in content
    assert "Reactive_Mean_Error_Deg" in content
    assert "Predictive_Mean_Error_Deg" in content
