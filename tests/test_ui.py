"""
Unit Tests for UI Widgets, Theme, and Strip Chart.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

import pytest
import pygame

from ui.theme import FontManager
from ui.widgets import Button, Slider, PipelineBar
from ui.plot_canvas import RealTimeErrorPlot


def test_font_manager_initialization():
    """Verify font manager initializes fonts properly."""
    FontManager.initialize()
    assert FontManager.title_font is not None
    assert FontManager.heading_font is not None
    assert FontManager.body_font is not None


def test_button_interaction():
    """Verify button responds to mouse clicks."""
    clicked = False

    def on_click():
        nonlocal clicked
        clicked = True

    btn = Button(pygame.Rect(10, 10, 100, 30), "TEST", callback=on_click)

    down_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (20, 20)})
    btn.handle_event(down_event)
    assert btn.is_pressed is True

    up_event = pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": (20, 20)})
    result = btn.handle_event(up_event)
    assert result is True
    assert clicked is True


def test_slider_value_change():
    """Verify slider maps mouse position to range accurately."""
    slider = Slider(pygame.Rect(10, 10, 200, 40), "SPEED", min_val=1.0, max_val=5.0, initial_val=2.0)
    assert slider.value == 2.0

    mid_x = slider.track_rect.centerx
    mid_y = slider.track_rect.centery
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (mid_x, mid_y)})
    slider.handle_event(event)
    assert abs(slider.value - 3.0) < 0.2


def test_real_time_error_plot():
    """Verify error plot correctly appends and caps measurement buffer."""
    plot = RealTimeErrorPlot(pygame.Rect(0, 0, 500, 150), max_history=10)
    for i in range(25):
        plot.append(float(i), float(i * 0.5))

    assert len(plot.reactive_history) == 10
    assert len(plot.predictive_history) == 10
    assert plot.reactive_history[-1] == 24.0


def test_pipeline_bar_interaction():
    """Verify PipelineBar responds to clicks and smoothly advances in auto mode."""
    bar = PipelineBar(pygame.Rect(16, 62, 1248, 38))
    assert bar.active_stage == 0
    assert bar.is_auto is True

    # Test clicking Stage 1 (Predict)
    click_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (240, 75)})
    handled = bar.handle_event(click_event)
    assert handled is True
    assert bar.active_stage == 1
    assert bar.is_auto is False  # manual click locks to selected stage

    # Test clicking AUTO toggle
    auto_click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (bar.rect.right - 40, 75)})
    bar.handle_event(auto_click)
    assert bar.is_auto is True

    # Test auto advancement
    for _ in range(180):
        bar.update_auto(is_sim_running=True)
    assert bar.active_stage == 2  # smoothly advanced from 1 to 2