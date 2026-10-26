"""
Simulation Arena and Telemetry Visualization Renderer.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

import math
from typing import Tuple, List, Optional
import sys
from pathlib import Path
import pygame

# Allow direct execution of this script from anywhere or subfolders
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    WORLD_LEFT, WORLD_RIGHT, WORLD_TOP, WORLD_BOTTOM,
    TURRET_BARREL_LENGTH,
    COLOR_PANEL, COLOR_PANEL_ALT, COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_NAVY, COLOR_WHITE, COLOR_PURE_WHITE, COLOR_MUTED, COLOR_DARK_GREY,
    COLOR_REACTIVE, COLOR_PREDICTIVE, COLOR_TARGET, COLOR_SENSOR_DELAY,
    COLOR_ARENA_BG, COLOR_ARENA_GRID, COLOR_ACCENT_BLUE
)
from ui.theme import (
    FontManager, draw_card, draw_text, draw_badge, draw_rotated_ellipse
)


class ArenaRenderer:
    """
    Renders synchronized comparative simulation arenas with
    clean light styling, laser targeting vectors, and Kalman covariance ellipses.
    Strictly clips simulation graphics to the arena field to maintain zero visual spillover.
    """

    def __init__(self, left_rect: pygame.Rect, right_rect: pygame.Rect):
        self.left_rect = left_rect
        self.right_rect = right_rect

    def set_rects(self, left_rect: pygame.Rect, right_rect: pygame.Rect) -> None:
        """Update arena boundaries dynamically for different workspace layouts."""
        self.left_rect = left_rect
        self.right_rect = right_rect

    def world_to_screen(self, world_x: float, world_y: float, panel_rect: pygame.Rect) -> Tuple[int, int]:
        """Map normalized arena world coordinates into panel screen space."""
        margin_x = 16
        margin_top = 52
        margin_bottom = 16

        usable_w = panel_rect.width - 2 * margin_x
        usable_h = panel_rect.height - (margin_top + margin_bottom)

        u = (world_x - WORLD_LEFT) / max(1e-4, WORLD_RIGHT - WORLD_LEFT)
        v = (world_y - WORLD_TOP) / max(1e-4, WORLD_BOTTOM - WORLD_TOP)

        sx = panel_rect.x + margin_x + int(u * usable_w)
        sy = panel_rect.y + margin_top + int(v * usable_h)
        return sx, sy

    def screen_to_world(self, screen_x: float, screen_y: float, panel_rect: pygame.Rect) -> Tuple[float, float]:
        """Map panel screen space coordinates back into normalized arena world coordinates."""
        margin_x = 16
        margin_top = 52
        margin_bottom = 16

        usable_w = panel_rect.width - 2 * margin_x
        usable_h = panel_rect.height - (margin_top + margin_bottom)

        u = (screen_x - (panel_rect.x + margin_x)) / max(1e-4, usable_w)
        v = (screen_y - (panel_rect.y + margin_top)) / max(1e-4, usable_h)

        world_x = WORLD_LEFT + u * (WORLD_RIGHT - WORLD_LEFT)
        world_y = WORLD_TOP + v * (WORLD_BOTTOM - WORLD_TOP)
        return world_x, world_y

    def get_turret_base_pos(self, panel_rect: pygame.Rect) -> Tuple[int, int]:
        """Compute the turret base position in screen space for a given panel rectangle."""
        field_rect = pygame.Rect(
            panel_rect.x + 10, panel_rect.y + 48,
            panel_rect.width - 20, panel_rect.height - 58
        )
        return field_rect.centerx, field_rect.bottom - 18

    def render_reactive_arena(
        self,
        surface: pygame.Surface,
        turret,
        delayed_x: float,
        delayed_y: float,
        true_x: float,
        true_y: float,
        trail: List[Tuple[float, float]],
        error_deg: float,
        latency_frames: int
    ) -> None:
        """Render the left arena (Reactive System observing delayed sensor state)."""
        FontManager.initialize()

        # Outer card container
        draw_card(surface, self.left_rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)

        # Clear primary heading
        draw_text(
            surface, "REACTIVE TRACKING",
            self.left_rect.x + 14, self.left_rect.y + 10,
            color=COLOR_NAVY, font=FontManager.heading_font
        )

        # One short sentence plain-English explanation
        draw_text(
            surface, "Tracks where target was detected; delayed camera feed causes lag and overshoot.",
            self.left_rect.x + 14, self.left_rect.y + 28,
            color=COLOR_MUTED, font=FontManager.small_font
        )

        # Latency badge
        latency_ms = latency_frames * 16.7
        draw_badge(
            surface, f"CAMERA DELAY: {latency_frames}F ({latency_ms:.0f}ms)",
            self.left_rect.right - 14, self.left_rect.y + 11,
            COLOR_SENSOR_DELAY, align_right=True
        )

        # Simulation field area
        field_rect = pygame.Rect(
            self.left_rect.x + 10, self.left_rect.y + 48,
            self.left_rect.width - 20, self.left_rect.height - 58
        )
        pygame.draw.rect(surface, COLOR_ARENA_BG, field_rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_BORDER, field_rect, width=1, border_radius=6)

        # Subtle clean background gridlines
        for gx in range(1, 5):
            x_line = field_rect.x + int(gx * field_rect.width / 5)
            pygame.draw.line(surface, COLOR_ARENA_GRID, (x_line, field_rect.top), (x_line, field_rect.bottom), 1)
        for gy in range(1, 4):
            y_line = field_rect.y + int(gy * field_rect.height / 4)
            pygame.draw.line(surface, COLOR_ARENA_GRID, (field_rect.left, y_line), (field_rect.right, y_line), 1)

        screen_delayed = self.world_to_screen(delayed_x, delayed_y, self.left_rect)
        screen_true = self.world_to_screen(true_x, true_y, self.left_rect)

        # Set clipping to prevent lasers and graphics from spilling outside arena
        surface.set_clip(field_rect)

        # Motion trail ribbon
        if len(trail) >= 2:
            pts = [self.world_to_screen(tx, ty, self.left_rect) for tx, ty in trail[-40:]]
            if len(pts) >= 2:
                pygame.draw.lines(surface, (215, 225, 235), False, pts, 2)

        # Draw latency discrepancy line between delayed observation and actual true target
        pygame.draw.line(surface, (253, 230, 138), screen_delayed, screen_true, 1)

        # Draw True Target Ghost Dot (Shows where target actually is right now)
        pygame.draw.circle(surface, (148, 163, 184), screen_true, 9, width=1)
        draw_text(surface, "Target (t)", screen_true[0] + 12, screen_true[1] - 8, color=(100, 116, 139), font=FontManager.small_font)

        # Draw Delayed Sensor Observation (What reactive camera actually detects)
        pygame.draw.circle(surface, COLOR_SENSOR_DELAY, screen_delayed, 9)
        pygame.draw.circle(surface, COLOR_PURE_WHITE, screen_delayed, 3)
        draw_text(surface, f"Camera (t-{latency_frames}F)", screen_delayed[0] + 12, screen_delayed[1] - 8, color=COLOR_SENSOR_DELAY, font=FontManager.small_font)

        # Turret base position in screen space
        base_sx = field_rect.centerx
        base_sy = field_rect.bottom - 18
        turret.base_x = base_sx
        turret.base_y = base_sy

        # Laser targeting line toward delayed observation (dual-layer for glow effect)
        aim_x = base_sx + math.cos(turret.pan_angle) * 700
        aim_y = base_sy + math.sin(turret.pan_angle) * 700
        # Wide soft beam
        pygame.draw.line(surface, (254, 205, 211), (base_sx, base_sy), (int(aim_x), int(aim_y)), 3)
        # Sharp laser core
        pygame.draw.line(surface, COLOR_REACTIVE, (base_sx, base_sy), (int(aim_x), int(aim_y)), 1)

        # Turret Body
        self._render_turret(surface, base_sx, base_sy, turret.pan_angle, COLOR_REACTIVE)

        # Target Lock Ring if within tolerance
        if turret.is_locked:
            pygame.draw.circle(surface, COLOR_REACTIVE, screen_delayed, 16, width=2)
            pygame.draw.circle(surface, (254, 205, 211), screen_delayed, 20, width=1)

        # Reset clipping
        surface.set_clip(None)

        # Real-time Aiming Error Readout Pill
        draw_badge(
            surface, f"AIM ERROR: {error_deg:.2f}°",
            self.left_rect.x + 18, field_rect.bottom - 26, COLOR_REACTIVE
        )

    def render_predictive_arena(
        self,
        surface: pygame.Surface,
        turret,
        true_x: float,
        true_y: float,
        pred_x: float,
        pred_y: float,
        ellipse_params: Tuple[float, float, float],
        trail: List[Tuple[float, float]],
        error_deg: float,
        steps_ahead: int,
        is_coasting: bool
    ) -> None:
        """Render the right arena (Predictive System with Kalman state prediction)."""
        FontManager.initialize()

        # Outer card container
        draw_card(surface, self.right_rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)

        # Clear primary heading
        draw_text(
            surface, "PREDICTIVE TRACKING (KALMAN FILTER)",
            self.right_rect.x + 14, self.right_rect.y + 10,
            color=COLOR_NAVY, font=FontManager.heading_font
        )

        # One short sentence plain-English explanation
        draw_text(
            surface, "Predicts target velocity and aims ahead to intercept the future position.",
            self.right_rect.x + 14, self.right_rect.y + 28,
            color=COLOR_MUTED, font=FontManager.small_font
        )

        # Prediction Lead Badge
        status_tag = "COASTING" if is_coasting else f"PREDICTION LEAD: +{steps_ahead}F"
        status_col = COLOR_SENSOR_DELAY if is_coasting else COLOR_PREDICTIVE
        draw_badge(
            surface, status_tag,
            self.right_rect.right - 14, self.right_rect.y + 11,
            status_col, align_right=True
        )

        # Simulation field area
        field_rect = pygame.Rect(
            self.right_rect.x + 10, self.right_rect.y + 48,
            self.right_rect.width - 20, self.right_rect.height - 58
        )
        pygame.draw.rect(surface, COLOR_ARENA_BG, field_rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_BORDER, field_rect, width=1, border_radius=6)

        # Subtle clean background gridlines
        for gx in range(1, 5):
            x_line = field_rect.x + int(gx * field_rect.width / 5)
            pygame.draw.line(surface, COLOR_ARENA_GRID, (x_line, field_rect.top), (x_line, field_rect.bottom), 1)
        for gy in range(1, 4):
            y_line = field_rect.y + int(gy * field_rect.height / 4)
            pygame.draw.line(surface, COLOR_ARENA_GRID, (field_rect.left, y_line), (field_rect.right, y_line), 1)

        screen_true = self.world_to_screen(true_x, true_y, self.right_rect)
        screen_pred = self.world_to_screen(pred_x, pred_y, self.right_rect)

        # Set clipping to prevent lasers and graphics from spilling outside arena
        surface.set_clip(field_rect)

        # Motion trail ribbon
        if len(trail) >= 2:
            pts = [self.world_to_screen(tx, ty, self.right_rect) for tx, ty in trail[-40:]]
            if len(pts) >= 2:
                pygame.draw.lines(surface, (215, 225, 235), False, pts, 2)

        # Draw Kalman 95% Uncertainty Ellipse around predicted future target
        semi_major, semi_minor, angle_deg = ellipse_params
        draw_rotated_ellipse(
            surface, screen_pred,
            semi_major=semi_major, semi_minor=semi_minor,
            angle_deg=angle_deg, color=COLOR_PREDICTIVE, alpha=45, outline_width=1
        )

        # Ground-Truth Target
        pygame.draw.circle(surface, COLOR_TARGET, screen_true, 9)
        pygame.draw.circle(surface, COLOR_PURE_WHITE, screen_true, 3)
        draw_text(surface, "Target (t)", screen_true[0] + 12, screen_true[1] - 8, color=COLOR_TARGET, font=FontManager.small_font)

        # Predictive Ghost Target (Extrapolated +H frames)
        pygame.draw.circle(surface, COLOR_PREDICTIVE, screen_pred, 9)
        pygame.draw.circle(surface, COLOR_PURE_WHITE, screen_pred, 3)
        lead_label = f"Kalman Coasting (+{steps_ahead}F)" if is_coasting else f"Kalman Aim (+{steps_ahead}F)"
        draw_text(surface, lead_label, screen_pred[0] + 12, screen_pred[1] - 8, color=COLOR_PREDICTIVE, font=FontManager.small_font)

        # Lead extrapolation dashed vector
        pygame.draw.line(surface, (16, 185, 129), screen_true, screen_pred, 2)

        # Turret base position in screen space
        base_sx = field_rect.centerx
        base_sy = field_rect.bottom - 18
        turret.base_x = base_sx
        turret.base_y = base_sy

        # Laser targeting line aimed at predicted position (dual-layer for glow effect)
        aim_x = base_sx + math.cos(turret.pan_angle) * 700
        aim_y = base_sy + math.sin(turret.pan_angle) * 700
        # Wide soft beam
        pygame.draw.line(surface, (167, 243, 208), (base_sx, base_sy), (int(aim_x), int(aim_y)), 3)
        # Sharp laser core
        pygame.draw.line(surface, COLOR_PREDICTIVE, (base_sx, base_sy), (int(aim_x), int(aim_y)), 1)

        # Turret Body
        self._render_turret(surface, base_sx, base_sy, turret.pan_angle, COLOR_PREDICTIVE)

        # Target Lock Ring if within tolerance
        if turret.is_locked:
            pygame.draw.circle(surface, COLOR_PREDICTIVE, screen_pred, 16, width=2)
            pygame.draw.circle(surface, (167, 243, 208), screen_pred, 20, width=1)

        # Reset clipping
        surface.set_clip(None)

        # Real-time Aiming Error Readout Pill
        draw_badge(
            surface, f"AIM ERROR: {error_deg:.2f}°",
            self.right_rect.x + 18, field_rect.bottom - 26, COLOR_PREDICTIVE
        )

    def _render_turret(
        self,
        surface: pygame.Surface,
        bx: int,
        by: int,
        angle: float,
        accent_color: Tuple[int, int, int]
    ) -> None:
        """Render virtual robotic turret with crisp modern styling in light theme."""
        barrel_len = 38
        barrel_w = 6

        end_x = bx + math.cos(angle) * barrel_len
        end_y = by + math.sin(angle) * barrel_len

        # Concentric metallic gimbal pedestal rings
        pygame.draw.circle(surface, (203, 213, 225), (bx, by), 16)
        pygame.draw.circle(surface, (241, 245, 249), (bx, by), 14)
        pygame.draw.circle(surface, (226, 232, 240), (bx, by), 11)
        pygame.draw.circle(surface, accent_color, (bx, by), 8)
        pygame.draw.circle(surface, COLOR_PURE_WHITE, (bx, by), 3)

        # Main barrel with outline
        pygame.draw.line(surface, COLOR_NAVY, (bx, by), (int(end_x), int(end_y)), barrel_w + 2)
        pygame.draw.line(surface, accent_color, (bx, by), (int(end_x), int(end_y)), barrel_w)

        # Muzzle tip
        pygame.draw.circle(surface, COLOR_NAVY, (int(end_x), int(end_y)), 4)
        pygame.draw.circle(surface, COLOR_PURE_WHITE, (int(end_x), int(end_y)), 1)


if __name__ == "__main__":
    print("=" * 65)
    print(" ArenaRenderer UI Component (Direct Module Test)")
    print("=" * 65)
    print("[OK] ArenaRenderer module imported and ready.")
    print("To launch the full interactive simulation, run:")
    print("  python main.py\n")