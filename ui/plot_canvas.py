"""
High-Resolution Scientific Error Oscilloscope and Historical Analysis Chart.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

from typing import List
import sys
from pathlib import Path
import numpy as np
import pygame

# Allow direct execution of this script from anywhere or subfolders
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    COLOR_PANEL, COLOR_PANEL_ALT, COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_NAVY, COLOR_WHITE, COLOR_PURE_WHITE, COLOR_MUTED, COLOR_DARK_GREY,
    COLOR_REACTIVE, COLOR_PREDICTIVE
)
from ui.theme import (
    FontManager, draw_card, draw_text, draw_badge
)


class RealTimeErrorPlot:
    """
    Renders a clean, uncluttered dual-trace tracking error oscilloscope
    in light theme with shaded lock tolerances, statistical callouts, and subtle gridlines.
    """

    def __init__(self, rect: pygame.Rect, max_history: int = 400, title: str = "ANGULAR TRACKING ERROR TELEMETRY"):
        self.rect = rect
        self.max_history = max_history
        self.title = title

        self.margin_left = 46
        self.margin_right = 16
        self.margin_top = 34
        self.margin_bottom = 24

        self._update_graph_rect()

        self.y_max = 40.0
        self.reactive_history: List[float] = []
        self.predictive_history: List[float] = []

    def set_rect(self, rect: pygame.Rect) -> None:
        """Update chart dimensions dynamically."""
        self.rect = rect
        self._update_graph_rect()

    def _update_graph_rect(self) -> None:
        """Recalculate internal canvas dimensions."""
        self.graph_rect = pygame.Rect(
            self.rect.x + self.margin_left,
            self.rect.y + self.margin_top,
            max(20, self.rect.width - (self.margin_left + self.margin_right)),
            max(20, self.rect.height - (self.margin_top + self.margin_bottom))
        )

    def reset(self) -> None:
        """Clear measurement buffers."""
        self.reactive_history.clear()
        self.predictive_history.clear()

    def append(self, reactive_err: float, predictive_err: float) -> None:
        """Append newest frame samples."""
        self.reactive_history.append(float(reactive_err))
        self.predictive_history.append(float(predictive_err))

        if len(self.reactive_history) > self.max_history:
            self.reactive_history.pop(0)
            self.predictive_history.pop(0)

    def render(self, surface: pygame.Surface) -> None:
        """Render the scientific oscilloscope chart in light theme."""
        FontManager.initialize()

        draw_card(surface, self.rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)

        # Title
        draw_text(
            surface, self.title.upper(),
            self.rect.x + 14, self.rect.y + 10,
            color=COLOR_NAVY, font=FontManager.heading_font
        )

        curr_r = self.reactive_history[-1] if self.reactive_history else 0.0
        curr_p = self.predictive_history[-1] if self.predictive_history else 0.0

        draw_badge(surface, f"REACTIVE: {curr_r:.2f}°", self.rect.right - 260, self.rect.y + 8, COLOR_REACTIVE)
        draw_badge(surface, f"PREDICTIVE: {curr_p:.2f}°", self.rect.right - 130, self.rect.y + 8, COLOR_PREDICTIVE)

        # Inner graph area
        pygame.draw.rect(surface, (250, 252, 255), self.graph_rect, border_radius=4)
        pygame.draw.rect(surface, COLOR_BORDER, self.graph_rect, width=1, border_radius=4)

        # Shaded Acceptable Lock Band (0 to 3.0 degrees)
        lock_px = int((3.0 / self.y_max) * self.graph_rect.height)
        if lock_px > 0:
            lock_rect = pygame.Rect(
                self.graph_rect.left,
                self.graph_rect.bottom - lock_px,
                self.graph_rect.width,
                lock_px
            )
            pygame.draw.rect(surface, (236, 253, 245), lock_rect)
            pygame.draw.line(
                surface, (167, 243, 208),
                (self.graph_rect.left, self.graph_rect.bottom - lock_px),
                (self.graph_rect.right, self.graph_rect.bottom - lock_px),
                1
            )
            draw_text(
                surface, "LOCK TOLERANCE (≤3°)",
                self.graph_rect.right - 130, self.graph_rect.bottom - lock_px - 14,
                color=(5, 150, 105), font=FontManager.small_font
            )

        # Horizontal Gridlines and Y-Axis Ticks
        y_ticks = [0.0, 10.0, 20.0, 30.0, 40.0]
        for y_val in y_ticks:
            fraction = y_val / self.y_max
            py = self.graph_rect.bottom - int(fraction * self.graph_rect.height)
            pygame.draw.line(surface, (235, 241, 248), (self.graph_rect.left, py), (self.graph_rect.right, py), 1)
            draw_text(
                surface, f"{int(y_val)}°",
                self.graph_rect.left - 6, py - 6,
                color=COLOR_MUTED, font=FontManager.small_font, align_right=True
            )

        # Plot Traces
        n_samples = len(self.reactive_history)
        if n_samples >= 2:
            dx = self.graph_rect.width / max(1, self.max_history - 1)
            start_offset = (self.max_history - n_samples) * dx

            r_points = []
            p_points = []

            for i in range(n_samples):
                px = self.graph_rect.left + int(start_offset + i * dx)

                r_val = min(self.y_max, max(0.0, self.reactive_history[i]))
                p_val = min(self.y_max, max(0.0, self.predictive_history[i]))

                ry = self.graph_rect.bottom - int((r_val / self.y_max) * self.graph_rect.height)
                py = self.graph_rect.bottom - int((p_val / self.y_max) * self.graph_rect.height)

                r_points.append((px, ry))
                p_points.append((px, py))

            # Draw reactive error trace (rose)
            pygame.draw.lines(surface, COLOR_REACTIVE, False, r_points, 2)
            # Draw predictive error trace (emerald)
            pygame.draw.lines(surface, COLOR_PREDICTIVE, False, p_points, 2)