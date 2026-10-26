"""
Interactive UI Widgets: Pipeline Bar, Navigation Bar, Action Buttons, and Compact Sliders.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

from typing import List, Optional, Callable, Tuple
import sys
from pathlib import Path
import pygame

# Allow direct execution of this script from anywhere or subfolders
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    COLOR_PANEL, COLOR_PANEL_ALT, COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_NAVY, COLOR_WHITE, COLOR_PURE_WHITE, COLOR_MUTED, COLOR_DARK_GREY, COLOR_ACCENT_BLUE,
    COLOR_PREDICTIVE, COLOR_REACTIVE, COLOR_SENSOR_DELAY
)
from ui.theme import FontManager, draw_card, draw_text


class PipelineBar:
    """
    Interactive 4-Stage Scientific Pipeline Navigation Banner.
    Communicates: DETECT -> PREDICT -> CONTROL -> EVALUATE.
    Allows clicking individual stages to inspect them without erratic fluctuating,
    and supports an AUTO-tour mode with smooth transitions.
    """

    STAGES = [
        ("1. DETECT (CV)", "OpenCV HSV segmentation & spatial moments isolate target centroid."),
        ("2. PREDICT (KALMAN)", "Kalman 4D filter estimates velocity & extrapolates +8F ahead."),
        ("3. CONTROL (PID)", "Shortest-arc PID controller steers virtual turret with anti-windup."),
        ("4. EVALUATE (KPI)", "Quantitative benchmark comparing reactive vs predictive tracking.")
    ]

    def __init__(self, rect: pygame.Rect, on_stage_changed: Optional[Callable[[int], None]] = None):
        self.rect = rect
        self.on_stage_changed = on_stage_changed
        self.active_stage = 0
        self.is_auto = True
        self.hover_idx = -1
        self.hover_auto = False
        self.auto_timer = 0
        self.auto_interval_frames = 180  # 3.0 seconds per stage in AUTO mode

    def set_rect(self, rect: pygame.Rect) -> None:
        self.rect = rect

    def set_stage(self, stage_idx: int, manual: bool = True) -> None:
        self.active_stage = max(0, min(len(self.STAGES) - 1, stage_idx))
        if manual:
            self.is_auto = False
        if self.on_stage_changed:
            self.on_stage_changed(self.active_stage)

    def update_auto(self, is_sim_running: bool = True) -> None:
        """Smoothly advance active stage in AUTO mode without erratic frame jitter."""
        if self.is_auto and is_sim_running:
            self.auto_timer += 1
            if self.auto_timer >= self.auto_interval_frames:
                self.auto_timer = 0
                self.active_stage = (self.active_stage + 1) % len(self.STAGES)
                if self.on_stage_changed:
                    self.on_stage_changed(self.active_stage)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse hover and clicks on pipeline stages and AUTO toggle."""
        seg_w = 175
        auto_w = 68
        auto_rect = pygame.Rect(self.rect.right - auto_w - 10, self.rect.y + 5, auto_w, self.rect.height - 10)

        if event.type == pygame.MOUSEMOTION:
            self.hover_auto = auto_rect.collidepoint(event.pos)
            self.hover_idx = -1
            for idx in range(len(self.STAGES)):
                seg_rect = pygame.Rect(self.rect.x + 8 + idx * (seg_w + 14), self.rect.y + 5, seg_w, self.rect.height - 10)
                if seg_rect.collidepoint(event.pos):
                    self.hover_idx = idx
                    break
            return False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if auto_rect.collidepoint(event.pos):
                self.is_auto = not self.is_auto
                self.auto_timer = 0
                return True

            for idx in range(len(self.STAGES)):
                seg_rect = pygame.Rect(self.rect.x + 8 + idx * (seg_w + 14), self.rect.y + 5, seg_w, self.rect.height - 10)
                if seg_rect.collidepoint(event.pos):
                    self.set_stage(idx, manual=True)
                    return True

        return False

    def render(self, surface: pygame.Surface, custom_narrative: Optional[str] = None) -> None:
        """Render interactive pipeline cards with active and hover states in light theme."""
        FontManager.initialize()
        draw_card(surface, self.rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=6)

        seg_w = 175
        auto_w = 68
        auto_rect = pygame.Rect(self.rect.right - auto_w - 10, self.rect.y + 5, auto_w, self.rect.height - 10)

        for idx, (name, _) in enumerate(self.STAGES):
            is_active = (idx == self.active_stage)
            is_hover = (idx == self.hover_idx)
            seg_rect = pygame.Rect(self.rect.x + 8 + idx * (seg_w + 14), self.rect.y + 5, seg_w, self.rect.height - 10)

            if is_active:
                pygame.draw.rect(surface, (235, 243, 255), seg_rect, border_radius=5)
                pygame.draw.rect(surface, COLOR_ACCENT_BLUE, seg_rect, width=2, border_radius=5)
                text_col = COLOR_ACCENT_BLUE
            elif is_hover:
                pygame.draw.rect(surface, (241, 245, 249), seg_rect, border_radius=5)
                pygame.draw.rect(surface, COLOR_BORDER_LIGHT, seg_rect, width=1, border_radius=5)
                text_col = COLOR_NAVY
            else:
                text_col = COLOR_MUTED

            draw_text(
                surface, name,
                seg_rect.centerx, seg_rect.centery,
                color=text_col, font=FontManager.subheading_font, align_center=True
            )

            # Arrow separator
            if idx < len(self.STAGES) - 1:
                arrow_x = seg_rect.right + 7
                draw_text(
                    surface, "→",
                    arrow_x, self.rect.centery - 6,
                    color=COLOR_DARK_GREY, font=FontManager.body_font, align_center=True
                )

        # Render AUTO / MANUAL mode toggle pill
        if self.is_auto:
            auto_bg = (236, 253, 245)
            auto_border = COLOR_PREDICTIVE
            auto_text = "AUTO"
            auto_col = COLOR_PREDICTIVE
        else:
            auto_bg = (254, 243, 199)
            auto_border = COLOR_SENSOR_DELAY
            auto_text = "MANUAL"
            auto_col = COLOR_SENSOR_DELAY

        if self.hover_auto:
            auto_bg = tuple(min(255, c + 10) for c in auto_bg)

        pygame.draw.rect(surface, auto_bg, auto_rect, border_radius=4)
        pygame.draw.rect(surface, auto_border, auto_rect, width=1, border_radius=4)
        draw_text(
            surface, auto_text,
            auto_rect.centerx, auto_rect.centery,
            color=auto_col, font=FontManager.small_font, align_center=True
        )

        # Narrative explanation in remaining space
        narrative_x = self.rect.x + 8 + 4 * (seg_w + 14) + 6
        narrative = custom_narrative if custom_narrative else self.STAGES[self.active_stage][1]
        draw_text(
            surface, f"STATUS:  {narrative}",
            narrative_x, self.rect.centery - 7,
            color=COLOR_NAVY, font=FontManager.body_font
        )


class NavTabBar:
    """Segmented 4-workspace navigation tab bar with clean light aesthetic."""

    def __init__(self, rect: pygame.Rect, tabs: List[str], active_idx: int = 0, on_change: Optional[Callable[[int], None]] = None):
        self.rect = rect
        self.tabs = tabs
        self.active_idx = active_idx
        self.on_change = on_change
        self.hover_idx = -1

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse hover and clicks on tab buttons."""
        if not self.tabs:
            return False

        tab_w = self.rect.width // len(self.tabs)

        if event.type == pygame.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                rel_x = event.pos[0] - self.rect.x
                self.hover_idx = min(len(self.tabs) - 1, max(0, rel_x // tab_w))
            else:
                self.hover_idx = -1

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                rel_x = event.pos[0] - self.rect.x
                clicked_idx = min(len(self.tabs) - 1, max(0, rel_x // tab_w))
                if clicked_idx != self.active_idx:
                    self.active_idx = clicked_idx
                    if self.on_change:
                        self.on_change(self.active_idx)
                    return True
        return False

    def render(self, surface: pygame.Surface) -> None:
        """Render segmented workspace pills with clean active highlight in light mode."""
        FontManager.initialize()
        draw_card(surface, self.rect, bg_color=COLOR_PANEL_ALT, border_color=COLOR_BORDER, border_radius=6)

        tab_w = self.rect.width // len(self.tabs)
        for i, name in enumerate(self.tabs):
            tab_rect = pygame.Rect(self.rect.x + i * tab_w + 2, self.rect.y + 2, tab_w - 4, self.rect.height - 4)
            is_active = (i == self.active_idx)
            is_hover = (i == self.hover_idx)

            if is_active:
                pygame.draw.rect(surface, COLOR_PANEL, tab_rect, border_radius=5)
                pygame.draw.rect(surface, COLOR_BORDER_LIGHT, tab_rect, width=1, border_radius=5)
                text_color = COLOR_ACCENT_BLUE
            elif is_hover:
                pygame.draw.rect(surface, (241, 245, 249), tab_rect, border_radius=5)
                text_color = COLOR_NAVY
            else:
                text_color = COLOR_MUTED

            draw_text(
                surface, name,
                tab_rect.centerx, tab_rect.centery,
                color=text_color,
                font=FontManager.subheading_font,
                align_center=True
            )


class ActionButton:
    """Refined flat action button with shortcut hint and hover animation."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        hotkey_tag: str = "",
        is_primary: bool = False,
        accent_color: Tuple[int, int, int] = COLOR_PREDICTIVE,
        callback: Optional[Callable[[], None]] = None
    ):
        self.rect = rect
        self.label = label
        self.hotkey_tag = hotkey_tag
        self.is_primary = is_primary
        self.accent_color = accent_color
        self.callback = callback
        self.is_hovered = False
        self.is_pressed = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse hover, down press, and release click."""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            return False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_pressed = True
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.is_pressed
            self.is_pressed = False
            if was_pressed and self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
                return True

        return False

    def render(self, surface: pygame.Surface) -> None:
        """Render modern flat action button with hover and active states in light theme."""
        FontManager.initialize()

        if self.is_primary:
            if self.is_pressed:
                bg_col = (29, 78, 216)
            elif self.is_hovered:
                bg_col = (59, 130, 246)
            else:
                bg_col = self.accent_color
            border_col = bg_col
            text_col = COLOR_PURE_WHITE
        else:
            if self.is_pressed:
                bg_col = (226, 232, 240)
            elif self.is_hovered:
                bg_col = (241, 245, 249)
            else:
                bg_col = COLOR_PANEL
            border_col = COLOR_BORDER_LIGHT if (self.is_hovered or self.is_pressed) else COLOR_BORDER
            text_col = COLOR_NAVY

        draw_card(surface, self.rect, bg_color=bg_col, border_color=border_col, border_radius=6)

        # Label and keyboard shortcut badge
        if self.hotkey_tag:
            full_txt = f"{self.label}  {self.hotkey_tag}"
        else:
            full_txt = self.label

        draw_text(
            surface, full_txt,
            self.rect.centerx, self.rect.centery,
            color=text_col,
            font=FontManager.subheading_font,
            align_center=True
        )


class CompactSlider:
    """Refined parameter slider with label, track, and numerical readout in light theme."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        min_val: float,
        max_val: float,
        initial_val: float,
        step: float = 1.0,
        unit: str = "",
        callback: Optional[Callable[[float], None]] = None
    ):
        self.rect = rect
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.step = step
        self.unit = unit
        self.callback = callback

        self.is_dragging = False
        self.track_height = 4
        self.track_margin_x = 12
        self.track_y_offset = 28

        self._update_track_rect()

    def set_rect(self, rect: pygame.Rect) -> None:
        """Dynamically update slider bounding rect."""
        self.rect = rect
        self._update_track_rect()

    def _update_track_rect(self) -> None:
        """Calculate slider track coordinate area."""
        self.track_rect = pygame.Rect(
            self.rect.x + self.track_margin_x,
            self.rect.y + self.track_y_offset,
            self.rect.width - 2 * self.track_margin_x,
            self.track_height
        )

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle drag events."""
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hitbox = pygame.Rect(self.track_rect.x - 8, self.track_rect.y - 10, self.track_rect.width + 16, 24)
            if hitbox.collidepoint(event.pos):
                self.is_dragging = True
                changed = self._update_from_pos(event.pos[0])

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_dragging = False

        elif event.type == pygame.MOUSEMOTION and self.is_dragging:
            changed = self._update_from_pos(event.pos[0])

        return changed

    def _update_from_pos(self, mouse_x: int) -> bool:
        """Map mouse coordinate to parameter value."""
        clamped_x = max(self.track_rect.left, min(self.track_rect.right, mouse_x))
        fraction = (clamped_x - self.track_rect.left) / self.track_rect.width
        raw_val = self.min_val + fraction * (self.max_val - self.min_val)

        if self.step > 0:
            stepped = round(raw_val / self.step) * self.step
        else:
            stepped = raw_val

        stepped = max(self.min_val, min(self.max_val, stepped))
        if abs(stepped - self.value) > 1e-4:
            self.value = stepped
            if self.callback:
                self.callback(self.value)
            return True
        return False

    def render(self, surface: pygame.Surface) -> None:
        """Render slider with clean light theme styling."""
        FontManager.initialize()
        draw_card(surface, self.rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=6)

        draw_text(
            surface, self.label.upper(),
            self.rect.x + 12, self.rect.y + 7,
            color=COLOR_MUTED, font=FontManager.small_font
        )

        if self.step >= 1.0:
            val_str = f"{int(self.value)}{self.unit}"
        else:
            val_str = f"{self.value:.1f}{self.unit}"

        draw_text(
            surface, val_str,
            self.rect.right - 12, self.rect.y + 7,
            color=COLOR_NAVY, font=FontManager.subheading_font,
            align_right=True
        )

        # Light track background
        pygame.draw.rect(surface, (226, 232, 240), self.track_rect, border_radius=2)

        fraction = (self.value - self.min_val) / max(1e-4, self.max_val - self.min_val)
        fill_w = int(fraction * self.track_rect.width)
        fill_rect = pygame.Rect(self.track_rect.x, self.track_rect.y, fill_w, self.track_height)
        pygame.draw.rect(surface, COLOR_ACCENT_BLUE, fill_rect, border_radius=2)

        thumb_x = self.track_rect.left + fill_w
        thumb_y = self.track_rect.centery
        thumb_col = (255, 255, 255)
        pygame.draw.circle(surface, thumb_col, (thumb_x, thumb_y), 6)
        pygame.draw.circle(surface, COLOR_ACCENT_BLUE, (thumb_x, thumb_y), 6, width=2)


# Compatibility Aliases for test suites and scripts
Button = ActionButton
Slider = CompactSlider