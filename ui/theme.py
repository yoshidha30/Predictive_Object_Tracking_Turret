"""
Theme Styling, Typography, and Drawing Primitives for Scientific Visualization.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

from typing import Tuple, Optional
import sys
from pathlib import Path
import pygame

# Allow direct execution of this script from anywhere or subfolders
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    COLOR_BG, COLOR_PANEL, COLOR_PANEL_ALT, COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_NAVY, COLOR_WHITE, COLOR_PURE_WHITE, COLOR_MUTED, COLOR_DARK_GREY,
    COLOR_REACTIVE, COLOR_PREDICTIVE, COLOR_TARGET, COLOR_SENSOR_DELAY, COLOR_ACCENT_BLUE
)


class FontManager:
    """Manages cached system typography with crisp visual hierarchy and high readability."""
    _initialized = False
    hero_font = None
    title_font = None
    heading_font = None
    subheading_font = None
    body_font = None
    small_font = None
    code_font = None

    @classmethod
    def initialize(cls) -> None:
        if cls._initialized:
            return
        if not pygame.font.get_init():
            pygame.font.init()

        primary_font = "segoeui"
        code_font_name = "consolas"

        try:
            cls.hero_font = pygame.font.SysFont(primary_font, 26, bold=True)
            cls.title_font = pygame.font.SysFont(primary_font, 16, bold=True)
            cls.heading_font = pygame.font.SysFont(primary_font, 14, bold=True)
            cls.subheading_font = pygame.font.SysFont(primary_font, 12, bold=True)
            cls.body_font = pygame.font.SysFont(primary_font, 12)
            cls.small_font = pygame.font.SysFont(primary_font, 10)
            cls.code_font = pygame.font.SysFont(code_font_name, 11)
        except Exception:
            cls.hero_font = pygame.font.SysFont(None, 28, bold=True)
            cls.title_font = pygame.font.SysFont(None, 18, bold=True)
            cls.heading_font = pygame.font.SysFont(None, 15, bold=True)
            cls.subheading_font = pygame.font.SysFont(None, 13, bold=True)
            cls.body_font = pygame.font.SysFont(None, 13)
            cls.small_font = pygame.font.SysFont(None, 11)
            cls.code_font = pygame.font.SysFont(None, 12)

        cls._initialized = True


def draw_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    bg_color: Tuple[int, int, int] = COLOR_PANEL,
    border_color: Tuple[int, int, int] = COLOR_BORDER,
    border_radius: int = 8,
    border_width: int = 1
) -> None:
    """Render a clean scientific card surface with smooth corners and subtle border."""
    pygame.draw.rect(surface, bg_color, rect, border_radius=border_radius)
    if border_width > 0:
        pygame.draw.rect(surface, border_color, rect, width=border_width, border_radius=border_radius)


def draw_text(
    surface: pygame.Surface,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = COLOR_NAVY,
    font: Optional[pygame.font.Font] = None,
    align_right: bool = False,
    align_center: bool = False
) -> pygame.Rect:
    """Render anti-aliased text with precise alignment support and dark navy contrast."""
    if font is None:
        FontManager.initialize()
        font = FontManager.body_font

    rendered = font.render(str(text), True, color)
    rect = rendered.get_rect()
    if align_right:
        rect.topright = (x, y)
    elif align_center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)

    surface.blit(rendered, rect)
    return rect


def draw_kpi_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    value_str: str,
    subtitle: str,
    accent_color: Tuple[int, int, int],
    badge_text: Optional[str] = None,
    secondary_str: Optional[str] = None
) -> None:
    """Render a prominent scientific KPI metric card in light theme with clear units."""
    FontManager.initialize()
    draw_card(surface, rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)

    # Accent top highlight strip (3px)
    highlight_rect = pygame.Rect(rect.x + 8, rect.y, rect.width - 16, 3)
    pygame.draw.rect(surface, accent_color, highlight_rect, border_radius=2)

    # Title label (subtle slate)
    draw_text(
        surface, title.upper(),
        rect.x + 14, rect.y + 11,
        color=COLOR_MUTED, font=FontManager.small_font
    )

    # Primary Hero Value (dark navy, bold, prominent)
    draw_text(
        surface, value_str,
        rect.x + 14, rect.y + 25,
        color=COLOR_NAVY, font=FontManager.hero_font
    )

    # Optional secondary telemetry line (e.g. instantaneous vs cumulative)
    if secondary_str:
        draw_text(
            surface, secondary_str,
            rect.x + 14, rect.y + 57,
            color=COLOR_NAVY, font=FontManager.subheading_font
        )
        draw_text(
            surface, subtitle,
            rect.x + 14, rect.y + 73,
            color=COLOR_MUTED, font=FontManager.small_font
        )
    else:
        # Plain-English Subtitle / Unit explanation
        draw_text(
            surface, subtitle,
            rect.x + 14, rect.y + 60,
            color=COLOR_MUTED, font=FontManager.small_font
        )

    # Optional status badge in top right
    if badge_text:
        draw_badge(surface, badge_text, rect.right - 14, rect.y + 11, accent_color, align_right=True)


def draw_pipeline_banner(
    surface: pygame.Surface,
    rect: pygame.Rect,
    active_stage: int,
    live_narrative: str
) -> None:
    """
    Renders the live explanatory pipeline narrative banner:
    DETECT -> PREDICT -> CONTROL -> EVALUATE
    """
    FontManager.initialize()
    draw_card(surface, rect, bg_color=COLOR_PANEL_ALT, border_color=COLOR_BORDER, border_radius=6)

    stages = ["1. DETECT (CV)", "2. PREDICT (KALMAN)", "3. CONTROL (PID)", "4. EVALUATE (KPI)"]
    seg_w = 170

    for idx, name in enumerate(stages):
        is_current = (idx == active_stage)
        seg_rect = pygame.Rect(rect.x + 8 + idx * seg_w, rect.y + 5, seg_w - 12, rect.height - 10)

        if is_current:
            pygame.draw.rect(surface, (235, 243, 255), seg_rect, border_radius=4)
            pygame.draw.rect(surface, COLOR_ACCENT_BLUE, seg_rect, width=1, border_radius=4)
            text_col = COLOR_ACCENT_BLUE
        else:
            text_col = COLOR_MUTED

        draw_text(
            surface, name,
            seg_rect.centerx, seg_rect.centery,
            color=text_col, font=FontManager.subheading_font, align_center=True
        )

        if idx < len(stages) - 1:
            draw_text(
                surface, "→",
                rect.x + 8 + (idx + 1) * seg_w - 10, rect.centery - 6,
                color=COLOR_DARK_GREY, font=FontManager.body_font
            )

    # Narrative explanation on the right
    narrative_x = rect.x + 8 + 4 * seg_w + 10
    draw_text(
        surface, f"STATUS:  {live_narrative}",
        narrative_x, rect.centery - 7,
        color=COLOR_NAVY, font=FontManager.body_font
    )


def draw_badge(
    surface: pygame.Surface,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    align_right: bool = False
) -> pygame.Rect:
    """Render a clean light-themed pill badge with soft pastel tint and crisp border."""
    FontManager.initialize()
    font = FontManager.small_font
    rendered = font.render(str(text), True, color)
    text_w, text_h = rendered.get_size()

    pad_h = 8
    pad_v = 3
    badge_w = text_w + 2 * pad_h
    badge_h = text_h + 2 * pad_v

    if align_right:
        rx = x - badge_w
    else:
        rx = x

    pill_rect = pygame.Rect(rx, y, badge_w, badge_h)

    # Soft pastel background tint
    bg_col = tuple(min(255, int(c * 0.12 + 255 * 0.88)) for c in color)

    pygame.draw.rect(surface, bg_col, pill_rect, border_radius=4)
    pygame.draw.rect(surface, color, pill_rect, width=1, border_radius=4)
    surface.blit(rendered, (rx + pad_h, y + pad_v))
    return pill_rect


def draw_rotated_ellipse(
    surface: pygame.Surface,
    center: Tuple[int, int],
    semi_major: float,
    semi_minor: float,
    angle_deg: float,
    color: Tuple[int, int, int],
    alpha: int = 40,
    outline_width: int = 1
) -> None:
    """Render an anti-aliased rotated 2D Kalman covariance ellipse in light mode."""
    w = max(4, int(semi_major * 2))
    h = max(4, int(semi_minor * 2))
    diag = int((w**2 + h**2)**0.5) + 4

    temp_surf = pygame.Surface((diag, diag), pygame.SRCALPHA)
    cx, cy = diag // 2, diag // 2

    e_rect = pygame.Rect(cx - int(semi_major), cy - int(semi_minor), w, h)
    fill_color = (color[0], color[1], color[2], alpha)
    pygame.draw.ellipse(temp_surf, fill_color, e_rect)

    if outline_width > 0:
        outline_color = (color[0], color[1], color[2], min(255, alpha * 3))
        pygame.draw.ellipse(temp_surf, outline_color, e_rect, width=outline_width)

    rotated = pygame.transform.rotate(temp_surf, -angle_deg)
    r_rect = rotated.get_rect(center=center)
    surface.blit(rotated, r_rect)