"""
Target Kinematics Engine.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

import math
from typing import List, Tuple, Optional
import sys
from pathlib import Path

# Allow direct execution of this script from anywhere or subfolders
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    WORLD_LEFT, WORLD_RIGHT, WORLD_TOP, WORLD_BOTTOM,
    BALL_RADIUS, INITIAL_X, INITIAL_Y, INITIAL_VX, INITIAL_VY
)


class TargetEngine:
    """
    Simulates ground-truth motion for the tracked target across
    5 distinct kinematic trajectory patterns.
    """

    def __init__(self, mode: int = 1):
        self.mode = mode
        self.radius = BALL_RADIUS
        self.x = float(INITIAL_X)
        self.y = float(INITIAL_Y)
        self.vx = float(INITIAL_VX)
        self.vy = float(INITIAL_VY)
        self.base_speed = math.hypot(INITIAL_VX, INITIAL_VY)
        self.speed_multiplier = 1.0

        # Harmonic & circular trajectory parameters
        self.time_elapsed = 0.0
        self.center_x = (WORLD_LEFT + WORLD_RIGHT) / 2.0
        self.center_y = (WORLD_TOP + WORLD_BOTTOM) / 2.0

        # Trajectory trail for visual rendering
        self.trail: List[Tuple[float, float]] = [(self.x, self.y)]
        self.max_trail_length = 120
        self.frame_count = 0

    def reset(self, mode: Optional[int] = None) -> None:
        """Reset target kinematics to the starting state."""
        if mode is not None:
            self.mode = mode
        self.x = float(INITIAL_X)
        self.y = float(INITIAL_Y)
        self.vx = float(INITIAL_VX)
        self.vy = float(INITIAL_VY)
        self.time_elapsed = 0.0
        self.frame_count = 0
        self.trail.clear()
        self.trail.append((self.x, self.y))

    def set_speed_multiplier(self, multiplier: float) -> None:
        """Dynamically adjust target velocity scale."""
        self.speed_multiplier = max(0.2, min(8.0, float(multiplier)))

    def set_speed(self, speed: float) -> None:
        """Alias for set_speed_multiplier for backwards compatibility."""
        self.set_speed_multiplier(speed)

    def update(self, dt: float, mouse_pos: Optional[Tuple[float, float]] = None) -> Tuple[float, float, float, float]:
        """
        Advance target kinematics by time step dt.
        Returns: (x, y, vx, vy)
        """
        self.frame_count += 1
        self.time_elapsed += dt
        eff_speed = self.speed_multiplier

        if self.mode == 1:
            # Mode 1: Linear Bounce (Constant Velocity + Boundary Reflection)
            self._update_linear_bounce(dt, eff_speed)

        elif self.mode == 2:
            # Mode 2: Evasive Zig-Zag (90-degree orthogonal turn every 180 frames)
            if self.frame_count > 0 and self.frame_count % 180 == 0:
                self.vx, self.vy = -self.vy, self.vx
            self._update_linear_bounce(dt, eff_speed)

        elif self.mode == 3:
            # Mode 3: Harmonic Wave (Sinusoidal Lissajous motion)
            freq_x = 0.9 * eff_speed
            freq_y = 1.4 * eff_speed
            span_x = (WORLD_RIGHT - WORLD_LEFT - 4 * self.radius) * 0.42
            span_y = (WORLD_BOTTOM - WORLD_TOP - 4 * self.radius) * 0.40

            prev_x, prev_y = self.x, self.y
            self.x = self.center_x + span_x * math.sin(freq_x * self.time_elapsed)
            self.y = self.center_y + span_y * math.cos(freq_y * self.time_elapsed)
            self.vx = (self.x - prev_x) / max(dt, 1e-4)
            self.vy = (self.y - prev_y) / max(dt, 1e-4)

        elif self.mode == 4:
            # Mode 4: Circular Orbit (Continuous angular acceleration)
            radius_orbit = min(WORLD_RIGHT - WORLD_LEFT, WORLD_BOTTOM - WORLD_TOP) * 0.32
            omega = 1.3 * eff_speed
            prev_x, prev_y = self.x, self.y
            self.x = self.center_x + radius_orbit * math.cos(omega * self.time_elapsed)
            self.y = self.center_y + radius_orbit * math.sin(omega * self.time_elapsed)
            self.vx = (self.x - prev_x) / max(dt, 1e-4)
            self.vy = (self.y - prev_y) / max(dt, 1e-4)

        elif self.mode == 5:
            # Mode 5: Interactive Cursor / Manual Guidance
            if mouse_pos is not None:
                target_mx, target_my = mouse_pos
                prev_x, prev_y = self.x, self.y
                alpha = min(1.0, 7.0 * dt * eff_speed)
                self.x += alpha * (target_mx - self.x)
                self.y += alpha * (target_my - self.y)
                self.vx = (self.x - prev_x) / max(dt, 1e-4)
                self.vy = (self.y - prev_y) / max(dt, 1e-4)
            else:
                self._update_linear_bounce(dt, eff_speed)

        # Enforce boundary limits
        self._enforce_bounds()

        # Update historical trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

        return self.x, self.y, self.vx, self.vy

    def _update_linear_bounce(self, dt: float, eff_speed: float) -> None:
        """Update position with boundary reflections."""
        scale = eff_speed * (60.0 * dt)
        self.x += self.vx * scale
        self.y += self.vy * scale

        # Boundary checks with elastic reflection
        if self.x + self.radius >= WORLD_RIGHT:
            self.x = WORLD_RIGHT - self.radius
            self.vx = -abs(self.vx)
        elif self.x - self.radius <= WORLD_LEFT:
            self.x = WORLD_LEFT + self.radius
            self.vx = abs(self.vx)

        if self.y + self.radius >= WORLD_BOTTOM:
            self.y = WORLD_BOTTOM - self.radius
            self.vy = -abs(self.vy)
        elif self.y - self.radius <= WORLD_TOP:
            self.y = WORLD_TOP + self.radius
            self.vy = abs(self.vy)

    def _enforce_bounds(self) -> None:
        """Clamp position strictly inside arena."""
        self.x = max(WORLD_LEFT + self.radius, min(WORLD_RIGHT - self.radius, self.x))
        self.y = max(WORLD_TOP + self.radius, min(WORLD_BOTTOM - self.radius, self.y))

    def get_state(self) -> Tuple[float, float, float, float]:
        """Return current state tuple (x, y, vx, vy)."""
        return self.x, self.y, self.vx, self.vy

    def get_trail(self) -> List[Tuple[float, float]]:
        """Return historical position coordinates."""
        return list(self.trail)


if __name__ == "__main__":
    print("=" * 65)
    print(" TargetEngine Kinematics Simulator (Direct Module Test)")
    print("=" * 65)
    engine = TargetEngine(mode=1)
    print(f"Mode 1 Initial: x={engine.x:.1f}, y={engine.y:.1f}, vx={engine.vx:.1f}, vy={engine.vy:.1f}")
    for f in range(1, 11):
        x, y, vx, vy = engine.update(1.0 / 60.0)
        print(f"  Frame {f:02d}: x={x:.2f}, y={y:.2f}, vx={vx:.2f}, vy={vy:.2f}")

    print("\n[OK] TargetEngine is functioning properly.")
    print("To launch the full interactive graphical simulation application, run:")
    print("  python main.py\n")
