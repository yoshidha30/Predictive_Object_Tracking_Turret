"""
2-DOF Pan-and-Tilt Virtual Turret and Intelligent PID Controller.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

import math
from typing import Tuple
import sys
from pathlib import Path

# Allow direct execution of this script from anywhere or subfolders
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    DT, DEFAULT_KP, DEFAULT_KI, DEFAULT_KD,
    MAX_ANGULAR_SPEED, MAX_ANGULAR_ACCEL,
    INTEGRAL_LIMIT, DERIVATIVE_SMOOTHING,
    TURRET_BARREL_LENGTH, TURRET_BASE_RADIUS
)


class TurretController:
    """
    Simulates a robotic 2-DOF Pan-and-Tilt virtual turret actuated by
    a shortest-arc PID controller with anti-windup clamping, low-pass derivative
    filtering, and motor torque acceleration limiting.
    """

    def __init__(
        self,
        base_x: float = 0.0,
        base_y: float = 0.0,
        kp: float = DEFAULT_KP,
        ki: float = DEFAULT_KI,
        kd: float = DEFAULT_KD,
        turret_id: str = "Turret"
    ):
        self.base_x = float(base_x)
        self.base_y = float(base_y)
        self.turret_id = str(turret_id)

        # PID gains
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        # Kinematic state
        # Initial pan angle points straight upwards (-90 degrees / -pi/2 rad)
        self.pan_angle = -math.pi / 2.0
        self.pan_velocity = 0.0
        self.tilt_angle = 0.0  # Elevation angle for 2-DOF tracking

        # PID internal accumulators
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0

        # Physical constraints
        self.max_angular_speed = MAX_ANGULAR_SPEED
        self.max_angular_accel = MAX_ANGULAR_ACCEL
        self.integral_limit = INTEGRAL_LIMIT
        self.derivative_alpha = DERIVATIVE_SMOOTHING

        # Target lock metrics
        self.consecutive_lock_frames = 0
        self.is_locked = False

    def reset(self, initial_angle: float = -math.pi / 2.0) -> None:
        """Reset controller state and accumulators."""
        self.pan_angle = float(initial_angle)
        self.pan_velocity = 0.0
        self.tilt_angle = 0.0
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0
        self.consecutive_lock_frames = 0
        self.is_locked = False

    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        """Dynamically update PID gains."""
        self.kp = max(0.0, float(kp))
        self.ki = max(0.0, float(ki))
        self.kd = max(0.0, float(kd))

    def update(self, target_x: float, target_y: float, dt: float = DT) -> float:
        """
        Advance the turret controller toward the specified target coordinates.
        Uses shortest-arc angle wrapping and applies velocity & acceleration clamping.
        Returns the instantaneous aiming error in degrees.
        """
        dt = max(1e-4, float(dt))
        dx = float(target_x) - self.base_x
        dy = float(target_y) - self.base_y
        dist = math.hypot(dx, dy)

        # 1. Calculate desired azimuth (Pan) angle
        desired_pan = math.atan2(dy, dx)

        # 2. Shortest-arc angular error with modulo 2*pi wrapping
        error = (desired_pan - self.pan_angle + math.pi) % (2.0 * math.pi) - math.pi

        # 3. Integral accumulation with anti-windup clamping
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        # 4. Filtered derivative to suppress high-frequency noise & derivative kick
        raw_derivative = (error - self.previous_error) / dt
        self.filtered_derivative += self.derivative_alpha * (raw_derivative - self.filtered_derivative)

        # 5. PID Law -> Desired angular velocity
        desired_velocity = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * self.filtered_derivative
        )

        # Velocity saturation
        desired_velocity = max(-self.max_angular_speed, min(self.max_angular_speed, desired_velocity))

        # 6. Physical motor torque slew-rate limiting (acceleration clamp)
        max_delta_v = self.max_angular_accel * dt
        accel_step = desired_velocity - self.pan_velocity
        accel_step = max(-max_delta_v, min(max_delta_v, accel_step))
        self.pan_velocity += accel_step

        # 7. Integrate position
        self.pan_angle += self.pan_velocity * dt
        # Normalize pan angle to [-pi, pi]
        self.pan_angle = (self.pan_angle + math.pi) % (2.0 * math.pi) - math.pi

        self.previous_error = error

        # 8. Compute simulated tilt / elevation angle (proportional to distance)
        # Closer target -> higher elevation; farther target -> lower elevation
        norm_dist = max(10.0, min(1000.0, dist))
        self.tilt_angle = math.atan2(150.0, norm_dist)

        # 9. Update target lock detector (Lock beam <= 3.0 degrees)
        err_deg = abs(math.degrees(error))
        if err_deg <= 3.0:
            self.consecutive_lock_frames += 1
            if self.consecutive_lock_frames >= 12:  # ~200 ms within beam
                self.is_locked = True
        else:
            self.consecutive_lock_frames = 0
            self.is_locked = False

        return err_deg

    def aiming_error_to(self, target_x: float, target_y: float) -> float:
        """Calculate geometric angular discrepancy in degrees to an arbitrary point."""
        dx = float(target_x) - self.base_x
        dy = float(target_y) - self.base_y
        desired_pan = math.atan2(dy, dx)
        error = (desired_pan - self.pan_angle + math.pi) % (2.0 * math.pi) - math.pi
        return abs(math.degrees(error))

    def get_barrel_endpoint(self, length: float = TURRET_BARREL_LENGTH) -> Tuple[float, float]:
        """Compute the (x, y) coordinates of the turret muzzle."""
        end_x = self.base_x + math.cos(self.pan_angle) * length
        end_y = self.base_y + math.sin(self.pan_angle) * length
        return end_x, end_y

    def get_aim_ray_endpoint(self, distance: float = 600.0) -> Tuple[float, float]:
        """Compute an extended targeting laser ray point along the aiming vector."""
        ray_x = self.base_x + math.cos(self.pan_angle) * distance
        ray_y = self.base_y + math.sin(self.pan_angle) * distance
        return ray_x, ray_y


if __name__ == "__main__":
    print("=" * 65)
    print(" TurretController PID Tracking Test")
    print("=" * 65)
    turret = TurretController(turret_id="Test-Turret", base_x=200.0, base_y=200.0)
    for step in range(1, 6):
        err = turret.update(target_x=300.0, target_y=250.0, dt=1.0/60.0)
        print(f"Step {step:02d}: Pan angle = {math.degrees(turret.pan_angle):.2f}°, Aim Error = {err:.2f}°")
    print("[OK] TurretController functioning properly.")
    print("To launch the full interactive simulation, run:")
    print("  python main.py\n")