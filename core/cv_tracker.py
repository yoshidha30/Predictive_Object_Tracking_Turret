"""
Computer Vision Target Detection and Sensor Degradation Subsystem.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

from collections import deque
from dataclasses import dataclass
import random
from typing import Optional, Tuple
import sys
from pathlib import Path
import cv2
import numpy as np
import pygame

# Allow direct execution of this script from anywhere or subfolders
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    HSV_LOWER_BLUE, HSV_UPPER_BLUE, CV_MIN_CONTOUR_AREA,
    BALL_RADIUS, DEFAULT_DELAY_FRAMES, MAX_LATENCY_FRAMES,
    COLOR_BLUE, COLOR_BG, COLOR_EMERALD, COLOR_RED, COLOR_WHITE, COLOR_AMBER
)


@dataclass
class CVDetection:
    """Structured result from the computer vision observation pipeline."""
    detected: bool
    meas_x: Optional[float]
    meas_y: Optional[float]
    delayed_true_x: float
    delayed_true_y: float
    true_x: float
    true_y: float
    latency_frames: int
    noise_std: float
    is_occluded: bool
    pip_surface: pygame.Surface


class CVTracker:
    """
    Simulates a physical camera sensor with configurable observation latency,
    measurement noise, and occlusions, coupled with an OpenCV HSV centroid detector.
    """

    def __init__(self, initial_x: float, initial_y: float, latency_frames: int = DEFAULT_DELAY_FRAMES):
        self.latency_frames = int(latency_frames)
        self.noise_std = 0.0          # Pixel standard deviation for Gaussian noise
        self.occlusion_prob = 0.0     # Probability of missed detection / occlusion [0.0, 1.0]

        # Delay buffer for past ground truth positions
        self.history = deque(
            [(float(initial_x), float(initial_y))] * (MAX_LATENCY_FRAMES + 2),
            maxlen=MAX_LATENCY_FRAMES + 5
        )

        # Offscreen camera crop surface for OpenCV processing
        self.crop_size = 140
        self.camera_surface = pygame.Surface((self.crop_size, self.crop_size))

        # Diagnostic Picture-in-Picture (PiP) viewport surface (170x115)
        self.pip_width = 170
        self.pip_height = 115
        self.pip_surface = pygame.Surface((self.pip_width, self.pip_height))

        # Cache font for PiP overlay
        if not pygame.font.get_init():
            pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 11)

        # Telemetry cache for external views
        self.last_mask: Optional[np.ndarray] = None
        self.last_detected: bool = False
        self.last_occluded: bool = False

    def reset(self, initial_x: float, initial_y: float) -> None:
        """Reset the internal latency history buffer."""
        self.history.clear()
        for _ in range(MAX_LATENCY_FRAMES + 2):
            self.history.append((float(initial_x), float(initial_y)))
        self.last_mask = None
        self.last_detected = False
        self.last_occluded = False

    def set_latency(self, frames: int) -> None:
        """Set observation delay in frames."""
        self.latency_frames = max(0, min(MAX_LATENCY_FRAMES, int(frames)))

    def set_noise(self, std_dev: float) -> None:
        """Set Gaussian measurement noise standard deviation."""
        self.noise_std = max(0.0, min(15.0, float(std_dev)))

    def set_occlusion(self, prob: float) -> None:
        """Set probability of occlusion / missed detection [0.0, 1.0]."""
        self.occlusion_prob = max(0.0, min(1.0, float(prob)))

    def process_frame(self, true_x: float, true_y: float) -> CVDetection:
        """
        Process the current frame:
        1. Append true position to latency queue.
        2. Retrieve position from L frames in the past.
        3. Check for occlusion / missed detection.
        4. Add synthetic Gaussian sensor noise.
        5. Run genuine OpenCV HSV thresholding & spatial moments.
        6. Generate the diagnostic PiP telemetry viewport.
        """
        self.history.append((float(true_x), float(true_y)))

        # Retrieve delayed ground-truth position
        delay_index = -1 - self.latency_frames
        try:
            delayed_x, delayed_y = self.history[delay_index]
        except IndexError:
            delayed_x, delayed_y = self.history[0]

        # Simulate occlusion / camera dropout
        is_occluded = (random.random() < self.occlusion_prob)

        # Add Gaussian measurement noise if not occluded
        noisy_x = delayed_x
        noisy_y = delayed_y
        if self.noise_std > 0.01 and not is_occluded:
            noisy_x += random.gauss(0.0, self.noise_std)
            noisy_y += random.gauss(0.0, self.noise_std)

        # Run OpenCV Detection Pipeline
        detected = False
        meas_x: Optional[float] = None
        meas_y: Optional[float] = None
        mask_pip = None

        if not is_occluded:
            # Render target into offscreen camera buffer
            self.camera_surface.fill(COLOR_BG)
            center = self.crop_size // 2
            pygame.draw.circle(
                self.camera_surface, COLOR_BLUE,
                (center, center), int(BALL_RADIUS)
            )

            # Convert Pygame surface to NumPy RGB array -> OpenCV HSV
            rgb_array = pygame.surfarray.array3d(self.camera_surface)
            rgb_array = np.transpose(rgb_array, (1, 0, 2))
            hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)

            # HSV Thresholding
            mask = cv2.inRange(hsv, HSV_LOWER_BLUE, HSV_UPPER_BLUE)
            mask_pip = mask.copy()

            # Find external contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) >= CV_MIN_CONTOUR_AREA:
                    moments = cv2.moments(largest)
                    if moments["m00"] > 1e-4:
                        local_cx = moments["m10"] / moments["m00"]
                        local_cy = moments["m01"] / moments["m00"]
                        offset_x = local_cx - center
                        offset_y = local_cy - center
                        meas_x = float(noisy_x + offset_x)
                        meas_y = float(noisy_y + offset_y)
                        detected = True

        # Save latest detection telemetry
        self.last_mask = mask_pip
        self.last_detected = detected
        self.last_occluded = is_occluded

        # Render Diagnostic Picture-in-Picture (PiP) viewport
        self._render_pip(mask_pip, detected, is_occluded)

        return CVDetection(
            detected=detected,
            meas_x=meas_x,
            meas_y=meas_y,
            delayed_true_x=delayed_x,
            delayed_true_y=delayed_y,
            true_x=true_x,
            true_y=true_y,
            latency_frames=self.latency_frames,
            noise_std=self.noise_std,
            is_occluded=is_occluded,
            pip_surface=self.pip_surface
        )

    def _render_pip(self, mask_img: Optional[np.ndarray], detected: bool, is_occluded: bool) -> None:
        """Draw the real-time diagnostic CV camera and mask overlay."""
        self.pip_surface.fill((10, 14, 20))

        # Border
        border_color = COLOR_EMERALD if detected else (COLOR_RED if is_occluded else COLOR_AMBER)
        pygame.draw.rect(self.pip_surface, border_color, (0, 0, self.pip_width, self.pip_height), 1, border_radius=4)

        # Header title
        title_surf = self.font.render("CV SENSOR DIAGNOSTIC", True, COLOR_WHITE)
        self.pip_surface.blit(title_surf, (8, 5))

        # Status text
        if is_occluded:
            status_text = "STATUS: OCCLUDED / DROP"
            status_color = COLOR_RED
        elif detected:
            status_text = "STATUS: TARGET LOCKED"
            status_color = COLOR_EMERALD
        else:
            status_text = "STATUS: SEARCHING"
            status_color = COLOR_AMBER

        status_surf = self.font.render(status_text, True, status_color)
        self.pip_surface.blit(status_surf, (8, 20))

        # Latency & Noise readouts
        lat_surf = self.font.render(f"LATENCY: {self.latency_frames} f ({self.latency_frames * 16.7:.0f} ms)", True, (180, 190, 205))
        self.pip_surface.blit(lat_surf, (8, 35))

        noise_surf = self.font.render(f"NOISE \u03c3: {self.noise_std:.1f} px | OCCL: {int(self.occlusion_prob * 100)}%", True, (180, 190, 205))
        self.pip_surface.blit(noise_surf, (8, 50))

        # Draw binary mask mini preview
        preview_rect = pygame.Rect(8, 67, 154, 40)
        pygame.draw.rect(self.pip_surface, (20, 26, 36), preview_rect, border_radius=3)

        if mask_img is not None and detected:
            # Downsample mask to fit preview box
            small_mask = cv2.resize(mask_img, (preview_rect.width - 4, preview_rect.height - 4), interpolation=cv2.INTER_NEAREST)
            # Create a 3-channel visual surface (Emerald green tint for detected binary mask)
            color_mask = np.zeros((small_mask.shape[0], small_mask.shape[1], 3), dtype=np.uint8)
            color_mask[small_mask > 0] = [16, 185, 129]
            color_mask = np.transpose(color_mask, (1, 0, 2))
            mask_surf = pygame.surfarray.make_surface(color_mask)
            self.pip_surface.blit(mask_surf, (preview_rect.x + 2, preview_rect.y + 2))
            # Crosshair over preview center
            cx = preview_rect.centerx
            cy = preview_rect.centery
            pygame.draw.line(self.pip_surface, COLOR_WHITE, (cx - 6, cy), (cx + 6, cy), 1)
            pygame.draw.line(self.pip_surface, COLOR_WHITE, (cx, cy - 6), (cx, cy + 6), 1)
        else:
            no_sig = self.font.render("[NO OPTICAL SIGNAL]", True, COLOR_RED)
            self.pip_surface.blit(no_sig, (preview_rect.x + 12, preview_rect.y + 12))

    def get_pip_surface(self, width: int = 440, height: int = 300) -> pygame.Surface:
        """
        Return the diagnostic PiP surface scaled or rendered to the requested viewport size.
        Supports native 170x115 or high-definition 440x300 for the architecture inspection view.
        """
        if width == self.pip_width and height == self.pip_height:
            return self.pip_surface

        # Dedicated high-definition diagnostic panel for How It Works workspace
        hi_surf = pygame.Surface((width, height))
        hi_surf.fill((10, 14, 20))
        border_color = COLOR_EMERALD if self.last_detected else (COLOR_RED if self.last_occluded else COLOR_AMBER)
        pygame.draw.rect(hi_surf, border_color, (0, 0, width, height), 1, border_radius=6)

        # Fonts
        if not pygame.font.get_init():
            pygame.font.init()
        title_font = pygame.font.SysFont("consolas", 12, bold=True)
        body_font = pygame.font.SysFont("consolas", 11)

        # Header title
        title_surf = title_font.render("OPENCV COMPUTER VISION TELEMETRY", True, COLOR_WHITE)
        hi_surf.blit(title_surf, (14, 10))

        # Status text
        if self.last_occluded:
            status_text = "STATUS: OCCLUDED / DROP (KALMAN COASTING)"
            status_color = COLOR_RED
        elif self.last_detected:
            status_text = "STATUS: TARGET LOCKED (COLOR SEGMENTATION ACTIVE)"
            status_color = COLOR_EMERALD
        else:
            status_text = "STATUS: SEARCHING / LOW CONFIDENCE"
            status_color = COLOR_AMBER

        status_surf = title_font.render(status_text, True, status_color)
        hi_surf.blit(status_surf, (14, 28))

        # Diagnostic parameter readouts
        lat_ms = self.latency_frames * 16.7
        lines = [
            f"CAMERA TRANSMISSION LATENCY:  {self.latency_frames} frames ({lat_ms:.1f} ms FIFO queue)",
            f"GAUSSIAN MEASUREMENT NOISE:   \u03c3 = {self.noise_std:.1f} px | OCCLUSION PROB: {int(self.occlusion_prob * 100)}%",
            f"HSV SEGMENTATION BOUNDS:     Hue [90, 130], Sat [80, 255], Val [80, 255]",
            f"CENTROID LOCALIZATION:        Spatial Moments (Cx = M10/M00, Cy = M01/M00)",
            f"MIN CONTOUR AREA THRESHOLD:   {CV_MIN_CONTOUR_AREA:.0f} px (suppresses spurious noise)"
        ]
        for idx, line in enumerate(lines):
            l_surf = body_font.render(line, True, (175, 190, 210))
            hi_surf.blit(l_surf, (14, 50 + idx * 17))

        # Binary Mask Preview Area
        preview_rect = pygame.Rect(14, 142, width - 28, height - 152)
        pygame.draw.rect(hi_surf, (16, 22, 32), preview_rect, border_radius=4)
        pygame.draw.rect(hi_surf, (30, 42, 60), preview_rect, width=1, border_radius=4)

        if self.last_mask is not None and self.last_detected:
            # Scale mask to fit inside the preview container
            mask_w = preview_rect.width - 8
            mask_h = preview_rect.height - 8
            small_mask = cv2.resize(self.last_mask, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST)
            color_mask = np.zeros((small_mask.shape[0], small_mask.shape[1], 3), dtype=np.uint8)
            color_mask[small_mask > 0] = [16, 185, 129]  # Emerald green tint
            color_mask = np.transpose(color_mask, (1, 0, 2))
            mask_surf = pygame.surfarray.make_surface(color_mask)
            hi_surf.blit(mask_surf, (preview_rect.x + 4, preview_rect.y + 4))

            # Crosshairs and reticle over preview center
            cx = preview_rect.centerx
            cy = preview_rect.centery
            pygame.draw.line(hi_surf, COLOR_WHITE, (cx - 16, cy), (cx + 16, cy), 1)
            pygame.draw.line(hi_surf, COLOR_WHITE, (cx, cy - 16), (cx, cy + 16), 1)
            pygame.draw.circle(hi_surf, COLOR_EMERALD, (cx, cy), 22, width=1)

            reticle_lbl = body_font.render("SEGMENTED CENTROID LOCKED", True, COLOR_EMERALD)
            hi_surf.blit(reticle_lbl, (preview_rect.x + 8, preview_rect.bottom - 18))
        else:
            no_sig = title_font.render("[NO OPTICAL SIGNAL DETECTED — COASTING]", True, COLOR_RED)
            hi_surf.blit(no_sig, (preview_rect.centerx - 140, preview_rect.centery - 8))

        return hi_surf


if __name__ == "__main__":
    print("=" * 65)
    print(" CVTracker Computer Vision Pipeline Test")
    print("=" * 65)
    tracker = CVTracker(initial_x=160.0, initial_y=350.0, latency_frames=4)
    detection = tracker.process_frame(165.0, 355.0)
    print(f"Target at (165, 355) -> Detected: {detection.detected}, Meas: ({detection.meas_x}, {detection.meas_y})")
    print("[OK] CVTracker functioning properly.")
    print("To launch the full interactive simulation, run:")
    print("  python main.py\n")