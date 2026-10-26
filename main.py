"""
Predictive Object-Tracking Turret — CV + IR Integration.
Main Application Entrypoint: Clean, Professional Scientific Research Dashboard.
"""

import sys
import os
import subprocess
import math
import csv
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pygame

from core.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS, DT,
    WORKSPACE_OVERVIEW, WORKSPACE_SIMULATION, WORKSPACE_EXPERIMENTS, WORKSPACE_HOW_IT_WORKS,
    WORKSPACE_TABS, TRAJECTORY_MODES,
    DEFAULT_DELAY_FRAMES, DEFAULT_PREDICTION_STEPS, DEFAULT_TOTAL_FRAMES,
    RESULTS_DIR, EXPERIMENT_JSON,
    COLOR_BG, COLOR_PANEL, COLOR_PANEL_ALT, COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_NAVY, COLOR_WHITE, COLOR_PURE_WHITE, COLOR_MUTED, COLOR_DARK_GREY,
    COLOR_REACTIVE, COLOR_PREDICTIVE, COLOR_TARGET, COLOR_SENSOR_DELAY, COLOR_ACCENT_BLUE
)
from core.target_engine import TargetEngine
from core.cv_tracker import CVTracker
from core.kalman_predictor import KalmanPredictor
from core.turret_controller import TurretController

from ui.theme import (
    FontManager, draw_card, draw_text, draw_badge, draw_kpi_card
)
from ui.widgets import NavTabBar, ActionButton, CompactSlider, PipelineBar
from ui.renderer import ArenaRenderer
from ui.plot_canvas import RealTimeErrorPlot


class PredictiveTurretProductApp:
    """
    Professional CV + Intelligent Robotics Scientific Simulation Product.
    Communicates: DETECT -> PREDICT -> CONTROL -> EVALUATE.
    Designed for clarity, elegance, and immediate understanding.
    """

    def __init__(self, initial_mode: int = 1, initial_workspace: int = WORKSPACE_OVERVIEW):
        pygame.init()
        pygame.display.set_caption("Predictive Object-Tracking Turret — CV + IR Simulation")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        FontManager.initialize()

        # Core Mathematical & Sensing Engines
        init_m = max(1, min(5, int(initial_mode)))
        self.target = TargetEngine(mode=init_m)
        self.cv_tracker = CVTracker(self.target.x, self.target.y, latency_frames=DEFAULT_DELAY_FRAMES)
        self.kalman = KalmanPredictor(self.target.x, self.target.y, dt=DT)
        self.reactive_turret = TurretController(turret_id="Reactive-Turret")
        self.predictive_turret = TurretController(turret_id="Predictive-Turret")

        # Telemetry & Experiment Buffers
        self.reactive_errors: List[float] = []
        self.predictive_errors: List[float] = []
        self.reactive_locked_frames = 0
        self.predictive_locked_frames = 0
        self.frame_number = 0
        self.max_test_frames = DEFAULT_TOTAL_FRAMES
        self.prediction_steps = DEFAULT_PREDICTION_STEPS

        # Persistent Historical Results Across Modes (1 to 5)
        self.mode_results = self._load_saved_results()

        # State Control
        self.is_paused = False
        self.is_presentation_mode = False
        self.is_trial_complete = False
        self.active_workspace = max(0, min(3, int(initial_workspace)))
        mode_label = TRAJECTORY_MODES.get(init_m, f"Mode {init_m}")
        self.status_message = f"SIMULATION RUNNING: {mode_label}"
        self.status_is_error = False

        # --- DYNAMIC LAYOUT RECTANGLES (1280 x 760) ---
        self.header_rect = pygame.Rect(16, 12, 1248, 44)
        self.nav_tabs_rect = pygame.Rect(440, 14, 520, 34)

        # Workspace 0: Overview Layout (Simplicity & Dominant Simulation Views)
        self.ov_pipeline_rect = pygame.Rect(16, 62, 1248, 36)
        self.ov_left_arena = pygame.Rect(16, 106, 616, 440)
        self.ov_right_arena = pygame.Rect(648, 106, 616, 440)
        self.ov_kpi_rect = pygame.Rect(16, 556, 1248, 92)
        self.ov_action_rect = pygame.Rect(16, 658, 1248, 52)

        # Workspace 1: Live Simulation Layout (Includes Tuning Sliders & Actuator Telemetry)
        self.sim_left_arena = pygame.Rect(16, 64, 616, 380)
        self.sim_right_arena = pygame.Rect(648, 64, 616, 380)
        self.sim_slider_rect = pygame.Rect(16, 452, 1248, 68)
        self.sim_kpi_rect = pygame.Rect(16, 528, 1248, 86)
        self.sim_action_rect = pygame.Rect(16, 624, 1248, 50)

        # Workspace 2: Experiments & Results Layout (Benchmark Matrix + Full-Scale Oscilloscope)
        self.exp_table_rect = pygame.Rect(16, 64, 1248, 220)
        self.exp_plot_rect = pygame.Rect(16, 294, 1248, 340)
        self.exp_action_rect = pygame.Rect(16, 644, 1248, 52)

        # Workspace 3: How It Works / Viva Defense Layout
        self.hiw_summary_rect = pygame.Rect(16, 64, 1248, 150)
        self.hiw_cv_rect = pygame.Rect(16, 224, 480, 520)
        self.hiw_math_rect = pygame.Rect(512, 224, 752, 520)

        # Presentation Mode Layout (Maximized Arenas for Projector)
        self.pres_left_arena = pygame.Rect(16, 20, 616, 550)
        self.pres_right_arena = pygame.Rect(648, 20, 616, 550)
        self.pres_kpi_rect = pygame.Rect(16, 582, 1248, 110)
        self.pres_pipeline_rect = pygame.Rect(16, 700, 1248, 38)

        # Visual Renderers & Telemetry Plot
        self.arena_renderer = ArenaRenderer(self.ov_left_arena, self.ov_right_arena)
        self.overview_plot = RealTimeErrorPlot(pygame.Rect(16, 556, 1248, 100), max_history=300)
        self.experiments_plot = RealTimeErrorPlot(self.exp_plot_rect, max_history=450, title="Multi-Mode Error Oscilloscope (3.0° Lock Tolerance)")
        self.pipeline_bar = PipelineBar(self.ov_pipeline_rect)
        self.latest_cv_result = self.cv_tracker.process_frame(self.target.x, self.target.y)

        # Build Interactive Controls
        self._init_controls()

    def _init_controls(self) -> None:
        """Create navigation tabs, action buttons, and tuning sliders."""
        self.tab_bar = NavTabBar(
            self.nav_tabs_rect,
            tabs=WORKSPACE_TABS,
            active_idx=self.active_workspace,
            on_change=self._set_workspace
        )

        # Overview Action Buttons (Essential Only)
        btn_h = 36
        y_btns = self.ov_action_rect.y + 8
        self.btn_pause = ActionButton(
            pygame.Rect(self.ov_action_rect.x + 12, y_btns, 120, btn_h),
            label="PAUSE", hotkey_tag="[SPACE]", is_primary=True,
            accent_color=COLOR_ACCENT_BLUE, callback=self.toggle_pause
        )
        self.btn_reset = ActionButton(
            pygame.Rect(self.ov_action_rect.x + 138, y_btns, 132, btn_h),
            label="RESET TRIAL", hotkey_tag="[R]", is_primary=False,
            accent_color=COLOR_SENSOR_DELAY, callback=self.reset_experiment
        )
        self.btn_mode = ActionButton(
            pygame.Rect(self.ov_action_rect.x + 276, y_btns, 194, btn_h),
            label="TRAJECTORY: MODE 1", hotkey_tag="[M]", is_primary=False,
            accent_color=COLOR_NAVY, callback=self.next_mode
        )
        self._update_mode_button_label()

        self.btn_export = ActionButton(
            pygame.Rect(self.ov_action_rect.x + 476, y_btns, 130, btn_h),
            label="EXPORT CSV", hotkey_tag="[E]", is_primary=False,
            accent_color=COLOR_PREDICTIVE, callback=self.export_csv
        )
        self.btn_open_folder = ActionButton(
            pygame.Rect(self.ov_action_rect.x + 612, y_btns, 144, btn_h),
            label="OPEN FOLDER", hotkey_tag="[O]", is_primary=False,
            accent_color=COLOR_ACCENT_BLUE, callback=self.open_results_folder
        )
        self.btn_presentation = ActionButton(
            pygame.Rect(self.ov_action_rect.right - 156, y_btns, 146, btn_h),
            label="PRESENTATION", hotkey_tag="[P]", is_primary=False,
            accent_color=COLOR_ACCENT_BLUE, callback=self.toggle_presentation_mode
        )

        # Secondary Sliders (Displayed in Simulation Workspace)
        y_sliders = self.sim_slider_rect.y + 10
        slider_w = 296
        slider_h = 44
        self.slider_speed = CompactSlider(
            pygame.Rect(self.sim_slider_rect.x + 14, y_sliders, slider_w, slider_h),
            label="Target Speed", min_val=1.0, max_val=8.0, initial_val=self.target.speed_multiplier,
            step=0.5, unit="x", callback=self._on_speed_changed
        )
        self.slider_latency = CompactSlider(
            pygame.Rect(self.sim_slider_rect.x + 324, y_sliders, slider_w, slider_h),
            label="Camera Delay", min_val=0.0, max_val=18.0, initial_val=float(self.cv_tracker.latency_frames),
            step=1.0, unit="f", callback=self._on_latency_changed
        )
        self.slider_noise = CompactSlider(
            pygame.Rect(self.sim_slider_rect.x + 634, y_sliders, slider_w, slider_h),
            label="Sensor Noise", min_val=0.0, max_val=6.0, initial_val=self.cv_tracker.noise_std,
            step=0.5, unit="px", callback=self._on_noise_changed
        )
        self.slider_pred_steps = CompactSlider(
            pygame.Rect(self.sim_slider_rect.x + 944, y_sliders, slider_w, slider_h),
            label="Prediction Horizon", min_val=1.0, max_val=16.0, initial_val=float(DEFAULT_PREDICTION_STEPS),
            step=1.0, unit="f", callback=self._on_prediction_steps_changed
        )

        self.buttons = [self.btn_pause, self.btn_reset, self.btn_mode, self.btn_export, self.btn_open_folder, self.btn_presentation]
        self.sliders = [self.slider_speed, self.slider_latency, self.slider_noise, self.slider_pred_steps]

    def _update_mode_button_label(self) -> None:
        """Update mode button label to current trajectory mode name."""
        m_name = TRAJECTORY_MODES.get(self.target.mode, f"Mode {self.target.mode}")
        short_name = m_name.split(":")[-1].strip()
        self.btn_mode.label = f"MODE: {short_name.upper()}"

    def _set_workspace(self, idx: int) -> None:
        """Switch active workspace view."""
        self.active_workspace = idx

    def _on_speed_changed(self, val: float) -> None:
        self.target.set_speed_multiplier(val)

    def _on_latency_changed(self, val: float) -> None:
        self.cv_tracker.set_latency(int(val))

    def _on_noise_changed(self, val: float) -> None:
        self.cv_tracker.set_noise(float(val))

    def _on_prediction_steps_changed(self, val: float) -> None:
        self.prediction_steps = max(0, int(val))

    def open_results_folder(self) -> None:
        """Open the results directory in the native OS file explorer."""
        try:
            target_dir = RESULTS_DIR.resolve()
            if sys.platform == "win32":
                os.startfile(str(target_dir))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target_dir)])
            else:
                subprocess.Popen(["xdg-open", str(target_dir)])
            self.status_message = f"Opened Folder: {target_dir.name}/ (Results Explorer)"
            self.status_is_error = False
            print(f">>> [EXPLORER] Opened results folder: {target_dir}")
        except Exception as err:
            self.status_message = f"Could not open folder: {err}"
            self.status_is_error = True

    def toggle_pause(self) -> None:
        """Toggle pause state. If trial is complete, locks simulation until user resets."""
        if self.is_trial_complete:
            self.status_message = "TRIAL COMPLETE (1,800 frames locked). Press [R] to start a new trial."
            return
        self.is_paused = not self.is_paused
        if not self.is_paused:
            self.btn_pause.label = "PAUSE"
            self.btn_pause.accent_color = COLOR_ACCENT_BLUE
            self.status_message = "SIMULATION RUNNING"
        else:
            self.btn_pause.label = "RESUME"
            self.btn_pause.accent_color = COLOR_PREDICTIVE
            self.status_message = "SIMULATION PAUSED"

    def toggle_presentation_mode(self) -> None:
        self.is_presentation_mode = not self.is_presentation_mode

    def next_mode(self) -> None:
        """Cycle to next trajectory mode."""
        curr = self.target.mode
        new_m = 1 if curr >= 5 else curr + 1
        self.target.reset(mode=new_m)
        self.reset_experiment()
        self._update_mode_button_label()
        self.status_message = f"ACTIVE: {TRAJECTORY_MODES.get(new_m, '')}"

    def reset_experiment(self) -> None:
        """Reset trial state and filter accumulators for a fresh 1,800-frame run."""
        self.target.reset()
        self.cv_tracker.reset(self.target.x, self.target.y)
        self.kalman.reset(self.target.x, self.target.y)
        self.reactive_turret.reset()
        self.predictive_turret.reset()
        self.overview_plot.reset()
        self.experiments_plot.reset()

        self.frame_number = 0
        self.is_trial_complete = False
        self.is_paused = False
        self.btn_pause.label = "PAUSE"
        self.btn_pause.accent_color = COLOR_ACCENT_BLUE
        self.btn_pause.is_primary = True
        self.btn_reset.is_primary = False
        self.reactive_errors.clear()
        self.predictive_errors.clear()
        self.reactive_locked_frames = 0
        self.predictive_locked_frames = 0
        self.latest_cv_result = self.cv_tracker.process_frame(self.target.x, self.target.y)
        self.status_message = f"NEW TRIAL: {TRAJECTORY_MODES.get(self.target.mode, '')}"

    def export_csv(self) -> None:
        """Export frame-by-frame and summary telemetry to CSV in results/ directory."""
        if not self.reactive_errors:
            self.status_message = "Export failed: No trial data recorded."
            self.status_is_error = True
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = RESULTS_DIR / f"tracking_mode{self.target.mode}_{timestamp}.csv"

        try:
            r_mean = float(np.mean(self.reactive_errors))
            p_mean = float(np.mean(self.predictive_errors))
            r_rmse = float(np.sqrt(np.mean(np.square(self.reactive_errors))))
            p_rmse = float(np.sqrt(np.mean(np.square(self.predictive_errors))))
            improvement = ((r_mean - p_mean) / r_mean * 100.0) if r_mean > 1e-3 else 0.0

            with filename.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Mode", "Frame", "Time_Seconds", "Reactive_Angular_Error_Deg", "Predictive_Angular_Error_Deg"])
                for idx, (r, p) in enumerate(zip(self.reactive_errors, self.predictive_errors), start=1):
                    w.writerow([self.target.mode, idx, round(idx * DT, 4), round(r, 4), round(p, 4)])
                w.writerow([])
                w.writerow(["SUMMARY_STATISTICS"])
                w.writerow(["Mode", self.target.mode, TRAJECTORY_MODES.get(self.target.mode, "")])
                w.writerow(["Total_Frames", len(self.reactive_errors)])
                w.writerow(["Reactive_Mean_Error_Deg", round(r_mean, 4)])
                w.writerow(["Predictive_Mean_Error_Deg", round(p_mean, 4)])
                w.writerow(["Mean_Error_Reduction_Percent", round(improvement, 2)])
                w.writerow(["Reactive_RMSE_Deg", round(r_rmse, 4)])
                w.writerow(["Predictive_RMSE_Deg", round(p_rmse, 4)])
                w.writerow(["Export_Timestamp", datetime.now().isoformat()])

            self.mode_results[self.target.mode] = {
                "reactive_mean": round(r_mean, 4),
                "predictive_mean": round(p_mean, 4),
                "improvement": round(improvement, 2),
                "reactive_rmse": round(r_rmse, 4),
                "predictive_rmse": round(p_rmse, 4),
                "frames": len(self.reactive_errors),
                "completed_at": datetime.now().isoformat(timespec="seconds")
            }
            self._save_persistent_results()
            self.status_message = f"CSV Exported: results/{filename.name} [Click 'OPEN FOLDER' or Press O]"
            self.status_is_error = False
            print(f"\n============================================================")
            print(f"[CSV EXPORT] File successfully saved:")
            print(f"Path: {filename.resolve()}")
            print(f"Folder: {RESULTS_DIR.resolve()}")
            print(f"============================================================\n")
        except Exception as err:
            self.status_message = f"Export failed: {err}"
            self.status_is_error = True

    def _load_saved_results(self) -> Dict[int, Any]:
        """Load experiment summary JSON."""
        res = {1: None, 2: None, 3: None, 4: None, 5: None}
        if EXPERIMENT_JSON.exists():
            try:
                with EXPERIMENT_JSON.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if str(k).isdigit():
                            res[int(k)] = v
            except Exception:
                pass
        return res

    def _save_persistent_results(self) -> None:
        """Save experiment summary JSON."""
        try:
            s = {str(k): v for k, v in self.mode_results.items() if v is not None}
            with EXPERIMENT_JSON.open("w", encoding="utf-8") as f:
                json.dump(s, f, indent=4)
        except Exception:
            pass

    def run(self) -> None:
        """Main 60 FPS application loop."""
        running = True
        while running:
            self.clock.tick(FPS)
            mouse_pos = pygame.mouse.get_pos()

            # --- EVENT PROCESSING ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.toggle_pause()
                    elif event.key == pygame.K_p:
                        self.toggle_presentation_mode()
                    elif event.key == pygame.K_r:
                        self.reset_experiment()
                    elif event.key == pygame.K_m:
                        self.next_mode()
                    elif event.key == pygame.K_e:
                        self.export_csv()
                    elif event.key == pygame.K_o:
                        self.open_results_folder()
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                        self._set_workspace(event.key - pygame.K_1)
                        self.tab_bar.active_idx = self.active_workspace
                    elif event.key == pygame.K_ESCAPE:
                        if self.is_presentation_mode:
                            self.is_presentation_mode = False
                        else:
                            running = False

                if self.is_presentation_mode:
                    pass
                else:
                    self.tab_bar.handle_event(event)
                    self.pipeline_bar.handle_event(event)

                    # Dispatch widget events based on active workspace
                    if self.active_workspace == WORKSPACE_OVERVIEW:
                        self.btn_pause.handle_event(event)
                        self.btn_reset.handle_event(event)
                        self.btn_mode.handle_event(event)
                        self.btn_export.handle_event(event)
                        self.btn_open_folder.handle_event(event)
                        self.btn_presentation.handle_event(event)

                    elif self.active_workspace == WORKSPACE_SIMULATION:
                        for btn in self.buttons:
                            btn.handle_event(event)
                        for slider in self.sliders:
                            slider.handle_event(event)

                    elif self.active_workspace == WORKSPACE_EXPERIMENTS:
                        self.btn_pause.handle_event(event)
                        self.btn_reset.handle_event(event)
                        self.btn_mode.handle_event(event)
                        self.btn_export.handle_event(event)
                        self.btn_open_folder.handle_event(event)

                        # Click to directly switch mode from benchmark table rows
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            start_y = self.exp_table_rect.y + 78
                            for m_idx in range(1, 6):
                                row_r = pygame.Rect(self.exp_table_rect.x + 12, start_y + (m_idx - 1) * 26 - 2, self.exp_table_rect.width - 24, 24)
                                if row_r.collidepoint(event.pos):
                                    self.target.reset(mode=m_idx)
                                    self.reset_experiment()
                                    self._update_mode_button_label()
                                    self.status_message = f"ACTIVE BENCHMARK: {TRAJECTORY_MODES.get(m_idx, '')}"
                                    break

            # --- SIMULATION COMPUTATION STEP ---
            if not self.is_paused and not self.is_trial_complete:
                self.frame_number += 1
                self.pipeline_bar.update_auto(is_sim_running=True)

                # Determine active arena rectangles
                if self.is_presentation_mode:
                    curr_left = self.pres_left_arena
                    curr_right = self.pres_right_arena
                elif self.active_workspace == WORKSPACE_SIMULATION:
                    curr_left = self.sim_left_arena
                    curr_right = self.sim_right_arena
                else:
                    curr_left = self.ov_left_arena
                    curr_right = self.ov_right_arena

                self.arena_renderer.set_rects(curr_left, curr_right)

                # 1. Update Target Kinematics (with interactive screen-to-world mapping)
                sim_mouse = None
                if self.target.mode == 5 and mouse_pos is not None:
                    if curr_left.collidepoint(mouse_pos):
                        sim_mouse = self.arena_renderer.screen_to_world(mouse_pos[0], mouse_pos[1], curr_left)
                    elif curr_right.collidepoint(mouse_pos):
                        sim_mouse = self.arena_renderer.screen_to_world(mouse_pos[0], mouse_pos[1], curr_right)

                true_x, true_y, true_vx, true_vy = self.target.update(DT, mouse_pos=sim_mouse)

                # 2. Camera Sensing & OpenCV Detection
                det = self.cv_tracker.process_frame(true_x, true_y)
                self.latest_cv_result = det

                # 3. Kalman Filter Estimation Step
                meas = (det.meas_x, det.meas_y) if det.detected else None
                self.kalman.step(meas)

                # 4. Multi-Step Extrapolation (+H frames ahead)
                pred_x, pred_y, ellipse_params = self.kalman.predict_future(steps=self.prediction_steps)

                # 5. Dual Turret Control Update
                base_rx, base_ry = self.arena_renderer.get_turret_base_pos(curr_left)
                self.reactive_turret.base_x = base_rx
                self.reactive_turret.base_y = base_ry

                base_px, base_py = self.arena_renderer.get_turret_base_pos(curr_right)
                self.predictive_turret.base_x = base_px
                self.predictive_turret.base_y = base_py

                # Reactive Turret points at delayed observation
                r_target = self.arena_renderer.world_to_screen(det.delayed_true_x, det.delayed_true_y, curr_left)
                r_actual = self.arena_renderer.world_to_screen(true_x, true_y, curr_left)
                self.reactive_turret.update(r_target[0], r_target[1], dt=DT)
                r_err = self.reactive_turret.aiming_error_to(r_actual[0], r_actual[1])

                # Predictive Turret points at Kalman future state
                p_target = self.arena_renderer.world_to_screen(pred_x, pred_y, curr_right)
                p_actual = self.arena_renderer.world_to_screen(true_x, true_y, curr_right)
                self.predictive_turret.update(p_target[0], p_target[1], dt=DT)
                p_err = self.predictive_turret.aiming_error_to(p_actual[0], p_actual[1])

                # Telemetry Accumulation
                self.reactive_errors.append(r_err)
                self.predictive_errors.append(p_err)
                if self.reactive_turret.is_locked:
                    self.reactive_locked_frames += 1
                if self.predictive_turret.is_locked:
                    self.predictive_locked_frames += 1

                # Update Error Charts
                self.overview_plot.append(r_err, p_err)
                self.experiments_plot.append(r_err, p_err)

                if self.frame_number >= self.max_test_frames:
                    self.frame_number = self.max_test_frames
                    self.is_trial_complete = True
                    self.is_paused = True
                    self.btn_pause.label = "LOCKED"
                    self.btn_pause.accent_color = COLOR_PREDICTIVE
                    self.btn_pause.is_primary = False
                    self.btn_reset.is_primary = True
                    self.export_csv()
                    self.status_message = f"TRIAL COMPLETE ({self.max_test_frames}f) — Score Locked | Press [R] New Trial | [O] Open CSV Folder"

            # --- RENDERING DISPATCH ---
            self.screen.fill(COLOR_BG)

            if self.is_presentation_mode:
                self._render_presentation_mode()
            elif self.active_workspace == WORKSPACE_OVERVIEW:
                self._render_overview_workspace()
            elif self.active_workspace == WORKSPACE_SIMULATION:
                self._render_simulation_workspace()
            elif self.active_workspace == WORKSPACE_EXPERIMENTS:
                self._render_experiments_workspace()
            elif self.active_workspace == WORKSPACE_HOW_IT_WORKS:
                self._render_how_it_works_workspace()

            pygame.display.flip()

        pygame.quit()
        sys.exit(0)

    # -------------------------------------------------------------
    # SECTION 1: OVERVIEW (THE STREAMLINED EXECUTIVE DASHBOARD)
    # -------------------------------------------------------------
    def _render_overview_workspace(self) -> None:
        """Render the neat, spacious, focused dashboard: Arenas, 3 KPIs, and Essential Controls."""
        self._render_header()

        r_err_disp = self.reactive_errors[-1] if self.reactive_errors else 0.0
        p_err_disp = self.predictive_errors[-1] if self.predictive_errors else 0.0
        delayed_x = self.latest_cv_result.delayed_true_x if self.latest_cv_result else self.target.x
        delayed_y = self.latest_cv_result.delayed_true_y if self.latest_cv_result else self.target.y

        # 1. Plain-English Pipeline Banner
        self.pipeline_bar.set_rect(self.ov_pipeline_rect)
        self.pipeline_bar.render(self.screen)

        # 2. Main Synchronized Simulation Arenas (Dominating the screen)
        self.arena_renderer.set_rects(self.ov_left_arena, self.ov_right_arena)

        self.arena_renderer.render_reactive_arena(
            self.screen, self.reactive_turret,
            delayed_x=delayed_x, delayed_y=delayed_y,
            true_x=self.target.x, true_y=self.target.y,
            trail=self.target.get_trail(),
            error_deg=r_err_disp,
            latency_frames=self.cv_tracker.latency_frames
        )

        pred_x, pred_y, ellipse_params = self.kalman.predict_future(steps=self.prediction_steps)
        self.arena_renderer.render_predictive_arena(
            self.screen, self.predictive_turret,
            true_x=self.target.x, true_y=self.target.y,
            pred_x=pred_x, pred_y=pred_y,
            ellipse_params=ellipse_params,
            trail=self.target.get_trail(),
            error_deg=p_err_disp,
            steps_ahead=self.prediction_steps,
            is_coasting=self.kalman.is_coasting
        )

        # 3. Primary 4 Comparison Metric Cards (Reactive MAE, Predictive MAE, Error Reduction, Lock Rate)
        self._render_primary_4_kpis(self.ov_kpi_rect)

        # 4. Essential Action Controls Bar
        self._render_overview_controls(self.ov_action_rect)

    def _render_primary_4_kpis(self, rect: pygame.Rect) -> None:
        """
        Render the 4 essential scientific performance indicators:
        1. Reactive tracking error (cumulative MAE + live error)
        2. Predictive tracking error (cumulative MAE + live error)
        3. Error reduction percentage (with RMSE comparison)
        4. Target lock rate (% frames within <=3.0° tolerance)
        """
        card_w = (rect.width - 36) // 4
        card_h = rect.height

        n_samples = len(self.reactive_errors)
        r_err_disp = self.reactive_errors[-1] if self.reactive_errors else 0.0
        p_err_disp = self.predictive_errors[-1] if self.predictive_errors else 0.0

        if n_samples > 10:
            r_mean = float(np.mean(self.reactive_errors))
            p_mean = float(np.mean(self.predictive_errors))
            r_rmse = float(np.sqrt(np.mean(np.square(self.reactive_errors))))
            p_rmse = float(np.sqrt(np.mean(np.square(self.predictive_errors))))
            reduction = ((r_mean - p_mean) / r_mean * 100.0) if r_mean > 1e-3 else 0.0
            r_lock_pct = (self.reactive_locked_frames / n_samples) * 100.0
            p_lock_pct = (self.predictive_locked_frames / n_samples) * 100.0

            r_val_str = f"{r_mean:.2f}°"
            p_val_str = f"{p_mean:.2f}°"
            red_val_str = f"{reduction:+.1f}%"
            lock_str = f"{p_lock_pct:.0f}% vs {r_lock_pct:.0f}%"

            if self.is_trial_complete:
                badge_text = "FINAL SCORE" if reduction > 0 else "NO GAIN"
                lock_badge = "TRIAL LOCKED"
                r_sec = f"Final: {r_err_disp:.2f}° (Locked)"
                p_sec = f"Final: {p_err_disp:.2f}° (Locked)"
            else:
                badge_text = "ACCURACY GAIN" if reduction > 0 else "NO GAIN"
                lock_badge = "BEAM LOCKED" if self.predictive_turret.is_locked else "SEARCHING"
                r_sec = f"Live: {r_err_disp:.2f}°"
                p_sec = f"Live: {p_err_disp:.2f}°"

            red_sec = f"RMSE: {p_rmse:.2f}° vs {r_rmse:.2f}°"
            lock_sec = "Predictive vs Reactive"
        else:
            r_val_str = "Collecting..."
            p_val_str = "Collecting..."
            red_val_str = "—"
            lock_str = "—"
            badge_text = "CALCULATING"
            lock_badge = "CALIBRATING"
            r_sec = "Live: —"
            p_sec = "Live: —"
            red_sec = "RMSE: —"
            lock_sec = "Predictive vs Reactive"

        cards_data = [
            ("Reactive Error (MAE)", r_val_str, "Lagged Camera (without prediction)", COLOR_REACTIVE, None, r_sec),
            ("Predictive Error (MAE)", p_val_str, "Kalman Compensated Target", COLOR_PREDICTIVE, None, p_sec),
            ("Error Reduction", red_val_str, "Tracking Accuracy Gain with Lead", COLOR_PREDICTIVE if not red_val_str.startswith("-") else COLOR_REACTIVE, badge_text, red_sec),
            ("Target Lock Rate (≤3°)", lock_str, "Beam Alignment Retention", COLOR_ACCENT_BLUE, lock_badge, lock_sec)
        ]

        for i, (title, val, sub, col, badge, sec) in enumerate(cards_data):
            c_rect = pygame.Rect(rect.x + i * (card_w + 12), rect.y, card_w, card_h)
            draw_kpi_card(self.screen, c_rect, title, val, sub, col, badge, secondary_str=sec)

    def _render_primary_3_kpis(self, rect: pygame.Rect) -> None:
        """Backwards compatibility alias for _render_primary_4_kpis."""
        self._render_primary_4_kpis(rect)

    def _render_overview_controls(self, rect: pygame.Rect) -> None:
        """Render essential controls: Start/Pause, Reset, Trajectory, Blind Zone, Cannon, SFX, Export, Folder, and Presentation mode."""
        draw_card(self.screen, rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)

        y_btns = rect.y + 8
        for btn in self.buttons:
            btn.rect.y = y_btns
            btn.render(self.screen)

        # Trial Progress Track & Frame Counter
        prog_x = self.btn_open_folder.rect.right + 18
        prog_w = max(10, self.btn_presentation.rect.left - prog_x - 14)
        curr_f = min(self.frame_number, self.max_test_frames)
        pct = (curr_f / self.max_test_frames) * 100.0

        if self.is_trial_complete:
            prog_label = f"LOCKED ({self.max_test_frames}f)"
            prog_col = COLOR_PREDICTIVE
        else:
            prog_label = f"{curr_f}/{self.max_test_frames}f ({pct:.0f}%)"
            prog_col = COLOR_NAVY

        draw_text(self.screen, prog_label, prog_x, y_btns + 3, color=prog_col, font=FontManager.small_font)

        # 5px Sleek Progress Bar
        bar_y = y_btns + 20
        bar_w = prog_w
        pygame.draw.rect(self.screen, (226, 232, 240), (prog_x, bar_y, bar_w, 5), border_radius=2)
        fill_w = int((curr_f / self.max_test_frames) * bar_w)
        fill_col = COLOR_PREDICTIVE if self.is_trial_complete else COLOR_ACCENT_BLUE
        if fill_w > 0:
            pygame.draw.rect(self.screen, fill_col, (prog_x, bar_y, fill_w, 5), border_radius=2)

    # -------------------------------------------------------------
    # SECTION 2: LIVE SIMULATION WORKSPACE (TUNING & SENSING)
    # -------------------------------------------------------------
    def _render_simulation_workspace(self) -> None:
        """Render live simulation with parameter tuning sliders and flight diagnostics."""
        self._render_header()

        r_err_disp = self.reactive_errors[-1] if self.reactive_errors else 0.0
        p_err_disp = self.predictive_errors[-1] if self.predictive_errors else 0.0
        delayed_x = self.latest_cv_result.delayed_true_x if self.latest_cv_result else self.target.x
        delayed_y = self.latest_cv_result.delayed_true_y if self.latest_cv_result else self.target.y

        self.arena_renderer.set_rects(self.sim_left_arena, self.sim_right_arena)

        self.arena_renderer.render_reactive_arena(
            self.screen, self.reactive_turret,
            delayed_x=delayed_x, delayed_y=delayed_y,
            true_x=self.target.x, true_y=self.target.y,
            trail=self.target.get_trail(),
            error_deg=r_err_disp,
            latency_frames=self.cv_tracker.latency_frames
        )

        pred_x, pred_y, ellipse_params = self.kalman.predict_future(steps=self.prediction_steps)
        self.arena_renderer.render_predictive_arena(
            self.screen, self.predictive_turret,
            true_x=self.target.x, true_y=self.target.y,
            pred_x=pred_x, pred_y=pred_y,
            ellipse_params=ellipse_params,
            trail=self.target.get_trail(),
            error_deg=p_err_disp,
            steps_ahead=self.prediction_steps,
            is_coasting=self.kalman.is_coasting
        )

        # Tuning Sliders Panel
        draw_card(self.screen, self.sim_slider_rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)
        draw_text(self.screen, "SIMULATION PARAMETERS & SENSOR NOISE TUNING", self.sim_slider_rect.x + 14, self.sim_slider_rect.y + 6, color=COLOR_NAVY, font=FontManager.small_font)
        y_sl = self.sim_slider_rect.y + 20
        for s in self.sliders:
            s.set_rect(pygame.Rect(s.rect.x, y_sl, s.rect.width, s.rect.height))
            s.render(self.screen)

        # Telemetry KPI cards (4 detailed cards)
        self._render_kpi_cards(self.sim_kpi_rect)

        # Simulation Action Bar
        draw_card(self.screen, self.sim_action_rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)
        y_b = self.sim_action_rect.y + 8
        for btn in self.buttons:
            btn.rect.y = y_b
            btn.render(self.screen)

    # -------------------------------------------------------------
    # SECTION 3: EXPERIMENTS & RESULTS WORKSPACE
    # -------------------------------------------------------------
    def _render_experiments_workspace(self) -> None:
        """Render the 5-mode benchmark matrix table, full-scale oscilloscope, and export controls."""
        self._render_header()

        # 1. Benchmark Table
        draw_card(self.screen, self.exp_table_rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)
        draw_text(
            self.screen, "MULTI-MODE TRACKING BENCHMARK (EMPIRICAL RESULTS)",
            self.exp_table_rect.x + 14, self.exp_table_rect.y + 10,
            color=COLOR_NAVY, font=FontManager.title_font
        )
        draw_text(
            self.screen, "Compares Reactive vs Predictive tracking under identical 8-frame camera latency and Gaussian noise.",
            self.exp_table_rect.x + 14, self.exp_table_rect.y + 30,
            color=COLOR_MUTED, font=FontManager.small_font
        )

        cols = [
            ("MODE / TRAJECTORY", 220),
            ("REACTIVE MEAN", 160),
            ("PREDICTIVE MEAN", 160),
            ("IMPROVEMENT", 150),
            ("REACTIVE RMSE", 150),
            ("PREDICTIVE RMSE", 150),
            ("STATUS", 170)
        ]
        start_x = self.exp_table_rect.x + 16
        header_y = self.exp_table_rect.y + 54

        curr_x = start_x
        for col_name, col_w in cols:
            draw_text(self.screen, col_name, curr_x, header_y, color=COLOR_MUTED, font=FontManager.subheading_font)
            curr_x += col_w

        pygame.draw.line(
            self.screen, COLOR_BORDER,
            (start_x, header_y + 18), (self.exp_table_rect.right - 16, header_y + 18), 1
        )

        # Table Rows for Modes 1-5
        row_y = header_y + 24
        for m_id in range(1, 6):
            m_name = TRAJECTORY_MODES.get(m_id, f"Mode {m_id}")
            res = self.mode_results.get(m_id)
            is_active_mode = (self.target.mode == m_id)

            row_bg = (235, 243, 255) if is_active_mode else ((248, 250, 252) if m_id % 2 == 0 else COLOR_PANEL)
            row_rect = pygame.Rect(start_x - 4, row_y - 2, self.exp_table_rect.width - 24, 24)
            pygame.draw.rect(self.screen, row_bg, row_rect, border_radius=4)
            if is_active_mode:
                pygame.draw.rect(self.screen, COLOR_ACCENT_BLUE, row_rect, width=1, border_radius=4)

            curr_x = start_x
            draw_text(self.screen, m_name, curr_x, row_y + 2, color=COLOR_NAVY, font=FontManager.body_font)
            curr_x += 220

            if res:
                r_m = res.get('reactive_mean')
                p_m = res.get('predictive_mean')
                imp_val = res.get('improvement')
                r_rmse = res.get('reactive_rmse')
                p_rmse = res.get('predictive_rmse')
                f_count = res.get('frames', 0)

                r_m_str = f"{r_m:.2f}°" if r_m is not None else "—"
                p_m_str = f"{p_m:.2f}°" if p_m is not None else "—"
                imp_str = f"{imp_val:+.1f}%" if imp_val is not None else "—"
                imp_col = COLOR_PREDICTIVE if (imp_val is not None and imp_val >= 0) else COLOR_REACTIVE
                r_rmse_str = f"{r_rmse:.2f}°" if r_rmse is not None else "—"
                p_rmse_str = f"{p_rmse:.2f}°" if p_rmse is not None else "—"

                draw_text(self.screen, r_m_str, curr_x, row_y + 2, color=COLOR_REACTIVE, font=FontManager.body_font)
                curr_x += 160
                draw_text(self.screen, p_m_str, curr_x, row_y + 2, color=COLOR_PREDICTIVE, font=FontManager.body_font)
                curr_x += 160
                draw_text(self.screen, imp_str, curr_x, row_y + 2, color=imp_col, font=FontManager.body_font)
                curr_x += 150
                draw_text(self.screen, r_rmse_str, curr_x, row_y + 2, color=COLOR_MUTED, font=FontManager.body_font)
                curr_x += 150
                draw_text(self.screen, p_rmse_str, curr_x, row_y + 2, color=COLOR_MUTED, font=FontManager.body_font)
                curr_x += 150
                draw_text(self.screen, f"RECORDED ({f_count}f)", curr_x, row_y + 2, color=COLOR_PREDICTIVE, font=FontManager.small_font)
            else:
                tag = "ACTIVE (IN PROGRESS)" if is_active_mode else "PENDING BENCHMARK"
                draw_text(self.screen, "—", curr_x, row_y + 2, color=COLOR_MUTED, font=FontManager.body_font)
                curr_x += 160
                draw_text(self.screen, "—", curr_x, row_y + 2, color=COLOR_MUTED, font=FontManager.body_font)
                curr_x += 160
                draw_text(self.screen, "—", curr_x, row_y + 2, color=COLOR_MUTED, font=FontManager.body_font)
                curr_x += 150
                draw_text(self.screen, "—", curr_x, row_y + 2, color=COLOR_MUTED, font=FontManager.body_font)
                curr_x += 150
                draw_text(self.screen, "—", curr_x, row_y + 2, color=COLOR_MUTED, font=FontManager.body_font)
                curr_x += 150
                draw_text(self.screen, tag, curr_x, row_y + 2, color=COLOR_ACCENT_BLUE if is_active_mode else COLOR_MUTED, font=FontManager.small_font)

            row_y += 26

        # 2. Full-Scale Oscilloscope
        self.experiments_plot.render(self.screen)

        # 3. Action Bar
        draw_card(self.screen, self.exp_action_rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)
        y_b = self.exp_action_rect.y + 8
        self.btn_pause.rect.y = y_b
        self.btn_reset.rect.y = y_b
        self.btn_mode.rect.y = y_b
        self.btn_export.rect.y = y_b
        self.btn_open_folder.rect.y = y_b

        self.btn_pause.render(self.screen)
        self.btn_reset.render(self.screen)
        self.btn_mode.render(self.screen)
        self.btn_export.render(self.screen)
        self.btn_open_folder.render(self.screen)

        if self.status_message:
            stat_col = COLOR_REACTIVE if self.status_is_error else (COLOR_PREDICTIVE if ("CSV" in self.status_message or "Folder" in self.status_message) else COLOR_NAVY)
            txt = f"• {self.status_message}"
            max_w = self.exp_action_rect.right - (self.btn_open_folder.rect.right + 24)
            font_to_use = FontManager.subheading_font
            if font_to_use.size(txt)[0] > max_w:
                font_to_use = FontManager.small_font
            draw_text(
                self.screen, txt,
                self.btn_open_folder.rect.right + 18, y_b + (10 if font_to_use == FontManager.small_font else 9),
                color=stat_col, font=font_to_use
            )

    # -------------------------------------------------------------
    # SECTION 4: HOW IT WORKS / VIVA DEFENSE WORKSPACE
    # -------------------------------------------------------------
    def _render_how_it_works_workspace(self) -> None:
        """Render clean engineering pipeline architecture, OpenCV stream, and mathematical formulations."""
        self._render_header()

        # 1. Pipeline Architecture Summary
        draw_card(self.screen, self.hiw_summary_rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)
        draw_text(
            self.screen, "END-TO-END ENGINEERING PIPELINE ARCHITECTURE",
            self.hiw_summary_rect.x + 14, self.hiw_summary_rect.y + 10,
            color=COLOR_NAVY, font=FontManager.title_font
        )
        draw_text(
            self.screen, "The system addresses sensor latency and visual occlusions through mathematical state estimation.",
            self.hiw_summary_rect.x + 14, self.hiw_summary_rect.y + 30,
            color=COLOR_MUTED, font=FontManager.small_font
        )

        steps = [
            ("1. OPENCV SENSING", "Captures delayed frames, applies HSV mask (Hue 90-130), calculates spatial moments for centroid."),
            ("2. DELAY QUEUE", "Models hardware camera transmission lag with a FIFO buffer (8 frames = 133ms delay)."),
            ("3. KALMAN PREDICTION", "Estimates 4D state [x, y, vx, vy] and projects trajectory +8 steps ahead into future space."),
            ("4. ROBOTIC CONTROL", "PID controller computes shortest-arc angular velocity commands to steer virtual turret."),
            ("5. EVALUATION", "Compares reactive error against predictive error with RMSE telemetry and lock percentage.")
        ]
        card_w = 236
        for idx, (title, desc) in enumerate(steps):
            c_rect = pygame.Rect(self.hiw_summary_rect.x + 14 + idx * 244, self.hiw_summary_rect.y + 54, card_w, 84)
            draw_card(self.screen, c_rect, bg_color=COLOR_PANEL_ALT, border_color=COLOR_BORDER, border_radius=6)
            draw_text(self.screen, title, c_rect.x + 10, c_rect.y + 8, color=COLOR_ACCENT_BLUE, font=FontManager.subheading_font)
            self._render_multiline_text(desc, c_rect.x + 10, c_rect.y + 26, card_w - 20, font=FontManager.small_font, color=COLOR_MUTED)

        # 2. Live OpenCV Camera Diagnostic Viewport
        draw_card(self.screen, self.hiw_cv_rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)
        draw_text(
            self.screen, "LIVE OPENCV CAMERA STREAM (PIP)",
            self.hiw_cv_rect.x + 14, self.hiw_cv_rect.y + 10,
            color=COLOR_NAVY, font=FontManager.heading_font
        )
        draw_badge(
            self.screen, "60 FPS HSV MASK",
            self.hiw_cv_rect.right - 14, self.hiw_cv_rect.y + 10,
            COLOR_PREDICTIVE, align_right=True
        )

        pip_surf = self.cv_tracker.get_pip_surface(width=440, height=300)
        pip_rect = pygame.Rect(self.hiw_cv_rect.x + 20, self.hiw_cv_rect.y + 40, 440, 300)
        self.screen.blit(pip_surf, pip_rect)
        pygame.draw.rect(self.screen, COLOR_BORDER, pip_rect, width=1, border_radius=4)

        notes_y = self.hiw_cv_rect.y + 354
        draw_text(self.screen, "OpenCV Processing Details:", self.hiw_cv_rect.x + 20, notes_y, color=COLOR_NAVY, font=FontManager.subheading_font)
        notes = [
            "• Color Segmentation: HSV InRange thresholding isolates target marker.",
            "• Spatial Moments: Centroid calculated via Cx = M10/M00, Cy = M01/M00.",
            "• Noise Resistance: Small contours (<30 px) filtered out as sensor noise.",
            "• Occlusion Robustness: Zero contours trigger Kalman coasting mode."
        ]
        for idx, line in enumerate(notes):
            draw_text(self.screen, line, self.hiw_cv_rect.x + 20, notes_y + 22 + idx * 22, color=COLOR_MUTED, font=FontManager.small_font)

        # 3. Mathematical Equations for Viva Defense
        draw_card(self.screen, self.hiw_math_rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)
        draw_text(
            self.screen, "MATHEMATICAL FORMULATIONS & CONTROL THEORY (VIVA GUIDE)",
            self.hiw_math_rect.x + 14, self.hiw_math_rect.y + 10,
            color=COLOR_NAVY, font=FontManager.heading_font
        )

        math_boxes = [
            ("1. Kalman State & Prediction Model",
             "State: x_k = [x, y, v_x, v_y]^T,   dt = 0.0167 s\n"
             "Predict: x_k|k-1 = F * x_k-1|k-1,   P_k|k-1 = F * P_k-1 * F^T + Q\n"
             "Future Extrapolation (+H steps): x_k+H = [x + H*dt*v_x,  y + H*dt*v_y]^T"),

            ("2. Joseph-Form Covariance Update (Numerical Stability)",
             "Innovation: y_k = z_k - H * x_k|k-1,   S_k = H * P_k|k-1 * H^T + R\n"
             "Kalman Gain: K_k = P_k|k-1 * H^T * S_k^-1\n"
             "Joseph Covariance: P_k|k = (I - K*H) * P_k|k-1 * (I - K*H)^T + K*R*K^T\n"
             "Maintains positive semi-definiteness under floating point roundoff errors."),

            ("3. Shortest-Arc 2-DOF Pan-and-Tilt PID Control",
             "Angular Error: e_theta = atan2(sin(theta_target - theta), cos(theta_target - theta))\n"
             "Control Output: u(t) = K_p * e(t) + K_i * Integral(e) + K_d * de/dt\n"
             "Safety Constraints: |Integral| <= 0.8 rad (anti-windup),  |w| <= 2.8 rad/s,  |alpha| <= 14.0 rad/s^2")
        ]

        m_y = self.hiw_math_rect.y + 42
        for title, formula in math_boxes:
            b_rect = pygame.Rect(self.hiw_math_rect.x + 18, m_y, self.hiw_math_rect.width - 36, 140)
            draw_card(self.screen, b_rect, bg_color=COLOR_PANEL_ALT, border_color=COLOR_BORDER, border_radius=6)
            draw_text(self.screen, title, b_rect.x + 12, b_rect.y + 8, color=COLOR_ACCENT_BLUE, font=FontManager.subheading_font)

            f_lines = formula.split("\n")
            for f_idx, line in enumerate(f_lines):
                draw_text(self.screen, line, b_rect.x + 14, b_rect.y + 30 + f_idx * 24, color=COLOR_NAVY, font=FontManager.code_font)

            m_y += 154

    # -------------------------------------------------------------
    # SECTION 5: FULL-SCREEN PRESENTATION MODE
    # -------------------------------------------------------------
    def _render_presentation_mode(self) -> None:
        """Render high-visibility view for classroom and projector presentations."""
        r_err_disp = self.reactive_errors[-1] if self.reactive_errors else 0.0
        p_err_disp = self.predictive_errors[-1] if self.predictive_errors else 0.0
        delayed_x = self.latest_cv_result.delayed_true_x if self.latest_cv_result else self.target.x
        delayed_y = self.latest_cv_result.delayed_true_y if self.latest_cv_result else self.target.y

        self.arena_renderer.set_rects(self.pres_left_arena, self.pres_right_arena)

        self.arena_renderer.render_reactive_arena(
            self.screen, self.reactive_turret,
            delayed_x=delayed_x, delayed_y=delayed_y,
            true_x=self.target.x, true_y=self.target.y,
            trail=self.target.get_trail(),
            error_deg=r_err_disp,
            latency_frames=self.cv_tracker.latency_frames
        )

        pred_x, pred_y, ellipse_params = self.kalman.predict_future(steps=self.prediction_steps)
        self.arena_renderer.render_predictive_arena(
            self.screen, self.predictive_turret,
            true_x=self.target.x, true_y=self.target.y,
            pred_x=pred_x, pred_y=pred_y,
            ellipse_params=ellipse_params,
            trail=self.target.get_trail(),
            error_deg=p_err_disp,
            steps_ahead=self.prediction_steps,
            is_coasting=self.kalman.is_coasting
        )

        self._render_primary_3_kpis(self.pres_kpi_rect)

        # Presentation Mode Bottom Banner
        draw_card(self.screen, self.pres_pipeline_rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=6)
        curr_f = min(self.frame_number, self.max_test_frames)
        pct = (curr_f / self.max_test_frames) * 100.0

        if self.is_trial_complete:
            pres_msg = f"TRIAL COMPLETE ({self.max_test_frames}f / 30.0s) — Locked | [R] New Trial | [O] Folder | [P] Dashboard"
            pres_col = COLOR_PREDICTIVE
        else:
            pres_msg = f"FACULTY PRESENTATION MODE — TRIAL: {curr_f}/{self.max_test_frames}f ({pct:.0f}%) | [SPACE] Pause | [R] Reset | [P] Normal Dashboard"
            pres_col = COLOR_NAVY

        draw_text(
            self.screen, pres_msg,
            self.pres_pipeline_rect.centerx, self.pres_pipeline_rect.centery,
            color=pres_col, font=FontManager.subheading_font, align_center=True
        )

    # -------------------------------------------------------------
    # COMMON SUB-COMPONENTS
    # -------------------------------------------------------------
    def _render_header(self) -> None:
        """Render top navigation header."""
        draw_card(self.screen, self.header_rect, bg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_radius=8)

        draw_text(
            self.screen, "PREDICTIVE OBJECT-TRACKING TURRET",
            self.header_rect.x + 14, self.header_rect.y + 7,
            color=COLOR_NAVY, font=FontManager.title_font
        )
        draw_text(
            self.screen, "COMPUTER VISION + MOTION PREDICTION + ROBOTIC CONTROL",
            self.header_rect.x + 14, self.header_rect.y + 26,
            color=COLOR_MUTED, font=FontManager.small_font
        )

        # Segmented Navigation Tabs
        self.tab_bar.render(self.screen)

        # Mode Badge
        curr_m_name = TRAJECTORY_MODES.get(self.target.mode, "").split(":")[-1].strip()
        mode_badge = draw_badge(
            self.screen, f"MODE {self.target.mode}: {curr_m_name.upper()}",
            self.header_rect.right - 14, self.header_rect.y + 12,
            COLOR_ACCENT_BLUE, align_right=True
        )

        # Trial Progress Badge
        curr_f = min(self.frame_number, self.max_test_frames)
        pct = (curr_f / self.max_test_frames) * 100.0
        if self.is_trial_complete:
            prog_txt = "TRIAL COMPLETE (1800f)"
            prog_col = COLOR_PREDICTIVE
        else:
            prog_txt = f"TRIAL: {curr_f}/{self.max_test_frames}f ({pct:.0f}%)"
            prog_col = COLOR_SENSOR_DELAY if self.is_paused else COLOR_ACCENT_BLUE

        draw_badge(
            self.screen, prog_txt,
            mode_badge.left - 10, self.header_rect.y + 12,
            prog_col, align_right=True
        )

    def _render_kpi_cards(self, rect: pygame.Rect) -> None:
        """Render 4 detailed diagnostic metric cards for the Simulation workspace."""
        card_w = (rect.width - 36) // 4
        card_h = rect.height

        n_samples = len(self.reactive_errors)
        if n_samples > 15:
            r_mean = float(np.mean(self.reactive_errors))
            p_mean = float(np.mean(self.predictive_errors))
            reduction = ((r_mean - p_mean) / r_mean * 100.0) if r_mean > 1e-3 else 0.0
            r_lock_pct = (self.reactive_locked_frames / n_samples) * 100.0
            p_lock_pct = (self.predictive_locked_frames / n_samples) * 100.0

            r_mean_str = f"{r_mean:.2f}°"
            p_mean_str = f"{p_mean:.2f}°"
            red_str = f"{reduction:+.1f}%"
            lock_str = f"{p_lock_pct:.0f}% vs {r_lock_pct:.0f}%"
        else:
            r_mean_str = "Collecting..."
            p_mean_str = "Collecting..."
            red_str = "—"
            lock_str = "—"

        cards_data = [
            ("Reactive Mean Error", r_mean_str, "Lagged Camera Observation", COLOR_REACTIVE, None),
            ("Predictive Mean Error", p_mean_str, "Kalman Compensated Target", COLOR_PREDICTIVE, None),
            ("Error Reduction", red_str, "Lead Tracking Accuracy Gain", COLOR_PREDICTIVE if red_str.startswith("+") else COLOR_REACTIVE, "ACCURACY GAIN"),
            ("Lock Retention (≤3°)", lock_str, "Predictive vs Reactive Lock", COLOR_ACCENT_BLUE, "FIRING READY")
        ]

        for i, (title, val, sub, col, badge) in enumerate(cards_data):
            c_rect = pygame.Rect(rect.x + i * (card_w + 12), rect.y, card_w, card_h)
            draw_kpi_card(self.screen, c_rect, title, val, sub, col, badge)

    def _render_multiline_text(
        self, text: str, x: int, y: int, max_w: int,
        font: pygame.font.Font, color: Tuple[int, int, int]
    ) -> None:
        """Render multi-line wrapped text."""
        words = text.split(" ")
        lines = []
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            w, _ = font.size(test_line)
            if w <= max_w:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))

        for idx, line in enumerate(lines):
            draw_text(self.screen, line, x, y + idx * 14, color=color, font=font)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predictive Object-Tracking Turret Lab")
    parser.add_argument("pos_arg", nargs="?", type=str, default=None, help="Mode (1-5) or Workspace (1-4)")
    parser.add_argument("--mode", "-m", type=int, default=None, help="Trajectory mode (1: Bounce, 2: ZigZag, 3: Wave, 4: Orbit, 5: Cursor)")
    parser.add_argument("--workspace", "-w", type=int, default=None, help="Workspace index (1: Overview, 2: Simulation, 3: Experiments, 4: How It Works)")
    args = parser.parse_args()

    mode = args.mode
    ws = (args.workspace - 1) if (args.workspace is not None and 1 <= args.workspace <= 4) else WORKSPACE_OVERVIEW

    if args.pos_arg is not None:
        val = int(args.pos_arg) if args.pos_arg.isdigit() else 1
        if mode is None:
            mode = val

    app = PredictiveTurretProductApp(initial_mode=mode or 1, initial_workspace=ws)
    app.run()