"""
Central Configuration and Parameter Definitions.
Predictive Object-Tracking Turret — CV + IR Integration.
"""

from pathlib import Path
import numpy as np

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
BACKUP_DIR = BASE_DIR / "backup_stage_a"
EXPERIMENT_JSON = BASE_DIR / "experiment_results.json"

# Ensure runtime directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Display & Screen Layout (Optimized for standard 1080p and 768p displays)
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 760
FPS = 60
DT = 1.0 / FPS

# Navigation Workspace Indices
WORKSPACE_OVERVIEW = 0
WORKSPACE_SIMULATION = 1
WORKSPACE_EXPERIMENTS = 2
WORKSPACE_HOW_IT_WORKS = 3

WORKSPACE_TABS = [
    "OVERVIEW",
    "SIMULATION",
    "EXPERIMENTS",
    "HOW IT WORKS"
]

# Color Palette: Premium Scientific Light Theme
COLOR_BG = (244, 246, 250)            # Soft off-white canvas (#F4F6FA)
COLOR_PANEL = (255, 255, 255)         # Crisp pure white surface (#FFFFFF)
COLOR_PANEL_ALT = (248, 250, 252)     # Subtle recessed light surface (#F8FAFC)
COLOR_BORDER = (226, 232, 240)        # Clean structural border (#E2E8F0)
COLOR_BORDER_LIGHT = (203, 213, 225)  # Focused / active border (#CBD5E1)

COLOR_NAVY = (15, 23, 42)             # Deep navy text (#0F172A)
COLOR_TEXT_PRIMARY = (15, 23, 42)     # Primary text
COLOR_WHITE = (15, 23, 42)            # Alias for text callers expecting primary text
COLOR_PURE_WHITE = (255, 255, 255)    # Pure white highlight/accent
COLOR_GREY = (100, 116, 139)          # Secondary slate (#64748B)
COLOR_MUTED = (100, 116, 139)         # Secondary labels (#64748B)
COLOR_TEXT_MUTED = (100, 116, 139)
COLOR_DARK_GREY = (203, 213, 225)     # Subtle dividers and track backgrounds (#CBD5E1)

# Semantic Functional Colors
COLOR_REACTIVE = (225, 29, 72)        # Rose Coral (#E11D48) - Reactive / Delayed
COLOR_PREDICTIVE = (5, 150, 105)      # Emerald Precision (#059669) - Compensated
COLOR_TARGET = (2, 132, 199)          # Electric Sky Blue (#0284C7) - Ground Truth
COLOR_SENSOR_DELAY = (217, 119, 6)    # Solar Amber (#D97706) - Delayed Observation
COLOR_ACCENT_BLUE = (37, 99, 235)     # Primary Blue Accent (#2563EB)
COLOR_PRIMARY_BLUE = (37, 99, 235)

COLOR_AMBER = (217, 119, 6)
COLOR_EMERALD = (5, 150, 105)
COLOR_RED = (225, 29, 72)
COLOR_CYAN = (2, 132, 199)
COLOR_PURPLE = (124, 58, 237)
COLOR_BLUE = (37, 99, 235)

# Simulation Arena Field Colors
COLOR_ARENA_BG = (250, 252, 255)      # Clean light arena canvas (#FAFCFF)
COLOR_ARENA_GRID = (235, 241, 248)    # Subtle spatial gridlines (#EBF1F8)

# Simulation Arena Bounds (Normalized World Coordinates)
WORLD_LEFT = 50.0
WORLD_RIGHT = 1150.0
WORLD_TOP = 150.0
WORLD_BOTTOM = 550.0
BALL_RADIUS = 12.0

# Initial Target Kinematics
INITIAL_X = 160.0
INITIAL_Y = 350.0
INITIAL_VX = 4.5
INITIAL_VY = 3.0

# Latency and Extrapolation Defaults
DEFAULT_DELAY_FRAMES = 8
DEFAULT_PREDICTION_STEPS = 8
MAX_LATENCY_FRAMES = 20

# Experiment Duration
DEFAULT_DURATION_SECONDS = 30
DEFAULT_TOTAL_FRAMES = FPS * DEFAULT_DURATION_SECONDS

# Computer Vision (CV) Parameters
HSV_LOWER_BLUE = np.array([90, 80, 80], dtype=np.uint8)
HSV_UPPER_BLUE = np.array([130, 255, 255], dtype=np.uint8)
CV_MIN_CONTOUR_AREA = 30.0
CV_CROP_SIZE = 120

# Kalman Filter Defaults
DEFAULT_PROCESS_NOISE_COV = 0.04
DEFAULT_MEASUREMENT_NOISE_COV = 1.2

# PID Controller Parameters
DEFAULT_KP = 3.2
DEFAULT_KI = 0.02
DEFAULT_KD = 0.035
MAX_ANGULAR_SPEED = 2.8      # rad/s (approx 160 deg/s)
MAX_ANGULAR_ACCEL = 14.0     # rad/s^2 (physical motor torque limit)
INTEGRAL_LIMIT = 0.8         # rad anti-windup clamping
DERIVATIVE_SMOOTHING = 0.15  # Low-pass filter coefficient for derivative

# Turret Dimensions and Anchor Offsets
TURRET_BARREL_LENGTH = 44.0
TURRET_BASE_RADIUS = 14.0

# Trajectory Mode Names
TRAJECTORY_MODES = {
    1: "Mode 1: Linear Bounce",
    2: "Mode 2: Evasive Zig-Zag",
    3: "Mode 3: Harmonic Wave",
    4: "Mode 4: Circular Orbit",
    5: "Mode 5: Interactive Cursor"
}