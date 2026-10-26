
import pygame
import cv2
import numpy as np
import math
import csv
from datetime import datetime

# =====================================================
# 1. INITIALIZATION
# =====================================================

pygame.init()

WIDTH, HEIGHT = 1200, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "Predictive Tracking Lab | FCV + Intelligent Robotics"
)

BACKGROUND = (18, 24, 38)
PANEL = (30, 40, 58)
WHITE = (240, 244, 248)
GREEN = (50, 220, 150)
BLUE = (50, 150, 255)
YELLOW = (255, 210, 60)
GREY = (130, 145, 165)
RED = (255, 105, 115)
PURPLE = (170, 130, 255)

title_font = pygame.font.SysFont("consolas", 27)
heading_font = pygame.font.SysFont("consolas", 18)
small_font = pygame.font.SysFont("consolas", 14)

clock = pygame.time.Clock()

# =====================================================
# 2. EXPERIMENT SETTINGS
# =====================================================

TEST_FPS = 60
TEST_DURATION_SECONDS = 30
TEST_FRAMES = TEST_FPS * TEST_DURATION_SECONDS

DELAY_FRAMES = 8
PREDICTION_STEPS = DELAY_FRAMES

LEFT, RIGHT = 55, 1145
TOP, BOTTOM = 205, 510
BALL_RADIUS = 12

INITIAL_X = 150.0
INITIAL_Y = 350.0
INITIAL_VX = 4.0
INITIAL_VY = 2.5

movement_mode = 1

# Stores completed experiment summaries in memory.
# Results remain available when switching modes.
mode_results = {
    1: None,
    2: None
}

# =====================================================
# 3. COMPUTER VISION SETTINGS
# =====================================================

LOWER_BLUE = np.array([90, 80, 80])
UPPER_BLUE = np.array([130, 255, 255])

vision_surface = pygame.Surface((WIDTH, HEIGHT))

# =====================================================
# 4. KALMAN FILTER
# State = [x, y, vx, vy]
# Measurement = [x, y]
# =====================================================

def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    kf.processNoiseCov = (
        np.eye(4, dtype=np.float32) * 0.03
    )

    kf.measurementNoiseCov = (
        np.eye(2, dtype=np.float32) * 1.0
    )

    kf.errorCovPost = np.eye(4, dtype=np.float32)

    kf.statePost = np.array([
        [INITIAL_X],
        [INITIAL_Y],
        [0.0],
        [0.0]
    ], dtype=np.float32)

    return kf


kalman = create_kalman_filter()

# =====================================================
# 5. PID CONTROLLER
# =====================================================

KP = 3.0
KI = 0.015
KD = 0.025

MAX_ANGULAR_SPEED = 2.5
INTEGRAL_LIMIT = 0.8
DERIVATIVE_SMOOTHING = 0.15


class PIDTracker:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.angle = -math.pi / 2
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0

    def reset(self):
        self.angle = -math.pi / 2
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0

    def update(self, target_x, target_y, dt):
        desired_angle = math.atan2(
            target_y - self.y,
            target_x - self.x
        )

        error = (
            desired_angle - self.angle + math.pi
        ) % (2 * math.pi) - math.pi

        self.integral += error * dt
        self.integral = max(
            -INTEGRAL_LIMIT,
            min(INTEGRAL_LIMIT, self.integral)
        )

        raw_derivative = (
            error - self.previous_error
        ) / max(dt, 0.001)

        self.filtered_derivative += (
            DERIVATIVE_SMOOTHING
            * (raw_derivative - self.filtered_derivative)
        )

        output = (
            KP * error
            + KI * self.integral
            + KD * self.filtered_derivative
        )

        output = max(
            -MAX_ANGULAR_SPEED,
            min(MAX_ANGULAR_SPEED, output)
        )

        self.angle += output * dt
        self.previous_error = error

    def aiming_error(self, target_x, target_y):
        desired_angle = math.atan2(
            target_y - self.y,
            target_x - self.x
        )

        error = (
            desired_angle - self.angle + math.pi
        ) % (2 * math.pi) - math.pi

        return abs(math.degrees(error))


reactive_turret = PIDTracker(307, 545)
predictive_turret = PIDTracker(892, 545)

# =====================================================
# 6. SIMULATION STATE
# =====================================================

ball_x = INITIAL_X
ball_y = INITIAL_Y
velocity_x = INITIAL_VX
velocity_y = INITIAL_VY

position_history = [
    (ball_x, ball_y)
] * (DELAY_FRAMES + 1)

reactive_errors = []
predictive_errors = []

frame_number = 0
test_complete = False
status_message = "RUNNING | M: Change mode | R: Reset | E: Export"

# =====================================================
# 7. DRAWING HELPERS
# =====================================================

def draw_text(message, x, y, colour=WHITE, font=small_font):
    rendered = font.render(str(message), True, colour)
    screen.blit(rendered, (x, y))


def map_to_panel(world_x, world_y, panel_x):
    x_fraction = (world_x - LEFT) / (RIGHT - LEFT)
    y_fraction = (world_y - TOP) / (BOTTOM - TOP)

    screen_x = panel_x + 20 + x_fraction * 525
    screen_y = 185 + y_fraction * 315

    return int(screen_x), int(screen_y)


def draw_turret(turret, colour):
    length = 55

    end_x = turret.x + math.cos(turret.angle) * length
    end_y = turret.y + math.sin(turret.angle) * length

    pygame.draw.circle(
        screen, colour,
        (int(turret.x), int(turret.y)), 14
    )

    pygame.draw.line(
        screen, colour,
        (int(turret.x), int(turret.y)),
        (int(end_x), int(end_y)), 7
    )

    pygame.draw.circle(
        screen, WHITE,
        (int(end_x), int(end_y)), 4
    )


def draw_error_graph(rect, reactive_values, predictive_values):
    pygame.draw.rect(screen, PANEL, rect, border_radius=10)

    draw_text(
        "ANGULAR ERROR OVER TIME",
        rect.x + 18, rect.y + 10, WHITE, heading_font
    )

    graph = pygame.Rect(
        rect.x + 55,
        rect.y + 48,
        rect.width - 75,
        rect.height - 65
    )

    pygame.draw.rect(
        screen, BACKGROUND, graph, border_radius=5
    )

    for i in range(5):
        value = 45 - i * 11.25
        y = graph.y + i * graph.height // 4

        pygame.draw.line(
            screen, GREY,
            (graph.x, y), (graph.right, y), 1
        )

        draw_text(
            f"{value:.1f}",
            rect.x + 3, y - 7, GREY
        )

    draw_text(
        "RED: REACTIVE",
        graph.x + 10, graph.y + 5, RED
    )

    draw_text(
        "GREEN: PREDICTIVE",
        graph.x + 180, graph.y + 5, GREEN
    )

    if len(reactive_values) < 2:
        draw_text(
            "Collecting measurements...",
            graph.x + 10, graph.y + 30, WHITE
        )
        return

    reactive_values = reactive_values[-300:]
    predictive_values = predictive_values[-300:]

    def make_points(values):
        points = []

        for i, value in enumerate(values):
            x = graph.x + int(
                i * graph.width / max(len(values) - 1, 1)
            )

            clipped = max(0.0, min(45.0, value))

            y = graph.bottom - int(
                clipped / 45.0 * graph.height
            )

            points.append((x, y))

        return points

    pygame.draw.lines(
        screen, RED, False,
        make_points(reactive_values), 2
    )

    pygame.draw.lines(
        screen, GREEN, False,
        make_points(predictive_values), 2
    )


# =====================================================
# 8. SAVE COMPLETED MODE SUMMARY
# =====================================================

def save_mode_summary():
    if not reactive_errors or not predictive_errors:
        return

    reactive_mean = float(np.mean(reactive_errors))
    predictive_mean = float(np.mean(predictive_errors))

    improvement = (
        (reactive_mean - predictive_mean)
        / reactive_mean * 100
        if reactive_mean > 0.01 else 0.0
    )

    mode_results[movement_mode] = {
        "reactive_mean": reactive_mean,
        "predictive_mean": predictive_mean,
        "improvement": improvement,
        "frames": frame_number
    }


# =====================================================
# 9. RESET EXPERIMENT
# =====================================================

def reset_experiment():
    global ball_x, ball_y, velocity_x, velocity_y
    global position_history, kalman
    global reactive_errors, predictive_errors
    global frame_number, test_complete, status_message

    ball_x, ball_y = INITIAL_X, INITIAL_Y
    velocity_x, velocity_y = INITIAL_VX, INITIAL_VY

    position_history = [
        (ball_x, ball_y)
    ] * (DELAY_FRAMES + 1)

    kalman = create_kalman_filter()

    reactive_turret.reset()
    predictive_turret.reset()

    reactive_errors = []
    predictive_errors = []

    frame_number = 0
    test_complete = False

    status_message = (
        "RUNNING | M: Change mode | R: Reset | E: Export"
    )


# =====================================================
# 10. CSV EXPORT
# =====================================================

def export_csv():
    global status_message

    if not reactive_errors:
        status_message = "No measurements available to export"
        return

    filename = (
        "tracking_mode"
        + str(movement_mode)
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".csv"
    )

    try:
        with open(
            filename, "w",
            newline="", encoding="utf-8"
        ) as csv_file:

            writer = csv.writer(csv_file)

            writer.writerow([
                "Mode",
                "Frame",
                "Time_Seconds",
                "Reactive_Angular_Error_Deg",
                "Predictive_Angular_Error_Deg"
            ])

            for index, (reactive, predictive) in enumerate(
                zip(reactive_errors, predictive_errors),
                start=1
            ):
                writer.writerow([
                    movement_mode,
                    index,
                    round(index / TEST_FPS, 4),
                    round(reactive, 4),
                    round(predictive, 4)
                ])

            reactive_mean = float(np.mean(reactive_errors))
            predictive_mean = float(np.mean(predictive_errors))

            improvement = (
                (reactive_mean - predictive_mean)
                / reactive_mean * 100
                if reactive_mean > 0.01 else 0.0
            )

            writer.writerow([])
            writer.writerow(["SUMMARY"])
            writer.writerow(["Mode", movement_mode])
            writer.writerow([
                "Reactive_Mean_Error_Deg",
                round(reactive_mean, 4)
            ])
            writer.writerow([
                "Predictive_Mean_Error_Deg",
                round(predictive_mean, 4)
            ])
            writer.writerow([
                "Mean_Error_Reduction_Percent",
                round(improvement, 2)
            ])

        status_message = "CSV saved: " + filename

    except OSError as error:
        status_message = "CSV export failed: " + str(error)


# =====================================================
# 11. EXPERIMENT COMPARISON PANEL
# =====================================================

def draw_comparison_panel():
    rect = pygame.Rect(25, 780, 1150, 105)

    pygame.draw.rect(
        screen, PANEL, rect, border_radius=10
    )

    draw_text(
        "COMPLETED EXPERIMENT COMPARISON",
        42, 790, WHITE, heading_font
    )

    draw_text(
        "MODE",
        45, 823, GREY
    )

    draw_text(
        "REACTIVE",
        205, 823, RED
    )

    draw_text(
        "PREDICTIVE",
        405, 823, GREEN
    )

    draw_text(
        "ERROR REDUCTION",
        640, 823, PURPLE
    )

    for index, mode in enumerate([1, 2]):
        y = 845 + index * 20
        result = mode_results[mode]

        mode_name = (
            "1 - Normal" if mode == 1
            else "2 - Sudden changes"
        )

        draw_text(mode_name, 45, y, YELLOW)

        if result is None:
            draw_text("Not completed", 205, y, GREY)
            draw_text("Not completed", 405, y, GREY)
            draw_text("--", 640, y, GREY)
        else:
            draw_text(
                f"{result['reactive_mean']:.2f} deg",
                205, y, RED
            )

            draw_text(
                f"{result['predictive_mean']:.2f} deg",
                405, y, GREEN
            )

            draw_text(
                f"{result['improvement']:.1f}%",
                640, y, PURPLE
            )

            if result["frames"] != TEST_FRAMES:
                draw_text(
                    "(incomplete)", 790, y, YELLOW
                )


# =====================================================
# 12. MAIN LOOP
# =====================================================

running = True

while running:

    clock.tick(TEST_FPS)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_r:
                reset_experiment()

            elif event.key == pygame.K_m:
                movement_mode = (
                    2 if movement_mode == 1 else 1
                )
                reset_experiment()

            elif event.key == pygame.K_e:
                export_csv()

    # -------------------------------------------------
    # UPDATE SIMULATION
    # -------------------------------------------------

    if not test_complete:

        # Sudden direction changes every 300 frames in Mode 2.
        if movement_mode == 2 and frame_number > 0:
            if frame_number % 300 == 0:
                velocity_x, velocity_y = (
                    -velocity_y, velocity_x
                )

        ball_x += velocity_x
        ball_y += velocity_y

        if ball_x + BALL_RADIUS >= RIGHT:
            ball_x = RIGHT - BALL_RADIUS
            velocity_x = -abs(velocity_x)

        elif ball_x - BALL_RADIUS <= LEFT:
            ball_x = LEFT + BALL_RADIUS
            velocity_x = abs(velocity_x)

        if ball_y + BALL_RADIUS >= BOTTOM:
            ball_y = BOTTOM - BALL_RADIUS
            velocity_y = -abs(velocity_y)

        elif ball_y - BALL_RADIUS <= TOP:
            ball_y = TOP + BALL_RADIUS
            velocity_y = abs(velocity_y)

        frame_number += 1

        position_history.append((ball_x, ball_y))

        delayed_x, delayed_y = position_history[
            -1 - DELAY_FRAMES
        ]

        # Simulate a camera frame for OpenCV.
        vision_surface.fill((18, 24, 38))

        pygame.draw.circle(
            vision_surface, BLUE,
            (int(delayed_x), int(delayed_y)),
            BALL_RADIUS
        )

        frame_rgb = pygame.surfarray.array3d(vision_surface)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))

        frame_hsv = cv2.cvtColor(
            frame_rgb, cv2.COLOR_RGB2HSV
        )

        mask = cv2.inRange(
            frame_hsv, LOWER_BLUE, UPPER_BLUE
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detected_x = None
        detected_y = None

        if contours:
            largest = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest) > 50:
                moments = cv2.moments(largest)

                if moments["m00"] != 0:
                    detected_x = (
                        moments["m10"] / moments["m00"]
                    )
                    detected_y = (
                        moments["m01"] / moments["m00"]
                    )

        # Kalman correction and prediction.
        if detected_x is not None and detected_y is not None:
            measurement = np.array([
                [detected_x],
                [detected_y]
            ], dtype=np.float32)

            kalman.correct(measurement)

        state = kalman.predict()

        estimated_x = float(state[0, 0])
        estimated_y = float(state[1, 0])
        estimated_vx = float(state[2, 0])
        estimated_vy = float(state[3, 0])

        future_x = (
            estimated_x + estimated_vx * PREDICTION_STEPS
        )

        future_y = (
            estimated_y + estimated_vy * PREDICTION_STEPS
        )

        future_x = max(LEFT, min(RIGHT, future_x))
        future_y = max(TOP, min(BOTTOM, future_y))

        reference_x = (
            detected_x if detected_x is not None else delayed_x
        )

        reference_y = (
            detected_y if detected_y is not None else delayed_y
        )

        reactive_target = map_to_panel(
            reference_x, reference_y, 25
        )

        predictive_target = map_to_panel(
            future_x, future_y, 610
        )

        reactive_actual = map_to_panel(
            ball_x, ball_y, 25
        )

        predictive_actual = map_to_panel(
            ball_x, ball_y, 610
        )

        dt = 1.0 / TEST_FPS

        reactive_turret.update(
            reactive_target[0], reactive_target[1], dt
        )

        predictive_turret.update(
            predictive_target[0], predictive_target[1], dt
        )

        reactive_error = reactive_turret.aiming_error(
            reactive_actual[0], reactive_actual[1]
        )

        predictive_error = predictive_turret.aiming_error(
            predictive_actual[0], predictive_actual[1]
        )

        reactive_errors.append(reactive_error)
        predictive_errors.append(predictive_error)

        if frame_number >= TEST_FRAMES:
            test_complete = True
            save_mode_summary()

            status_message = (
                "TEST COMPLETE | M: Change mode | "
                "R: New trial | E: Export CSV"
            )

    # =================================================
    # 13. LIVE METRICS
    # =================================================

    reactive_mean = (
        float(np.mean(reactive_errors))
        if reactive_errors else 0.0
    )

    predictive_mean = (
        float(np.mean(predictive_errors))
        if predictive_errors else 0.0
    )

    reactive_max = (
        float(np.max(reactive_errors))
        if reactive_errors else 0.0
    )

    predictive_max = (
        float(np.max(predictive_errors))
        if predictive_errors else 0.0
    )

    improvement = (
        (reactive_mean - predictive_mean)
        / reactive_mean * 100
        if reactive_mean > 0.01 else 0.0
    )

    # =================================================
    # 14. DRAW DASHBOARD
    # =================================================

    screen.fill(BACKGROUND)

    draw_text(
        "PREDICTIVE TRACKING LAB",
        30, 20, WHITE, title_font
    )

    draw_text(
        "FOUNDATION OF COMPUTER VISION | INTELLIGENT ROBOTICS",
        32, 60, GREEN
    )

    pygame.draw.rect(
        screen, PANEL,
        (25, 100, 565, 470),
        border_radius=12
    )

    pygame.draw.rect(
        screen, PANEL,
        (610, 100, 565, 470),
        border_radius=12
    )

    draw_text(
        "REACTIVE TRACKER",
        45, 115, RED, heading_font
    )

    draw_text(
        "PREDICTIVE TRACKER",
        630, 115, GREEN, heading_font
    )

    draw_text(
        f"Observation delay: {DELAY_FRAMES} frames",
        45, 145
    )

    draw_text(
        f"Prediction horizon: {PREDICTION_STEPS} frames",
        630, 145
    )

    draw_text(
        "MODE 1: NORMAL" if movement_mode == 1
        else "MODE 2: SUDDEN DIRECTION CHANGES",
        45, 170, YELLOW
    )

    reactive_actual = map_to_panel(ball_x, ball_y, 25)
    predictive_actual = map_to_panel(ball_x, ball_y, 610)

    delayed_position = position_history[
        -1 - DELAY_FRAMES
    ]

    reactive_reference = map_to_panel(
        delayed_position[0],
        delayed_position[1],
        25
    )

    pygame.draw.circle(
        screen, BLUE, reactive_actual, BALL_RADIUS
    )

    pygame.draw.circle(
        screen, BLUE, predictive_actual, BALL_RADIUS
    )

    pygame.draw.circle(
        screen, YELLOW, reactive_reference, 19, 2
    )

    pygame.draw.circle(
        screen, GREEN, predictive_target, 12, 2
    )

    pygame.draw.line(
        screen, GREY,
        (int(reactive_turret.x), int(reactive_turret.y)),
        reactive_actual, 1
    )

    pygame.draw.line(
        screen, GREY,
        (int(predictive_turret.x), int(predictive_turret.y)),
        predictive_actual, 1
    )

    draw_turret(reactive_turret, RED)
    draw_turret(predictive_turret, PURPLE)

    # Live performance panel
    pygame.draw.rect(
        screen, PANEL,
        (25, 590, 1150, 115),
        border_radius=12
    )

    draw_text(
        "LIVE EXPERIMENT PERFORMANCE",
        45, 603, WHITE, heading_font
    )

    draw_text(
        f"Reactive mean error: {reactive_mean:.2f} deg",
        45, 634, RED
    )

    draw_text(
        f"Predictive mean error: {predictive_mean:.2f} deg",
        45, 658, GREEN
    )

    draw_text(
        f"Error reduction: {improvement:.1f}%",
        630, 634, PURPLE
    )

    draw_text(
        f"Maximum errors: R={reactive_max:.1f} deg | "
        f"P={predictive_max:.1f} deg",
        630, 658, WHITE
    )

    # Progress and controls
    progress = frame_number / TEST_FRAMES

    pygame.draw.rect(
        screen, PANEL,
        (25, 713, 1150, 40),
        border_radius=8
    )

    draw_text(
        f"FRAME: {frame_number}/{TEST_FRAMES}",
        40, 725, YELLOW
    )

    draw_text(
        f"PROGRESS: {progress * 100:.1f}%",
        300, 725, YELLOW
    )

    draw_text("M: CHANGE MODE", 540, 725, YELLOW)
    draw_text("R: RESET", 760, 725, YELLOW)
    draw_text("E: EXPORT CSV", 900, 725, YELLOW)

    draw_text(
        status_message,
        40, 755,
        GREEN if test_complete else WHITE
    )

    # Completed results from both modes
    draw_comparison_panel()

    # Error history graph
    draw_error_graph(
        pygame.Rect(25, 900, 1150, 160),
        reactive_errors,
        predictive_errors
    )

    pygame.display.flip()

pygame.quit()
