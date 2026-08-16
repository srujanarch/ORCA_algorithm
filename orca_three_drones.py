"""
===============================================================
ORCA THREE-DRONE ADVANCED SWARM SIMULATOR
===============================================================

Standalone 2-D autonomous swarm simulation.

Features
--------
1. Three autonomous drones
2. RVO2-style ORCA
3. LP1 / LP2 / LP3
4. Drone-drone collision avoidance
5. Static circular obstacles
6. Moving circular obstacles
7. Waypoint navigation
8. Maximum velocity constraint
9. Maximum acceleration constraint
10. Smooth drone motion
11. Drone heading
12. Communication radius
13. Communication-link visualization
14. Battery / energy estimation
15. Collision detection
16. Minimum separation monitoring
17. Goal completion
18. Velocity vectors
19. Real-time telemetry
20. GIF animation

Requirements
------------
numpy
matplotlib
pillow

Install if necessary:

    pip install numpy matplotlib pillow

Run:

    python3 orca_three_drones.py

Output:

    orca_advanced_three_drones.gif
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import (
    FuncAnimation,
    PillowWriter
)


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

EPSILON = 1e-8

DT = 0.05

TIME_HORIZON = 3.0

MAX_SIMULATION_TIME = 45.0

GOAL_TOLERANCE = 0.30

OUTPUT_FILE = "orca_advanced_three_drones.gif"

FPS = 20


# ============================================================
# VECTOR FUNCTIONS
# ============================================================

def det(v, w):
    """
    2-D determinant.

    det(v,w) =
        vx * wy - vy * wx
    """

    return (
        v[0] * w[1]
        -
        v[1] * w[0]
    )


def abs_sq(v):
    """
    Squared magnitude.
    """

    return float(
        np.dot(v, v)
    )


def normalize(v):
    """
    Normalize a 2-D vector.
    """

    length = np.linalg.norm(v)

    if length < EPSILON:
        return np.zeros(2)

    return v / length


def clamp_magnitude(v, maximum):
    """
    Limit vector magnitude.
    """

    length = np.linalg.norm(v)

    if length <= maximum:
        return v.copy()

    if length < EPSILON:
        return np.zeros(2)

    return (
        v / length
        * maximum
    )


# ============================================================
# ORCA LINE
# ============================================================

class Line:
    """
    ORCA half-plane.

    The permitted region lies on the left side
    of the directed line.

    Constraint:

        det(direction, point - velocity) <= 0
    """

    def __init__(
        self,
        point,
        direction
    ):

        self.point = np.asarray(
            point,
            dtype=float
        )

        self.direction = np.asarray(
            direction,
            dtype=float
        )


# ============================================================
# DRONE
# ============================================================

class Drone:

    def __init__(
        self,
        drone_id,
        position,
        waypoints,
        radius=0.45,
        max_speed=2.0,
        max_acceleration=2.5,
        communication_radius=5.0
    ):

        self.id = drone_id

        self.position = np.asarray(
            position,
            dtype=float
        )

        self.waypoints = [
            np.asarray(
                waypoint,
                dtype=float
            )
            for waypoint in waypoints
        ]

        self.current_waypoint = 0

        self.radius = radius

        self.max_speed = max_speed

        self.max_acceleration = (
            max_acceleration
        )

        self.communication_radius = (
            communication_radius
        )

        self.velocity = np.zeros(2)

        self.heading = 0.0

        self.reached_goal = False

        # ----------------------------------------------------
        # Battery
        # ----------------------------------------------------

        self.battery = 100.0

        self.energy_used = 0.0

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        self.history = [
            self.position.copy()
        ]

        self.velocity_history = [
            self.velocity.copy()
        ]

        self.heading_history = [
            self.heading
        ]

        self.battery_history = [
            self.battery
        ]

    # ========================================================
    # CURRENT GOAL
    # ========================================================

    @property
    def goal(self):

        if (
            self.current_waypoint
            >= len(self.waypoints)
        ):

            return self.waypoints[-1]

        return self.waypoints[
            self.current_waypoint
        ]

    # ========================================================
    # PREFERRED VELOCITY
    # ========================================================

    def preferred_velocity(self):

        if self.reached_goal:

            return np.zeros(2)

        target = self.goal

        displacement = (
            target
            - self.position
        )

        distance = np.linalg.norm(
            displacement
        )

        # ----------------------------------------------------
        # Waypoint reached
        # ----------------------------------------------------

        if distance <= GOAL_TOLERANCE:

            if (
                self.current_waypoint
                <
                len(self.waypoints) - 1
            ):

                self.current_waypoint += 1

                target = self.goal

                displacement = (
                    target
                    - self.position
                )

                distance = np.linalg.norm(
                    displacement
                )

            else:

                self.reached_goal = True

                return np.zeros(2)

        if distance < EPSILON:

            return np.zeros(2)

        # ----------------------------------------------------
        # Slow down near waypoint
        # ----------------------------------------------------

        speed = min(
            self.max_speed,
            distance * 1.2
        )

        return (
            displacement
            / distance
            * speed
        )

    # ========================================================
    # APPLY ACCELERATION LIMIT
    # ========================================================

    def apply_motion_model(
        self,
        desired_velocity,
        dt
    ):

        velocity_change = (
            desired_velocity
            - self.velocity
        )

        max_change = (
            self.max_acceleration
            * dt
        )

        velocity_change = (
            clamp_magnitude(
                velocity_change,
                max_change
            )
        )

        new_velocity = (
            self.velocity
            + velocity_change
        )

        new_velocity = (
            clamp_magnitude(
                new_velocity,
                self.max_speed
            )
        )

        self.velocity = new_velocity

    # ========================================================
    # UPDATE STATE
    # ========================================================

    def update_state(
        self,
        dt
    ):

        self.position += (
            self.velocity
            * dt
        )

        speed = np.linalg.norm(
            self.velocity
        )

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        if speed > 1e-5:

            self.heading = math.atan2(
                self.velocity[1],
                self.velocity[0]
            )

        # ----------------------------------------------------
        # Battery model
        #
        # Very simple simulation model:
        # higher speed = higher consumption.
        # ----------------------------------------------------

        consumption = (
            0.0015
            + 0.0020 * speed
            + 0.0010 * speed * speed
        )

        self.battery = max(
            0.0,
            self.battery
            - consumption
        )

        self.energy_used += (
            consumption
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        self.history.append(
            self.position.copy()
        )

        self.velocity_history.append(
            self.velocity.copy()
        )

        self.heading_history.append(
            self.heading
        )

        self.battery_history.append(
            self.battery
        )


# ============================================================
# OBSTACLE
# ============================================================

class Obstacle:

    def __init__(
        self,
        obstacle_id,
        position,
        radius,
        velocity=None,
        moving=False,
        bounds=None
    ):

        self.id = obstacle_id

        self.position = np.asarray(
            position,
            dtype=float
        )

        self.radius = radius

        if velocity is None:

            self.velocity = np.zeros(2)

        else:

            self.velocity = np.asarray(
                velocity,
                dtype=float
            )

        self.moving = moving

        self.bounds = bounds

        self.history = [
            self.position.copy()
        ]

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        dt
    ):

        if not self.moving:

            self.history.append(
                self.position.copy()
            )

            return

        self.position += (
            self.velocity
            * dt
        )

        # ----------------------------------------------------
        # Bounce from bounds
        # ----------------------------------------------------

        if self.bounds is not None:

            xmin, xmax, ymin, ymax = (
                self.bounds
            )

            if self.position[0] < xmin:

                self.position[0] = xmin

                self.velocity[0] *= -1

            elif self.position[0] > xmax:

                self.position[0] = xmax

                self.velocity[0] *= -1

            if self.position[1] < ymin:

                self.position[1] = ymin

                self.velocity[1] *= -1

            elif self.position[1] > ymax:

                self.position[1] = ymax

                self.velocity[1] *= -1

        self.history.append(
            self.position.copy()
        )


# ============================================================
# ORCA LINE FOR DRONE-DRONE
# ============================================================

def compute_orca_line(
    drone,
    other,
    time_horizon,
    time_step
):
    """
    Standard reciprocal ORCA.

    Both drones take 50% responsibility.
    """

    relative_position = (
        other.position
        - drone.position
    )

    relative_velocity = (
        drone.velocity
        - other.velocity
    )

    distance_sq = abs_sq(
        relative_position
    )

    combined_radius = (
        drone.radius
        + other.radius
    )

    combined_radius_sq = (
        combined_radius
        * combined_radius
    )

    line = Line(
        np.zeros(2),
        np.zeros(2)
    )

    # ========================================================
    # NOT CURRENTLY COLLIDING
    # ========================================================

    if distance_sq > combined_radius_sq:

        inv_time_horizon = (
            1.0
            / time_horizon
        )

        w = (
            relative_velocity
            -
            inv_time_horizon
            * relative_position
        )

        w_length_sq = abs_sq(w)

        dot_product = np.dot(
            w,
            relative_position
        )

        # ----------------------------------------------------
        # Cutoff circle
        # ----------------------------------------------------

        if (
            dot_product < 0.0
            and
            dot_product * dot_product
            >
            combined_radius_sq
            * w_length_sq
        ):

            w_length = math.sqrt(
                w_length_sq
            )

            if w_length > EPSILON:

                unit_w = (
                    w
                    / w_length
                )

            else:

                unit_w = np.array(
                    [1.0, 0.0]
                )

            line.direction = np.array(
                [
                    unit_w[1],
                    -unit_w[0]
                ]
            )

            u = (
                combined_radius
                / time_horizon
                -
                w_length
            ) * unit_w

        # ----------------------------------------------------
        # Collision cone
        # ----------------------------------------------------

        else:

            leg = math.sqrt(
                max(
                    0.0,
                    distance_sq
                    - combined_radius_sq
                )
            )

            if det(
                relative_position,
                w
            ) > 0.0:

                line.direction = np.array(
                    [
                        relative_position[0]
                        * leg
                        -
                        relative_position[1]
                        * combined_radius,

                        relative_position[0]
                        * combined_radius
                        +
                        relative_position[1]
                        * leg
                    ]
                ) / distance_sq

            else:

                line.direction = -np.array(
                    [
                        relative_position[0]
                        * leg
                        +
                        relative_position[1]
                        * combined_radius,

                        -relative_position[0]
                        * combined_radius
                        +
                        relative_position[1]
                        * leg
                    ]
                ) / distance_sq

            dot_product_2 = np.dot(
                relative_velocity,
                line.direction
            )

            u = (
                dot_product_2
                * line.direction
                -
                relative_velocity
            )

    # ========================================================
    # ALREADY COLLIDING
    # ========================================================

    else:

        inv_time_step = (
            1.0
            / time_step
        )

        w = (
            relative_velocity
            -
            inv_time_step
            * relative_position
        )

        w_length = np.linalg.norm(
            w
        )

        if w_length > EPSILON:

            unit_w = (
                w
                / w_length
            )

        else:

            position_length = np.linalg.norm(
                relative_position
            )

            if position_length > EPSILON:

                unit_w = (
                    -relative_position
                    / position_length
                )

            else:

                unit_w = np.array(
                    [1.0, 0.0]
                )

        line.direction = np.array(
            [
                unit_w[1],
                -unit_w[0]
            ]
        )

        u = (
            combined_radius
            * inv_time_step
            -
            w_length
        ) * unit_w

    # ========================================================
    # 50/50 RESPONSIBILITY
    # ========================================================

    line.point = (
        drone.velocity
        + 0.5 * u
    )

    return line


# ============================================================
# ORCA LINE FOR OBSTACLE
# ============================================================

def compute_obstacle_orca_line(
    drone,
    obstacle,
    time_horizon,
    time_step
):
    """
    ORCA-style constraint against an obstacle.

    Unlike drone-drone ORCA, the drone takes FULL
    responsibility for avoiding an obstacle.

    The obstacle can be stationary or moving.
    """

    relative_position = (
        obstacle.position
        - drone.position
    )

    relative_velocity = (
        drone.velocity
        - obstacle.velocity
    )

    distance_sq = abs_sq(
        relative_position
    )

    combined_radius = (
        drone.radius
        + obstacle.radius
    )

    combined_radius_sq = (
        combined_radius
        * combined_radius
    )

    line = Line(
        np.zeros(2),
        np.zeros(2)
    )

    # ========================================================
    # OUTSIDE COLLISION
    # ========================================================

    if distance_sq > combined_radius_sq:

        inv_tau = (
            1.0
            / time_horizon
        )

        w = (
            relative_velocity
            -
            inv_tau
            * relative_position
        )

        w_length_sq = abs_sq(w)

        dot_product = np.dot(
            w,
            relative_position
        )

        # ----------------------------------------------------
        # Cutoff circle
        # ----------------------------------------------------

        if (
            dot_product < 0.0
            and
            dot_product * dot_product
            >
            combined_radius_sq
            * w_length_sq
        ):

            w_length = math.sqrt(
                w_length_sq
            )

            if w_length > EPSILON:

                unit_w = (
                    w
                    / w_length
                )

            else:

                unit_w = np.array(
                    [1.0, 0.0]
                )

            line.direction = np.array(
                [
                    unit_w[1],
                    -unit_w[0]
                ]
            )

            u = (
                combined_radius
                * inv_tau
                -
                w_length
            ) * unit_w

        # ----------------------------------------------------
        # Cone leg
        # ----------------------------------------------------

        else:

            leg = math.sqrt(
                max(
                    0.0,
                    distance_sq
                    - combined_radius_sq
                )
            )

            if det(
                relative_position,
                w
            ) > 0.0:

                line.direction = np.array(
                    [
                        relative_position[0]
                        * leg
                        -
                        relative_position[1]
                        * combined_radius,

                        relative_position[0]
                        * combined_radius
                        +
                        relative_position[1]
                        * leg
                    ]
                ) / distance_sq

            else:

                line.direction = -np.array(
                    [
                        relative_position[0]
                        * leg
                        +
                        relative_position[1]
                        * combined_radius,

                        -relative_position[0]
                        * combined_radius
                        +
                        relative_position[1]
                        * leg
                    ]
                ) / distance_sq

            dot_product_2 = np.dot(
                relative_velocity,
                line.direction
            )

            u = (
                dot_product_2
                * line.direction
                -
                relative_velocity
            )

    # ========================================================
    # COLLISION
    # ========================================================

    else:

        inv_dt = (
            1.0
            / time_step
        )

        w = (
            relative_velocity
            -
            inv_dt
            * relative_position
        )

        w_length = np.linalg.norm(w)

        if w_length > EPSILON:

            unit_w = (
                w
                / w_length
            )

        else:

            distance = np.linalg.norm(
                relative_position
            )

            if distance > EPSILON:

                unit_w = (
                    -relative_position
                    / distance
                )

            else:

                unit_w = np.array(
                    [1.0, 0.0]
                )

        line.direction = np.array(
            [
                unit_w[1],
                -unit_w[0]
            ]
        )

        u = (
            combined_radius
            * inv_dt
            -
            w_length
        ) * unit_w

    # ========================================================
    # FULL RESPONSIBILITY
    # ========================================================

    line.point = (
        drone.velocity
        + u
    )

    return line


# ============================================================
# LP1
# ============================================================

def linear_program_1(
    lines,
    line_no,
    radius,
    opt_velocity,
    direction_opt,
    result
):
    """
    RVO2-style 1-D linear program.
    """

    line = lines[line_no]

    dot_product = np.dot(
        line.point,
        line.direction
    )

    discriminant = (
        dot_product * dot_product
        +
        radius * radius
        -
        abs_sq(line.point)
    )

    if discriminant < 0.0:

        return False, result

    sqrt_discriminant = math.sqrt(
        discriminant
    )

    t_left = (
        -dot_product
        -
        sqrt_discriminant
    )

    t_right = (
        -dot_product
        +
        sqrt_discriminant
    )

    # --------------------------------------------------------
    # Previous constraints
    # --------------------------------------------------------

    for i in range(
        line_no
    ):

        other = lines[i]

        denominator = det(
            line.direction,
            other.direction
        )

        numerator = det(
            other.direction,
            line.point
            -
            other.point
        )

        # Parallel
        if abs(
            denominator
        ) <= EPSILON:

            if numerator < 0.0:

                return False, result

            continue

        t = (
            numerator
            / denominator
        )

        if denominator >= 0.0:

            t_right = min(
                t_right,
                t
            )

        else:

            t_left = max(
                t_left,
                t
            )

        if t_left > t_right:

            return False, result

    # --------------------------------------------------------
    # Select optimal t
    # --------------------------------------------------------

    if direction_opt:

        if np.dot(
            line.direction,
            opt_velocity
        ) > 0.0:

            t = t_right

        else:

            t = t_left

    else:

        t = np.dot(
            line.direction,
            opt_velocity
            -
            line.point
        )

        t = max(
            t_left,
            min(
                t_right,
                t
            )
        )

    result = (
        line.point
        +
        t * line.direction
    )

    return True, result


# ============================================================
# LP2
# ============================================================

def linear_program_2(
    lines,
    radius,
    opt_velocity,
    direction_opt
):
    """
    RVO2-style 2-D linear program.
    """

    # --------------------------------------------------------
    # Initial solution
    # --------------------------------------------------------

    if direction_opt:

        result = (
            opt_velocity
            * radius
        )

    elif (
        abs_sq(opt_velocity)
        >
        radius * radius
    ):

        result = (
            normalize(opt_velocity)
            * radius
        )

    else:

        result = (
            opt_velocity.copy()
        )

    # --------------------------------------------------------
    # Process constraints
    # --------------------------------------------------------

    for i in range(
        len(lines)
    ):

        violation = det(
            lines[i].direction,
            lines[i].point
            -
            result
        )

        if violation > 0.0:

            temp_result = (
                result.copy()
            )

            success, candidate = (
                linear_program_1(
                    lines,
                    i,
                    radius,
                    opt_velocity,
                    direction_opt,
                    result
                )
            )

            if not success:

                return (
                    i,
                    temp_result
                )

            result = candidate

    return (
        len(lines),
        result
    )


# ============================================================
# LP3
# ============================================================

def linear_program_3(
    lines,
    begin_line,
    radius,
    result
):
    """
    RVO2-style 3rd-stage LP.

    Used when LP2 encounters an infeasible
    constraint.
    """

    distance = 0.0

    for i in range(
        begin_line,
        len(lines)
    ):

        current_distance = det(
            lines[i].direction,
            lines[i].point
            -
            result
        )

        if (
            current_distance
            <= distance
        ):

            continue

        projected_lines = []

        # ----------------------------------------------------
        # Project previous lines onto current line
        # ----------------------------------------------------

        for j in range(i):

            new_line = Line(
                np.zeros(2),
                np.zeros(2)
            )

            determinant = det(
                lines[i].direction,
                lines[j].direction
            )

            # ------------------------------------------------
            # Parallel
            # ------------------------------------------------

            if abs(
                determinant
            ) <= EPSILON:

                if np.dot(
                    lines[i].direction,
                    lines[j].direction
                ) > 0.0:

                    continue

                new_line.point = (
                    0.5
                    *
                    (
                        lines[i].point
                        +
                        lines[j].point
                    )
                )

            # ------------------------------------------------
            # Non-parallel
            # ------------------------------------------------

            else:

                new_line.point = (
                    lines[i].point
                    +
                    (
                        det(
                            lines[j].direction,
                            lines[i].point
                            -
                            lines[j].point
                        )
                        /
                        determinant
                    )
                    *
                    lines[i].direction
                )

            direction = (
                lines[j].direction
                -
                lines[i].direction
            )

            new_line.direction = normalize(
                direction
            )

            # Degenerate direction
            if (
                np.linalg.norm(
                    new_line.direction
                )
                <
                EPSILON
            ):

                continue

            projected_lines.append(
                new_line
            )

        # ----------------------------------------------------
        # Previous solution
        # ----------------------------------------------------

        temp_result = (
            result.copy()
        )

        direction_opt = np.array(
            [
                -lines[i].direction[1],
                lines[i].direction[0]
            ]
        )

        line_fail, candidate = (
            linear_program_2(
                projected_lines,
                radius,
                direction_opt,
                True
            )
        )

        if (
            line_fail
            <
            len(projected_lines)
        ):

            result = temp_result

        else:

            result = candidate

        distance = det(
            lines[i].direction,
            lines[i].point
            -
            result
        )

    return result


# ============================================================
# COMPUTE NEW ORCA VELOCITY
# ============================================================

def compute_new_velocity(
    drone,
    other_drones,
    obstacles
):
    """
    Build all ORCA constraints and solve the
    velocity-selection problem.
    """

    preferred_velocity = (
        drone.preferred_velocity()
    )

    lines = []

    # --------------------------------------------------------
    # Drone-drone constraints
    # --------------------------------------------------------

    for other in other_drones:

        line = compute_orca_line(
            drone,
            other,
            TIME_HORIZON,
            DT
        )

        lines.append(
            line
        )

    # --------------------------------------------------------
    # Obstacle constraints
    # --------------------------------------------------------

    for obstacle in obstacles:

        line = (
            compute_obstacle_orca_line(
                drone,
                obstacle,
                TIME_HORIZON,
                DT
            )
        )

        lines.append(
            line
        )

    # --------------------------------------------------------
    # LP2
    # --------------------------------------------------------

    line_fail, result = (
        linear_program_2(
            lines,
            drone.max_speed,
            preferred_velocity,
            False
        )
    )

    # --------------------------------------------------------
    # LP3
    # --------------------------------------------------------

    if line_fail < len(lines):

        result = linear_program_3(
            lines,
            line_fail,
            drone.max_speed,
            result
        )

    # --------------------------------------------------------
    # Speed limit
    # --------------------------------------------------------

    result = clamp_magnitude(
        result,
        drone.max_speed
    )

    return result


# ============================================================
# COMMUNICATION
# ============================================================

def communication_links(
    drones
):
    """
    Determine which drones are within
    communication range.
    """

    links = []

    for i in range(
        len(drones)
    ):

        for j in range(
            i + 1,
            len(drones)
        ):

            distance = np.linalg.norm(
                drones[i].position
                -
                drones[j].position
            )

            radius = min(
                drones[i].communication_radius,
                drones[j].communication_radius
            )

            if distance <= radius:

                links.append(
                    (
                        i,
                        j
                    )
                )

    return links


# ============================================================
# COLLISION CHECK
# ============================================================

def check_drone_collisions(
    drones
):
    """
    Check drone-drone collisions.
    """

    collisions = []

    for i in range(
        len(drones)
    ):

        for j in range(
            i + 1,
            len(drones)
        ):

            distance = np.linalg.norm(
                drones[i].position
                -
                drones[j].position
            )

            minimum_distance = (
                drones[i].radius
                +
                drones[j].radius
            )

            if (
                distance
                <
                minimum_distance
            ):

                collisions.append(
                    (
                        i,
                        j
                    )
                )

    return collisions


# ============================================================
# DRONE-OBSTACLE COLLISION
# ============================================================

def check_obstacle_collisions(
    drones,
    obstacles
):
    """
    Check drone-obstacle collisions.
    """

    collisions = []

    for drone in drones:

        for obstacle in obstacles:

            distance = np.linalg.norm(
                drone.position
                -
                obstacle.position
            )

            minimum_distance = (
                drone.radius
                +
                obstacle.radius
            )

            if (
                distance
                <
                minimum_distance
            ):

                collisions.append(
                    (
                        drone.id,
                        obstacle.id
                    )
                )

    return collisions


# ============================================================
# MINIMUM SEPARATION
# ============================================================

def minimum_drone_clearance(
    drones
):
    """
    Minimum boundary-to-boundary
    clearance between drones.
    """

    minimum = float("inf")

    for i in range(
        len(drones)
    ):

        for j in range(
            i + 1,
            len(drones)
        ):

            distance = np.linalg.norm(
                drones[i].position
                -
                drones[j].position
            )

            clearance = (
                distance
                -
                drones[i].radius
                -
                drones[j].radius
            )

            minimum = min(
                minimum,
                clearance
            )

    return minimum


# ============================================================
# SIMULATION
# ============================================================

def run_simulation(
    drones,
    obstacles
):
    """
    Main synchronous simulation.
    """

    max_steps = int(
        MAX_SIMULATION_TIME
        / DT
    )

    print()
    print("=" * 72)
    print("ADVANCED ORCA THREE-DRONE SWARM SIMULATION")
    print("=" * 72)

    print()
    print("Configuration")
    print("-" * 72)

    print(
        f"Number of drones       : "
        f"{len(drones)}"
    )

    print(
        f"Number of obstacles    : "
        f"{len(obstacles)}"
    )

    print(
        f"Time step              : "
        f"{DT:.3f} s"
    )

    print(
        f"ORCA time horizon      : "
        f"{TIME_HORIZON:.2f} s"
    )

    print(
        f"Maximum simulation     : "
        f"{MAX_SIMULATION_TIME:.1f} s"
    )

    print()

    for drone in drones:

        print(
            f"Drone {drone.id}:"
        )

        print(
            f"    Start    = "
            f"{drone.position}"
        )

        print(
            f"    Waypoints = "
            f"{len(drone.waypoints)}"
        )

        print(
            f"    Goal     = "
            f"{drone.waypoints[-1]}"
        )

        print(
            f"    Radius   = "
            f"{drone.radius:.2f} m"
        )

        print(
            f"    Max speed = "
            f"{drone.max_speed:.2f} m/s"
        )

        print()

    # ========================================================
    # SIMULATION
    # ========================================================

    collision_detected = False

    obstacle_collision_detected = False

    min_clearance = float("inf")

    min_obstacle_clearance = float("inf")

    completed = False

    for step in range(
        max_steps
    ):

        # ----------------------------------------------------
        # Check goal completion
        # ----------------------------------------------------

        for drone in drones:

            drone.preferred_velocity()

        all_finished = all(
            drone.reached_goal
            for drone in drones
        )

        if all_finished:

            completed = True

            print(
                f"All drones reached their goals "
                f"at t = {step * DT:.2f} s"
            )

            break

        # ----------------------------------------------------
        # Calculate all new velocities
        # BEFORE updating any drone
        # ----------------------------------------------------

        new_velocities = []

        for drone in drones:

            others = [
                other
                for other in drones
                if other.id != drone.id
            ]

            desired_velocity = (
                compute_new_velocity(
                    drone,
                    others,
                    obstacles
                )
            )

            new_velocities.append(
                desired_velocity
            )

        # ----------------------------------------------------
        # Motion model
        # ----------------------------------------------------

        for drone, desired_velocity in zip(
            drones,
            new_velocities
        ):

            if drone.reached_goal:

                drone.velocity = np.zeros(2)

            else:

                drone.apply_motion_model(
                    desired_velocity,
                    DT
                )

        # ----------------------------------------------------
        # Update drones
        # ----------------------------------------------------

        for drone in drones:

            drone.update_state(
                DT
            )

        # ----------------------------------------------------
        # Update obstacles
        # ----------------------------------------------------

        for obstacle in obstacles:

            obstacle.update(
                DT
            )

        # ----------------------------------------------------
        # Drone collisions
        # ----------------------------------------------------

        drone_collisions = (
            check_drone_collisions(
                drones
            )
        )

        if drone_collisions:

            collision_detected = True

        # ----------------------------------------------------
        # Obstacle collisions
        # ----------------------------------------------------

        obstacle_collisions = (
            check_obstacle_collisions(
                drones,
                obstacles
            )
        )

        if obstacle_collisions:

            obstacle_collision_detected = True

        # ----------------------------------------------------
        # Minimum drone clearance
        # ----------------------------------------------------

        current_clearance = (
            minimum_drone_clearance(
                drones
            )
        )

        min_clearance = min(
            min_clearance,
            current_clearance
        )

        # ----------------------------------------------------
        # Minimum obstacle clearance
        # ----------------------------------------------------

        for drone in drones:

            for obstacle in obstacles:

                distance = np.linalg.norm(
                    drone.position
                    -
                    obstacle.position
                )

                clearance = (
                    distance
                    -
                    drone.radius
                    -
                    obstacle.radius
                )

                min_obstacle_clearance = min(
                    min_obstacle_clearance,
                    clearance
                )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 72)
    print("FINAL SIMULATION RESULTS")
    print("=" * 72)

    print()

    for drone in drones:

        goal_distance = np.linalg.norm(
            drone.position
            -
            drone.goal
        )

        speed = np.linalg.norm(
            drone.velocity
        )

        print(
            f"Drone {drone.id}"
        )

        print(
            f"    Final position : "
            f"{drone.position}"
        )

        print(
            f"    Goal           : "
            f"{drone.goal}"
        )

        print(
            f"    Goal distance  : "
            f"{goal_distance:.3f} m"
        )

        print(
            f"    Final speed    : "
            f"{speed:.3f} m/s"
        )

        print(
            f"    Battery        : "
            f"{drone.battery:.2f} %"
        )

        print(
            f"    Energy used    : "
            f"{drone.energy_used:.4f}"
        )

        print()

    # --------------------------------------------------------
    # Collision statistics
    # --------------------------------------------------------

    print(
        f"Minimum drone clearance     : "
        f"{min_clearance:.4f} m"
    )

    print(
        f"Minimum obstacle clearance  : "
        f"{min_obstacle_clearance:.4f} m"
    )

    print()

    if not collision_detected:

        print(
            "Drone collision check      : PASS"
        )

    else:

        print(
            "Drone collision check      : FAIL"
        )

    if not obstacle_collision_detected:

        print(
            "Obstacle collision check   : PASS"
        )

    else:

        print(
            "Obstacle collision check   : FAIL"
        )

    if completed:

        print(
            "Mission completion          : SUCCESS"
        )

    else:

        print(
            "Mission completion          : TIMEOUT"
        )

    print()
    print("=" * 72)

    return drones, obstacles


# ============================================================
# CREATE SCENARIO
# ============================================================

def create_scenario():
    """
    Create a challenging three-drone environment.

    The drones cross the environment from different
    directions.

    There are:
        - static obstacles
        - moving obstacles
        - intermediate waypoints
    """

    drones = [

        # ----------------------------------------------------
        # DRONE 0
        # ----------------------------------------------------

        Drone(
            drone_id=0,

            position=[
                -9.0,
                -2.0
            ],

            waypoints=[
                [-5.0, -2.0],
                [0.0, 0.0],
                [5.0, 2.0],
                [9.0, 3.0]
            ],

            radius=0.45,

            max_speed=2.0,

            max_acceleration=2.5,

            communication_radius=6.0
        ),

        # ----------------------------------------------------
        # DRONE 1
        # ----------------------------------------------------

        Drone(
            drone_id=1,

            position=[
                9.0,
                2.5
            ],

            waypoints=[
                [5.0, 2.5],
                [1.5, 0.5],
                [-4.0, -0.5],
                [-9.0, -3.0]
            ],

            radius=0.45,

            max_speed=2.0,

            max_acceleration=2.5,

            communication_radius=6.0
        ),

        # ----------------------------------------------------
        # DRONE 2
        # ----------------------------------------------------

        Drone(
            drone_id=2,

            position=[
                0.0,
                -9.0
            ],

            waypoints=[
                [0.0, -6.0],
                [-1.5, -2.0],
                [1.0, 2.0],
                [0.0, 8.5]
            ],

            radius=0.45,

            max_speed=1.8,

            max_acceleration=2.2,

            communication_radius=6.0
        )
    ]

    # ========================================================
    # STATIC OBSTACLES
    # ========================================================

    obstacles = [

        Obstacle(
            obstacle_id=0,

            position=[
                -2.0,
                2.5
            ],

            radius=1.0,

            moving=False
        ),

        Obstacle(
            obstacle_id=1,

            position=[
                3.0,
                -2.0
            ],

            radius=1.0,

            moving=False
        ),

        Obstacle(
            obstacle_id=2,

            position=[
                0.0,
                4.5
            ],

            radius=0.8,

            moving=False
        )
    ]

    # ========================================================
    # MOVING OBSTACLES
    # ========================================================

    obstacles.append(

        Obstacle(
            obstacle_id=3,

            position=[
                -6.0,
                4.0
            ],

            radius=0.65,

            velocity=[
                1.5,
                -0.15
            ],

            moving=True,

            bounds=[
                -8.0,
                8.0,
                -6.0,
                6.0
            ]
        )
    )

    obstacles.append(

        Obstacle(
            obstacle_id=4,

            position=[
                6.0,
                -4.0
            ],

            radius=0.65,

            velocity=[
                -1.2,
                0.20
            ],

            moving=True,

            bounds=[
                -8.0,
                8.0,
                -6.0,
                6.0
            ]
        )
    )

    return drones, obstacles


# ============================================================
# ANIMATION
# ============================================================

def animate_simulation(
    drones,
    obstacles
):
    """
    Create detailed swarm animation.
    """

    frame_count = max(
        len(drone.history)
        for drone in drones
    )

    # ========================================================
    # Determine plot boundaries
    # ========================================================

    all_positions = []

    for drone in drones:

        all_positions.extend(
            drone.history
        )

        all_positions.extend(
            drone.waypoints
        )

    for obstacle in obstacles:

        all_positions.extend(
            obstacle.history
        )

    all_positions = np.asarray(
        all_positions
    )

    margin = 2.0

    xmin = (
        np.min(all_positions[:, 0])
        -
        margin
    )

    xmax = (
        np.max(all_positions[:, 0])
        +
        margin
    )

    ymin = (
        np.min(all_positions[:, 1])
        -
        margin
    )

    ymax = (
        np.max(all_positions[:, 1])
        +
        margin
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    ax.set_xlim(
        xmin,
        xmax
    )

    ax.set_ylim(
        ymin,
        ymax
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "X position (m)"
    )

    ax.set_ylabel(
        "Y position (m)"
    )

    ax.set_title(
        "Advanced ORCA — Three Drone Swarm"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    # ========================================================
    # Drone colors
    # ========================================================

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green"
    ]

    # ========================================================
    # Drone graphics
    # ========================================================

    drone_circles = []

    drone_labels = []

    drone_trails = []

    drone_velocity_arrows = []

    communication_circles = []

    # --------------------------------------------------------
    # Drones
    # --------------------------------------------------------

    for drone, color in zip(
        drones,
        colors
    ):

        circle = plt.Circle(
            drone.history[0],
            drone.radius,
            color=color,
            ec="black",
            linewidth=1.5,
            alpha=0.9,
            zorder=10
        )

        ax.add_patch(
            circle
        )

        drone_circles.append(
            circle
        )

        label = ax.text(
            drone.history[0][0],
            drone.history[0][1] + 0.65,
            f"D{drone.id}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=20
        )

        drone_labels.append(
            label
        )

        trail, = ax.plot(
            [],
            [],
            "-",
            color=color,
            linewidth=1.5,
            alpha=0.65,
            zorder=2
        )

        drone_trails.append(
            trail
        )

        # ----------------------------------------------------
        # Communication circle
        # ----------------------------------------------------

        communication_circle = plt.Circle(
            drone.history[0],
            drone.communication_radius,
            fill=False,
            linestyle="--",
            linewidth=0.8,
            color=color,
            alpha=0.20,
            zorder=1
        )

        ax.add_patch(
            communication_circle
        )

        communication_circles.append(
            communication_circle
        )

        # ----------------------------------------------------
        # Velocity arrow
        # ----------------------------------------------------

        arrow = ax.quiver(
            drone.position[0],
            drone.position[1],
            0.0,
            0.0,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.007,
            zorder=15
        )

        drone_velocity_arrows.append(
            arrow
        )

        # ----------------------------------------------------
        # Goal markers
        # ----------------------------------------------------

        for waypoint_index, waypoint in enumerate(
            drone.waypoints
        ):

            if (
                waypoint_index
                ==
                len(drone.waypoints) - 1
            ):

                marker = "X"

                size = 13

            else:

                marker = "."

                size = 8

            ax.plot(
                waypoint[0],
                waypoint[1],
                marker,
                color=color,
                markersize=size,
                markeredgewidth=2
            )

    # ========================================================
    # Obstacle graphics
    # ========================================================

    obstacle_circles = []

    obstacle_velocity_arrows = []

    for obstacle in obstacles:

        if obstacle.moving:

            color = "tab:red"

        else:

            color = "dimgray"

        circle = plt.Circle(
            obstacle.position,
            obstacle.radius,
            color=color,
            ec="black",
            linewidth=1.5,
            alpha=0.75,
            zorder=7
        )

        ax.add_patch(
            circle
        )

        obstacle_circles.append(
            circle
        )

        if obstacle.moving:

            arrow = ax.quiver(
                obstacle.position[0],
                obstacle.position[1],
                obstacle.velocity[0],
                obstacle.velocity[1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                width=0.006,
                zorder=8
            )

        else:

            arrow = None

        obstacle_velocity_arrows.append(
            arrow
        )

    # ========================================================
    # Communication links
    # ========================================================

    communication_lines = {}

    for i in range(
        len(drones)
    ):

        for j in range(
            i + 1,
            len(drones)
        ):

            line, = ax.plot(
                [],
                [],
                "--",
                linewidth=1.0,
                alpha=0.6,
                color="purple",
                zorder=3
            )

            communication_lines[
                (i, j)
            ] = line

    # ========================================================
    # Telemetry panel
    # ========================================================

    telemetry = ax.text(
        0.015,
        0.985,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.90
        ),
        zorder=30
    )

    # ========================================================
    # Mission information
    # ========================================================

    mission_text = ax.text(
        0.985,
        0.015,
        "",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.90
        ),
        zorder=30
    )

    # ========================================================
    # Pad histories
    # ========================================================

    drone_positions = []

    drone_velocities = []

    for drone in drones:

        positions = list(
            drone.history
        )

        velocities = list(
            drone.velocity_history
        )

        while len(positions) < frame_count:

            positions.append(
                positions[-1].copy()
            )

        while len(velocities) < frame_count:

            velocities.append(
                velocities[-1].copy()
            )

        drone_positions.append(
            positions
        )

        drone_velocities.append(
            velocities
        )

    obstacle_positions = []

    for obstacle in obstacles:

        positions = list(
            obstacle.history
        )

        while len(positions) < frame_count:

            positions.append(
                positions[-1].copy()
            )

        obstacle_positions.append(
            positions
        )

    # ========================================================
    # INIT
    # ========================================================

    def init():

        for trail in drone_trails:

            trail.set_data(
                [],
                []
            )

        telemetry.set_text("")

        mission_text.set_text("")

        return []

    # ========================================================
    # UPDATE
    # ========================================================

    def update(frame):

        # ====================================================
        # Drones
        # ====================================================

        for i, drone in enumerate(
            drones
        ):

            position = (
                drone_positions[i][frame]
            )

            velocity = (
                drone_velocities[i][frame]
            )

            # ------------------------------------------------
            # Drone body
            # ------------------------------------------------

            drone_circles[i].center = (
                position
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            drone_labels[i].set_position(
                (
                    position[0],
                    position[1] + 0.65
                )
            )

            # ------------------------------------------------
            # Trail
            # ------------------------------------------------

            trajectory = (
                drone_positions[i]
                [
                    :frame + 1
                ]
            )

            xs = [
                p[0]
                for p in trajectory
            ]

            ys = [
                p[1]
                for p in trajectory
            ]

            drone_trails[i].set_data(
                xs,
                ys
            )

            # ------------------------------------------------
            # Communication radius
            # ------------------------------------------------

            communication_circles[i].center = (
                position
            )

            # ------------------------------------------------
            # Velocity arrow
            # ------------------------------------------------

            drone_velocity_arrows[i].set_offsets(
                np.array(
                    [position]
                )
            )

            drone_velocity_arrows[i].set_UVC(
                velocity[0],
                velocity[1]
            )

        # ====================================================
        # Obstacles
        # ====================================================

        for i, obstacle in enumerate(
            obstacles
        ):

            position = (
                obstacle_positions[i][frame]
            )

            obstacle_circles[i].center = (
                position
            )

            if (
                obstacle_velocity_arrows[i]
                is not None
            ):

                obstacle_velocity_arrows[i].set_offsets(
                    np.array(
                        [position]
                    )
                )

                obstacle_velocity_arrows[i].set_UVC(
                    obstacle.velocity[0],
                    obstacle.velocity[1]
                )

        # ====================================================
        # Communication links
        # ====================================================

        links = communication_links(
            [
                create_snapshot_drone(
                    drones[i],
                    drone_positions[i][frame]
                )
                for i in range(
                    len(drones)
                )
            ]
        )

        for key, line in (
            communication_lines.items()
        ):

            i, j = key

            if key in links:

                p1 = (
                    drone_positions[i][frame]
                )

                p2 = (
                    drone_positions[j][frame]
                )

                line.set_data(
                    [
                        p1[0],
                        p2[0]
                    ],
                    [
                        p1[1],
                        p2[1]
                    ]
                )

                line.set_alpha(
                    0.75
                )

            else:

                line.set_data(
                    [],
                    []
                )

                line.set_alpha(
                    0.0
                )

        # ====================================================
        # Current drone separation
        # ====================================================

        minimum_clearance = float(
            "inf"
        )

        for i in range(
            len(drones)
        ):

            for j in range(
                i + 1,
                len(drones)
            ):

                distance = np.linalg.norm(
                    drone_positions[i][frame]
                    -
                    drone_positions[j][frame]
                )

                clearance = (
                    distance
                    -
                    drones[i].radius
                    -
                    drones[j].radius
                )

                minimum_clearance = min(
                    minimum_clearance,
                    clearance
                )

        # ====================================================
        # Current obstacle clearance
        # ====================================================

        obstacle_clearance = float(
            "inf"
        )

        for i, drone in enumerate(
            drones
        ):

            for j, obstacle in enumerate(
                obstacles
            ):

                distance = np.linalg.norm(
                    drone_positions[i][frame]
                    -
                    obstacle_positions[j][frame]
                )

                clearance = (
                    distance
                    -
                    drone.radius
                    -
                    obstacle.radius
                )

                obstacle_clearance = min(
                    obstacle_clearance,
                    clearance
                )

        # ====================================================
        # Telemetry
        # ====================================================

        simulation_time = (
            frame * DT
        )

        telemetry_lines = [
            "SWARM TELEMETRY",
            "------------------------------"
        ]

        for i, drone in enumerate(
            drones
        ):

            position = (
                drone_positions[i][frame]
            )

            velocity = (
                drone_velocities[i][frame]
            )

            speed = np.linalg.norm(
                velocity
            )

            heading = math.degrees(
                math.atan2(
                    velocity[1],
                    velocity[0]
                )
            )

            telemetry_lines.append(
                f"D{i}: "
                f"pos=({position[0]:5.2f},"
                f"{position[1]:5.2f}) "
                f"v={speed:4.2f} "
                f"hdg={heading:6.1f}° "
                f"bat={drone.battery_history[
                    min(
                        frame,
                        len(
                            drone.battery_history
                        ) - 1
                    )
                ]:5.1f}%"
            )

        telemetry.set_text(
            "\n".join(
                telemetry_lines
            )
        )

        # ====================================================
        # Mission status
        # ====================================================

        reached = 0

        for i, drone in enumerate(
            drones
        ):

            distance = np.linalg.norm(
                drone_positions[i][frame]
                -
                drone.waypoints[-1]
            )

            if distance <= GOAL_TOLERANCE:

                reached += 1

        mission_text.set_text(
            f"TIME       : "
            f"{simulation_time:6.2f} s\n"
            f"GOALS      : "
            f"{reached}/{len(drones)}\n"
            f"DRONE CLR. : "
            f"{minimum_clearance:6.3f} m\n"
            f"OBS CLR.   : "
            f"{obstacle_clearance:6.3f} m\n"
            f"COMMS      : "
            f"{len(links)}/3 links"
        )

        return []

    # ========================================================
    # ANIMATION
    # ========================================================

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        init_func=init,
        interval=1000 / FPS,
        blit=False
    )

    print()
    print(
        f"Generating animation "
        f"({frame_count} frames)..."
    )

    animation.save(
        OUTPUT_FILE,
        writer=PillowWriter(
            fps=FPS
        )
    )

    plt.close(fig)

    print(
        f"Animation saved to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# SNAPSHOT DRONE
# ============================================================

class SnapshotDrone:

    def __init__(
        self,
        original,
        position
    ):

        self.id = original.id

        self.position = (
            position.copy()
        )

        self.radius = (
            original.radius
        )

        self.communication_radius = (
            original.communication_radius
        )


def create_snapshot_drone(
    drone,
    position
):
    """
    Used only for animation communication
    calculations.
    """

    return SnapshotDrone(
        drone,
        position
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "Initializing advanced ORCA swarm..."
    )

    drones, obstacles = (
        create_scenario()
    )

    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    drones, obstacles = (
        run_simulation(
            drones,
            obstacles
        )
    )

    # --------------------------------------------------------
    # Generate animation
    # --------------------------------------------------------

    animate_simulation(
        drones,
        obstacles
    )

    print()
    print(
        "Simulation complete."
    )

    print(
        f"Open animation with:"
    )

    print(
        f"    xdg-open {OUTPUT_FILE}"
    )
