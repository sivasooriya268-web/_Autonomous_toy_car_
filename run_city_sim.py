#!/usr/bin/env python3
"""
run_city_sim.py - MuJoCo 3D simulation of the toy car driving through a small city.
No ROS. Needs:  pip install mujoco numpy

Setup:   python make_city.py          (writes city_sim.xml; also imported by this file)
Run:     python run_city_sim.py                # one trip in the 3D viewer
         python run_city_sim.py --batch 50     # 50 random trips, no window, results.csv
         python run_city_sim.py --seed 7       # repeat a specific trip
         python run_city_sim.py --level easy   # fewer obstacles (default is hard)
macOS:   start the viewer with  mjpython run_city_sim.py

What happens in each trip
-------------------------
  * The car starts at one intersection with a random heading and must drive to the
    destination marker (another intersection).
  * It knows the road map (a 3x3 grid of intersections) and finds the shortest route.
  * The overhead camera gives its pose; the ultrasonic sensor watches for obstacles.
  * Some roads are closed by barriers and some cars are parked beside the lane
    (--level hard: 4 closed roads + 5 parked cars, --level easy: 2 + 2). At least one
    closed road is always on the first planned route. The car only discovers a closed
    road when it gets there: it stops, waits, turns around and plans a new route that
    avoids that road. With several closed roads it may have to do this more than once.

Same pipeline as the real car:
  overhead webcam -> FakeOverheadCamera   laptop -> CityController   WiFi -> CommandLink
  ESP32 -> esp32_apply()                  servos -> MuJoCo velocity actuators
"""

import argparse
import contextlib
import csv
import math
import time
from collections import deque

import numpy as np
import mujoco
import mujoco.viewer

from make_city import ROAD_CENTERS, ROAD_HALF_WIDTH, ROAD_EXTENT, MAX_BARRIERS, MAX_PARKED

# =============================================================================
# Settings
# =============================================================================
XML_PATH = "city_sim.xml"

CONTROL_DT = 0.05            # laptop loop: 20 Hz
MAX_SIM_TIME = 300.0         # give up after this many simulated seconds

# --- Difficulty: (closed-road barriers, parked cars). Max is set in make_city.py ---
LEVELS = {"easy": (2, 2), "hard": (4, 5)}

# --- Navigation ---
BASE_SPEED = 6.0             # wheel rad/s on straight roads (about 0.18 m/s)
MAX_WHEEL = 10.0
LOOKAHEAD = 0.25             # m  aim this far ahead along the road centre line
KP = 3.0                     # wheel rad/s per rad of heading error
MAX_TURN = 6.0
DEADBAND = math.radians(2)
SWITCH_RADIUS = 0.10         # m  count an intersection as reached inside this radius
ARRIVE_DISTANCE = 0.12       # m  destination reached (estimated position)
SUCCESS_RADIUS = 0.15        # m  trip succeeds inside this (true position)

# --- Obstacle handling ---
SENSOR_OFFSET = 0.078        # m  sensor is this far ahead of the car's reference point
SLOW_DISTANCE = 0.55         # m  start slowing for a real obstacle
STOP_DISTANCE = 0.30         # m  fully stopped here
BLOCK_WAIT = 1.0             # s  waited this long behind an obstacle -> road is closed
ROAD_MARGIN = 0.02           # m  tolerance when asking "is this point on the road?"
PARKED_OFFSET = 0.16         # m  parked cars (8 cm wide) sit this far from the lane centre

EMERGENCY_DISTANCE = 0.08    # m  ESP32 refuses net-forward motion closer than this
SENSOR_NOISE_STD = 0.01
SENSOR_MIN, SENSOR_MAX = 0.02, 4.00

# --- Fake overhead camera ---
CAM_POS_NOISE = 0.005
CAM_YAW_NOISE = math.radians(1.5)
CAM_DROPOUT_PROB = 0.02
CAM_DROPOUT_FRAMES = (3, 9)
STALE_LIMIT = 0.4            # s  reuse the last good pose this long after a dropout

# --- WiFi link / ESP32 ---
LINK_DELAY = (0.10, 0.20)
PACKET_LOSS = 0.02
FAILSAFE_TIMEOUT = 0.5
SERVO_DRIFT = 0.15           # rad/s


# =============================================================================
# Road map (graph of intersections)
# =============================================================================
class RoadGraph:
    """Intersections are nodes (row, col); each road piece between two is an edge."""

    def __init__(self, centers):
        self.centers = list(centers)
        n = len(self.centers)
        self.nodes = [(r, c) for r in range(n) for c in range(n)]

    def xy(self, node):
        """World position of an intersection."""
        r, c = node
        return np.array([self.centers[c], self.centers[r]])

    def neighbors(self, node):
        r, c = node
        n = len(self.centers)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= r + dr < n and 0 <= c + dc < n:
                yield (r + dr, c + dc)

    def edges(self):
        """Every road piece once, as (node_a, node_b)."""
        return [(a, b) for a in self.nodes for b in self.neighbors(a) if b > a]

    def shortest_path(self, start, goal, blocked):
        """Breadth-first search. `blocked` is a set of frozenset({a, b}) closed roads."""
        came_from = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == goal:
                break
            for nb in self.neighbors(node):
                if nb in came_from or frozenset((node, nb)) in blocked:
                    continue
                came_from[nb] = node
                queue.append(nb)
        if goal not in came_from:
            return None                      # destination cut off
        path, node = [], goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        return path[::-1]


GRAPH = RoadGraph(ROAD_CENTERS)


def on_road(p):
    """Is world point p=(x, y) on the asphalt? Used to tell obstacles from buildings."""
    x, y = p
    half = ROAD_HALF_WIDTH + ROAD_MARGIN
    ext = ROAD_EXTENT + ROAD_MARGIN
    if abs(y) <= ext and any(abs(x - c) <= half for c in ROAD_CENTERS):
        return True                           # on a vertical road
    if abs(x) <= ext and any(abs(y - c) <= half for c in ROAD_CENTERS):
        return True                           # on a horizontal road
    return False


# =============================================================================
# Small helpers
# =============================================================================
def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def get_car_pose_2d(data, car_body_id):
    """(x, y, yaw) of the car in the world frame."""
    x, y = data.xpos[car_body_id][:2]
    rot = data.xmat[car_body_id].reshape(3, 3)
    return float(x), float(y), float(math.atan2(rot[1, 0], rot[0, 0]))


def measure_front_distance(model, data, site_id, car_body_id, rng):
    """HC-SR04 model: one mj_ray along the sensor site's +Z, +/-1 cm noise, 2 cm - 4 m."""
    origin = data.site_xpos[site_id].copy()
    direction = data.site_xmat[site_id].reshape(3, 3)[:, 2].copy()
    geom_hit = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, origin, direction, None, 1, car_body_id, geom_hit)
    if dist < 0:
        dist = SENSOR_MAX
    dist += rng.normal(0.0, SENSOR_NOISE_STD)
    return float(np.clip(dist, SENSOR_MIN, SENSOR_MAX))


def car_collided(model, data, car_body_id, floor_geom_id):
    """True if any part of the car touches something other than the floor."""
    for i in range(data.ncon):
        c = data.contact[i]
        root1 = model.body_rootid[model.geom_bodyid[c.geom1]]
        root2 = model.body_rootid[model.geom_bodyid[c.geom2]]
        car1, car2 = root1 == car_body_id, root2 == car_body_id
        if car1 == car2:
            continue
        other = c.geom2 if car1 else c.geom1
        if other != floor_geom_id:
            return True
    return False


# =============================================================================
# Scene setup: start, destination, obstacles
# =============================================================================
def choose_scene(rng, n_closed, n_parked):
    """
    Pick start and destination intersections and distinct road pieces for the obstacles
    (the first n_closed become closed roads, the rest get a parked car).
    Rules: the closed roads never cut the destination off completely, and at least one
    of them is on the first planned route, so the car always has to find a detour.
    """
    edges = GRAPH.edges()
    for _ in range(5000):
        i, j = rng.choice(len(GRAPH.nodes), 2, replace=False)
        start, goal = GRAPH.nodes[i], GRAPH.nodes[j]
        first_route = GRAPH.shortest_path(start, goal, set())
        if len(first_route) - 1 < 3:                    # trip of at least 3 road pieces
            continue
        picks = rng.choice(len(edges), n_closed + n_parked, replace=False)
        chosen = [edges[k] for k in picks]
        closed = {frozenset(e) for e in chosen[:n_closed]}
        if GRAPH.shortest_path(start, goal, closed) is None:
            continue                                     # destination cut off
        route_pieces = {frozenset(e) for e in zip(first_route, first_route[1:])}
        if n_closed and not (closed & route_pieces):
            continue                                     # force at least one detour
        return start, goal, chosen
    raise RuntimeError("could not build a valid scene")


def place_obstacles(model, data, ids, chosen_edges, n_closed, rng):
    """Move the used mocap obstacles onto their road pieces; park the rest outside the walls."""
    barrier_bodies = ids["obstacles"][:MAX_BARRIERS]
    parked_bodies = ids["obstacles"][MAX_BARRIERS:]

    for k, (a, b) in enumerate(chosen_edges):
        pa, pb = GRAPH.xy(a), GRAPH.xy(b)
        along = (pb - pa) / np.linalg.norm(pb - pa)
        normal = np.array([-along[1], along[0]])
        yaw = math.atan2(along[1], along[0])           # 0, 90, 180 or -90 degrees
        if k < n_closed:                                # barrier across the whole road
            body = barrier_bodies[k]
            p = pa + (pb - pa) * rng.uniform(0.45, 0.55)
            z = 0.10
        else:                                           # parked car beside the lane
            body = parked_bodies[k - n_closed]
            side = rng.choice([-1.0, 1.0])
            p = pa + (pb - pa) * rng.uniform(0.30, 0.70) + normal * side * PARKED_OFFSET
            z = 0.05
        mid = model.body_mocapid[body]
        data.mocap_pos[mid] = [p[0], p[1], z]
        data.mocap_quat[mid] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    # Obstacles this level does not use go far outside the city, where nothing can reach them.
    used_barriers, used_parked = n_closed, len(chosen_edges) - n_closed
    for k, body in enumerate(barrier_bodies[used_barriers:]):
        data.mocap_pos[model.body_mocapid[body]] = [20.0 + k, 20.0, 0.10]
    for k, body in enumerate(parked_bodies[used_parked:]):
        data.mocap_pos[model.body_mocapid[body]] = [20.0 + k, 22.0, 0.05]


def reset_episode(model, data, ids, rng, n_closed, n_parked):
    """Reset physics, put the car on its start intersection, place goal and obstacles."""
    start, goal, chosen = choose_scene(rng, n_closed, n_parked)
    mujoco.mj_resetData(model, data)

    sx, sy = GRAPH.xy(start)
    yaw = rng.uniform(-math.pi, math.pi)
    adr = ids["car_qpos_adr"]
    data.qpos[adr:adr + 2] = [sx, sy]                   # keep XML height, set x, y
    data.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    gx, gy = GRAPH.xy(goal)
    data.mocap_pos[model.body_mocapid[ids["goal_body"]]] = [gx, gy, 0.003]
    place_obstacles(model, data, ids, chosen, n_closed, rng)
    mujoco.mj_forward(model, data)
    return start, goal, chosen


# =============================================================================
# Fake overhead camera (pose only: the destination comes from the map)
# =============================================================================
class MarkerTracker:
    """A marker that is sometimes lost for a few frames in a row."""

    def __init__(self, rng):
        self.rng = rng
        self.frames_left = 0

    def visible(self):
        if self.frames_left > 0:
            self.frames_left -= 1
            return False
        if self.rng.random() < CAM_DROPOUT_PROB:
            self.frames_left = int(self.rng.integers(*CAM_DROPOUT_FRAMES))
            return False
        return True


class FakeOverheadCamera:
    def __init__(self, rng):
        self.rng = rng
        self.marker = MarkerTracker(rng)

    def observe(self, true_pose):
        """Noisy (x, y, yaw) of the car marker, or None if it was lost this frame."""
        if not self.marker.visible():
            return None
        x, y, yaw = true_pose
        return (x + self.rng.normal(0, CAM_POS_NOISE),
                y + self.rng.normal(0, CAM_POS_NOISE),
                yaw + self.rng.normal(0, CAM_YAW_NOISE))


class LastKnown:
    def __init__(self):
        self.value, self.t = None, -1e9

    def update(self, value, now):
        if value is not None:
            self.value, self.t = value, now

    def get(self, now):
        return self.value if (now - self.t) <= STALE_LIMIT else None


# =============================================================================
# Controller (laptop side). No MuJoCo code in here, so it can be reused on the real car.
# =============================================================================
class CityController:
    """
    States:  DRIVE    following the planned route
             WAIT     stopped behind an obstacle, deciding whether the road is closed
             ARRIVED  at the destination
             LOST     car marker not visible: stop
             NO_ROUTE every route is blocked

    Route following: the route is a list of intersections. For the current road piece
    the car aims at a point LOOKAHEAD metres ahead on the road centre line, which keeps
    it centred in its lane.  Hit points that land on a building are ignored (the map
    says buildings are expected); only hits on the road count as obstacles.
    """

    def __init__(self, start_node, goal_node):
        self.goal_node = goal_node
        self.goal_xy = GRAPH.xy(goal_node)
        self.blocked = set()                         # roads discovered to be closed
        self.prev_node = start_node                  # last intersection passed
        self.seg_start = GRAPH.xy(start_node)        # where the current road piece starts
        self.targets = GRAPH.shortest_path(start_node, goal_node, self.blocked)[1:]
        self.state = "DRIVE"
        self.obstacle_frames = 0                     # consecutive frames with an obstacle
        self.blocked_since = None
        self.replans = 0

    def _replan(self, pos):
        """The road ahead is closed: remember it, turn around, find another route."""
        self.blocked.add(frozenset((self.prev_node, self.targets[0])))
        self.replans += 1
        self.blocked_since, self.obstacle_frames = None, 0
        path = GRAPH.shortest_path(self.prev_node, self.goal_node, self.blocked)
        if path is None:
            self.state = "NO_ROUTE"
            return
        self.targets = path          # path[0] is the intersection behind us: go back first
        self.seg_start = pos.copy()

    def decide(self, pose, distance, now):
        """
        pose (x, y, yaw) or None, distance = ultrasonic reading (m), now = seconds.
        Returns (left_wheel_speed, right_wheel_speed) in rad/s.
        """
        if self.state == "NO_ROUTE":
            return 0.0, 0.0
        if pose is None:
            self.state = "LOST"
            return 0.0, 0.0
        pos = np.array(pose[:2])
        yaw = pose[2]

        if np.linalg.norm(pos - self.goal_xy) < ARRIVE_DISTANCE:
            self.state = "ARRIVED"
            return 0.0, 0.0

        # ---- 1. Is there a real obstacle ahead? (hit point on the road, not a building) ----
        obstacle = False
        if distance < SLOW_DISTANCE:
            reach = SENSOR_OFFSET + distance
            hit = pos + reach * np.array([math.cos(yaw), math.sin(yaw)])
            obstacle = on_road(hit)
        self.obstacle_frames = self.obstacle_frames + 1 if obstacle else 0
        confirmed = self.obstacle_frames >= 2        # two frames in a row: not just noise

        # ---- 2. Reached the next intersection? Move on to the next road piece ----
        target_xy = GRAPH.xy(self.targets[0])
        if np.linalg.norm(pos - target_xy) < SWITCH_RADIUS:
            self.prev_node = self.targets.pop(0)
            self.seg_start = target_xy
            if not self.targets:
                self.state = "ARRIVED"
                return 0.0, 0.0
            target_xy = GRAPH.xy(self.targets[0])

        # ---- 3. Steer toward a point ahead on the road centre line ----
        seg = target_xy - self.seg_start
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            aim = target_xy
        else:
            u = seg / seg_len
            progress = float(np.dot(pos - self.seg_start, u))
            aim = self.seg_start + u * min(max(progress, 0.0) + LOOKAHEAD, seg_len)
        heading_error = wrap_angle(math.atan2(aim[1] - pos[1], aim[0] - pos[0]) - yaw)

        err = 0.0 if abs(heading_error) < DEADBAND else heading_error
        turn = float(np.clip(KP * err, -MAX_TURN, MAX_TURN))
        forward = BASE_SPEED * max(0.0, 1.0 - abs(err) / (math.pi / 2))   # 0 beyond 90 deg

        # ---- 4. Slow down and stop for a real obstacle; give up on the road after a wait ----
        self.state = "DRIVE"
        if confirmed:
            forward *= float(np.clip((distance - STOP_DISTANCE) /
                                     (SLOW_DISTANCE - STOP_DISTANCE), 0.0, 1.0))
            if distance < STOP_DISTANCE + 0.04:
                self.state = "WAIT"
                if self.blocked_since is None:
                    self.blocked_since = now
                if now - self.blocked_since >= BLOCK_WAIT:
                    self._replan(pos)
                    return 0.0, 0.0
        else:
            self.blocked_since = None

        left = float(np.clip(forward - turn, -MAX_WHEEL, MAX_WHEEL))     # +turn = steer left
        right = float(np.clip(forward + turn, -MAX_WHEEL, MAX_WHEEL))
        return left, right


# =============================================================================
# Fake WiFi link and ESP32 safety
# =============================================================================
class CommandLink:
    """Laptop -> ESP32: random delay, packet loss and a failsafe timeout."""

    def __init__(self, rng):
        self.rng = rng
        self.pending = []
        self.cmd = (0.0, 0.0)
        self.cmd_sent_time = -1.0
        self.last_rx_time = 0.0

    def send(self, now, left, right):
        if self.rng.random() < PACKET_LOSS:
            return
        self.pending.append((now + self.rng.uniform(*LINK_DELAY), now, left, right))

    def receive(self, now):
        arrived = [p for p in self.pending if p[0] <= now]
        self.pending = [p for p in self.pending if p[0] > now]
        if arrived:
            newest = max(arrived, key=lambda p: p[1])
            if newest[1] > self.cmd_sent_time:
                self.cmd = (newest[2], newest[3])
                self.cmd_sent_time = newest[1]
            self.last_rx_time = now
        if now - self.last_rx_time > FAILSAFE_TIMEOUT:
            return 0.0, 0.0, True
        return self.cmd[0], self.cmd[1], False


def esp32_apply(left, right, distance):
    """ESP32 emergency stop: no net-forward motion when something is very close."""
    if distance < EMERGENCY_DISTANCE and (left + right) > 0:
        return 0.0, 0.0
    return left, right


# =============================================================================
# One trip
# =============================================================================
def run_episode(model, data, ids, rng, level="hard", viewer=None, verbose=False):
    lock = viewer.lock if viewer is not None else contextlib.nullcontext
    n_closed, n_parked = LEVELS[level]

    with lock():
        start, goal, chosen = reset_episode(model, data, ids, rng, n_closed, n_parked)
        goal_xy = GRAPH.xy(goal)

    controller = CityController(start, goal)
    camera = FakeOverheadCamera(rng)
    link = CommandLink(rng)
    pose_memory = LastKnown()
    drift = rng.uniform(-SERVO_DRIFT, SERVO_DRIFT, size=2)
    substeps = int(round(CONTROL_DT / model.opt.timestep))

    if verbose:
        names = ["closed road"] * n_closed + ["parked car"] * n_parked
        print(f"  level {level}: trip {start} -> {goal}, first route: {[start] + controller.targets}")
        for kind, (a, b) in zip(names, chosen):
            print(f"  {kind}: between {a} and {b}")

    collisions, in_collision = 0, False
    last_state = None
    success, reason, time_to_goal = False, "timeout", float("nan")
    true_goal_dist = float("nan")

    while data.time < MAX_SIM_TIME:
        if viewer is not None and not viewer.is_running():
            reason = "viewer closed"
            break
        wall_start = time.perf_counter()

        with lock():
            now = data.time

            # 1. Sense
            distance = measure_front_distance(
                model, data, ids["sensor_site"], ids["car_body"], rng)
            true_pose = get_car_pose_2d(data, ids["car_body"])
            pose_memory.update(camera.observe(true_pose), now)

            # 2. Decide on the laptop, send over the fake WiFi
            left, right = controller.decide(pose_memory.get(now), distance, now)
            link.send(now, left, right)

            # 3. ESP32: receive, failsafe, emergency stop, servos (with drift)
            rx_left, rx_right, _ = link.receive(now)
            rx_left, rx_right = esp32_apply(rx_left, rx_right, distance)
            data.ctrl[ids["left_act"]] = np.clip(rx_left + drift[0], -MAX_WHEEL, MAX_WHEEL)
            data.ctrl[ids["right_act"]] = np.clip(rx_right + drift[1], -MAX_WHEEL, MAX_WHEEL)

            # 4. Physics for one control period
            for _ in range(substeps):
                mujoco.mj_step(model, data)
                hit = car_collided(model, data, ids["car_body"], ids["floor_geom"])
                if hit and not in_collision:
                    collisions += 1
                in_collision = hit

            # 5. Success uses the TRUE position
            x, y, _ = get_car_pose_2d(data, ids["car_body"])
            true_goal_dist = math.hypot(goal_xy[0] - x, goal_xy[1] - y)
            if true_goal_dist < SUCCESS_RADIUS:
                success, reason, time_to_goal = True, "arrived", data.time

        if verbose and controller.state != last_state:
            print(f"  t={now:6.1f}s  state={controller.state:<8} "
                  f"distance={distance:4.2f} m  replans={controller.replans}  "
                  f"collisions={collisions}")
        last_state = controller.state

        if viewer is not None:
            viewer.sync()
            spare = CONTROL_DT - (time.perf_counter() - wall_start)
            if spare > 0:
                time.sleep(spare)

        if success:
            break
        if controller.state == "NO_ROUTE":
            reason = "no route"
            break

    with lock():
        data.ctrl[:] = 0.0

    return {
        "success": int(success),
        "time_s": round(time_to_goal, 1) if success else "",
        "collisions": collisions,
        "replans": controller.replans,
        "final_goal_dist_m": round(true_goal_dist, 3),
        "end_reason": reason,
    }


# =============================================================================
# Entry points
# =============================================================================
def lookup_ids(model):
    def find(kind, name):
        i = mujoco.mj_name2id(model, kind, name)
        if i < 0:
            raise RuntimeError(f"'{name}' not found in {XML_PATH} (run make_city.py first)")
        return i

    car_body = find(mujoco.mjtObj.mjOBJ_BODY, "car")
    return {
        "car_body": car_body,
        "car_qpos_adr": int(model.jnt_qposadr[model.body_jntadr[car_body]]),
        "goal_body": find(mujoco.mjtObj.mjOBJ_BODY, "goal"),
        "floor_geom": find(mujoco.mjtObj.mjOBJ_GEOM, "floor"),
        "sensor_site": find(mujoco.mjtObj.mjOBJ_SITE, "sensor_site"),
        "left_act": find(mujoco.mjtObj.mjOBJ_ACTUATOR, "left_vel_servo"),
        "right_act": find(mujoco.mjtObj.mjOBJ_ACTUATOR, "right_vel_servo"),
        "overhead_cam": find(mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam"),
        "obstacles": [find(mujoco.mjtObj.mjOBJ_BODY, f"obs_{i}")
                      for i in range(1, MAX_BARRIERS + MAX_PARKED + 1)],
    }


def run_viewer(model, data, ids, seed, level):
    rng = np.random.default_rng(seed)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = ids["overhead_cam"]       # switch to car_cam in the left panel
        print(f"Trip with seed {seed}, level {level}. Close the window to quit.")
        result = run_episode(model, data, ids, rng, level=level, viewer=viewer, verbose=True)
        print("Result:", result)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def run_batch(model, data, ids, n_runs, seed, csv_path, level):
    rows = []
    for ep in range(n_runs):
        rng = np.random.default_rng(seed + ep)
        result = run_episode(model, data, ids, rng, level=level)
        result["episode"], result["seed"] = ep, seed + ep
        rows.append(result)
        print(f"trip {ep + 1:3d}/{n_runs}: {result['end_reason']:<9} "
              f"time={result['time_s']!s:>6}  replans={result['replans']}  "
              f"collisions={result['collisions']}")

    fields = ["episode", "seed", "success", "time_s", "collisions", "replans",
              "final_goal_dist_m", "end_reason"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    wins = [r for r in rows if r["success"]]
    print("\n=== Summary ===")
    print(f"success rate     : {len(wins)}/{n_runs} ({100 * len(wins) / n_runs:.0f}%)")
    print(f"mean collisions  : {np.mean([r['collisions'] for r in rows]):.2f} per trip")
    print(f"mean replans     : {np.mean([r['replans'] for r in rows]):.2f} per trip")
    if wins:
        print(f"mean trip time   : {np.mean([r['time_s'] for r in wins]):.1f} s (successful trips)")
    print(f"saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo toy car in a city")
    parser.add_argument("--batch", type=int, default=0, help="run N random trips headless")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--level", choices=list(LEVELS), default="hard",
                        help="obstacle count: easy = 2 closed roads + 2 parked cars, hard = 4 + 5")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    ids = lookup_ids(model)

    if args.batch > 0:
        run_batch(model, data, ids, args.batch, args.seed, args.csv, args.level)
    else:
        run_viewer(model, data, ids, args.seed, args.level)


if __name__ == "__main__":
    main()
