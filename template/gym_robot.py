import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry
from pathlib import Path

from cspace import RectangularCSpace
from robot import DubinsCarParams
from robot import UnicycleRobot


GYM_DUBINS_CAR_ENV_ID = "CS440DubinsCar-v0"


def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])
def dist_2d(config1, config2):
    return np.linalg.norm(config1[:2] - config2[:2])

class GymDubinsCar(gym.Env):
    # Don't change these parameters...
    # That being said, these are the rewards we use for training but you might find others work better
    NUM_LIDAR_BEAMS = 16
    COLLISION_PENALTY = -30.0
    GOAL_REWARD = 10.0
    RAY_EPSILON = 1e-9
    ACTION_NOISE_MAGNITUDE = 0.1

    def __init__(self, car_params: DubinsCarParams, start_config, goal_config, max_steps=30):
        self.car_params = car_params
        self.start_config = np.array(start_config, dtype=np.float64)
        self.goal_config = np.array(goal_config, dtype=np.float64)
        self.max_steps = max_steps

        self.actions = np.array([
            [self.car_params.velocity, self.car_params.turn_rate],
            [self.car_params.velocity, 0.0],
            [self.car_params.velocity, -self.car_params.turn_rate],
        ], dtype=np.float64)
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.beam_offsets = np.linspace(0.0, 2.0 * np.pi, num=self.NUM_LIDAR_BEAMS, endpoint=False)
        self.lidar_segments = self._build_lidar_segments()
        self.workspace_diagonal = np.linalg.norm(
            self.car_params.robot.cspace.workspace_boundary[:, 1]
            - self.car_params.robot.cspace.workspace_boundary[:, 0]
        )

        initial_observation, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_observation.shape,
            dtype=initial_observation.dtype,
        )

    # --- Helpers for computing and visualizing lidar observations ---

    # convert the boundary and obstacles into a list of line segments that we can raycast against for lidar
    def _build_lidar_segments(self):
        boundary = self.car_params.robot.cspace.workspace_boundary
        x_min, x_max = boundary[0]
        y_min, y_max = boundary[1]
        corners = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ], dtype=np.float64)

        segments = []
        for i in range(4):
            segments.append((corners[i], corners[(i + 1) % 4]))

        for obstacle in self.car_params.robot.cspace.obstacles:
            obstacle = np.array(obstacle, dtype=np.float64)
            for i in range(obstacle.shape[0]):
                segments.append((obstacle[i], obstacle[(i + 1) % obstacle.shape[0]]))
        return segments

    # compute the distance along a ray from an origin in a given direction to a line segment (if it hits)
    def _ray_segment_distance(self, origin, direction, seg_start, seg_end):
        segment_direction = seg_end - seg_start
        denominator = cross_2d(direction, segment_direction)
        if abs(denominator) <= self.RAY_EPSILON:
            return None

        diff = seg_start - origin
        ray_distance = cross_2d(diff, segment_direction) / denominator
        segment_fraction = cross_2d(diff, direction) / denominator

        if ray_distance < 0.0:
            return None
        if segment_fraction < -self.RAY_EPSILON or segment_fraction > 1.0 + self.RAY_EPSILON:
            return None
        return ray_distance

    # compute the lidar points by raycasting against all segments and finding the closest hit for each beam
    def _compute_lidar_points(self):
        origin = self.config[:2]
        max_range = self.workspace_diagonal
        lidar_points = []

        for beam_offset in self.beam_offsets:
            beam_angle = self.config[2] + beam_offset
            direction = np.array([np.cos(beam_angle), np.sin(beam_angle)], dtype=np.float64)
            closest_distance = max_range

            for seg_start, seg_end in self.lidar_segments:
                hit_distance = self._ray_segment_distance(origin, direction, seg_start, seg_end)
                if hit_distance is None:
                    continue
                if hit_distance < closest_distance:
                    closest_distance = hit_distance

            lidar_points.append(origin + closest_distance * direction)

        return np.array(lidar_points, dtype=np.float64)

    # This method should be called whenever we update the current config, to recompute the lidar points and distances
    def update_lidar(self):
        self.lidar = self._compute_lidar_points()
        self.lidar_distances = np.linalg.norm(self.lidar - self.config[:2], axis=1)

    # A flexible render method that can show various combinations of the workspace, start/goal/current poses, path, and lidar beams
    def render(
            self,
            ax=None,
            path=None,
            show_current=True,
            show_start=True,
            show_goal=True,
            show_goal_tolerance=True,
            show_lidar=False,
            title=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cspace = self.car_params.robot.cspace
        cspace.draw_workspace(ax)

        if show_goal_tolerance:
            goal_region = Circle(self.goal_config[:2], self.car_params.goal_tolerance, 
                                 facecolor="green", edgecolor="green", alpha=0.15, label="Goal tolerance")
            ax.add_patch(goal_region)

        if show_start:
            cspace.draw_state(ax, self.start_config, color="blue", label="Start")
        if show_goal:
            cspace.draw_state(ax, self.goal_config, color="green", label="Goal pose")
        if show_current:
            cspace.draw_state(ax, self.config, color="orange", label="Current pose")

        if path is not None:
            path = np.asarray(path, dtype=np.float64)
            if path.ndim != 2 or path.shape[1] != 3:
                raise ValueError(f"Expected path with shape (n, 3), got {path.shape}")
            ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2, label="Path")

        if show_lidar:
            for lidar_point in self.lidar:
                ax.plot([self.config[0], lidar_point[0]], [self.config[1], lidar_point[1]], 
                        color="darkorange", alpha=0.4, linewidth=1)
            ax.scatter(self.lidar[:, 0], self.lidar[:, 1], color="darkorange", s=12, label="Lidar hits")

        ax.set_aspect("equal", adjustable="box")
        if title is not None:
            ax.set_title(title)

        handles, labels = ax.get_legend_handles_labels()
        deduped = {}
        for handle, label in zip(handles, labels):
            if label and label not in deduped:
                deduped[label] = handle
        if deduped:
            ax.legend(deduped.values(), deduped.keys())
        return fig, ax

    # --- Helpers for computing the observation features from the lidar, config, and goal state (and anything else you want) ---

   
        # TODO(IV): Improve this simple baseline feature vector.
        # Whatever you return here should always be a 1D numeric NumPy array with the same shape.
        
        # feature design 
        # 告訴 agent「現在的情況是什麼」讓 agent 決定下一步怎麼走:
        # robot position 
        x, y, theta = self.config 

    def make_observation(self):
        x, y, theta = self.config 
        goal_x, goal_y, goal_theta = self.goal_config

        goal_dx = goal_x - x
        goal_dy = goal_y - y
        goal_distance = np.sqrt(goal_dx**2 + goal_dy**2)

        goal_angle = np.arctan2(goal_dy, goal_dx)

        angle_diff = np.arctan2(
            np.sin(goal_angle - theta),
            np.cos(goal_angle - theta)
        )

        # 保護 lidar（避免空或變長）
        lidar = np.array(self.lidar_distances, dtype=np.float64)

        # 關鍵：固定長度（假設應該是16）
        FIXED_LIDAR_DIM = 16

        if len(lidar) < FIXED_LIDAR_DIM:
            lidar = np.pad(lidar, (0, FIXED_LIDAR_DIM - len(lidar)), constant_values=0)
        else:
            lidar = lidar[:FIXED_LIDAR_DIM]

        min_lidar = np.min(lidar)

        features = np.array([
            np.sin(theta),
            np.cos(theta),
            goal_dx,
            goal_dy,
            goal_distance,
            min_lidar,
            np.sin(angle_diff),
            np.cos(angle_diff)
        ], dtype=np.float64)

        # 永遠 8 + 16 = 24
        features = np.concatenate([features, lidar])

        return features

       
        # x, y, theta = self.config
        # goal_x, goal_y, _ = self.goal_config

        # goal_dx = goal_x - x
        # goal_dy = goal_y - y

        # return np.array([
        #     goal_dx,
        #     goal_dy,
        #     theta
        # ], dtype=np.float64)

    # --- The important methods for our gym environment: reset and step ---

    # Set the current config to the start, recompute lidar, reset the steps taken, and return the initial observation
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.config = self.start_config.copy()
        self.steps_taken = 0
        self.update_lidar()
        return self.make_observation(), {} # no info, though you can feel free to add some if you want!

    # The usual dynamics are deterministic, but we add some noise to the control
    # While this makes the problem harder to plan for, it can help learning algorithms generalize better 
    # and avoid overfitting to a perfectly deterministic environment
    def _sample_noisy_action(self, action):
        control = self.actions[action].copy()
        noise = self.np_random.uniform(
            low=-self.ACTION_NOISE_MAGNITUDE,
            high=self.ACTION_NOISE_MAGNITUDE,
            size=control.shape,
        )
        control += noise
        control_boundary = self.car_params.robot.control_boundary
        return np.clip(control, control_boundary[:, 0], control_boundary[:, 1])

    # Apply the (noisy) action, check for collisions along the way, then compute reward:
    # - If we collide, return a large negative reward (based on self.COLLISION_PENALTY) and end the episode
    # - Otherwise, reward is based on how much closer we got to the goal (so it can be positive or negative)
    # - If we reached the goal, return a large positive reward (based on self.GOAL_REWARD) and end the episode
    # - If we exceeded max steps, end the episode
    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected an integer in [0, {self.action_space.n - 1}]")
        
        # increment step count - if after this action we exceed max steps, episode will end truncated
        self.steps_taken += 1
        
        # apply noisy control
        intermediate_poses = self.car_params.robot.apply_constant_control(
            self.config,
            self._sample_noisy_action(action),
            self.car_params.control_duration,
        )
        
        # validate and if invalid motion then episode ends with collision penalty
        for intermediate_pos in intermediate_poses:
            if not self.car_params.robot.cspace.is_valid(intermediate_pos):
                return self.make_observation(), self.COLLISION_PENALTY, True, False, {}
        
        # compute shaped reward based on how much closer we got to the goal (can be positive or negative)
        prev_goal_dist = dist_2d(self.config, self.goal_config)
        next_goal_dist = dist_2d(intermediate_poses[-1], self.goal_config)
        reward = prev_goal_dist - next_goal_dist

        # update our config and recompute lidar
        self.config = intermediate_poses[-1]
        self.update_lidar()

        # check if we reached the TWO-DIMENSIONAL goal region (ignoring heading)
        if dist_2d(self.config, self.goal_config) <= self.car_params.goal_tolerance:
            return self.make_observation(), self.GOAL_REWARD, True, False, {}
        # check if we should truncate due to max steps
        if self.steps_taken >= self.max_steps:
            return self.make_observation(), reward, False, True, {}
        # standard step case - not done, not truncated, return shaped reward
        return self.make_observation(), reward, False, False, {}


def make_dubins_env(params_file, max_steps=30, verbose=False):
    import json

    params_file = Path(params_file)
    with open(params_file, "r") as f:
        data = json.load(f)

    if verbose:
        print(f"Initializing configuration space from {params_file.name}")
    workspace_boundary = np.array([
        [data["boundary"]["x_min"], data["boundary"]["x_max"]],
        [data["boundary"]["y_min"], data["boundary"]["y_max"]],
    ])
    cspace_boundary = np.concatenate([workspace_boundary, np.array([[0, 2 * np.pi]])], axis=0)
    obstacles = [np.array(obstacle) for obstacle in data["obstacles"]]
    cspace = RectangularCSpace(
        workspace_boundary=workspace_boundary,
        cspace_boundary=cspace_boundary,
        is_angular=np.array([False, False, True]),
        obstacles=obstacles,
        rectangle_height=data["robot_height"],
        rectangle_width=data["robot_width"],
        interpolation_delta=data["dt"],
    )

    start = np.array(data["start"], dtype=np.float64)
    goal = np.array(data["goal"], dtype=np.float64)
    robot = UnicycleRobot(cspace, interpolation_delta=data["dt"])
    car_params = DubinsCarParams(
        robot=robot,
        velocity=data["velocity"],
        turn_rate=data["turn_rate"],
        control_duration=data["control_duration"],
        goal_tolerance=data["goal_tolerance"],
    )

    env = GymDubinsCar(car_params, start, goal, max_steps=max_steps)
    env.env_name = params_file.stem
    env.params_file = params_file
    return env


if GYM_DUBINS_CAR_ENV_ID not in registry:
    register(
        id=GYM_DUBINS_CAR_ENV_ID,
        entry_point="gym_robot:make_dubins_env",
    )


