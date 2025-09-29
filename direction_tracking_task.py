"""Direction-tracking flight task."""
# ruff: noqa: F821

from typing import Optional
import numpy as np
from dm_control import mjcf

from dm_control.utils import rewards
from dm_control.composer.observation import observable
from dm_control import composer

from flybody.tasks.pattern_generators import (WingBeatPatternGenerator)
from flybody.tasks.task_utils import neg_quat
from flybody.tasks.base import Flying


class DirectionTracking(Flying):
    """Vision-based direction tracking flight with controllable Wing Beat Pattern Generator."""

    def __init__(self,
                 wbpg: WingBeatPatternGenerator,
                 floor_contacts_fatal: bool = True,
                 eye_camera_fovy: float = 150.,
                 eye_camera_size: int = 32,
                 target_height_range: tuple = (0.5, 0.8),
                 target_speed_range: tuple = (20, 40),
                 init_pos_x_range: Optional[tuple] = (-5, -5),
                 init_pos_y_range: Optional[tuple] = (0, 0),
                 **kwargs):
        """Task of learning a policy for tracking a direction while using a
            wing beat pattern generator with controllable wing beat frequency.

        Args:
            wpg: Wing beat generator.
            floor_contacts_fatal: Whether to terminate the episode when the fly
                contacts the floor.
            eye_camera_fovy: Field of view of the eye camera.
            eye_camera_size: Size of the eye camera.
            target_height_range: Range of target height.
            target_speed_range: Range of target speed.
            init_pos_x_range: Range of initial x position.
            init_pos_y_range: Range of initial y position.
            **kwargs: Arguments passed to the superclass constructor.
        """

        super().__init__(add_ghost=False,
                         num_user_actions=1,
                         eye_camera_fovy=eye_camera_fovy,
                         eye_camera_size=eye_camera_size,
                         **kwargs)
        self._wbpg = wbpg
        self._floor_contacts_fatal = floor_contacts_fatal
        self._eye_camera_size = eye_camera_size
        self._target_height_range = target_height_range
        self._target_speed_range = target_speed_range
        self._init_pos_x_range = init_pos_x_range
        self._init_pos_y_range = init_pos_y_range

        self._goal_position = np.array([0, 0, 0])
        self.heading_to_goal_angle =0.0
        self._pillars_initialized = False

        # Initialize the camera
        mjcf_root = self.root_entity.mjcf_model
        mjcf_root.worldbody.add(
            'camera',
            name='tracking_cam',
            mode='fixed',
            pos=[0, 0, 9],
            xyaxes=[1, 0, 0, 0, 1, 0],
            fovy=60
        )

        # Add a red ball at the position of goal
        self._goal = mjcf_root.worldbody.add(
            'site',
            name='goal_site',
            pos=[1, 0, 0],
            size=[0.05],
            rgba=[1, 0, 0, 1],
            type="sphere"
        )

        # Remove all light.
        for light in self._walker.mjcf_model.find_all('light'):
            light.remove()

        # Get wing joint indices into agent's action vector.
        self._wing_inds_action = self._walker._action_indices['wings']
        # Get 'user' index into agent's action vector (only one user action).
        self._user_idx_action = self._walker._action_indices['user'][0]

        # Dummy initialization.
        self._target_height = 0.
        self._target_speed = 0.

        self._target_zaxis = None
        self._ncol = None
        self._grid_axis = None

        # === Explicitly add/enable/disable vision task observables.
        # Fly observables.
        self._walker.observables.right_eye.enabled = True
        self._walker.observables.left_eye.enabled = True
        self._walker.observables.thorax_height.enabled = False
        # Task observables.
        self._walker.observables.add_observable('task_input', self.task_input)

    def get_hfield_height(self, x, y, physics):
        """Return hfield height at a hfield grid point closest to (x, y)."""

        hfield_half_size = physics.model.hfield_size[0, 0]
        self._ncol = physics.model.hfield_ncol[0]
        self._grid_axis = np.linspace(-hfield_half_size, hfield_half_size,
                                      self._ncol)

        # Get nearest indices.
        x_idx = np.argmin(np.abs(self._grid_axis - x))
        y_idx = np.argmin(np.abs(self._grid_axis - y))
        # physics.model.hfield_data is in range [0, 1], needs to be rescaled.
        elevation_z = physics.model.hfield_size[0, 2]  # z_top scaling.
        return elevation_z * physics.model.hfield_data[y_idx * self._ncol +
                                                       x_idx]

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        super().initialize_episode_mjcf(random_state)

        self._target_height = random_state.uniform(*self._target_height_range)
        self._target_speed = random_state.uniform(*self._target_speed_range)

        theta = np.deg2rad(self._body_pitch_angle)
        self._target_zaxis = np.array([np.sin(theta), 0, np.cos(theta)])

        mjcf_root = self.root_entity.mjcf_model
        goal_x = random_state.uniform(-3, 3)
        goal_y = random_state.uniform(-3, 3)
        goal_z = 2
        self._goal_position = np.array([goal_x, goal_y, goal_z])
        self._goal.set_attributes(pos=self._goal_position.tolist())

        # Initialize the pillars
        if not self._pillars_initialized:
            self._num_pillars = 100
            for i in range(self._num_pillars):

                x = random_state.uniform(-10, 10)
                y = random_state.uniform(-10, 10)
            
                
                while abs(x + 4.5) < 1.0 and abs(y) < 1.0:
                    x = random_state.uniform(-5, 5)
                    y = random_state.uniform(-5, 5)
            
                
                height = random_state.uniform(1.0, 3.0)
                radius = random_state.uniform(0.05, 0.15)
            
                pillar_body = mjcf_root.worldbody.add(
                    'body', name=f'pillar_{i}',
                    pos=[x, y, height/2]
                )
            
                pillar_body.add(
                    'geom',
                    type='cylinder',
                    size=[radius, height/2],
                    rgba=[0.4, 0.2, 0.1, 1],
                    mass=10.0,
                    contype=1,
                    conaffinity=1
                )
            self._pillars_initialized = True

    def initialize_episode(self, physics: 'mjcf.Physics',
                           random_state: np.random.RandomState):
        """Randomly selects a starting point and set the walker.

        Environment call sequence:
            check_termination, get_reward_factors, get_discount
        """
        super().initialize_episode(physics, random_state)

        init_x = random_state.uniform(*self._init_pos_x_range)
        init_y = random_state.uniform(*self._init_pos_y_range)

        # Reset wing pattern generator and get initial wing angles.
        initial_phase = random_state.uniform()
        init_wing_qpos = self._wbpg.reset(initial_phase=initial_phase)

        self._arena.initialize_episode(physics, random_state)

        # Initialize root position and orientation.
        hfield_height = self.get_hfield_height(init_x, init_y, physics)
        init_z = hfield_height + self._target_height
        self._walker.set_pose(physics, np.array([init_x, init_y, init_z]),
                              neg_quat(self._up_dir))

        # Initialize wing qpos.
        physics.bind(self._wing_joints).qpos = init_wing_qpos

        # If enabled, initialize leg joint angles in retracted position.
        if self._leg_joints:
            physics.bind(self._leg_joints).qpos = self._leg_springrefs

        if self._initialize_qvel:
            # Only initialize linear CoM velocity, not rotational velocity.
            init_vel, _ = self._walker.get_velocity(physics)
            self._walker.set_velocity(
                physics, [self._target_speed, init_vel[1], init_vel[2]])

    def before_step(self, physics: 'mjcf.Physics', action, random_state: np.random.RandomState):
        # Get target wing joint angles at beat frequency requested by the agent.
        base_freq, rel_range = self._wbpg.base_beat_freq, self._wbpg.rel_freq_range
        act = action[self._user_idx_action]  # Action in [-1, 1].
        ctrl_freq = base_freq * (1 + rel_range * act)
        ctrl = self._wbpg.step(ctrl_freq=ctrl_freq)  # Returns position control.

        length = physics.bind(self._wing_joints).qpos
        # Convert position control to force control.
        action[self._wing_inds_action] += (ctrl - length)

        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        """Returns the factorized reward terms."""

        # Height.
        xpos, _ = self._walker.get_pose(physics)
        current_height = (xpos[2] - self.get_hfield_height(*xpos[:2], physics))
        height = rewards.tolerance(current_height,
                                   bounds=(self._target_height,
                                           self._target_height),
                                   sigmoid='linear',
                                   margin=0.15,
                                   value_at_margin=0)
        # print(f"Height : {height}")

        velocity, _ = self._walker.get_velocity(physics)

        # Maintain certain speed.
        speed = rewards.tolerance(np.linalg.norm(velocity),
                                  bounds=(self._target_speed,
                                          self._target_speed),
                                  sigmoid='linear',
                                  margin=1.1 * self._target_speed,
                                  value_at_margin=0.0)
        # print(f"speed : {speed}")

        # Keep zero egocentric side speed.
        vel = self.observables['walker/velocimeter'](physics)
        side_speed = rewards.tolerance(vel[1],
                                       bounds=(0, 0),
                                       sigmoid='linear',
                                       margin=10,
                                       value_at_margin=0.0)
        # print(f"side_speed : {side_speed}")

        # World z-axis, to replace root quaternion reward above.
        current_zaxis = self.observables['walker/world_zaxis'](physics)
        angle = np.arccos(np.dot(self._target_zaxis, current_zaxis))
        world_zaxis = rewards.tolerance(angle,
                                        bounds=(0, 0),
                                        sigmoid='linear',
                                        margin=np.pi,
                                        value_at_margin=0.0)
        # print(f"world_zaxis : {world_zaxis}")

        # Reward for leg retraction during flight.
        qpos_diff = physics.bind(self._leg_joints).qpos - self._leg_springrefs
        retract_legs = rewards.tolerance(qpos_diff,
                                         bounds=(0, 0),
                                         sigmoid='linear',
                                         margin=4.,
                                         value_at_margin=0.0)
        # print(f"retract_legs : {retract_legs}")
        
        # Reward for keeping certain distance with the obstacles
        min_pillar_distance = np.inf

        for i in range(self._num_pillars):
            pillar_pos = physics.named.data.xpos[f"pillar_{i}"]
            distance = np.linalg.norm(xpos - pillar_pos)
            if distance < min_pillar_distance:
                min_pillar_distance = distance

        safe_distance = 0.2
        avoidance_penalty = 0.0
        if min_pillar_distance < safe_distance:
            avoidance_penalty = -0.9 * (safe_distance - min_pillar_distance)
        
        # print(f"avoidence : {avoidance_penalty}")


        # Reward for the distance to the plume source
        goal_distance = np.linalg.norm(xpos - self._goal_position)
        goal_proximity = rewards.tolerance(goal_distance,
                                           bounds=(0.0, 0.0),
                                           margin=10.0,
                                           sigmoid='linear',
                                           value_at_margin=0.0)
        # print(f"goal : {goal_proximity}")
        
        # Reward for smaller angle between HD and GD
        angle = self._get_heading_angle(physics)[0]
        heading_reward = rewards.tolerance(
            angle,
            bounds=(0, 0), 
            sigmoid='linear', 
            margin=np.pi,
            value_at_margin=0.0
        )
        # print(f"head : {heading_reward}")

        weights = {
            'height': 1.0,
            'speed': 0.2,
            'side_speed': 0.3,
            'world_zaxis': 0.1,
            'avoidance': 1.0,
            'goal': 1.5,
            'heading': 0.5,
            'retract_legs': 0.2,
        }
    
        reward = (
            weights['height'] * height +
            weights['speed'] * speed +
            weights['side_speed'] * side_speed +
            weights['world_zaxis'] * world_zaxis +
            weights['avoidance'] * avoidance_penalty +
            weights['goal'] * goal_proximity +
            weights['heading'] * heading_reward 
            # weights['retract_legs'] * retract_legs
        )

        # Normalization
        normalizer = sum(abs(w) for w in weights.values())
        reward_normalized = (reward / normalizer + 1) / 2
        
        # print(f"Reward: {reward_normalized:.4f}")
        return reward_normalized


    def check_floor_contact(self, physics):
        """Check if fly collides with floor geom."""
        world_id = 0
        for contact in physics.data.contact:
            # If the contact is not active, continue.
            if contact.efc_address < 0:
                continue
            # Check floor contact.
            body1 = physics.model.geom_bodyid[contact.geom1]
            body2 = physics.model.geom_bodyid[contact.geom2]
            if body1 == world_id or body2 == world_id:
                return True
        return False

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        fly_pos, _ = self._walker.get_pose(physics)

        fly_pos_xy = fly_pos[:2]
        goal_pos_xy = self._goal_position[:2]

        distance = np.linalg.norm(fly_pos_xy - goal_pos_xy)

        if distance < 0.1:
            return True

        if self._floor_contacts_fatal:
            return self.check_floor_contact(physics) or super().check_termination(physics)
        else:
            return super().check_termination(physics)
            
    def _get_heading_angle(self, physics):
        """Calculate the angle between goal position and head position"""
        fly_pos, fly_quat = self._walker.get_pose(physics)
        w, x, y, z = fly_quat

        rot_matrix = np.array([
            [1-2*(y**2+z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(x**2+z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x**2 + y**2)]
        ])

        agent_forward = rot_matrix[:, 0]

        goal_vec = np.array(self._goal_position) - np.array(fly_pos)
        goal_vec /= np.linalg.norm(goal_vec)

        dot = np.clip(np.dot(agent_forward, goal_vec), -1.0, 1.0)
        angle = np.arccos(dot)

        # Right is negative, Left is positive
        cross = np.cross(agent_forward, goal_vec)
        if np.dot(cross, rot_matrix[:, 2]) < 0:
            angle = -angle

        return np.array([angle], dtype=np.float32)

    @property
    def target_height(self):
        return self._target_height

    @property
    def target_speed(self):
        return self._target_speed

    @composer.observable
    def task_input(self):
        """Task-specific input, framed as an observable."""
        def get_task_input(physics: 'mjcf.Physics'):
            self.heading_to_goal_angle = self._get_heading_angle(physics)
            return np.hstack([self._target_height, self._target_speed, self.heading_to_goal_angle])
        return observable.Generic(get_task_input)