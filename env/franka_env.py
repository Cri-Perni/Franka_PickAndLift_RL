import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import (
    SIMULATION_DT,
    GRAVITY,
    ROBOT_URDF,
    ROBOT_HOME_POSE,
    CUBE_SIZE,
    CUBE_INITIAL_POS,
    CUBE_INITIAL_HEIGHT,
    CUBE_COLOR,
    CUBE_FRICTION,
    ARM_VELOCITY_SCALE,
    GRIPPER_CLOSE_VELOCITY,
    GRIPPER_OPEN_VELOCITY,
    MAX_STEPS,
    ACTION_DIM,
    OBSERVATION_DIM,
    DifficultyConfig,
)
from .utils import (
    to_numpy,
    find_robot_links,
    compute_gripper_center,
    compute_finger_width,
)
from .rewards import (
    reward_approach,
    reward_grasp,
    reward_lift,
    penalty_premature_closing,
    penalty_base_collision,
    check_success,
)

class FrankaReachEnv(gym.Env):
    """Ambiente di reinforcement learning per il task di pick-and-place con Franka Emika Panda"""
    
    def __init__(self, render_mode=False, difficulty=3):
        """Inizializza l'ambiente
        
        Args:
            render_mode: Se True, mostra il viewer 3D
            difficulty: Livello di difficoltà (0=easy, 1=medium, 2+=hard)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.max_steps = MAX_STEPS
        self.current_step = 0
        
        self._init_genesis()
        self._setup_scene()
        self._setup_spaces()
    
    def _init_genesis(self):
        """Inizializza il backend Genesis"""
        target_backend = gs.gpu if self.render_mode else gs.cpu
        gs.init(backend=target_backend, logging_level='warning')
        
        self.scene = gs.Scene(
            show_viewer=self.render_mode,
            rigid_options=gs.options.RigidOptions(
                dt=SIMULATION_DT,
                gravity=GRAVITY,
            ),
        )
    
    def _setup_scene(self):
        """Configura la scena con piano, robot e cubo"""
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=ROBOT_URDF)
        )
        
        self.left_finger_name, self.right_finger_name, self.hand_name = (
            find_robot_links(self.robot)
        )
        
        print(f"🔧 Config: L='{self.left_finger_name}', "
              f"R='{self.right_finger_name}', Hand='{self.hand_name}'")
        
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=CUBE_SIZE, pos=CUBE_INITIAL_POS),
            surface=gs.surfaces.Default(color=CUBE_COLOR),
            material=gs.materials.Rigid(friction=CUBE_FRICTION)
        )
        
        self.scene.build()
    
    def _setup_spaces(self):
        """Definisce gli spazi di azione e osservazione"""
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Resetta l'ambiente a uno stato iniziale
        
        Args:
            seed: Seed per la generazione casuale
            options: Opzioni aggiuntive
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        self.current_step = 0
        
        config = DifficultyConfig.get(self.difficulty)
        
        self._reset_robot(config)
        self._reset_cube(config)
        
        self.scene.step()
        
        return self._get_obs(), {}
    
    def _reset_robot(self, config):
        """Resetta la posizione del robot
        
        Args:
            config: Configurazione di difficoltà
        """
        qpos = self.robot.get_dofs_position()
        
        home_pose = torch.tensor(
            ROBOT_HOME_POSE, device=qpos.device, dtype=qpos.dtype
        )
        qpos[1:7] = home_pose
        qpos[0] = np.random.uniform(*config['joint_0_range'])
        
        self.robot.set_dofs_position(qpos)
        self.robot.set_dofs_velocity(torch.zeros(9, device=qpos.device))
    
    def _reset_cube(self, config):
        """Resetta la posizione del cubo
        
        Args:
            config: Configurazione di difficoltà
        """
        qpos = self.robot.get_dofs_position()
        
        angle = np.random.uniform(*config['angle_range'])
        radius = np.random.uniform(*config['radius_range'])
        
        cube_x = radius * np.cos(angle)
        cube_y = radius * np.sin(angle)
        
        self.cube.set_pos([cube_x, cube_y, CUBE_INITIAL_HEIGHT])
        self.cube.set_dofs_velocity(torch.zeros(6, device=qpos.device))

    def step(self, action):
        """Esegue un passo di simulazione
        
        Args:
            action: Azione da eseguire
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        motor_commands = self._compute_motor_commands(action)
        
        self.robot.control_dofs_velocity(motor_commands)
        self.scene.step()
        
        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def _compute_motor_commands(self, action):
        """Converte le azioni in comandi motori
        
        Args:
            action: Azione normalizzata [-1, 1]
            
        Returns:
            numpy.ndarray: Comandi di velocità per i motori
        """
        motor_commands = np.zeros(ACTION_DIM, dtype=np.float32)
        
        motor_commands[:7] = action[:7] * ARM_VELOCITY_SCALE
        
        grip_vel = GRIPPER_CLOSE_VELOCITY if action[7] > 0 else GRIPPER_OPEN_VELOCITY
        motor_commands[7] = grip_vel
        motor_commands[8] = grip_vel
        
        return motor_commands

    def _get_obs(self):
        """Costruisce il vettore di osservazione
        
        Returns:
            numpy.ndarray: Vettore di osservazione (27D)
        """
        qpos = self.robot.get_dofs_position()
        qvel = self.robot.get_dofs_velocity()
        cube_pos = self.cube.get_pos()
        cube_vel = self.cube.get_dofs_velocity()
        
        if self.left_finger_name and self.right_finger_name:
            ee_pos = compute_gripper_center(
                self.robot, self.left_finger_name, self.right_finger_name
            )
        else:
            ee_pos = to_numpy(self.robot.get_link(self.hand_name).get_pos())
        
        np_ee = ee_pos if isinstance(ee_pos, np.ndarray) else to_numpy(ee_pos)
        np_cube = to_numpy(cube_pos)
        relative_pos = np_cube - np_ee
        
        return np.concatenate([
            to_numpy(qpos),
            to_numpy(qvel),
            relative_pos,
            np_cube,
            to_numpy(cube_vel)[:3]
        ]).astype(np.float32)

    def _compute_reward(self, action):
        """Calcola la reward per lo stato corrente
        
        Args:
            action: Azione eseguita
            
        Returns:
            tuple: (reward, terminated)
        """
        cube_pos = to_numpy(self.cube.get_pos())
        gripper_center = compute_gripper_center(
            self.robot, self.left_finger_name, self.right_finger_name
        )
        finger_width = compute_finger_width(
            self.robot, self.left_finger_name, self.right_finger_name
        )
        
        delta = gripper_center - cube_pos
        dist_xy = np.linalg.norm(delta[:2])
        dist_z = np.abs(delta[2])
        cube_z = cube_pos[2]
        
        is_closing = action is not None and to_numpy(action)[7] > 0
        
        reward = 0.0
        reward += reward_approach(dist_xy, dist_z, finger_width)
        
        grasp_reward, is_holding = reward_grasp(
            is_closing, finger_width, dist_xy, gripper_center
        )
        reward += grasp_reward
        reward += reward_lift(cube_z, is_holding)
        reward += penalty_base_collision(gripper_center)
        reward += penalty_premature_closing(is_closing, finger_width, dist_xy)
        
        terminated, success_reward = check_success(
            cube_z, self.current_step, self.max_steps
        )
        reward += success_reward
        
        return reward, terminated