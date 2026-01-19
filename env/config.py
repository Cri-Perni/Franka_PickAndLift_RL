"""
Configurazioni e costanti per l'ambiente Franka Reach.
"""

import numpy as np


# === PARAMETRI SIMULAZIONE ===
SIMULATION_DT = 0.01
GRAVITY = (0.0, 0.0, -9.81)

# === PARAMETRI ROBOT ===
ROBOT_URDF = 'xml/franka_emika_panda/panda.xml'
ROBOT_HOME_POSE = [-0.2, 0.0, -2.2, 0.0, 2.0, 0.78]

# === PARAMETRI CUBO ===
CUBE_SIZE = (0.045, 0.045, 0.045)
CUBE_INITIAL_POS = (0.6, 0.0, 0.02)
CUBE_INITIAL_HEIGHT = 0.02
CUBE_COLOR = (0.0, 1.0, 0.0, 1.0)  # Verde
CUBE_FRICTION = 1.5

# === PARAMETRI CONTROLLO ===
ARM_VELOCITY_SCALE = 2.0
GRIPPER_CLOSE_VELOCITY = -0.4  # Velocità ridotta per evitare compenetrazione
GRIPPER_OPEN_VELOCITY = 1.0

# === PARAMETRI AMBIENTE ===
MAX_STEPS = 500
ACTION_DIM = 9
OBSERVATION_DIM = 27

# === PARAMETRI REWARD ===
# Success
SUCCESS_HEIGHT = 0.20  # Altezza per considerare il task completato (m)
SUCCESS_BASE_REWARD = 100.0
SUCCESS_TIME_BONUS = 1.0  # Reward per step risparmiato

# Grasp
GRASP_MIN_WIDTH = 0.035  # Larghezza minima gripper per considerare una presa (m)
GRASP_MAX_WIDTH = 0.075  # Larghezza massima gripper per considerare una presa (m)
GRASP_MAX_DISTANCE_XY = 0.05  # Distanza massima in XY per considerare una presa (m)
GRASP_BASE_REWARD = 2.0
GRASP_DRAG_PENALTY = -1.0
GRASP_MIN_HEIGHT = 0.05  # Altezza minima gripper per evitare dragging (m)

# Lift
LIFT_MIN_HEIGHT = 0.03  # Altezza minima cubo per iniziare a premiare (m)
LIFT_BASE_REWARD = 5.0
LIFT_HEIGHT_MULTIPLIER = 20.0

# Approach
APPROACH_DISTANCE_WEIGHT = 2.0
APPROACH_ZONE_THRESHOLD = 0.10  # Soglia per "zona calda" (m)
APPROACH_ZONE_BONUS = 0.5
APPROACH_VERTICAL_WEIGHT = 2.0
APPROACH_OPEN_GRIPPER_THRESHOLD = 0.04  # Larghezza minima per bonus (m)
APPROACH_OPEN_GRIPPER_BONUS = 0.2

# Penalties
PENALTY_PREMATURE_CLOSE_DISTANCE = 0.10  # Distanza oltre la quale chiudere è penalizzato (m)
PENALTY_PREMATURE_CLOSE_WIDTH = 0.04  # Larghezza sotto la quale penalizzare (m)
PENALTY_PREMATURE_CLOSE_VALUE = -0.5
PENALTY_BASE_COLLISION_DISTANCE = 0.15  # Distanza minima dalla base (m)
PENALTY_BASE_COLLISION_VALUE = -2.0

# === CONFIGURAZIONI DI DIFFICOLTÀ ===
class DifficultyConfig:
    """Configurazioni per diversi livelli di difficoltà"""
    
    EASY = {
        'name': 'easy',
        'joint_0_range': (0.0, 0.0),  # Fisso
        'angle_range': (-0.3, 0.3),
        'radius_range': (0.45, 0.55),
    }
    
    MEDIUM = {
        'name': 'medium',
        'joint_0_range': (-0.5, 0.5),
        'angle_range': (-1.5, 1.5),
        'radius_range': (0.40, 0.65),
    }
    
    HARD = {
        'name': 'hard',
        'joint_0_range': (-np.pi, np.pi),
        'angle_range': (-np.pi, np.pi),
        'radius_range': (0.35, 0.70),
    }
    
    @classmethod
    def get(cls, difficulty_level):
        """Ottiene la configurazione per il livello specificato
        
        Args:
            difficulty_level: 0 (easy), 1 (medium), 2+ (hard)
        """
        if difficulty_level == 0:
            return cls.EASY
        elif difficulty_level == 1:
            return cls.MEDIUM
        else:
            return cls.HARD
