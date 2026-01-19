"""
Modulo per il calcolo delle reward e penalty per l'ambiente Franka Reach.
Contiene tutte le componenti modulari del sistema di ricompense.
"""

import numpy as np
from .config import (
    APPROACH_DISTANCE_WEIGHT,
    APPROACH_ZONE_THRESHOLD,
    APPROACH_ZONE_BONUS,
    APPROACH_VERTICAL_WEIGHT,
    APPROACH_OPEN_GRIPPER_THRESHOLD,
    APPROACH_OPEN_GRIPPER_BONUS,
    GRASP_MIN_WIDTH,
    GRASP_MAX_WIDTH,
    GRASP_MAX_DISTANCE_XY,
    GRASP_BASE_REWARD,
    GRASP_DRAG_PENALTY,
    GRASP_MIN_HEIGHT,
    LIFT_MIN_HEIGHT,
    LIFT_BASE_REWARD,
    LIFT_HEIGHT_MULTIPLIER,
    PENALTY_PREMATURE_CLOSE_DISTANCE,
    PENALTY_PREMATURE_CLOSE_WIDTH,
    PENALTY_PREMATURE_CLOSE_VALUE,
    PENALTY_BASE_COLLISION_DISTANCE,
    PENALTY_BASE_COLLISION_VALUE,
    SUCCESS_HEIGHT,
    SUCCESS_BASE_REWARD,
    SUCCESS_TIME_BONUS,
)


def reward_approach(dist_xy, dist_z, finger_width):
    """Incoraggia ad avvicinarsi al cubo con le dita aperte
    
    Args:
        dist_xy: Distanza in piano XY tra gripper e cubo
        dist_z: Distanza verticale tra gripper e cubo
        finger_width: Larghezza corrente del gripper
        
    Returns:
        float: Reward per l'avvicinamento
    """
    reward = 0.0
    reward -= (dist_xy * APPROACH_DISTANCE_WEIGHT)
    
    if dist_xy < APPROACH_ZONE_THRESHOLD:
        reward += APPROACH_ZONE_BONUS
        reward -= (dist_z * APPROACH_VERTICAL_WEIGHT)
        
        if finger_width > APPROACH_OPEN_GRIPPER_THRESHOLD:
            reward += APPROACH_OPEN_GRIPPER_BONUS
            
    return reward


def reward_grasp(is_closing, finger_width, dist_xy, gripper_center):
    """Premia la presa stabile
    
    Args:
        is_closing: Se il gripper sta chiudendo
        finger_width: Larghezza corrente del gripper
        dist_xy: Distanza in piano XY tra gripper e cubo
        gripper_center: Posizione centrale del gripper
        
    Returns:
        tuple: (reward, is_holding)
    """
    is_holding = (
        is_closing and 
        (finger_width > GRASP_MIN_WIDTH) and 
        (finger_width < GRASP_MAX_WIDTH) and
        (dist_xy < GRASP_MAX_DISTANCE_XY)
    )
    
    if not is_holding:
        return 0.0, False
        
    reward = GRASP_BASE_REWARD
    
    if gripper_center[2] < GRASP_MIN_HEIGHT:
        reward += GRASP_DRAG_PENALTY
        
    return reward, is_holding


def reward_lift(cube_z, is_holding):
    """Premia il sollevamento (solo se sta tenendo l'oggetto)
    
    Args:
        cube_z: Altezza corrente del cubo
        is_holding: Se il gripper sta tenendo il cubo
        
    Returns:
        float: Reward per il sollevamento
    """
    if not is_holding or cube_z <= LIFT_MIN_HEIGHT:
        return 0.0
        
    reward = LIFT_BASE_REWARD
    reward += (cube_z * LIFT_HEIGHT_MULTIPLIER)
    return reward


def penalty_premature_closing(is_closing, finger_width, dist_xy):
    """Punisce chiudere le dita lontano dall'oggetto
    
    Args:
        is_closing: Se il gripper sta chiudendo
        finger_width: Larghezza corrente del gripper
        dist_xy: Distanza in piano XY tra gripper e cubo
        
    Returns:
        float: Penalty per chiusura prematura
    """
    if (is_closing and 
        dist_xy > PENALTY_PREMATURE_CLOSE_DISTANCE and 
        finger_width < PENALTY_PREMATURE_CLOSE_WIDTH):
        return PENALTY_PREMATURE_CLOSE_VALUE
    return 0.0


def penalty_base_collision(gripper_center):
    """Evita che il robot si colpisca da solo
    
    Args:
        gripper_center: Posizione centrale del gripper
        
    Returns:
        float: Penalty per collisione con base
    """
    dist_base = np.linalg.norm(gripper_center[:2])
    if dist_base < PENALTY_BASE_COLLISION_DISTANCE:
        return PENALTY_BASE_COLLISION_VALUE
    return 0.0


def check_success(cube_z, current_step, max_steps):
    """Verifica condizione vittoria
    
    Args:
        cube_z: Altezza corrente del cubo
        current_step: Step corrente
        max_steps: Numero massimo di step
        
    Returns:
        tuple: (terminated, success_reward)
    """
    if cube_z >= SUCCESS_HEIGHT:
        step_left = max_steps - current_step
        return True, (SUCCESS_BASE_REWARD + (step_left * SUCCESS_TIME_BONUS))
    return False, 0.0
