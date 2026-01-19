"""
Funzioni di utilità per l'ambiente Franka Reach.
"""

import torch
import numpy as np


def to_numpy(x):
    """Converte un tensore PyTorch o altro tipo in numpy array
    
    Args:
        x: Tensore PyTorch, numpy array, o valore convertibile
        
    Returns:
        numpy.ndarray
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def find_robot_links(robot):
    """Identifica i link del gripper del robot
    
    Args:
        robot: Entità robot di Genesis
        
    Returns:
        tuple: (left_finger_name, right_finger_name, hand_name)
    """
    left_finger_name = None
    right_finger_name = None
    hand_name = "link7"  # Fallback per orientamento
    
    found_links = [l.name for l in robot.links]
    for name in found_links:
        if "left" in name and "finger" in name:
            left_finger_name = name
        if "right" in name and "finger" in name:
            right_finger_name = name
        if "hand" in name or "link7" in name:
            hand_name = name
    
    return left_finger_name, right_finger_name, hand_name


def compute_gripper_center(robot, left_finger_name, right_finger_name):
    """Calcola la posizione centrale del gripper
    
    Args:
        robot: Entità robot di Genesis
        left_finger_name: Nome del link del dito sinistro
        right_finger_name: Nome del link del dito destro
        
    Returns:
        numpy.ndarray: Posizione centrale del gripper
    """
    l_pos = to_numpy(robot.get_link(left_finger_name).get_pos())
    r_pos = to_numpy(robot.get_link(right_finger_name).get_pos())
    return (l_pos + r_pos) / 2.0


def compute_finger_width(robot, left_finger_name, right_finger_name):
    """Calcola la distanza tra le dita del gripper
    
    Args:
        robot: Entità robot di Genesis
        left_finger_name: Nome del link del dito sinistro
        right_finger_name: Nome del link del dito destro
        
    Returns:
        float: Distanza tra le dita
    """
    l_pos = to_numpy(robot.get_link(left_finger_name).get_pos())
    r_pos = to_numpy(robot.get_link(right_finger_name).get_pos())
    return np.linalg.norm(l_pos - r_pos)
