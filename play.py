import gymnasium as gym
from stable_baselines3 import PPO
from franka_env import FrankaReachEnv
import numpy as np
import time
import os

def main():
    # Nome del file salvato (senza .zip)
    model_name = "franka_new_final"
    
    if not os.path.exists(f"{model_name}.zip"):
        print(f"❌ ERRORE: Non trovo il file '{model_name}.zip'.")
        print("Controlla nella cartella se il file ha un nome diverso (es. dentro 'logs/')")
        return

    print(f"🧠 Caricamento cervello: {model_name}...")
    
    env = FrankaReachEnv(render_mode=True)
    
    
    model = PPO.load(model_name)
    
    obs, _ = env.reset()
    print("✅ AVVIO! Guarda la finestra 3D.")
    print("Premi Ctrl+C qui nel terminale per uscire.")

    while True:
        # L'AI decide l'azione
        action, _ = model.predict(obs, deterministic=True)
        
        # Eseguiamo
        obs, reward, done, _, _ = env.step(action)
        remain =env.max_steps - env.current_step
        
        # Rallentiamo la scena (60 FPS circa)
        time.sleep(0.016)
        
        if done:
            print(f"Target raggiunto/perso! Reward: {reward:.2f}, Steps remaining: {remain}")
            obs, _ = env.reset()
            time.sleep(0.5)

if __name__ == "__main__":
    main()
