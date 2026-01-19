import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from franka_env import FrankaReachEnv

def run_detailed_evaluation(model_path, num_episodes=100):
    # 1. Setup Ambiente (Senza render per velocità)
    env = FrankaReachEnv(render_mode=False)
    env = Monitor(env)
    
    # 2. Caricamento Modello su CPU
    print(f"📂 Caricamento modello: {model_path}...")
    model = PPO.load(model_path, device='cpu')
    
    print(f"🧪 Valutazione in corso su {num_episodes} episodi...")
    
    # 3. Valutazione manuale per tracciare i terminated
    rewards_list = []
    lengths_list = []
    successes_list = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_success = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Il successo è quando l'episodio termina con terminated=True
            # (non truncated, che indica solo timeout)
            if terminated:
                episode_success = True
                done = True
            elif truncated:
                done = True
        
        rewards_list.append(episode_reward)
        lengths_list.append(episode_length)
        successes_list.append(episode_success)
    
    # 4. Analisi dei dati
    rewards = np.array(rewards_list)
    lengths = np.array(lengths_list)
    successes = np.array(successes_list)
    
    success_rate = (np.sum(successes) / num_episodes) * 100
    
    mean_rew = np.mean(rewards)
    std_rew = np.std(rewards)
    min_rew = np.min(rewards)
    max_rew = np.max(rewards)
    
    # 5. Output formattato
    print("\n" + "═"*50)
    print(f"📊 REPORT PRESTAZIONI: {model_path}")
    print("═"*50)
    print(f"🏆 SUCCESS RATE:      {success_rate:.1f}% ({int(np.sum(successes))}/{num_episodes} episodi)")
    print(f"📈 REWARD MEDIA:      {mean_rew:.2f} (± {std_rew:.2f})")
    print(f"🔝 REWARD MASSIMA:    {max_rew:.2f}")
    print(f"📉 REWARD MINIMA:     {min_rew:.2f}")
    print(f"⏱️ DURATA MEDIA:      {np.mean(lengths):.1f} step")
    print("═"*50)
    
    # Analisi successi vs fallimenti
    if success_rate > 0:
        successful_episodes = rewards[successes]
        failed_episodes = rewards[~successes]
        
        print(f"✅ EPISODI DI SUCCESSO (terminated=True):")
        print(f"   Reward media: {np.mean(successful_episodes):.2f}")
        print(f"   Durata media: {np.mean(lengths[successes]):.1f} step")
        
        if len(failed_episodes) > 0:
            print(f"\n❌ EPISODI FALLITI (timeout senza terminated):")
            print(f"   Reward media: {np.mean(failed_episodes):.2f}")
            print(f"   Durata media: {np.mean(lengths[~successes]):.1f} step")
    else:
        print("❌ Nessun episodio completato con successo (cubo mai sollevato a 0.20m)")
        print("💡 Il robot raggiunge il timeout senza mai triggare terminated=True")
    
    print("═"*50)
    
    env.close()

if __name__ == "__main__":
    # Inserisci il nome esatto del tuo file .zip
    PATH = "franka_new_final.zip" 
    run_detailed_evaluation(PATH)