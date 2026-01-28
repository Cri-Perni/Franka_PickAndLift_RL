import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn
import os
import glob

# Importiamo l'ambiente dal file franka_env.py
from env.franka_env import FrankaReachEnv

# Definizione dell'architettura personalizzata
# 'pi' = Policy (Attore) -> Chi muove il robot
# 'vf' = Value Function (Critico) -> Chi giudica l'azione
policy_kwargs = dict(
    activation_fn=nn.Tanh,  # Tanh è OBBLIGATORIA per robotica continua (evita scatti bruschi)
    net_arch=dict(
        pi=[512, 512, 256], # Actor: 3 layer per gestire la complessità a 360°
        vf=[512, 512, 256]  # Critic: Separato e potente
    ),
    log_std_init=0, 
)

def get_latest_model(log_dir="logs/"):
    """Cerca il file .zip più recente nella cartella logs"""
    list_of_files = glob.glob(f'{log_dir}/*.zip') 
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def main():
    # --- 1. CONFIGURAZIONE BASE ---
    num_envs = 40
    log_dir = "./logs/"
    
    # Crea cartella logs se non esiste
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "="*50)
    print("🤖 GESTORE TRAINING ROBOT")
    print("="*50)
    print("1. 🆕 NUOVO Addestramento (Parte da zero)")
    print("2. 🔄 CONTINUA Addestramento (Carica ultimo checkpoint)")
    print("3. 📂 CONTINUA Addestramento (Scegli file specifico)")
    
    choice = input("\nScegli un'opzione (1/2/3): ").strip()
    
    model_path = None
    reset_timesteps = False

    # --- 2. LOGICA DI SCELTA ---
    if choice == "1":
        print("\n>>> Hai scelto: NUOVO MODELLO. Il cervello sarà resettato.")
        confirm = input("Sei sicuro? (s/n): ").lower()
        if confirm != 's': return
        model_path = None # Nessun caricamento

    elif choice == "2":
        print("\n>>> Cerco l'ultimo salvataggio...")
        latest = get_latest_model(log_dir)
        if latest:
            print(f"✅ Trovato: {latest}")
            model_path = latest
        else:
            print("❌ Nessun modello trovato in logs/! Inizio da zero.")
            model_path = None

    elif choice == "3":
        custom_path = input("Inserisci il percorso del file .zip (es. logs/mio_modello.zip): ")
        if os.path.exists(custom_path):
            model_path = custom_path
        else:
            print(f"❌ File non trovato: {custom_path}")
            return
    else:
        print("Scelta non valida.")
        return

    # --- 3. CREAZIONE AMBIENTE ---
    print(f"\n🚀 Avvio simulazione su {num_envs} processi...")
    env = make_vec_env(
        FrankaReachEnv, 
        n_envs=num_envs, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs={"render_mode": False}
    )

    # --- 4. INIZIALIZZAZIONE MODELLO ---
    if model_path is None:
        # NUOVO MODELLO
        print("🧠 Creazione NUOVO cervello PPO...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            device="cuda",              # Usa la GPU
            policy_kwargs=policy_kwargs,
    
            # --- PARAMETRI DI TRAINING ---
            learning_rate=3e-4,
    
            # Batch Size: Quanti dati passiamo alla GPU in un colpo solo per l'update.
            batch_size=2048,            
    
            # N_Steps: Quanti step raccoglie OGNI env prima di fare un update.
            # Se hai 20 env, il buffer totale sarà 20 * n_steps.
            # Per robotica, orizzonti lunghi aiutano.
            n_steps=1024,               
    
            gamma=0.99,                 # Sconto futuro
            gae_lambda=0.95,            # Stabilità del gradiente
    
            # --- IL SEGRETO PER IL 360 GRADI ---
            ent_coef=0.01,
    
            use_sde=True,               # ATTIVA gSDE!
            sde_sample_freq=4,          # Rampiona il rumore ogni 4 step (movimenti più coerenti)
    
            tensorboard_log="./tensorboard_logs/"
        )
        reset_timesteps = True
        save_name = "franka_new"
    else:
        # CARICAMENTO
        print(f"🧠 Caricamento pesi da: {model_path}")
        model = PPO.load(
            model_path,
            env=env,
            device="cuda",
            # custom_objects={"learning_rate": 0.0001} # Scommenta se vuoi abbassare il LR
        )
        reset_timesteps = True # Continua a contare gli step da dove era rimasto
        save_name = "franka_continued"

    # --- 5. TRAINING ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path=log_dir, 
        name_prefix=save_name
    )

    print(f"\n⚡ Inizio Training! Target: 6.000.000 steps.")
    print("   (Premi Ctrl+C per fermare e salvare)\n")
    
    try:
        model.learn(
            total_timesteps=6_000_000, 
            callback=checkpoint_callback, 
            reset_num_timesteps=reset_timesteps
        )
    except KeyboardInterrupt:
        print("\n🛑 Interrotto manualmente!")

    # --- 6. SALVATAGGIO FINALE ---
    final_name = f"{save_name}_final"
    model.save(final_name)
    print(f"✅ Salvato come '{final_name}.zip'")
    
    env.close()

if __name__ == "__main__":
    main()
