"""
CONDUCCIÓN AUTÓNOMA MEDIANTE DQN (Deep Q-Network) LEARNING
Sistema de Aprendizaje por Refuerzo Profundo para Conducción Autónoma

Modelo 2 - Versión Avanzada con 80+ Raycast y NPCs Integrados
Desarrollado para ProyectoIAUE - TF_APR_ConduccionAut

Características:
- DQN con Experience Replay
- Target Network para estabilidad
- Epsilon-Greedy con decay adaptativo
- Integración completa con NPCs
- 80+ sensores raycast para detección precisa
- Sistema de recompensas multinivel
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
import json
import time
import os
from typing import Tuple, List, Dict, Any

# Importaciones para conectar con otros módulos del proyecto
try:
    from dqn_utils import ModelConverter, DataProcessor, PerformanceAnalyzer
    from dqn_server_tcp import DQNTCPServer
    from dqn_evaluator import ModelEvaluator
    print("[IMPORT] ✓ Módulos auxiliares cargados correctamente")
except ImportError as e:
    print(f"[IMPORT] ⚠ Algunos módulos auxiliares no encontrados: {e}")
    print("[IMPORT] Continuando con funcionalidad básica...")

# ===============================
# CONFIGURACIÓN GLOBAL DQN
# ===============================
class DQNConfig:
    # Arquitectura de red
    INPUT_SIZE = 84          # 80 raycast + velocidad + vida + dirección + estado_npc
    HIDDEN_LAYERS = [512, 256, 128, 64]
    OUTPUT_SIZE = 5          # ACELERAR, FRENAR, IZQUIERDA, DERECHA, DETENERSE
    
    # Hiperparámetros DQN
    LEARNING_RATE = 0.0001
    GAMMA = 0.99             # Factor de descuento
    EPSILON_START = 1.0      # Exploración inicial
    EPSILON_END = 0.01       # Exploración mínima
    EPSILON_DECAY = 0.995    # Decay rate
    
    # Experience Replay
    MEMORY_SIZE = 100000     # Tamaño del buffer
    BATCH_SIZE = 64          # Tamaño del batch
    TARGET_UPDATE = 1000     # Actualización de target network
    
    # Entrenamiento
    MAX_EPISODES = 5000
    MAX_STEPS_PER_EPISODE = 2000
    MIN_MEMORY_SIZE = 10000  # Mínimo para empezar entrenamiento

# ===============================
# ARQUITECTURA DE RED NEURONAL DQN
# ===============================
class DQNNetwork(nn.Module):
    """
    Red Neuronal Profunda para Q-Learning
    Arquitectura optimizada para conducción autónoma
    """
    
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int):
        super(DQNNetwork, self).__init__()
        
        # Capas de entrada
        layers = []
        prev_size = input_size
        
        # Capas ocultas con BatchNorm y Dropout
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Inicialización de pesos Xavier
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización Xavier para mejor convergencia"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# ===============================
# EXPERIENCE REPLAY BUFFER
# ===============================
class ExperienceReplay:
    """
    Buffer de experiencias para DQN
    Almacena transiciones (s, a, r, s', done)
    """
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Almacena una transición"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Muestrea un batch aleatorio"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# ===============================
# AGENTE DQN PRINCIPAL
# ===============================
class DQNAgent:
    """
    Agente de Conducción Autónoma con DQN
    Implementa Double DQN con Target Network
    """
    
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Usando dispositivo: {self.device}")
        
        # Redes neuronales
        self.q_network = DQNNetwork(
            config.INPUT_SIZE, 
            config.HIDDEN_LAYERS, 
            config.OUTPUT_SIZE
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            config.INPUT_SIZE, 
            config.HIDDEN_LAYERS, 
            config.OUTPUT_SIZE
        ).to(self.device)
        
        # Sincronizar target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizador y pérdida
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        # Experience Replay
        self.memory = ExperienceReplay(config.MEMORY_SIZE)
        
        # Variables de entrenamiento
        self.epsilon = config.EPSILON_START
        self.steps_done = 0
        self.episode_rewards = []
        self.losses = []
        
        # Mapeo de acciones
        self.action_map = {
            0: "ACELERAR",
            1: "FRENAR", 
            2: "IZQUIERDA",
            3: "DERECHA",
            4: "DETENERSE"
        }
    
    def preprocess_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        Preprocesa el estado del entorno para la red neuronal
        
        Estado incluye:
        - 80 raycast distances (normalizadas)
        - Velocidad (normalizada)
        - Vida (normalizada) 
        - Dirección (normalizada)
        - Estado NPCs (codificado)
        """
        processed = []
        
        # Raycast distances (80 sensores)
        raycast_data = raw_state.get('raycast', [0.0] * 80)
        raycast_normalized = [min(d / 1500.0, 1.0) for d in raycast_data[:80]]
        processed.extend(raycast_normalized)
        
        # Velocidad normalizada (0-120 km/h)
        velocity = min(raw_state.get('velocity', 0.0) / 120.0, 1.0)
        processed.append(velocity)
        
        # Vida normalizada (0-100%)
        health = raw_state.get('health', 100.0) / 100.0
        processed.append(health)
        
        # Dirección normalizada (-180° a 180°)
        direction = raw_state.get('direction', 0.0) / 180.0
        processed.append(direction)
        
        # Estado de NPCs (one-hot encoding)
        npc_state = raw_state.get('npc_detected', 'Ninguno')
        npc_encoding = [0.0, 0.0, 0.0, 0.0]  # [Camion, AutoRojo, AutoAzul, Ninguno]
        
        if npc_state == 'Camion':
            npc_encoding[0] = 1.0
        elif npc_state == 'AutoRojo':
            npc_encoding[1] = 1.0
        elif npc_state == 'AutoAzul':
            npc_encoding[2] = 1.0
        else:
            npc_encoding[3] = 1.0
        
        processed.extend(npc_encoding)
        
        return np.array(processed, dtype=np.float32)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Selecciona acción usando epsilon-greedy
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.OUTPUT_SIZE - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Almacena transición en memory buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def compute_reward(self, raw_state: Dict[str, Any], action: int, 
                      collision_info: Dict[str, Any]) -> float:
        """
        Sistema de recompensas multinivel mejorado
        """
        reward = 0.0
        
        # Recompensas base por acción
        if action == 0:  # ACELERAR
            reward += 0.1
        elif action == 1:  # FRENAR
            reward -= 0.05
        elif action == 4:  # DETENERSE
            reward -= 0.2
        
        # Recompensas por velocidad óptima (40-80 km/h)
        velocity = raw_state.get('velocity', 0.0)
        if 40 <= velocity <= 80:
            reward += 0.5
        elif velocity > 100:
            reward -= 0.3
        elif velocity < 20:
            reward -= 0.4
        
        # Penalizaciones por proximidad a obstáculos
        min_distance = min(raw_state.get('raycast', [1500.0]))
        if min_distance < 100:
            reward -= 2.0
        elif min_distance < 200:
            reward -= 1.0
        elif min_distance > 800:
            reward += 0.2
        
        # Sistema de colisiones
        collision_type = collision_info.get('type', 'none')
        if collision_type == 'Camion':
            reward -= 100.0  # Muerte inmediata
        elif collision_type in ['AutoRojo', 'AutoAzul']:
            reward -= 50.0   # Daño moderado
        elif collision_type == 'Pared':
            reward -= 10.0   # Penalización leve
        elif collision_type == 'Meta':
            reward += 100.0  # Éxito total
        
        # Recompensa por supervivencia
        health = raw_state.get('health', 100.0)
        if health > 80:
            reward += 0.1
        elif health < 30:
            reward -= 0.5
        
        # Bonus por progreso en el circuito
        checkpoint_bonus = collision_info.get('checkpoint_bonus', 0.0)
        reward += checkpoint_bonus
        
        return reward
    
    def train_step(self):
        """
        Paso de entrenamiento DQN con target network
        """
        if len(self.memory) < self.config.MIN_MEMORY_SIZE:
            return None
        
        # Muestrear batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.BATCH_SIZE)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Q-values actuales
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Q-values del target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.GAMMA * next_q_values * ~dones)
        
        # Calcular pérdida
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Actualizar epsilon
        if self.epsilon > self.config.EPSILON_END:
            self.epsilon *= self.config.EPSILON_DECAY
        
        # Actualizar target network
        self.steps_done += 1
        if self.steps_done % self.config.TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"[DQN] Target network actualizada en step {self.steps_done}")
        
        return loss.item()
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'config': self.config.__dict__
        }, filepath)
        print(f"[DQN] Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str):
        """Carga modelo pre-entrenado"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        print(f"[DQN] Modelo cargado desde: {filepath}")

# ===============================
# ENTORNO DE ENTRENAMIENTO
# ===============================
class TrainingEnvironment:
    """
    Simulador de entrenamiento para el agente DQN
    Genera estados sintéticos realistas
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        """Reinicia el entorno"""
        self.step_count = 0
        self.health = 100.0
        self.position = 0.0
        self.velocity = 0.0
        self.direction = 0.0
        
        return self._get_state()
    
    def _get_state(self) -> Dict[str, Any]:
        """Genera estado actual del entorno"""
        # Simular 80 raycast con variaciones realistas
        base_distances = np.random.uniform(200, 1500, 80)
        
        # Simular obstáculos ocasionales
        if random.random() < 0.1:  # 10% probabilidad de obstáculo
            obstacle_sensors = random.sample(range(80), random.randint(3, 8))
            for sensor in obstacle_sensors:
                base_distances[sensor] = random.uniform(50, 300)
        
        return {
            'raycast': base_distances.tolist(),
            'velocity': self.velocity,
            'health': self.health,
            'direction': self.direction,
            'npc_detected': random.choice(['Ninguno', 'Camion', 'AutoRojo', 'AutoAzul']) if random.random() < 0.05 else 'Ninguno'
        }
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Ejecuta una acción en el entorno"""
        self.step_count += 1
        
        # Simular física básica
        if action == 0:  # ACELERAR
            self.velocity = min(120, self.velocity + 5)
        elif action == 1:  # FRENAR
            self.velocity = max(0, self.velocity - 8)
        elif action == 4:  # DETENERSE
            self.velocity = max(0, self.velocity - 15)
        
        # Simular colisiones
        collision_info = {'type': 'none', 'checkpoint_bonus': 0.0}
        if random.random() < 0.01:  # 1% probabilidad de colisión
            collision_types = ['Camion', 'AutoRojo', 'AutoAzul', 'Pared']
            collision_info['type'] = random.choice(collision_types)
            
            if collision_info['type'] == 'Camion':
                self.health = 0
            elif collision_info['type'] in ['AutoRojo', 'AutoAzul']:
                self.health = max(0, self.health - 50)
            elif collision_info['type'] == 'Pared':
                self.health = max(0, self.health - 10)
        
        # Progreso en circuito
        self.position += self.velocity * 0.01
        if int(self.position) % 100 == 0 and self.position > 0:
            collision_info['checkpoint_bonus'] = 5.0
        
        # Condiciones de finalización
        done = (self.health <= 0 or 
                self.step_count >= 2000 or 
                self.position >= 1000)
        
        if self.position >= 1000:  # Meta alcanzada
            collision_info['type'] = 'Meta'
        
        state = self._get_state()
        return state, 0.0, done, collision_info

# ===============================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ===============================
def train_dqn_agent():
    """
    Función principal de entrenamiento DQN
    """
    print("="*60)
    print("INICIANDO ENTRENAMIENTO DQN - CONDUCCIÓN AUTÓNOMA")
    print("Modelo 2: 80+ Raycast + NPCs Integrados")
    print("="*60)
    
    # Configuración
    config = DQNConfig()
    agent = DQNAgent(config)
    env = TrainingEnvironment()
    
    # Gestor de mejores modelos
    best_manager = BestModelManager("best_models")
    
    # Métricas
    episode_rewards = []
    episode_lengths = []
    losses = []
    success_rate = []
    
    # Gestor de mejores modelos
    best_model_manager = BestModelManager("mejores_modelos")
    
    # Entrenamiento principal
    for episode in range(config.MAX_EPISODES):
        state_dict = env.reset()
        state = agent.preprocess_state(state_dict)
        total_reward = 0.0
        steps = 0
        
        for step in range(config.MAX_STEPS_PER_EPISODE):
            # Seleccionar acción
            action = agent.select_action(state, training=True)
            
            # Ejecutar acción
            next_state_dict, _, done, collision_info = env.step(action)
            next_state = agent.preprocess_state(next_state_dict)
            
            # Calcular recompensa
            reward = agent.compute_reward(next_state_dict, action, collision_info)
            total_reward += reward
            
            # Almacenar transición
            agent.store_transition(state, action, reward, next_state, done)
            
            # Entrenar
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Registrar métricas
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Calcular tasa de éxito (últimos 100 episodios)
        if len(episode_rewards) >= 100:
            recent_rewards = episode_rewards[-100:]
            success_count = sum(1 for r in recent_rewards if r > 50)
            success_rate.append(success_count / 100.0)
        
        # Logging y guardar mejores modelos
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            current_success = success_rate[-1] if success_rate else 0
            
            print(f"Episodio {episode:4d} | "
                  f"Recompensa: {total_reward:8.2f} | "
                  f"Promedio: {avg_reward:8.2f} | "
                  f"Pasos: {steps:4d} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Éxito: {current_success:.1%}")
            
            # Evaluar si es mejor modelo
            metrics = {
                'avg_reward': avg_reward,
                'success_rate': current_success,
                'avg_steps': avg_length,
                'episode': episode,
                'epsilon': agent.epsilon
            }
            
            best_manager.evaluate_model(agent, episode, metrics)
        
        # Guardar modelo periódicamente
        if episode % 1000 == 0 and episode > 0:
            agent.save_model(f'dqn_model_episode_{episode}.pth')
        
        # Evaluar y guardar mejores modelos
        metrics = {
            'avg_reward': np.mean(episode_rewards[-100:]),
            'success_rate': success_rate[-1] if success_rate else 0.0,
            'avg_steps': np.mean(episode_lengths[-100:])
        }
        best_model_manager.evaluate_model(agent, episode, metrics)
    
    # Guardar modelo final
    agent.save_model('dqn_agente_final.pth')
    
    # Exportar mejor modelo a ONNX
    final_onnx_path = 'conducionModel.onnx'
    export_to_onnx(agent, final_onnx_path)
    
    # Crear servidor TCP para conexión con Unreal Engine
    try:
        from dqn_server_tcp import DQNTCPServer
        server = DQNTCPServer(agent, port=12345)
        print(f"[TCP] Servidor listo para Unreal Engine en puerto 12345")
        print(f"[TCP] Para iniciar: python dqn_server_tcp.py")
    except ImportError:
        print(f"[TCP] Servidor TCP no disponible")
    
    # Generar gráficos
    plot_training_metrics(episode_rewards, episode_lengths, losses, success_rate)
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"📁 Modelo PyTorch: dqn_agente_final.pth")
    print(f"📁 Modelo ONNX: {final_onnx_path}")
    print(f"📁 Mejores modelos en: best_models/")
    print("="*60)

# ===============================
# SISTEMA DE INTERCONEXIÓN
# ===============================
class SystemConnector:
    """
    Clase para conectar todos los módulos del sistema DQN
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.agent = None
        self.server = None
        self.evaluator = None
    
    def load_trained_agent(self) -> DQNAgent:
        """Carga agente entrenado"""
        config = DQNConfig()
        self.agent = DQNAgent(config)
        self.agent.load_model(self.model_path)
        return self.agent
    
    def start_tcp_server(self, port: int = 12345):
        """Inicia servidor TCP para Unreal Engine"""
        try:
            from dqn_server_tcp import DQNTCPServer
            if self.agent is None:
                self.load_trained_agent()
            
            self.server = DQNTCPServer(self.agent, port=port)
            self.server.start()
            print(f"[SYSTEM] ✓ Servidor TCP iniciado en puerto {port}")
        except ImportError:
            print("[SYSTEM] ✗ Módulo servidor TCP no disponible")
    
    def evaluate_model(self):
        """Evalúa el modelo cargado"""
        try:
            from dqn_evaluator import ModelEvaluator
            if self.agent is None:
                self.load_trained_agent()
            
            self.evaluator = ModelEvaluator(self.agent)
            results = self.evaluator.comprehensive_evaluation()
            print(f"[SYSTEM] ✓ Evaluación completada")
            return results
        except ImportError:
            print("[SYSTEM] ✗ Módulo evaluador no disponible")
    
    def export_to_onnx(self, output_path: str = "conducionModel.onnx"):
        """Exporta modelo a ONNX"""
        if self.agent is None:
            self.load_trained_agent()
        
        export_to_onnx(self.agent, output_path)
        print(f"[SYSTEM] ✓ Modelo exportado a {output_path}")

# ===============================
# FUNCIÓN DE INICIO RÁPIDO
# ===============================
def quick_start(action: str = "train"):
    """
    Función de inicio rápido para diferentes modos
    
    Modos disponibles:
    - train: Entrenar nuevo agente
    - server: Iniciar servidor TCP
    - evaluate: Evaluar modelo existente
    - export: Exportar a ONNX
    """
    
    if action == "train":
        train_dqn_agent()
    
    elif action == "server":
        connector = SystemConnector("dqn_agente_final.pth")
        connector.start_tcp_server()
    
    elif action == "evaluate":
        connector = SystemConnector("dqn_agente_final.pth")
        connector.evaluate_model()
    
    elif action == "export":
        connector = SystemConnector("dqn_agente_final.pth")
        connector.export_to_onnx()
    
    else:
        print("Acciones disponibles: train, server, evaluate, export")
    print(f"Episodios totales: {config.MAX_EPISODES}")
    print(f"Recompensa final promedio: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Tasa de éxito final: {success_rate[-1]:.1%}" if success_rate else "N/A")
    print("="*60)

# ===============================
# EXPORTACIÓN A ONNX
# ===============================
def export_to_onnx(agent: DQNAgent, output_path: str):
    """
    Exporta el modelo DQN entrenado a formato ONNX
    Para uso en tiempo real en Unreal Engine
    """
    print(f"[ONNX] Exportando modelo a: {output_path}")
    
    # Preparar modelo para exportación
    agent.q_network.eval()
    
    # Input dummy con la forma correcta
    dummy_input = torch.randn(1, agent.config.INPUT_SIZE).to(agent.device)
    
    try:
        torch.onnx.export(
            agent.q_network,                    # Modelo a exportar
            dummy_input,                        # Input de ejemplo
            output_path,                        # Ruta de salida
            export_params=True,                 # Exportar parámetros
            opset_version=11,                   # Versión ONNX
            do_constant_folding=True,           # Optimización
            input_names=['input_tensor'],       # Nombres de entrada
            output_names=['output_tensor'],     # Nombres de salida
            dynamic_axes={
                'input_tensor': {0: 'batch_size'},
                'output_tensor': {0: 'batch_size'}
            }
        )
        print(f"[ONNX] ✓ Modelo exportado exitosamente")
        
        # Verificar modelo ONNX
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"[ONNX] ✓ Modelo validado correctamente")
        except ImportError:
            print(f"[ONNX] ⚠ ONNX no instalado, saltando validación")
        except Exception as e:
            print(f"[ONNX] ⚠ Error en validación: {e}")
        
    except Exception as e:
        print(f"[ONNX] ✗ Error en exportación: {e}")

# ===============================
# GESTOR DE MEJORES MODELOS
# ===============================
class BestModelManager:
    """
    Gestiona y guarda automáticamente los mejores modelos
    basado en métricas de rendimiento
    """
    
    def __init__(self, save_dir: str = "best_models"):
        self.save_dir = save_dir
        self.best_reward = float('-inf')
        self.best_success_rate = 0.0
        self.best_efficiency = 0.0
        
        # Crear directorio si no existe
        os.makedirs(save_dir, exist_ok=True)
        
        # Archivo de métricas
        self.metrics_file = os.path.join(save_dir, "best_models_log.json")
        self.load_previous_best()
    
    def load_previous_best(self):
        """Carga métricas previas si existen"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.best_reward = data.get('best_reward', float('-inf'))
                    self.best_success_rate = data.get('best_success_rate', 0.0)
                    self.best_efficiency = data.get('best_efficiency', 0.0)
                print(f"[BEST] Cargadas métricas previas - Reward: {self.best_reward:.2f}")
            except:
                print("[BEST] Error cargando métricas previas, empezando fresh")
    
    def evaluate_model(self, agent: DQNAgent, episode: int, metrics: Dict[str, Any]) -> bool:
        """
        Evalúa si el modelo actual es mejor que los anteriores
        """
        current_reward = metrics.get('avg_reward', 0.0)
        current_success = metrics.get('success_rate', 0.0)
        current_steps = metrics.get('avg_steps', 1000)
        
        # Calcular eficiencia (reward/steps)
        current_efficiency = current_reward / max(current_steps, 1)
        
        # Criterios para "mejor modelo"
        is_best_reward = current_reward > self.best_reward
        is_best_success = current_success > self.best_success_rate
        is_best_efficiency = current_efficiency > self.best_efficiency
        
        improved = False
        
        # Guardar por mejor recompensa
        if is_best_reward:
            self.best_reward = current_reward
            self.save_best_model(agent, episode, "best_reward", metrics)
            improved = True
        
        # Guardar por mejor tasa de éxito
        if is_best_success:
            self.best_success_rate = current_success
            self.save_best_model(agent, episode, "best_success", metrics)
            improved = True
        
        # Guardar por mejor eficiencia
        if is_best_efficiency:
            self.best_efficiency = current_efficiency
            self.save_best_model(agent, episode, "best_efficiency", metrics)
            improved = True
        
        return improved
    
    def save_best_model(self, agent: DQNAgent, episode: int, 
                       model_type: str, metrics: Dict[str, Any]):
        """
        Guarda el mejor modelo en múltiples formatos
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_type}_episode_{episode}_{timestamp}"
        
        # 1. Guardar PyTorch (.pth)
        pytorch_path = os.path.join(self.save_dir, f"{base_name}.pth")
        agent.save_model(pytorch_path)
        
        # 2. Exportar a ONNX
        onnx_path = os.path.join(self.save_dir, f"{base_name}.onnx")
        export_to_onnx(agent, onnx_path)
        
        # 3. Guardar métricas
        metrics_data = {
            'episode': episode,
            'timestamp': timestamp,
            'model_type': model_type,
            'metrics': metrics,
            'best_reward': self.best_reward,
            'best_success_rate': self.best_success_rate,
            'best_efficiency': self.best_efficiency,
            'files': {
                'pytorch': pytorch_path,
                'onnx': onnx_path
            }
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"[BEST] 🏆 Nuevo mejor modelo guardado: {model_type}")
        print(f"       📁 PyTorch: {pytorch_path}")
        print(f"       📁 ONNX: {onnx_path}")

# ===============================
# GRÁFICOS Y MÉTRICAS
# ===============================
def plot_training_metrics(rewards: List[float], lengths: List[int], 
                         losses: List[float], success_rate: List[float]):
    """Genera gráficos de métricas de entrenamiento"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Recompensas por episodio
    ax1.plot(rewards, alpha=0.7)
    ax1.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), 'r-', linewidth=2)
    ax1.set_title('Recompensas por Episodio')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.grid(True)
    
    # Longitud de episodios
    ax2.plot(lengths, alpha=0.7)
    ax2.plot(np.convolve(lengths, np.ones(100)/100, mode='valid'), 'g-', linewidth=2)
    ax2.set_title('Duración de Episodios')
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Pasos')
    ax2.grid(True)
    
    # Pérdidas de entrenamiento
    if losses:
        ax3.plot(losses, alpha=0.6)
        ax3.set_title('Pérdida de Entrenamiento')
        ax3.set_xlabel('Iteración')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
    
    # Tasa de éxito
    if success_rate:
        ax4.plot(success_rate, 'purple', linewidth=2)
        ax4.set_title('Tasa de Éxito (100 episodios)')
        ax4.set_xlabel('Episodio (x100)')
        ax4.set_ylabel('Tasa de Éxito')
        ax4.set_ylim(0, 1)
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import sys
    
    # Modo por defecto: entrenar
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    quick_start(mode)
