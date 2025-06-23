"""
UTILIDADES Y HERRAMIENTAS PARA DQN
Funciones auxiliares para el sistema de conducción autónoma

Características:
- Conversión de modelos PyTorch a ONNX
- Validación de modelos
- Herramientas de debugging
- Utilidades de datos
- Conectores para integración del sistema
"""

import torch
import torch.onnx
import numpy as np
import json
import os
import time
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

# CONEXIÓN PRINCIPAL CON OTROS MÓDULOS
def initialize_system_connections():
    """
    Inicializa las conexiones entre todos los módulos del sistema
    """
    try:
        # Importar módulos principales
        from dqn_agente_avanzado import DQNAgent, DQNConfig
        from dqn_server_tcp import DQNTCPServer
        from dqn_evaluator import ModelEvaluator
        
        print("[SYSTEM] ✓ Todos los módulos conectados correctamente")
        return True
    except ImportError as e:
        print(f"[SYSTEM] ⚠ Error conectando módulos: {e}")
        return False

class ModelConverter:
    """
    Conversor de modelos para diferentes formatos
    """
    
    @staticmethod
    def pytorch_to_onnx(model: torch.nn.Module, 
                       input_shape: Tuple[int, ...], 
                       output_path: str,
                       input_names: List[str] = None,
                       output_names: List[str] = None):
        """
        Convierte modelo PyTorch a ONNX
        """
        model.eval()
        
        # Crear input dummy
        dummy_input = torch.randn(1, *input_shape)
        
        # Nombres por defecto
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    input_names[0]: {0: 'batch_size'},
                    output_names[0]: {0: 'batch_size'}
                }
            )
            print(f"✅ Modelo convertido exitosamente: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error en conversión: {e}")
            return False
    
    @staticmethod
    def validate_onnx_model(onnx_path: str, input_shape: Tuple[int, ...]) -> bool:
        """
        Valida que el modelo ONNX funciona correctamente
        """
        try:
            import onnxruntime as ort
            
            # Crear sesión ONNX
            session = ort.InferenceSession(onnx_path)
            
            # Crear input de prueba
            test_input = np.random.randn(1, *input_shape).astype(np.float32)
            
            # Ejecutar inferencia
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: test_input})
            
            print(f"✅ Modelo ONNX validado: {onnx_path}")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {result[0].shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error validando modelo ONNX: {e}")
            return False

class DataProcessor:
    """
    Procesador de datos para entrenamiento y evaluación
    """
    
    @staticmethod
    def normalize_sensor_data(raycast_data: List[float], 
                            max_distance: float = 1500.0) -> List[float]:
        """
        Normaliza datos de sensores raycast
        """
        return [min(d / max_distance, 1.0) for d in raycast_data]
    
    @staticmethod
    def encode_npc_state(npc_type: str) -> List[float]:
        """
        Codifica el estado de NPCs en one-hot
        """
        encoding = [0.0, 0.0, 0.0, 0.0]  # [Camion, AutoRojo, AutoAzul, Ninguno]
        
        if npc_type == 'Camion':
            encoding[0] = 1.0
        elif npc_type == 'AutoRojo':
            encoding[1] = 1.0
        elif npc_type == 'AutoAzul':
            encoding[2] = 1.0
        else:
            encoding[3] = 1.0
        
        return encoding
    
    @staticmethod
    def create_state_vector(raycast: List[float], 
                          velocity: float,
                          health: float, 
                          direction: float,
                          npc_type: str) -> np.ndarray:
        """
        Crea vector de estado completo para el agente
        """
        # Normalizar raycast
        raycast_norm = DataProcessor.normalize_sensor_data(raycast[:80])
        
        # Normalizar otras variables
        vel_norm = min(velocity / 120.0, 1.0)  # 0-120 km/h
        health_norm = health / 100.0  # 0-100%
        dir_norm = direction / 180.0  # -180° a +180°
        
        # Codificar NPCs
        npc_encoding = DataProcessor.encode_npc_state(npc_type)
        
        # Construir vector final
        state_vector = raycast_norm + [vel_norm, health_norm, dir_norm] + npc_encoding
        
        return np.array(state_vector, dtype=np.float32)

class DebugTools:
    """
    Herramientas de debugging para el sistema DQN
    """
    
    @staticmethod
    def visualize_raycast(raycast_data: List[float], 
                         save_path: str = 'raycast_visualization.png'):
        """
        Visualiza los datos de raycast en forma circular
        """
        # Convertir a coordenadas polares
        angles = np.linspace(0, 2*np.pi, len(raycast_data), endpoint=False)
        distances = np.array(raycast_data)
        
        # Crear gráfico polar
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot raycast
        ax.plot(angles, distances, 'b-', linewidth=2, alpha=0.7)
        ax.fill(angles, distances, alpha=0.25, color='blue')
        
        # Configurar
        ax.set_ylim(0, max(distances) * 1.1)
        ax.set_title('Visualización Raycast 360°', pad=20)
        ax.grid(True)
        
        # Marcar zonas peligrosas
        danger_threshold = 200
        danger_mask = distances < danger_threshold
        if np.any(danger_mask):
            danger_angles = angles[danger_mask]
            danger_distances = distances[danger_mask]
            ax.scatter(danger_angles, danger_distances, c='red', s=50, alpha=0.8, label='Zona Peligrosa')
            ax.legend()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualización guardada: {save_path}")
    
    @staticmethod
    def log_state_info(state_dict: Dict[str, Any], action: str, reward: float):
        """
        Log detallado del estado del agente
        """
        print("="*50)
        print("ESTADO DEL AGENTE")
        print("="*50)
        
        # Información básica
        print(f"Velocidad: {state_dict.get('velocity', 0):.1f} km/h")
        print(f"Vida: {state_dict.get('health', 100):.1f}%")
        print(f"Dirección: {state_dict.get('direction', 0):.1f}°")
        print(f"NPC Detectado: {state_dict.get('npc_detected', 'Ninguno')}")
        
        # Información de sensores
        raycast = state_dict.get('raycast', [])
        if raycast:
            min_dist = min(raycast)
            max_dist = max(raycast)
            avg_dist = np.mean(raycast)
            
            print(f"\nSENSORES RAYCAST:")
            print(f"  Distancia mínima: {min_dist:.1f}m")
            print(f"  Distancia máxima: {max_dist:.1f}m")
            print(f"  Distancia promedio: {avg_dist:.1f}m")
            
            # Detectar sectores críticos
            danger_count = sum(1 for d in raycast if d < 200)
            if danger_count > 0:
                print(f"  ⚠️ Sensores en zona crítica: {danger_count}/{len(raycast)}")
        
        # Acción y recompensa
        print(f"\nACCIÓN TOMADA: {action}")
        print(f"RECOMPENSA: {reward:.2f}")
        print("="*50)

class ConfigManager:
    """
    Gestor de configuraciones del sistema
    """
    
    @staticmethod
    def save_config(config_dict: Dict[str, Any], filepath: str):
        """
        Guarda configuración en JSON
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"Configuración guardada: {filepath}")
    
    @staticmethod
    def load_config(filepath: str) -> Dict[str, Any]:
        """
        Carga configuración desde JSON
        """
        if not os.path.exists(filepath):
            print(f"⚠️ Archivo de configuración no encontrado: {filepath}")
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"Configuración cargada: {filepath}")
        return config
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Configuración por defecto del sistema
        """
        return {
            "dqn": {
                "learning_rate": 0.0001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "memory_size": 100000,
                "batch_size": 64,
                "target_update": 1000
            },
            "environment": {
                "max_episodes": 5000,
                "max_steps_per_episode": 2000,
                "sensor_range": 1500.0,
                "num_raycast": 80
            },
            "rewards": {
                "goal_reached": 100.0,
                "collision_truck": -100.0,
                "collision_car": -50.0,
                "collision_wall": -10.0,
                "optimal_speed_bonus": 0.5,
                "survival_bonus": 0.1
            },
            "tcp": {
                "host": "127.0.0.1",
                "port": 9999,
                "timeout": 30
            }
        }

def create_demo_files():
    """
    Crea archivos de demostración y configuración
    """
    print("Creando archivos de demostración...")
    
    # Configuración por defecto
    config = ConfigManager.get_default_config()
    ConfigManager.save_config(config, 'dqn_config.json')
    
    # Datos de ejemplo para visualización
    demo_raycast = np.random.uniform(100, 1500, 80).tolist()
    demo_state = {
        'raycast': demo_raycast,
        'velocity': 65.0,
        'health': 85.0,
        'direction': 15.0,
        'npc_detected': 'AutoRojo'
    }
    
    # Visualizar raycast de ejemplo
    DebugTools.visualize_raycast(demo_raycast, 'demo_raycast.png')
    
    # Log de estado de ejemplo
    DebugTools.log_state_info(demo_state, 'ACELERAR', 2.5)
    
    print("✅ Archivos de demostración creados")

if __name__ == "__main__":
    create_demo_files()
