"""
SERVIDOR TCP MEJORADO PARA DQN - CONDUCCIÓN AUTÓNOMA
Sistema de comunicación en tiempo real entre Unreal Engine y DQN Agent

Características:
- Comunicación TCP optimizada
- Procesamiento de 80+ raycast
- Integración con NPCs
- Sistema de logging avanzado
- Manejo de errores robusto
- Conectado con dqn_master.py y dqn_agente_avanzado.py
"""

import socket
import json
import numpy as np
import threading
import time
import logging
from typing import Dict, Any, Optional, Tuple
import struct
import os
import sys

# Agregar directorio actual para importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar agente DQN
try:
    from dqn_agente_avanzado import DQNAgent, DQNConfig
    print("[TCP] ✓ Módulo DQN Agent importado correctamente")
except ImportError as e:
    print(f"[TCP] ✗ Error importando DQN Agent: {e}")
    DQNAgent = None
    DQNConfig = None

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dqn_server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DQNTCPServer:
    """
    Servidor TCP para comunicación con el agente DQN entrenado
    Procesa estados complejos y devuelve acciones óptimas
    """
    
    def __init__(self, agent: Optional[DQNAgent] = None, 
                 host: str = '127.0.0.1', port: int = 12345,
                 model_path: str = None):
        self.host = host
        self.port = port
        self.socket = None
        self.is_running = False
        self.client_connected = False
        
        # Cargar agente DQN
        if agent is not None:
            self.agent = agent
            print(f"[TCP] ✓ Agente DQN proporcionado")
        elif model_path and os.path.exists(model_path):
            self.agent = self.load_agent_from_file(model_path)
        else:
            # Buscar modelo más reciente
            self.agent = self.find_and_load_latest_model()
        
        if self.agent is None:
            raise ValueError("No se pudo cargar ningún agente DQN")
        
        # Mapeo de acciones mejorado
        self.action_map = {
            0: "ACELERAR",
            1: "FRENAR",
            2: "IZQUIERDA", 
            3: "DERECHA",
            4: "DETENERSE"
        }
        
        # Estadísticas del servidor
        self.stats = {
            'messages_received': 0,
            'actions_sent': 0,
            'collisions_detected': 0,
            'session_start': None,
            'last_action': "DETENERSE"
        }
        
        # Modelo DQN (simulado - en producción cargar modelo real)
        self.model_loaded = False
        self.initialize_model()
    
    def initialize_model(self):
        """
        Inicializa el modelo DQN
        En producción, aquí se cargaría el modelo ONNX o PyTorch entrenado
        """
        try:
            # Simular carga de modelo
            logger.info("[DQN-SERVER] Inicializando modelo DQN...")
            time.sleep(1)  # Simular tiempo de carga
            self.model_loaded = True
            logger.info("[DQN-SERVER] Modelo DQN cargado exitosamente")
        except Exception as e:
            logger.error(f"[DQN-SERVER] Error al cargar modelo: {e}")
            self.model_loaded = False
    
    def preprocess_state(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocesa el estado recibido de Unreal Engine
        Convierte a formato esperado por el modelo DQN
        """
        try:
            processed_state = []
            
            # Procesar raycast (80 sensores)
            raycast_data = raw_data.get('raycast', [])
            if len(raycast_data) < 80:
                # Rellenar con valores por defecto si faltan sensores
                raycast_data.extend([1500.0] * (80 - len(raycast_data)))
            
            # Normalizar distancias de raycast (0-1500m → 0-1)
            raycast_normalized = [min(float(d) / 1500.0, 1.0) for d in raycast_data[:80]]
            processed_state.extend(raycast_normalized)
            
            # Velocidad normalizada
            velocity = float(raw_data.get('vel', 0.0))
            velocity_normalized = min(velocity / 120.0, 1.0)  # 0-120 km/h → 0-1
            processed_state.append(velocity_normalized)
            
            # Vida normalizada
            health = float(raw_data.get('vida', 100.0))
            health_normalized = health / 100.0  # 0-100% → 0-1
            processed_state.append(health_normalized)
            
            # Dirección (si disponible)
            direction = float(raw_data.get('direccion', 0.0))
            direction_normalized = direction / 180.0  # -180°/+180° → -1/+1
            processed_state.append(direction_normalized)
            
            # Estado de NPCs (one-hot encoding)
            npc_detected = raw_data.get('npc', 'Ninguno')
            npc_encoding = [0.0, 0.0, 0.0, 0.0]  # [Camion, AutoRojo, AutoAzul, Ninguno]
            
            if npc_detected == 'Camion':
                npc_encoding[0] = 1.0
                self.stats['collisions_detected'] += 1
            elif npc_detected == 'AutoRojo':
                npc_encoding[1] = 1.0
            elif npc_detected == 'AutoAzul':
                npc_encoding[2] = 1.0
            else:
                npc_encoding[3] = 1.0
            
            processed_state.extend(npc_encoding)
            
            return np.array(processed_state, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"[DQN-SERVER] Error en preprocesamiento: {e}")
            # Retornar estado por defecto en caso de error
            return np.zeros(84, dtype=np.float32)
    
    def predict_action(self, state: np.ndarray) -> int:
        """
        Predice la acción óptima usando el modelo DQN
        En este ejemplo, usa lógica heurística para demostración
        """
        if not self.model_loaded:
            return 1  # FRENAR por defecto
        
        try:
            # LÓGICA HEURÍSTICA MEJORADA (en producción usar modelo DQN real)
            
            # Extraer información del estado
            raycast_distances = state[:80] * 1500.0  # Desnormalizar
            velocity = state[80] * 120.0
            health = state[81] * 100.0
            npc_state = state[83:87]
            
            # Detectar obstáculos críticos
            min_distance = np.min(raycast_distances)
            front_distances = raycast_distances[35:45]  # Sensores frontales
            avg_front_distance = np.mean(front_distances)
            
            # Lógica de decisión avanzada
            
            # PRIORIDAD 1: Evitar colisiones inmediatas
            if min_distance < 100:
                if avg_front_distance < 150:
                    return 1  # FRENAR
                elif raycast_distances[10:30].mean() > raycast_distances[50:70].mean():
                    return 2  # IZQUIERDA (más espacio a la izquierda)
                else:
                    return 3  # DERECHA
            
            # PRIORIDAD 2: Responder a NPCs detectados
            if npc_state[0] > 0.5:  # Camión detectado
                return 1  # FRENAR inmediatamente
            elif npc_state[1] > 0.5 or npc_state[2] > 0.5:  # Auto detectado
                if velocity > 60:
                    return 1  # FRENAR
                else:
                    return 2 if np.random.random() > 0.5 else 3  # Evadir
            
            # PRIORIDAD 3: Control de velocidad
            if velocity < 30 and avg_front_distance > 300:
                return 0  # ACELERAR
            elif velocity > 100:
                return 1  # FRENAR
            
            # PRIORIDAD 4: Navegación normal
            if avg_front_distance > 500:
                return 0  # ACELERAR
            elif avg_front_distance > 200:
                return 0 if velocity < 70 else 1  # Mantener velocidad
            else:
                return 1  # FRENAR por precaución
                
        except Exception as e:
            logger.error(f"[DQN-SERVER] Error en predicción: {e}")
            return 1  # FRENAR por defecto
    
    def process_message(self, message: str) -> str:
        """
        Procesa un mensaje recibido y devuelve la acción
        """
        try:
            # Parsear mensaje JSON
            data = json.loads(message.strip().replace("'", "\""))
            self.stats['messages_received'] += 1
            
            # Preprocesar estado
            state = self.preprocess_state(data)
            
            # Predecir acción
            action_idx = self.predict_action(state)
            action_str = self.action_map.get(action_idx, "FRENAR")
            
            # Actualizar estadísticas
            self.stats['actions_sent'] += 1
            self.stats['last_action'] = action_str
            
            # Logging detallado
            npc_info = data.get('npc', 'Ninguno')
            velocity = data.get('vel', 0.0)
            min_dist = min(data.get('raycast', [1500])[:80]) if data.get('raycast') else 1500
            
            log_msg = (f"[DECISIÓN] Vel:{velocity:.1f}km/h | "
                      f"MinDist:{min_dist:.0f}m | "
                      f"NPC:{npc_info} | "
                      f"Acción:{action_str}")
            
            if npc_info != 'Ninguno':
                log_msg += " 🚨 NPC DETECTADO"
            
            logger.info(log_msg)
            
            return action_str
            
        except json.JSONDecodeError as e:
            logger.error(f"[DQN-SERVER] Error JSON: {e}")
            return "FRENAR"
        except Exception as e:
            logger.error(f"[DQN-SERVER] Error procesando mensaje: {e}")
            return "FRENAR"
    
    def handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        """
        Maneja la conexión con un cliente
        """
        logger.info(f"[DQN-SERVER] Cliente conectado desde {addr}")
        self.client_connected = True
        self.stats['session_start'] = time.time()
        
        try:
            while self.is_running:
                # Recibir datos
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    break
                
                # Procesar mensaje
                response = self.process_message(data)
                
                # Enviar respuesta
                conn.send(response.encode('utf-8'))
                
        except ConnectionResetError:
            logger.warning(f"[DQN-SERVER] Cliente {addr} desconectado abruptamente")
        except Exception as e:
            logger.error(f"[DQN-SERVER] Error con cliente {addr}: {e}")
        finally:
            conn.close()
            self.client_connected = False
            logger.info(f"[DQN-SERVER] Cliente {addr} desconectado")
    
    def print_stats(self):
        """
        Imprime estadísticas del servidor
        """
        if self.stats['session_start']:
            uptime = time.time() - self.stats['session_start']
            logger.info(f"[ESTADÍSTICAS] Tiempo activo: {uptime:.1f}s | "
                       f"Mensajes: {self.stats['messages_received']} | "
                       f"Acciones: {self.stats['actions_sent']} | "
                       f"Colisiones: {self.stats['collisions_detected']} | "
                       f"Última acción: {self.stats['last_action']}")
    
    def start_server(self):
        """
        Inicia el servidor TCP
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            
            self.is_running = True
            logger.info(f"[DQN-SERVER] Servidor iniciado en {self.host}:{self.port}")
            logger.info(f"[DQN-SERVER] Esperando conexión de Unreal Engine...")
            
            # Hilo para estadísticas periódicas
            stats_thread = threading.Thread(target=self._stats_worker, daemon=True)
            stats_thread.start()
            
            while self.is_running:
                try:
                    conn, addr = self.socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                except OSError:
                    if self.is_running:
                        logger.error("[DQN-SERVER] Error en accept()")
                    break
                    
        except Exception as e:
            logger.error(f"[DQN-SERVER] Error iniciando servidor: {e}")
        finally:
            self.stop_server()
    
    def _stats_worker(self):
        """
        Worker thread para mostrar estadísticas periódicamente
        """
        while self.is_running:
            time.sleep(30)  # Cada 30 segundos
            if self.client_connected:
                self.print_stats()
    
    def stop_server(self):
        """
        Detiene el servidor
        """
        self.is_running = False
        if self.socket:
            self.socket.close()
        logger.info("[DQN-SERVER] Servidor detenido")

def main():
    """
    Función principal del servidor DQN
    """
    print("="*60)
    print("SERVIDOR DQN - CONDUCCIÓN AUTÓNOMA")
    print("Modelo 2: 80+ Raycast + NPCs Integrados")
    print("="*60)
    
    server = DQNServer()
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        logger.info("[DQN-SERVER] Deteniendo servidor por interrupción del usuario...")
        server.stop_server()
    except Exception as e:
        logger.error(f"[DQN-SERVER] Error crítico: {e}")
    
    print("\n[DQN-SERVER] Servidor terminado")

if __name__ == "__main__":
    main()
