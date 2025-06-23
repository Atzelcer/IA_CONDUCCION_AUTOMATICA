"""
SERVIDOR TCP Q-LEARNING CLÁSICO - MODELO 1
Sistema de conducción autónoma con Q-Table tradicional
Versión mejorada con logging y manejo de errores avanzado

Características:
- Q-Learning con tabla de estados discretizada
- Comunicación TCP robusta
- Sistema de logging detallado
- Manejo de errores mejorado
"""

import socket
import json
import numpy as np
import logging
import time
from typing import Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('q_learning_server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# === CONFIGURACIÓN Q-LEARNING ===
try:
    q_table = np.load("q_table_agenteRL.npy")
    logger.info(f"[Q-LEARNING] Q-Table cargada: {q_table.shape}")
except FileNotFoundError:
    logger.error("[Q-LEARNING] Error: No se encontró q_table_agenteRL.npy")
    logger.info("[Q-LEARNING] Ejecutar primero: python entrenar_agente_q_table_complejo.py")
    exit(1)

n_states = q_table.shape[0]
acciones_map = {0: "FRENAR", 1: "AVANZAR", 2: "IZQUIERDA", 3: "DERECHA"}

def get_state(obs):
    """
    Discretiza el estado de observación para la Q-Table
    """
    return hash(tuple(np.round(obs, 1))) % n_states

def iniciar_servidor():
    """
    Servidor TCP mejorado para Q-Learning clásico
    """
    host = '127.0.0.1'
    puerto = 9999

    logger.info("="*60)
    logger.info("SERVIDOR Q-LEARNING - CONDUCCIÓN AUTÓNOMA")
    logger.info("Modelo 1: Q-Table Clásico")
    logger.info("="*60)

    try:
        servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        servidor.bind((host, puerto))
        servidor.listen(1)
        logger.info(f"[SERVIDOR] Esperando conexión en {host}:{puerto}...")

        conn, addr = servidor.accept()
        logger.info(f"[SERVIDOR] Cliente conectado desde {addr}")

        # Estadísticas
        mensajes_procesados = 0
        acciones_enviadas = 0
        colisiones_detectadas = 0
        inicio_sesion = time.time()

        while True:
            try:
                datos = conn.recv(1024).decode('utf-8')
                if not datos:
                    break

                # Parsear datos
                info = json.loads(datos.strip().replace("'", "\""))
                mensajes_procesados += 1

                # Extraer información mejorada
                izq = float(info.get("izq", 0))
                der = float(info.get("der", 0))
                vel = float(info.get("vel", 0))
                vida = float(info.get("vida", 100))
                npc = info.get("npc", "Ninguno")

                # Procesar estado para Q-Table
                estado = get_state([izq, der, vel, vida])
                accion_idx = int(np.argmax(q_table[estado]))
                accion_str = acciones_map.get(accion_idx, "FRENAR")

                acciones_enviadas += 1

                # Detectar colisiones
                if npc != "Ninguno":
                    colisiones_detectadas += 1

                # Logging detallado
                log = (f"[DECISIÓN] izq={izq:.1f}m, der={der:.1f}m, "
                       f"vel={vel:.1f}km/h, vida={vida:.1f}% → {accion_str}")
                
                if npc != "Ninguno":
                    log += f" 🚨 COLISIÓN: {npc}"
                
                logger.info(log)

                # Estadísticas periódicas
                if mensajes_procesados % 100 == 0:
                    tiempo_transcurrido = time.time() - inicio_sesion
                    logger.info(f"[STATS] Mensajes: {mensajes_procesados} | "
                               f"Acciones: {acciones_enviadas} | "
                               f"Colisiones: {colisiones_detectadas} | "
                               f"Tiempo: {tiempo_transcurrido:.1f}s")

                # Enviar respuesta
                conn.send(accion_str.encode('utf-8'))

            except json.JSONDecodeError as e:
                logger.error(f"[ERROR JSON] {e}")
                conn.send(b"FRENAR")
            except Exception as e:
                logger.error(f"[ERROR] {e}")
                conn.send(b"FRENAR")

    except KeyboardInterrupt:
        logger.info("[SERVIDOR] Detenido por usuario")
    except Exception as e:
        logger.error(f"[SERVIDOR] Error crítico: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
        if 'servidor' in locals():
            servidor.close()
        logger.info("[SERVIDOR] Conexión cerrada")

if __name__ == "__main__":
    iniciar_servidor()
