"""
EVALUADOR Y ANALIZADOR DE RENDIMIENTO DQN
Sistema de evaluación para el agente de conducción autónoma

Características:
- Métricas de rendimiento detalladas
- Análisis de comportamiento del agente
- Generación de reportes
- Visualización de resultados
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from typing import List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime

class DQNEvaluator:
    """
    Evaluador de rendimiento para el agente DQN
    """
    
    def __init__(self):
        self.episodes_data = []
        self.actions_history = []
        self.collision_history = []
        self.metrics = {}
        
        # Configurar estilo de gráficos
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def record_episode(self, episode_data: Dict[str, Any]):
        """
        Registra datos de un episodio
        """
        episode_data['timestamp'] = datetime.now().isoformat()
        self.episodes_data.append(episode_data)
    
    def record_action(self, action: str, state_info: Dict[str, Any]):
        """
        Registra una acción tomada por el agente
        """
        action_record = {
            'action': action,
            'timestamp': time.time(),
            'velocity': state_info.get('velocity', 0),
            'min_distance': state_info.get('min_distance', 1500),
            'npc_detected': state_info.get('npc_detected', 'Ninguno')
        }
        self.actions_history.append(action_record)
    
    def record_collision(self, collision_type: str, episode: int, step: int):
        """
        Registra una colisión
        """
        collision_record = {
            'type': collision_type,
            'episode': episode,
            'step': step,
            'timestamp': time.time()
        }
        self.collision_history.append(collision_record)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de rendimiento
        """
        if not self.episodes_data:
            return {}
        
        # Métricas básicas
        rewards = [ep.get('total_reward', 0) for ep in self.episodes_data]
        steps = [ep.get('steps', 0) for ep in self.episodes_data]
        successes = [ep.get('success', False) for ep in self.episodes_data]
        
        # Métricas de colisiones
        collision_types = [c['type'] for c in self.collision_history]
        
        # Métricas de acciones
        action_counts = {}
        for action_record in self.actions_history:
            action = action_record['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        self.metrics = {
            # Rendimiento general
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            
            # Eficiencia
            'avg_steps': np.mean(steps),
            'std_steps': np.std(steps),
            'success_rate': np.mean(successes) * 100,
            
            # Seguridad
            'total_collisions': len(self.collision_history),
            'collisions_per_episode': len(self.collision_history) / len(self.episodes_data),
            'fatal_collisions': collision_types.count('Camion'),
            'moderate_collisions': collision_types.count('AutoRojo') + collision_types.count('AutoAzul'),
            'minor_collisions': collision_types.count('Pared'),
            
            # Comportamiento
            'action_diversity': len(action_counts) / 5.0,  # Normalizado por número total de acciones
            'most_common_action': max(action_counts.keys(), key=action_counts.get) if action_counts else 'N/A',
            
            # Consistencia
            'reward_stability': 1.0 / (1.0 + np.std(rewards)),  # Mayor es mejor
            'episode_consistency': 1.0 / (1.0 + np.std(steps))
        }
        
        return self.metrics
    
    def generate_report(self, save_path: str = 'dqn_evaluation_report.txt'):
        """
        Genera un reporte detallado de evaluación
        """
        metrics = self.calculate_metrics()
        
        report = f"""
{'='*80}
REPORTE DE EVALUACIÓN - AGENTE DQN CONDUCCIÓN AUTÓNOMA
{'='*80}
Fecha del reporte: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Episodios evaluados: {len(self.episodes_data)}
Acciones registradas: {len(self.actions_history)}

MÉTRICAS DE RENDIMIENTO:
{'-'*40}
Recompensa promedio: {metrics.get('avg_reward', 0):.2f} ± {metrics.get('std_reward', 0):.2f}
Recompensa máxima: {metrics.get('max_reward', 0):.2f}
Recompensa mínima: {metrics.get('min_reward', 0):.2f}

MÉTRICAS DE EFICIENCIA:
{'-'*40}
Pasos promedio por episodio: {metrics.get('avg_steps', 0):.1f} ± {metrics.get('std_steps', 0):.1f}
Tasa de éxito: {metrics.get('success_rate', 0):.1f}%

MÉTRICAS DE SEGURIDAD:
{'-'*40}
Total de colisiones: {metrics.get('total_collisions', 0)}
Colisiones por episodio: {metrics.get('collisions_per_episode', 0):.2f}
Colisiones fatales (Camión): {metrics.get('fatal_collisions', 0)}
Colisiones moderadas (Autos): {metrics.get('moderate_collisions', 0)}
Colisiones menores (Paredes): {metrics.get('minor_collisions', 0)}

MÉTRICAS DE COMPORTAMIENTO:
{'-'*40}
Diversidad de acciones: {metrics.get('action_diversity', 0):.2f} (0-1)
Acción más común: {metrics.get('most_common_action', 'N/A')}
Estabilidad de recompensas: {metrics.get('reward_stability', 0):.3f}
Consistencia de episodios: {metrics.get('episode_consistency', 0):.3f}

ANÁLISIS DETALLADO:
{'-'*40}
"""
        
        # Análisis de distribución de acciones
        if self.actions_history:
            action_counts = {}
            for action_record in self.actions_history:
                action = action_record['action']
                action_counts[action] = action_counts.get(action, 0) + 1
            
            report += "Distribución de acciones:\n"
            total_actions = sum(action_counts.values())
            for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_actions) * 100
                report += f"  {action}: {count} ({percentage:.1f}%)\n"
        
        # Análisis de colisiones por tipo
        if self.collision_history:
            collision_types = {}
            for collision in self.collision_history:
                ctype = collision['type']
                collision_types[ctype] = collision_types.get(ctype, 0) + 1
            
            report += "\nDistribución de colisiones:\n"
            for ctype, count in sorted(collision_types.items(), key=lambda x: x[1], reverse=True):
                report += f"  {ctype}: {count}\n"
        
        # Recomendaciones
        report += f"\nRECOMENDACIONES:\n{'-'*40}\n"
        
        if metrics.get('success_rate', 0) < 70:
            report += "• Baja tasa de éxito - Considerar ajustar recompensas o entrenar más episodios\n"
        
        if metrics.get('fatal_collisions', 0) > len(self.episodes_data) * 0.1:
            report += "• Muchas colisiones fatales - Mejorar detección de camiones\n"
        
        if metrics.get('action_diversity', 0) < 0.6:
            report += "• Baja diversidad de acciones - Revisar exploración del agente\n"
        
        if metrics.get('reward_stability', 0) < 0.5:
            report += "• Inestabilidad en recompensas - Considerar ajustar hiperparámetros\n"
        
        report += f"\n{'='*80}\n"
        
        # Guardar reporte
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"Reporte guardado en: {save_path}")
        
        return report
    
    def plot_performance_charts(self, save_dir: str = '.'):
        """
        Genera gráficos de rendimiento
        """
        if not self.episodes_data:
            print("No hay datos para graficar")
            return
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Recompensas por episodio
        ax1 = plt.subplot(3, 3, 1)
        rewards = [ep.get('total_reward', 0) for ep in self.episodes_data]
        episodes = range(len(rewards))
        
        plt.plot(episodes, rewards, alpha=0.7, label='Recompensa por episodio')
        if len(rewards) > 50:
            window = min(50, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'Media móvil ({window})')
        
        plt.title('Evolución de Recompensas')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Distribución de recompensas
        ax2 = plt.subplot(3, 3, 2)
        plt.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Media: {np.mean(rewards):.2f}')
        plt.title('Distribución de Recompensas')
        plt.xlabel('Recompensa')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Pasos por episodio
        ax3 = plt.subplot(3, 3, 3)
        steps = [ep.get('steps', 0) for ep in self.episodes_data]
        plt.plot(episodes, steps, alpha=0.7, color='green')
        if len(steps) > 50:
            window = min(50, len(steps) // 10)
            moving_avg = np.convolve(steps, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(steps)), moving_avg, 'orange', linewidth=2)
        
        plt.title('Duración de Episodios')
        plt.xlabel('Episodio')
        plt.ylabel('Pasos')
        plt.grid(True, alpha=0.3)
        
        # 4. Tasa de éxito acumulativa
        ax4 = plt.subplot(3, 3, 4)
        successes = [ep.get('success', False) for ep in self.episodes_data]
        cumulative_success = np.cumsum(successes) / np.arange(1, len(successes) + 1) * 100
        plt.plot(episodes, cumulative_success, color='purple', linewidth=2)
        plt.title('Tasa de Éxito Acumulativa')
        plt.xlabel('Episodio')
        plt.ylabel('Tasa de Éxito (%)')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        
        # 5. Distribución de acciones
        if self.actions_history:
            ax5 = plt.subplot(3, 3, 5)
            action_counts = {}
            for action_record in self.actions_history:
                action = action_record['action']
                action_counts[action] = action_counts.get(action, 0) + 1
            
            actions = list(action_counts.keys())
            counts = list(action_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(actions)))
            
            plt.pie(counts, labels=actions, autopct='%1.1f%%', colors=colors)
            plt.title('Distribución de Acciones')
        
        # 6. Colisiones por tipo
        if self.collision_history:
            ax6 = plt.subplot(3, 3, 6)
            collision_types = {}
            for collision in self.collision_history:
                ctype = collision['type']
                collision_types[ctype] = collision_types.get(ctype, 0) + 1
            
            types = list(collision_types.keys())
            counts = list(collision_types.values())
            colors = ['red', 'orange', 'yellow', 'lightblue'][:len(types)]
            
            plt.bar(types, counts, color=colors, alpha=0.7, edgecolor='black')
            plt.title('Colisiones por Tipo')
            plt.xlabel('Tipo de Colisión')
            plt.ylabel('Cantidad')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 7. Velocidad promedio por episodio
        if self.actions_history:
            ax7 = plt.subplot(3, 3, 7)
            # Agrupar velocidades por episodio (simulado)
            velocities = [action['velocity'] for action in self.actions_history if 'velocity' in action]
            if velocities:
                plt.hist(velocities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.axvline(np.mean(velocities), color='red', linestyle='--', label=f'Media: {np.mean(velocities):.1f}')
                plt.title('Distribución de Velocidades')
                plt.xlabel('Velocidad (km/h)')
                plt.ylabel('Frecuencia')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        # 8. Correlación recompensa-pasos
        ax8 = plt.subplot(3, 3, 8)
        if len(rewards) == len(steps):
            plt.scatter(steps, rewards, alpha=0.6, color='coral')
            z = np.polyfit(steps, rewards, 1)
            p = np.poly1d(z)
            plt.plot(steps, p(steps), "r--", alpha=0.8)
            
            # Calcular correlación
            correlation = np.corrcoef(steps, rewards)[0, 1]
            plt.title(f'Correlación Pasos vs Recompensa\nr = {correlation:.3f}')
            plt.xlabel('Pasos')
            plt.ylabel('Recompensa')
            plt.grid(True, alpha=0.3)
        
        # 9. Métricas de tiempo
        ax9 = plt.subplot(3, 3, 9)
        episode_times = [ep.get('duration', 0) for ep in self.episodes_data if 'duration' in ep]
        if episode_times:
            plt.plot(range(len(episode_times)), episode_times, alpha=0.7, color='brown')
            plt.title('Duración de Episodios (tiempo real)')
            plt.xlabel('Episodio')
            plt.ylabel('Duración (s)')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No hay datos de\nduración disponibles', 
                    ha='center', va='center', transform=ax9.transAxes,
                    fontsize=12, style='italic')
            plt.title('Duración de Episodios')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/dqn_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráficos guardados en: {save_dir}/dqn_performance_analysis.png")
    
    def export_data(self, filepath: str = 'dqn_evaluation_data.json'):
        """
        Exporta todos los datos a JSON
        """
        export_data = {
            'episodes': self.episodes_data,
            'actions': self.actions_history,
            'collisions': self.collision_history,
            'metrics': self.metrics,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Datos exportados a: {filepath}")

# Función de demostración
def demo_evaluation():
    """
    Demostración del evaluador con datos sintéticos
    """
    evaluator = DQNEvaluator()
    
    # Generar datos sintéticos de ejemplo
    print("Generando datos de evaluación de ejemplo...")
    
    for episode in range(100):
        # Simular episodio
        steps = np.random.randint(50, 500)
        success = np.random.random() > 0.3  # 70% tasa de éxito
        
        if success:
            reward = np.random.normal(80, 20)
        else:
            reward = np.random.normal(-10, 30)
        
        episode_data = {
            'episode': episode,
            'total_reward': reward,
            'steps': steps,
            'success': success,
            'duration': steps * 0.1  # Simular duración
        }
        
        evaluator.record_episode(episode_data)
        
        # Simular algunas acciones
        for _ in range(min(steps, 100)):
            action = np.random.choice(['ACELERAR', 'FRENAR', 'IZQUIERDA', 'DERECHA', 'DETENERSE'])
            state_info = {
                'velocity': np.random.uniform(0, 120),
                'min_distance': np.random.uniform(50, 1500),
                'npc_detected': np.random.choice(['Ninguno', 'Camion', 'AutoRojo', 'AutoAzul'], p=[0.8, 0.05, 0.075, 0.075])
            }
            evaluator.record_action(action, state_info)
        
        # Simular colisiones ocasionales
        if np.random.random() < 0.2:  # 20% probabilidad de colisión
            collision_type = np.random.choice(['Camion', 'AutoRojo', 'AutoAzul', 'Pared'], p=[0.1, 0.3, 0.3, 0.3])
            evaluator.record_collision(collision_type, episode, np.random.randint(0, steps))
    
    # Generar análisis
    print("\nGenerando reporte de evaluación...")
    evaluator.generate_report()
    
    print("\nGenerando gráficos de rendimiento...")
    evaluator.plot_performance_charts()
    
    print("\nExportando datos...")
    evaluator.export_data()
    
    print("\n¡Evaluación completada!")

if __name__ == "__main__":
    demo_evaluation()
