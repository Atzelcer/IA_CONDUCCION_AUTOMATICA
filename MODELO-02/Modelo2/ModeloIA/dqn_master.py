"""
SISTEMA MAESTRO DE CONDUCCIÓN AUTÓNOMA DQN
Conecta todos los módulos y gestiona los mejores pesos

Este archivo coordina:
- Entrenamiento del agente DQN
- Exportación automática a ONNX
- Servidor TCP para Unreal Engine
- Evaluación y métricas
- Guardado de mejores modelos

Uso:
    python dqn_master.py train      # Entrenar nuevo modelo
    python dqn_master.py server     # Iniciar servidor TCP
    python dqn_master.py evaluate   # Evaluar modelo existente
    python dqn_master.py export     # Exportar a ONNX
    python dqn_master.py all        # Ejecutar todo el pipeline
"""

import sys
import os
import json
import time
from pathlib import Path

# Configuración de rutas
CURRENT_DIR = Path(__file__).parent
MODELS_DIR = CURRENT_DIR / "models"
ONNX_DIR = CURRENT_DIR / "onnx_models"
LOGS_DIR = CURRENT_DIR / "logs"
BEST_MODELS_DIR = CURRENT_DIR / "best_models"

# Crear directorios necesarios
for directory in [MODELS_DIR, ONNX_DIR, LOGS_DIR, BEST_MODELS_DIR]:
    directory.mkdir(exist_ok=True)

class DQNMasterSystem:
    """
    Sistema maestro que conecta todos los módulos DQN
    """
    
    def __init__(self):
        self.config = None
        self.agent = None
        self.server = None
        self.evaluator = None
        self.best_manager = None
        
        # Registro de mejores modelos
        self.best_models_registry = {
            "best_reward": {"score": float('-inf'), "file": None, "onnx": None},
            "best_success": {"score": 0.0, "file": None, "onnx": None},
            "best_efficiency": {"score": 0.0, "file": None, "onnx": None},
            "latest": {"episode": 0, "file": None, "onnx": None}
        }
        
        self.load_registry()
    
    def load_registry(self):
        """Carga el registro de mejores modelos"""
        registry_file = BEST_MODELS_DIR / "models_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    self.best_models_registry = json.load(f)
                print(f"[MASTER] ✓ Registro de modelos cargado")
            except:
                print(f"[MASTER] ⚠ Error cargando registro, usando valores por defecto")
    
    def save_registry(self):
        """Guarda el registro de mejores modelos"""
        registry_file = BEST_MODELS_DIR / "models_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(self.best_models_registry, f, indent=2)
    
    def train_new_model(self):
        """Entrena un nuevo modelo DQN"""
        print("="*60)
        print("🚀 INICIANDO ENTRENAMIENTO DQN COMPLETO")
        print("="*60)
        
        try:
            # Importar módulos necesarios
            from dqn_agente_avanzado import train_dqn_agent, DQNConfig, DQNAgent
            
            # Configurar callback para mejores modelos
            original_train = train_dqn_agent
            
            # Entrenar modelo
            train_dqn_agent()
            
            print("[MASTER] ✓ Entrenamiento completado")
            
            # Verificar y registrar nuevos modelos
            self.check_new_models()
            
        except ImportError as e:
            print(f"[MASTER] ✗ Error importando módulos: {e}")
        except Exception as e:
            print(f"[MASTER] ✗ Error en entrenamiento: {e}")
    
    def check_new_models(self):
        """Verifica y registra nuevos modelos generados"""
        # Buscar archivos .pth recientes
        pth_files = list(CURRENT_DIR.glob("*.pth"))
        pth_files.extend(list(BEST_MODELS_DIR.glob("*.pth")))
        
        for pth_file in pth_files:
            # Generar ONNX si no existe
            onnx_file = self.ensure_onnx_exists(pth_file)
            
            # Actualizar registro
            if "best_reward" in pth_file.name:
                self.best_models_registry["best_reward"]["file"] = str(pth_file)
                self.best_models_registry["best_reward"]["onnx"] = str(onnx_file)
            elif "best_success" in pth_file.name:
                self.best_models_registry["best_success"]["file"] = str(pth_file)
                self.best_models_registry["best_success"]["onnx"] = str(onnx_file)
            elif "final" in pth_file.name:
                self.best_models_registry["latest"]["file"] = str(pth_file)
                self.best_models_registry["latest"]["onnx"] = str(onnx_file)
        
        self.save_registry()
        print(f"[MASTER] ✓ Registro de modelos actualizado")
    
    def ensure_onnx_exists(self, pth_file: Path) -> Path:
        """Asegura que existe la versión ONNX del modelo"""
        onnx_file = ONNX_DIR / f"{pth_file.stem}.onnx"
        
        if not onnx_file.exists():
            try:
                from dqn_agente_avanzado import DQNAgent, DQNConfig, export_to_onnx
                
                # Cargar modelo
                config = DQNConfig()
                agent = DQNAgent(config)
                agent.load_model(str(pth_file))
                
                # Exportar a ONNX
                export_to_onnx(agent, str(onnx_file))
                print(f"[MASTER] ✓ ONNX generado: {onnx_file.name}")
                
            except Exception as e:
                print(f"[MASTER] ✗ Error generando ONNX: {e}")
                # Crear archivo ONNX dummy
                onnx_file.touch()
        
        return onnx_file
    
    def start_tcp_server(self, port: int = 12345):
        """Inicia el servidor TCP para Unreal Engine"""
        print("="*60)
        print("🌐 INICIANDO SERVIDOR TCP PARA UNREAL ENGINE")
        print("="*60)
        
        try:
            from dqn_server_tcp import DQNTCPServer
            from dqn_agente_avanzado import DQNAgent, DQNConfig
            
            # Cargar mejor modelo disponible
            model_file = self.get_best_model("latest")
            if not model_file:
                print("[MASTER] ✗ No hay modelos disponibles")
                return
            
            # Cargar agente
            config = DQNConfig()
            agent = DQNAgent(config)
            agent.load_model(model_file)
            
            # Iniciar servidor
            server = DQNTCPServer(agent, port=port)
            print(f"[MASTER] ✓ Servidor TCP listo en puerto {port}")
            print(f"[MASTER] Usando modelo: {Path(model_file).name}")
            print("[MASTER] Presiona Ctrl+C para detener")
            
            server.start()
            
        except ImportError as e:
            print(f"[MASTER] ✗ Error importando servidor: {e}")
        except Exception as e:
            print(f"[MASTER] ✗ Error iniciando servidor: {e}")
    
    def evaluate_models(self):
        """Evalúa todos los modelos disponibles"""
        print("="*60)
        print("📊 EVALUANDO MODELOS DISPONIBLES")
        print("="*60)
        
        try:
            from dqn_evaluator import ModelEvaluator
            from dqn_agente_avanzado import DQNAgent, DQNConfig
            
            config = DQNConfig()
            
            for model_type, model_info in self.best_models_registry.items():
                model_file = model_info.get("file")
                if model_file and os.path.exists(model_file):
                    print(f"\n🔍 Evaluando {model_type}: {Path(model_file).name}")
                    
                    # Cargar y evaluar
                    agent = DQNAgent(config)
                    agent.load_model(model_file)
                    
                    evaluator = ModelEvaluator(agent)
                    results = evaluator.comprehensive_evaluation()
                    
                    # Mostrar resultados
                    print(f"   Recompensa promedio: {results['avg_reward']:.2f}")
                    print(f"   Tasa de éxito: {results['success_rate']:.1%}")
                    print(f"   Eficiencia: {results['efficiency']:.3f}")
        
        except ImportError as e:
            print(f"[MASTER] ✗ Error importando evaluador: {e}")
        except Exception as e:
            print(f"[MASTER] ✗ Error en evaluación: {e}")
    
    def export_all_onnx(self):
        """Exporta todos los modelos a ONNX"""
        print("="*60)
        print("📦 EXPORTANDO TODOS LOS MODELOS A ONNX")
        print("="*60)
        
        for model_type, model_info in self.best_models_registry.items():
            model_file = model_info.get("file")
            if model_file and os.path.exists(model_file):
                onnx_file = self.ensure_onnx_exists(Path(model_file))
                print(f"✓ {model_type}: {onnx_file.name}")
        
        # Copiar el mejor modelo a la ubicación principal
        best_latest = self.get_best_model("latest")
        if best_latest:
            main_onnx = CURRENT_DIR / "conducionModel.onnx"
            onnx_source = ONNX_DIR / f"{Path(best_latest).stem}.onnx"
            
            if onnx_source.exists():
                import shutil
                shutil.copy2(onnx_source, main_onnx)
                print(f"✓ Modelo principal: conducionModel.onnx")
    
    def get_best_model(self, model_type: str) -> str:
        """Obtiene la ruta del mejor modelo del tipo especificado"""
        model_info = self.best_models_registry.get(model_type, {})
        model_file = model_info.get("file")
        
        if model_file and os.path.exists(model_file):
            return model_file
        
        # Buscar archivos alternativos
        patterns = {
            "latest": ["*final*.pth", "*latest*.pth"],
            "best_reward": ["*best_reward*.pth", "*reward*.pth"],
            "best_success": ["*best_success*.pth", "*success*.pth"]
        }
        
        for pattern in patterns.get(model_type, []):
            files = list(CURRENT_DIR.glob(pattern))
            files.extend(list(BEST_MODELS_DIR.glob(pattern)))
            if files:
                return str(files[0])
        
        return None
    
    def run_full_pipeline(self):
        """Ejecuta el pipeline completo"""
        print("="*60)
        print("🔄 EJECUTANDO PIPELINE COMPLETO DQN")
        print("="*60)
        
        # 1. Entrenar modelo
        self.train_new_model()
        
        # 2. Exportar a ONNX
        self.export_all_onnx()
        
        # 3. Evaluar modelos
        self.evaluate_models()
        
        # 4. Mostrar resumen
        self.show_summary()
    
    def show_summary(self):
        """Muestra resumen del sistema"""
        print("\n" + "="*60)
        print("📋 RESUMEN DEL SISTEMA DQN")
        print("="*60)
        
        print("\n📁 Modelos disponibles:")
        for model_type, model_info in self.best_models_registry.items():
            model_file = model_info.get("file")
            onnx_file = model_info.get("onnx")
            
            if model_file and os.path.exists(model_file):
                print(f"  ✓ {model_type.upper()}")
                print(f"    PyTorch: {Path(model_file).name}")
                if onnx_file and os.path.exists(onnx_file):
                    print(f"    ONNX: {Path(onnx_file).name}")
                else:
                    print(f"    ONNX: ⚠ No disponible")
        
        print(f"\n📊 Directorios:")
        print(f"  🎯 Mejores modelos: {BEST_MODELS_DIR}")
        print(f"  📦 Modelos ONNX: {ONNX_DIR}")
        print(f"  📝 Logs: {LOGS_DIR}")
        
        print(f"\n🚀 Comandos disponibles:")
        print(f"  python dqn_master.py train    # Entrenar nuevo modelo")
        print(f"  python dqn_master.py server   # Servidor para Unreal Engine")
        print(f"  python dqn_master.py evaluate # Evaluar modelos")
        print(f"  python dqn_master.py export   # Exportar a ONNX")
        print("="*60)

def main():
    """Función principal"""
    system = DQNMasterSystem()
    
    if len(sys.argv) < 2:
        print("Uso: python dqn_master.py [train|server|evaluate|export|all]")
        system.show_summary()
        return
    
    command = sys.argv[1].lower()
    
    if command == "train":
        system.train_new_model()
    elif command == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
        system.start_tcp_server(port)
    elif command == "evaluate":
        system.evaluate_models()
    elif command == "export":
        system.export_all_onnx()
    elif command == "all":
        system.run_full_pipeline()
    else:
        print(f"Comando desconocido: {command}")
        system.show_summary()

if __name__ == "__main__":
    main()
