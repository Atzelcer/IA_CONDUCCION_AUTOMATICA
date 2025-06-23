#!/bin/bash
# ================================================================
# INSTALADOR AUTOMATICO - PROYECTOIAUE DQN (Linux/Mac)
# Sistema de Conducción Autónoma con Deep Q-Learning
# ================================================================

echo "================================================================"
echo "PROYECTOIAUE - INSTALADOR AUTOMATICO"
echo "Sistema de Conducción Autónoma con DQN Learning"
echo "================================================================"
echo

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no está instalado"
    echo "Por favor instala Python 3.10+ usando tu gestor de paquetes"
    exit 1
fi

echo "✅ Python detectado"
python3 --version

# Crear entorno virtual
echo
echo "📦 Creando entorno virtual..."
if [ -d "venv_dqn" ]; then
    echo "⚠️ El entorno virtual ya existe. ¿Desea recrearlo? (s/n)"
    read -r recreate
    if [[ $recreate == "s" || $recreate == "S" ]]; then
        rm -rf venv_dqn
        python3 -m venv venv_dqn
    fi
else
    python3 -m venv venv_dqn
fi

# Activar entorno virtual
echo
echo "🔧 Activando entorno virtual..."
source venv_dqn/bin/activate

# Actualizar pip
echo
echo "📈 Actualizando pip..."
python -m pip install --upgrade pip

# Instalar dependencias
echo
echo "📦 Instalando dependencias..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo
    echo "❌ Error instalando dependencias básicas"
    echo "Instalando dependencias una por una..."
    
    echo "Instalando NumPy..."
    pip install "numpy>=1.21.0"
    
    echo "Instalando Matplotlib..."
    pip install "matplotlib>=3.5.0"
    
    echo "Instalando PyTorch..."
    pip install "torch>=1.12.0" "torchvision>=0.13.0" "torchaudio>=0.12.0"
    
    echo "Instalando ONNX..."
    pip install "onnx>=1.12.0" "onnxruntime>=1.12.0"
    
    echo "Instalando utilidades..."
    pip install "seaborn>=0.11.0" "pandas>=1.4.0" "tqdm>=4.64.0"
fi

# Verificar instalación
echo
echo "🔍 Verificando instalación..."
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import numpy; print('✅ NumPy:', numpy.__version__)"
python -c "import matplotlib; print('✅ Matplotlib:', matplotlib.__version__)"
python -c "import onnxruntime; print('✅ ONNX Runtime:', onnxruntime.__version__)"

# Crear estructura de directorios
echo
echo "📁 Creando estructura de directorios..."
mkdir -p logs models results configs

# Generar configuración por defecto
echo
echo "⚙️ Generando configuración por defecto..."
python -c "
from dqn_utils import ConfigManager
config = ConfigManager.get_default_config()
ConfigManager.save_config(config, 'configs/dqn_default.json')
print('✅ Configuración creada: configs/dqn_default.json')
"

# Verificar Q-Table existente
echo
echo "🔍 Verificando archivos de modelo..."
if [ -f "q_table_agenteRL.npy" ]; then
    echo "✅ Q-Table encontrada: q_table_agenteRL.npy"
else
    echo "⚠️ Q-Table no encontrada"
    echo "Ejecutando entrenamiento básico..."
    python entrenar_agente_q_table_complejo.py
fi

# Verificar modelo ONNX
if [ -f "conducionModel.onnx" ]; then
    echo "✅ Modelo ONNX encontrado: conducionModel.onnx"
else
    echo "⚠️ Modelo ONNX no encontrado"
    echo "Se generará automáticamente durante el entrenamiento DQN"
fi

echo
echo "================================================================"
echo "✅ INSTALACIÓN COMPLETADA EXITOSAMENTE"
echo "================================================================"
echo
echo "🚀 COMANDOS DISPONIBLES:"
echo
echo "1. MODELO 1 - Q-Learning Clásico:"
echo "   python agente_q_table_tcp.py"
echo
echo "2. MODELO 2 - DQN Avanzado:"
echo "   python dqn_agente_avanzado.py          # Entrenamiento"
echo "   python dqn_server_tcp.py               # Servidor en tiempo real"
echo
echo "3. Evaluación y análisis:"
echo "   python dqn_evaluator.py"
echo
echo "4. Utilidades:"
echo "   python dqn_utils.py                    # Herramientas de debug"
echo
echo "================================================================"
echo "📋 PRÓXIMOS PASOS:"
echo
echo "1. Configurar Unreal Engine 4.27"
echo "2. Abrir TF_APR_ConduccionAut.uproject"
echo "3. Compilar el proyecto en Unreal Engine"
echo "4. Ejecutar uno de los servidores Python (comando arriba)"
echo "5. Iniciar la simulación en Unreal Engine"
echo
echo "ℹ️ Para más información, consultar README.md"
echo "================================================================"
echo

# Hacer ejecutables los scripts de Python
chmod +x *.py

echo "🔧 Permisos de ejecución configurados"
echo "Para activar el entorno virtual en futuras sesiones:"
echo "source venv_dqn/bin/activate"
