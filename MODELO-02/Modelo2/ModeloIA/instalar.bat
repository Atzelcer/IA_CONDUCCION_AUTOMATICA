@echo off
REM ================================================================
REM INSTALADOR AUTOMATICO - PROYECTOIAUE DQN
REM Sistema de Conducción Autónoma con Deep Q-Learning
REM ================================================================

echo.
echo ================================================================
echo PROYECTOIAUE - INSTALADOR AUTOMATICO
echo Sistema de Conducción Autónoma con DQN Learning
echo ================================================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no está instalado o no está en PATH
    echo Por favor instala Python 3.10+ desde: https://python.org
    pause
    exit /b 1
)

echo ✅ Python detectado
python --version

REM Crear entorno virtual
echo.
echo 📦 Creando entorno virtual...
if exist venv_dqn (
    echo ⚠️ El entorno virtual ya existe. ¿Desea recrearlo? (S/N)
    set /p recreate=
    if /i "%recreate%"=="S" (
        rmdir /s /q venv_dqn
        python -m venv venv_dqn
    )
) else (
    python -m venv venv_dqn
)

REM Activar entorno virtual
echo.
echo 🔧 Activando entorno virtual...
call venv_dqn\Scripts\activate.bat

REM Actualizar pip
echo.
echo 📈 Actualizando pip...
python -m pip install --upgrade pip

REM Instalar dependencias
echo.
echo 📦 Instalando dependencias...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ❌ Error instalando dependencias básicas
    echo Instalando dependencias una por una...
    
    echo Instalando NumPy...
    pip install numpy>=1.21.0
    
    echo Instalando Matplotlib...
    pip install matplotlib>=3.5.0
    
    echo Instalando PyTorch...
    pip install torch>=1.12.0 torchvision>=0.13.0 torchaudio>=0.12.0
    
    echo Instalando ONNX...
    pip install onnx>=1.12.0 onnxruntime>=1.12.0
    
    echo Instalando utilidades...
    pip install seaborn>=0.11.0 pandas>=1.4.0 tqdm>=4.64.0
)

REM Verificar instalación
echo.
echo 🔍 Verificando instalación...
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import numpy; print('✅ NumPy:', numpy.__version__)"
python -c "import matplotlib; print('✅ Matplotlib:', matplotlib.__version__)"
python -c "import onnxruntime; print('✅ ONNX Runtime:', onnxruntime.__version__)"

REM Crear estructura de directorios
echo.
echo 📁 Creando estructura de directorios...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "results" mkdir results
if not exist "configs" mkdir configs

REM Generar configuración por defecto
echo.
echo ⚙️ Generando configuración por defecto...
python -c "
from dqn_utils import ConfigManager
config = ConfigManager.get_default_config()
ConfigManager.save_config(config, 'configs/dqn_default.json')
print('✅ Configuración creada: configs/dqn_default.json')
"

REM Verificar Q-Table existente
echo.
echo 🔍 Verificando archivos de modelo...
if exist q_table_agenteRL.npy (
    echo ✅ Q-Table encontrada: q_table_agenteRL.npy
) else (
    echo ⚠️ Q-Table no encontrada
    echo Ejecutando entrenamiento básico...
    python entrenar_agente_q_table_complejo.py
)

REM Verificar modelo ONNX
if exist conducionModel.onnx (
    echo ✅ Modelo ONNX encontrado: conducionModel.onnx
) else (
    echo ⚠️ Modelo ONNX no encontrado
    echo Se generará automáticamente durante el entrenamiento DQN
)

echo.
echo ================================================================
echo ✅ INSTALACIÓN COMPLETADA EXITOSAMENTE
echo ================================================================
echo.
echo 🚀 COMANDOS DISPONIBLES:
echo.
echo 1. MODELO 1 - Q-Learning Clásico:
echo    python agente_q_table_tcp.py
echo.
echo 2. MODELO 2 - DQN Avanzado:
echo    python dqn_agente_avanzado.py          (Entrenamiento)
echo    python dqn_server_tcp.py               (Servidor en tiempo real)
echo.
echo 3. Evaluación y análisis:
echo    python dqn_evaluator.py
echo.
echo 4. Utilidades:
echo    python dqn_utils.py                    (Herramientas de debug)
echo.
echo ================================================================
echo 📋 PRÓXIMOS PASOS:
echo.
echo 1. Abrir Unreal Engine y cargar TF_APR_ConduccionAut.uproject
echo 2. Compilar el proyecto en Unreal Engine
echo 3. Ejecutar uno de los servidores Python (comando arriba)
echo 4. Iniciar la simulación en Unreal Engine
echo.
echo ℹ️ Para más información, consultar README.md
echo ================================================================
echo.

pause
