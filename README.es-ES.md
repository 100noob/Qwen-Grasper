# 1.YOLO-Edge-Perception

Este proyecto es un sistema de detección de bloques de colores de alto rendimiento diseñado específicamente para la **Raspberry Pi 5**. Cuenta con un flujo de trabajo completo, desde el entrenamiento del modelo **YOLOv11** (en PC) hasta el despliegue de inferencia optimizado utilizando el framework **ncnn** (en Raspberry Pi).

> **Nota**: Se recomienda encarecidamente utilizar **VS Code** con las extensiones de **Python** instaladas.

## 🚀 Características Principales
* **Pipeline de extremo a extremo**: Desde el entrenamiento del conjunto de datos en Roboflow hasta el despliegue en el mundo real en Raspberry Pi.
* **Sistema de construcción automatizado**: El proyecto utiliza `uv` para gestionar los entornos virtuales de Python y la instalación de dependencias (el uso de `uv` evita conflictos de versiones de pip).
* **Inferencia de alto rendimiento**: El módulo de `inference` utiliza el framework **ncnn** para lograr una detección fluida y en tiempo real en la Raspberry Pi 5.

## 📁 Estructura del Proyecto
```text
RPi-ColorBlock-Detection/
├── color_cube_train/          # Módulo de entrenamiento del modelo (Ejecutar en PC)
│   ├── src/                   # Scripts de entrenamiento principales
│   ├── source_data/           # Conjunto de datos bruto (70% Train, 20% Val, 10% Test)
│   ├── data.yaml              # Configuración del conjunto de datos de YOLO
│   ├── paths.py               # Utilidad de gestión de rutas
│   ├── requirements.txt       # Lista de dependencias de Python para entrenamiento
│   ├── yolo11s.pt             # Pesos pre-entrenados
│   └── runs/                  # Salidas y logs de entrenamiento
└── inference/                 # Módulo de despliegue en Raspberry Pi (Python + ncnn)
    ├── models/                # Modelos ncnn convertidos (.param / .bin)
    ├── requirements.txt       # Dependencias de inferencia (excluyendo libs globales)
    └── main.py
```

## 📊 Conjuntos de Datos y Recursos
Todos los conjuntos de datos utilizados en este proyecto fueron recolectados y anotados manualmente por el autor. ¡Son completamente de código abierto y gratuitos!

* **Dataset de Entrenamiento YOLO y Despliegue en Raspberry Pi 5**:
  [Color Cube Dataset (Roboflow)](https://app.roboflow.com/brian114-xv3lk/color-cube-gvk4q/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
  *Clases: 0: purple-cube, 1: green-cube, 2: orange-cube, 3: pink-cube, 4: yellow-cube (Soporta extensión hasta 7 clases)*

* **Dataset de Fine-Tuning y Pruebas de LLM**:
  [LLM Fine-Tuning Test Set (Roboflow)](https://app.roboflow.com/brian114-xv3lk/llm-fine-tuning-test-set-smglh/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

## 🛠️ Prerrequisitos: Instalación de `uv`
Este proyecto depende fuertemente de `uv` (un gestor de proyectos y paquetes de Python extremadamente rápido) para manejar los entornos virtuales y las dependencias. Evita eficazmente los conflictos de versiones de pip.

Si aún no has instalado `uv`, utiliza el instalador independiente oficial:

**En macOS y Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**En Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 🛠️ Instrucciones de Construcción
1. Entrenamiento del Modelo (color-cube-train)

* **Configuración del Entorno**: Utilizamos `uv` para crear rápidamente un entorno virtual de Python e instalar las dependencias desde `requirements.txt`. Se recomienda encarecidamente el uso de `uv` por su velocidad y capacidad para evitar conflictos de versiones de pip.
* **Prerrequisito Manual**: Por favor, instale PyTorch con soporte para CUDA previamente en Windows.

* **Comandos de Construcción**:
```bash
cd color_cube_train
uv venv
source .venv/bin/activate  # En Windows use: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## 🐧 Linux: Uso de Ubuntu (Raspberry Pi / PC)
* La configuración de Ubuntu es sencilla utilizando apt.
* 1. Actualizar listas de paquetes: ```sudo apt update && sudo apt upgrade -y```
* 2. Instalar cadena de herramientas de desarrollo:
```bash
sudo apt install -y build-essential git
```
* 3. Específico para Raspberry Pi (OpenCV)
* Para evitar conflictos de entorno virtual y problemas de compilación en Raspberry Pi, instale OpenCV globalmente: ```sudo apt install -y python3-opencv```

# 2.Interactive-Teaching-to-VLM-Dataset
```
finetune_inference/
├── 📂 click/                                # Etapa 1: Herramienta de Anotación
│   ├── 📂 build/                            # Archivos compilados
│   ├── 📂 destset/                          # Directorio de salida para JSONL anotados
│   ├── 📂 main/                             # Código fuente para interacción de clics
│   ├── 📂 model/                            # Pesos del modelo YOLO (.pt)
│   ├── 📂 my_dir/                           # [Creado por Usuario] Carpeta destino para imágenes procesadas
│   ├── 📂 src_images/                       # [Creado por Usuario] Imágenes fuente brutas para anotación
│   ├── paths.py                             # Configuración de rutas locales
│   ├── readme.txt
│   └── requirements.txt
│
├── 📂 Detection-to-VLM_Conversation_Format_Conversion/  # Etapa 2: Formateo de Datos
│   ├── 📂 JSONL/                            # Salida de JSONL conversacional convertida
│   ├── 📂 main/                             # Scripts de normalización y formateo
│   ├── 📂 src_images/                       # [Creado por Usuario] Enlace simbólico o copia de imágenes
│   ├── 📂 venv/                             # Entorno virtual
│   ├── paths.py
│   └── requirements.txt
│
├── 📂 lora_finetune_unsloth/                 # Etapa 3: Entrenamiento
│   ├── 📂 export_weight/                    # Adaptadores/pesos LoRA guardados
│   ├── 📂 imgs_and_json/
│   │   └── 📂 src_imgs/                     # [Creado por Usuario] Imágenes finales del dataset
│   ├── 📂 main/                             # Scripts de entrenamiento de Unsloth
│   ├── 📂 venv/                             # Entorno virtual
│   ├── paths.py
│   ├── pip.txt                              # Lista de dependencias
│   └── requirements.txt                     # Añadido
│
├── 📂 test_lora_weight/                      # Etapa 5: Validación de Pesos LoRA
│   ├── 📂 main/                             # Scripts de inferencia y evaluación mAP
│   ├── 📂 src/                              # Utilidades de evaluación (IoU, cálculo de mAP)
│   ├── 📂 test_set/                         # [Creado por Usuario] Conjunto de datos de prueba
│   │   ├── 📂 test_img/                     # [Creado por Usuario] Imágenes de prueba
│   │   └── 📂 test_lable/                   # [Creado por Usuario] Etiquetas GT en formato YOLO
│   ├── 📂 output_set/                       # Imágenes de salida con cajas delimitadoras
│   ├── 📂 pred_set/                         # Etiquetas predichas por el modelo
│   ├── paths.py                             # Configuración de rutas
│   └── requirements.txt                     # Dependencias
│
└── 📂 model_Finetune_test/                   # Etapa 4: Pruebas
    ├── 📂 main/                             # Scripts de pruebas de inferencia
    └── requirements.txt                     # Añadido
```

## Descripción General del Proyecto
1. **click**
Descripción: Una herramienta de anotación de datos utilizada para la detección interactiva de objetos. Captura las coordenadas absolutas de las cajas delimitadoras y registra las secuencias de clics específicas (orden) de los bloques de colores identificados en una imagen. Esto establece la verdad fundamental (ground truth) tanto para la ubicación como para la prioridad de agarre.

2. **Detection-to-VLM_Conversation_Format_Conversion**
Descripción: Un pipeline de procesamiento de datos que convierte la salida JSONL bruta de la herramienta de clics en un formato conversacional específico para VLM. Formatea los datos para incluir prompts de sistema estructurados, instrucciones del usuario y respuestas del asistente, diseñados específicamente para los requisitos de ajuste fino del modelo Qwen (Qwen-VL).

3. **lora_finetune_unsloth**
Descripción: El módulo de entrenamiento central utilizado para el ajuste fino del modelo multimodal grande Qwen. Aprovecha la librería Unsloth para implementar un entrenamiento LoRA (Low-Rank Adaptation) altamente eficiente en memoria. Este programa permite que el modelo aprenda tareas específicas —como contar bloques de colores y seguir un orden de agarre específico— basándose en el conjunto de datos convertido.
    * **`lora_finetune_unsloth/main/main.py`**: Este script actúa como el punto de entrada principal para el proceso de ajuste fino. Carga los datos JSON conversacionales formateados y las imágenes brutas, configura el Unsloth FastVisionModel con parámetros LoRA (para un entrenamiento eficiente en parámetros), inicializa el SFTTrainer, ejecuta el bucle de entrenamiento y, finalmente, exporta los pesos del adaptador ajustado y los archivos de configuración necesarios.

4. **model_Finetune_test**
Descripción: El módulo de inferencia y validación utilizado para probar el modelo Qwen-VL ajustado.
    * **`model_Finetune_test/main/main.py`**: Este script envía imágenes de prueba locales codificadas en base64 e instrucciones a un servidor de inferencia desplegado (a través de una API compatible con OpenAI). Solicita una salida JSON estructurada (incluyendo conteos de objetos, descripciones de colores y cajas delimitadoras) y posteriormente analiza la respuesta del servidor para dibujar visualmente las cajas delimitadoras predichas directamente sobre las imágenes originales, guardando los resultados localmente para su validación.

5. **test_lora_weight**
Descripción: Evalúa los pesos LoRA ajustados en un conjunto de pruebas designado, validando las coordenadas de predicción y las prioridades de agarre. Cuenta con capacidades de cómputo de mAP y guarda salidas anotadas visualmente.
    * **`test_lora_weight/main/main.py`**: Ejecuta predicciones sobre el conjunto de datos de prueba, consultando el modelo VLM y guardando las salidas de inferencia. Posteriormente, invoca las funciones de evaluación.
    * **Nota**: El directorio de pruebas (`test_set`) y sus subdirectorios (`test_img` para imágenes y `test_lable` para etiquetas de verdad fundamental en formato YOLO) deben ser creados y poblados manualmente por el usuario antes de ejecutar la evaluación.

### Creación de un Entorno Virtual usando `uv`
Para ejecutar `model_Finetune_test/main/main.py`, se recomienda utilizar `uv` (un gestor de proyectos y paquetes de Python extremadamente rápido) para crear un entorno virtual e instalar las dependencias.

**Paso 1: Instalar `uv`**
Si aún no has instalado `uv`, puedes hacerlo utilizando el instalador independiente oficial:
```bash
# En macOS y Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# En Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Paso 2: Crear un Entorno Virtual**
Navega al directorio `model_Finetune_test` y crea el entorno virtual:
```bash
cd finetune_inference/model_Finetune_test
uv venv
```

**Paso 3: Activar el Entorno Virtual**
```bash
# En Linux / macOS
source .venv/bin/activate

# En Windows
.venv\Scripts\activate
```

**Paso 4: Instalar Dependencias usando `uv`**
Usa `uv pip` para instalar rápidamente las librerías requeridas enumeradas en el archivo `requirements.txt`:
```bash
uv pip install -r requirements.txt
```

Una vez instalado, puedes configurar tus ajustes de API en `main.py` y ejecutar el script.

## 🚧 Estado del Proyecto: En Desarrollo
Las características descritas en la descripción del proyecto —específicamente la integración completa del brazo robótico y el bucle de inferencia de extremo a extremo— se encuentran actualmente en progreso.


## Agradecimientos
* [Ultralytics (YOLOv11)](https://github.com/ultralytics/ultralytics)
* [Tencent ncnn](https://github.com/Tencent/ncnn)
* [Roboflow Universe](https://universe.roboflow.com/)
* https://github.com/opencv/opencv
