# 🦙 Llama 3.1 — Interfaz Local

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interfaz_Web-red)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3.1-green)
![LangChain](https://img.shields.io/badge/LangChain-RAG-yellow)

**Sistema RAG** que emula el razonamiento de un **experto jurídico-normativo de la UNAH**.
Consulta documentos oficiales y recibe análisis estructurados, citas textuales, trazabilidad y recomendaciones prácticas.

Ideal para estudiantes, docentes y personal administrativo que necesiten respuestas precisas sobre reglamentos, estatutos y procedimientos universitarios.

## Características

- Emula el **razonamiento experto** con prompt maestro ultra-detallado
- Análisis estructurado en 6 secciones (Análisis preliminar → Marco normativo → Razonamiento → Conclusiones → Recomendación → Certeza)
- Soporte para:
  - Consultas simples
  - Análisis de **casos complejos** con múltiples variables
  - Casos de ejemplo predefinidos
- Interfaz web moderna con **Streamlit** (3 pestañas + descarga de informes)
- Modo consola incluido para pruebas rápidas
- Base vectorial con **ChromaDB** + embeddings multilingües optimizados
- Recuperación avanzada con **MMR** (máxima relevancia + diversidad)
- Citas completas con página y fragmento relevante

## Requisitos

Antes de instalar, asegúrate de tener:

- **Python 3.10+**
- **pip** actualizado
- **Ollama** instalado → https://ollama.ai
- Modelo descargado:
  ```bash
  ollama pull llama3.1
  ```
- **Git** para clonar el repositorio

## 🔧 Instalación

## 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/Pineda04/Sistema-Experto.git
```

## Entrar al proyecto

```bash
cd Sistema-Experto/
```

## Crear entorno virtual

### 🔹 Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 🔹 Windows (CMD)

```cmd
python -m venv venv
venv\Scripts\activate
```

### 🔹 Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 📥 Instalar dependencias

```bash
pip install -r requirements.txt
```

Luego crea la carpeta "documentos y coloca tus documentos oficiales:

```bash
mkdir documentos
# pega aquí tus documentos
```

## ▶️ Ejecutar la aplicación

```bash
streamlit run app.py
# El anterior comando es para ejecutarla desde su interfaz
```

o también se puede usar:

```bash
python test_console.py
# Este comando es par probar desde la terminal
```

---

## 🧠 Modelo utilizado

El archivo usa la variable:

```python
MODEL = "llama3.1"
```
Pero puede ser cambiada por cualquier otro modelo soportado por Ollama.

---

## 📁 Estructura del proyecto
```
Sistema-Experto/
├── app.py              → Interfaz web con Streamlit
├── rag_system.py       → Clase principal RAGSystemUNAH
├── test_console.py     → Modo consola
├── documentos/         → Aqui van los documentos a usar
├── chroma_db/          → Base vectorial (se crea automáticamente al ejecutar)
├── requirements.txt    → (librerias a instalar)
├── README.md
└── venv/               → (creado localmente)
```