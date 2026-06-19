# Implementacion de Streamlit y app movil

Este documento explica como se implementa el modelo en las dos interfaces del proyecto: la app web en Streamlit y la app movil en Expo/React Native.

## 1. Componentes

```text
app_streamlit.py
    App web local.

api_mobile.py
    Backend HTTP para la app movil.

src/mobile_inference.py
    Motor de inferencia usado por la API movil.

mobile_app/App.js
    Interfaz movil Expo/React Native.
```

Ambas apps usan la misma decision conceptual:

```text
imagen -> especie -> riesgo -> pais opcional -> respaldo binario -> decision final
```

## 2. Streamlit

Archivo:

```text
app_streamlit.py
```

Ejecutar:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app_streamlit.py
```

### 2.1 Carga de modelos

Streamlit usa:

```text
models/experiments/species_pilot51_b4_efficientnetv2b1_colab/best_model.keras
models/experiments/outputs_targeted_efficientnetb0_lowlr/best_model.keras
```

El primer modelo predice especie. El segundo modelo funciona como respaldo binario preventivo.

### 2.2 Carga de metadata

Archivos:

```text
data/metadata/species_pilot51_b4_scale_metadata.csv
data/metadata/species_pilot51_b4_scale_manifest.csv
```

La metadata permite convertir clase a:

```text
nombre comun
nombre cientifico
Venomous / Non Venomous
paises
continentes
```

El manifest permite calcular el prior geografico por pais o continente.

### 2.3 Flujo de usuario

El usuario:

```text
1. Carga una imagen.
2. Indica si conoce el pais.
3. Presiona Buscar.
4. Recibe especie, riesgo, confianza y top-5.
```

### 2.4 Resultado final

Streamlit muestra:

```text
Venenosa
No venenosa
No concluyente
```

Tambien muestra:

```text
especie seleccionada
nombre cientifico
clasificacion
top de candidatas
mensaje de seguridad
```

## 3. API movil

Archivo:

```text
api_mobile.py
```

Ejecutar:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api_mobile:app --host 0.0.0.0 --port 8000
```

Endpoints:

```text
GET /health
POST /predict
```

### 3.1 Health

`/health` sirve para confirmar que el backend esta prendido.

Respuesta esperada:

```json
{
  "status": "ok"
}
```

### 3.2 Predict

`/predict` recibe:

```text
image: archivo de imagen
country: pais o region opcional
```

Devuelve:

```text
decision
species
binary
model
safety_note
```

La respuesta incluye:

```text
model.decision_policy = B4 safety-first
```

Esto confirma que la API esta usando el perfil seguro B4.

## 4. Motor movil de inferencia

Archivo:

```text
src/mobile_inference.py
```

Funciones importantes:

```text
ensure_assets_exist()
predict_species()
predict_binary()
integrated_decision()
predict_image_bytes()
```

### 4.1 ensure_assets_exist

Valida que existan:

```text
modelo B4
clases B4
metadata B4
manifest B4
modelo binario
clases binarias
```

Si falta algo, lanza error antes de predecir.

### 4.2 predict_species

Hace:

```text
1. Carga modelo B4.
2. Preprocesa la imagen a 260x260.
3. Predice 51 probabilidades.
4. Obtiene top-5.
5. Aplica pais si existe.
6. Devuelve especie seleccionada y candidatas.
```

### 4.3 predict_binary

Hace:

```text
1. Carga modelo binario.
2. Preprocesa imagen a 224x224.
3. Calcula probabilidad de Venomous.
4. Aplica umbral preventivo.
```

### 4.4 integrated_decision

Combina especie y binario.

Reglas principales:

```text
Venenosa si la especie seleccionada es venenosa.
No concluyente si hay venenosa relevante en top-5.
No concluyente si el binario alerta y la especie no es muy segura.
No concluyente si la confianza es baja.
No venenosa solo si la evidencia es consistente.
```

## 5. App movil

Carpeta:

```text
mobile_app/
```

Ejecutar:

```powershell
cd mobile_app
npm install
npm start
```

Abrir con Expo Go.

### 5.1 Configuracion de API

En `mobile_app/App.js`:

```text
const API_URL = "http://192.168.10.19:8000";
```

Si cambia la red Wi-Fi, se debe cambiar esa IP por la IP local del computador.

El celular y el computador deben estar en la misma red.

### 5.2 Flujo movil

El usuario puede:

```text
Tomar foto con camara.
Elegir imagen de galeria.
Seleccionar pais o region.
Enviar al backend.
Ver resultado.
```

La app muestra:

```text
decision final
especie
nombre cientifico
confianza
region usada
modelo seguro B4
top-5 especies candidatas
nota de seguridad
```

## 6. Diferencia entre Streamlit y movil

Streamlit:

```text
Corre todo en un mismo archivo local.
Es ideal para pruebas, presentacion y depuracion.
```

Movil:

```text
La interfaz corre en el celular.
El modelo corre en el computador por API.
El celular envia imagen y recibe JSON.
```

## 7. Seguridad en ambas apps

Ambas apps deben mantener esta filosofia:

```text
Nunca forzar "No venenosa" si hay una senal venenosa relevante.
Usar "No concluyente" cuando la evidencia no sea suficiente.
Mostrar advertencia de no manipular la serpiente.
```

Mensaje usado:

```text
No manipules la serpiente. Si existe riesgo de mordedura, busca ayuda profesional.
```

## 8. Archivos que no deben faltar

Para Streamlit y movil:

```text
models/experiments/species_pilot51_b4_efficientnetv2b1_colab/best_model.keras
models/experiments/species_pilot51_b4_efficientnetv2b1_colab/class_names.json
models/experiments/outputs_targeted_efficientnetb0_lowlr/best_model.keras
models/experiments/outputs_targeted_efficientnetb0_lowlr/class_names.json
data/metadata/species_pilot51_b4_scale_metadata.csv
data/metadata/species_pilot51_b4_scale_manifest.csv
```

## 9. Prueba rapida

Backend:

```powershell
uvicorn api_mobile:app --host 0.0.0.0 --port 8000
```

Health:

```powershell
curl http://127.0.0.1:8000/health
```

Streamlit:

```powershell
streamlit run app_streamlit.py
```

Movil:

```powershell
cd mobile_app
npm start
```
