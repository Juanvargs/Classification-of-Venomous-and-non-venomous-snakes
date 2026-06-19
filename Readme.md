# Identificador de serpientes por especie y riesgo

Proyecto de vision por computador para identificar serpientes desde una imagen. El sistema primero predice la especie y despues convierte esa especie en una decision de riesgo: `Venenosa`, `No venenosa` o `No concluyente`.

La decision final del proyecto fue usar una politica conservadora: es mejor responder `No concluyente` cuando el modelo tiene duda que dar una falsa seguridad al usuario.

## Estado actual

Modelo principal:

```text
B4 EfficientNetV2B1
models/experiments/species_pilot51_b4_efficientnetv2b1_colab/best_model.keras
```

Modelo experimental conservado:

```text
B5 EfficientNetV2B3 300
models/experiments/species_pilot51_b5_efficientnetv2b3_300_colab/best_model.keras
```

Modelo de respaldo preventivo:

```text
outputs_targeted_efficientnetb0_lowlr
models/experiments/outputs_targeted_efficientnetb0_lowlr/best_model.keras
```

B4 queda como modelo activo porque fue el mas seguro para cliente. B5 mejora algunas metricas de especie, pero en la prueba grande produjo falsos seguros: serpientes venenosas que podian terminar como no venenosas. Por eso B5 queda como candidato de investigacion y no como decision automatica.

## Flujo del sistema

```text
Imagen del usuario
  -> preprocesamiento
  -> clasificador por especie
  -> top-5 especies candidatas
  -> metadata de especie
  -> ajuste opcional por pais o region
  -> respaldo binario venenosa/no venenosa
  -> regla integrada de seguridad
  -> resultado final
```

La clasificacion principal es multiclase por especie. La clasificacion venenosa/no venenosa no se aprende como unica salida principal; se deriva de la especie predicha usando metadata, y se valida con un modelo binario preventivo.

## Arquitectura

Arquitectura activa:

```text
EfficientNetV2B1 preentrenada en ImageNet
Entrada: 260x260x3
Salida: 51 clases de especie
Entrenamiento: transfer learning
Fase principal: cabeza nueva sobre base congelada
Fine-tuning: probado, pero no promovido si no mejora validacion
```

Arquitectura candidata:

```text
EfficientNetV2B3 preentrenada en ImageNet
Entrada: 300x300x3
Salida: 51 clases de especie
Uso: comparacion experimental B5
```

El modelo usa transferencia de aprendizaje: la red base ya viene entrenada con imagenes generales de ImageNet y se agrega una cabeza nueva para aprender las 51 especies del proyecto.

## Dataset activo

Dataset conservado:

```text
data/active/Snake Species Pilot 51 Targeted B4 Scale
train: 11104 imagenes
val: 1932 imagenes
test: 1252 imagenes
clases: 51 especies
```

Metadata activa:

```text
data/metadata/species_pilot51_b4_scale_metadata.csv
data/metadata/species_pilot51_b4_scale_manifest.csv
```

Los datasets anteriores B0, B1, B2 y B3 se usaron durante la investigacion, pero fueron retirados de la carpeta activa para dejar el proyecto limpio. Sus resultados quedan explicados en la documentacion.

## Resultados principales

B4 en test interno:

```text
accuracy: 0.63898
macro_f1: 0.63459
weighted_f1: 0.63978
top3_accuracy: 0.85783
top5_accuracy: 0.91374
support: 1252
```

B5 en test interno:

```text
accuracy: 0.64696
top3_accuracy: 0.86342
top5_accuracy: 0.90974
```

Decision:

```text
B4 activo para cliente.
B5 no activo por defecto porque tuvo falsos seguros en holdout completo.
```

## Regla de seguridad

La app combina especie, top-5, pais y modelo binario.

Regla resumida:

```text
Si la especie seleccionada es venenosa -> Venenosa.
Si la especie seleccionada es no venenosa, pero aparece una venenosa relevante en el top-5 -> No concluyente.
Si el binario preventivo alerta y la especie no es suficientemente confiable -> No concluyente.
Si la confianza de especie es baja -> No concluyente.
Solo si la especie no venenosa es consistente y segura -> No venenosa.
```

Esto protege al usuario contra el error mas grave: decir `No venenosa` cuando podria ser venenosa.

## Pais o region

El pais es opcional. Si el usuario lo conoce, el sistema lo usa como un prior geografico suave.

```text
GEO_PRIOR_ALPHA = 0.75
GEO_PRIOR_TOP_N = 10
```

Eso significa:

```text
El pais solo reordena las 10 especies visualmente mas probables.
No puede escoger cualquier especie del catalogo si visualmente no encaja.
Si el pais esta mal o no se conoce, el modelo sigue funcionando por imagen.
```

## Aplicaciones

Streamlit:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app_streamlit.py
```

API movil:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api_mobile:app --host 0.0.0.0 --port 8000
```

App movil:

```powershell
cd mobile_app
npm install
npm start
```

## Estructura actual

```text
app_streamlit.py        Aplicacion web local en Streamlit
api_mobile.py          API FastAPI para la app movil
mobile_app/            App movil Expo/React Native
src/mobile_inference.py Motor compartido de inferencia movil
src/                   Scripts de preparacion, evaluacion y entrenamiento
notebooks/             Notebook consolidado de entrenamiento en Colab
docs/                  Informes explicativos del proyecto
data/active/           Dataset activo B4 Scale
data/metadata/         Metadata y manifests de especies
data/external_tests/   Pruebas externas y holdout
models/experiments/    Modelos activos: B4, B5 y binario preventivo
```

## Documentacion

Informe tecnico completo:

```text
docs/informe_tecnico_proyecto.md
```

Implementacion de apps:

```text
docs/implementacion_apps.md
```

Notebook consolidado:

```text
notebooks/train_species_pilot51_consolidado_colab.ipynb
```

## Nota de seguridad

Este proyecto no reemplaza una identificacion profesional. Si hay riesgo de mordedura, el usuario debe mantenerse alejado de la serpiente y buscar ayuda profesional.
