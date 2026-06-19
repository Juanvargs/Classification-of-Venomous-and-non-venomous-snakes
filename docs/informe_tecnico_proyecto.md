# Informe tecnico del proyecto

## 1. Objetivo

El objetivo del proyecto es identificar serpientes a partir de imagenes y entregar una decision de riesgo entendible para el usuario final.

El sistema no se limita a decir `venenosa` o `no venenosa` desde el inicio. La estrategia final es mas informativa:

```text
imagen -> especie probable -> riesgo de la especie -> decision segura
```

Esto permite mostrar el nombre comun, el nombre cientifico, el top de especies candidatas y una clasificacion de riesgo.

## 2. Problema que se resolvio

Al inicio se trabajo con clasificacion binaria: `Venomous` y `Non Venomous`. Esa ruta es simple, pero tiene una limitacion importante: no explica que especie cree estar viendo. Si el modelo se equivoca, es dificil saber por que.

Despues se decidio empezar por clasificacion de especie. Esa decision fue correcta porque:

```text
1. Da una salida biologicamente mas completa.
2. Permite convertir especie a riesgo usando metadata.
3. Permite usar pais o region para mejorar la decision.
4. Permite analizar errores por especie y no solo por clase binaria.
5. Permite mostrar top-5 y controlar la incertidumbre.
```

## 3. Evolucion del proyecto

### 3.1 Baseline inicial

Se comenzo con una red convolucional usando transferencia de aprendizaje. La idea base fue usar una red ya entrenada y adaptarla al problema de serpientes.

En esta etapa se probaron modelos mas pequenos como MobileNetV2 y luego EfficientNetV2B0. Sirvieron para validar que el flujo funcionaba, pero no eran suficientemente fuertes para la etapa final.

### 3.2 Paso a clasificacion por especie

Se construyo un dataset piloto con especies de serpientes organizadas en carpetas:

```text
train/
val/
test/
```

Cada carpeta contiene subcarpetas por clase. Cada clase representa una especie o una carpeta de especie, por ejemplo:

```text
018_Agkistrodon_contortrix
065_Bogertophis_subocularis
```

El modelo aprende una clasificacion multiclase. En vez de dos salidas, la red tiene 51 salidas, una por especie.

### 3.3 B0

B0 uso EfficientNetV2B0 con imagenes de 224x224. Fue mejor que los pilotos anteriores, pero todavia tenia rendimiento limitado:

```text
accuracy: 0.58546
macro_f1: 0.57688
top3_accuracy: 0.82428
top5_accuracy: 0.89137
```

Sirvio como primer modelo fuerte, pero no quedo como final.

### 3.4 B1

B1 uso EfficientNetV2B1 con imagenes de 260x260. Mejoro frente a B0:

```text
accuracy: 0.62620
macro_f1: 0.62802
top3_accuracy: 0.83866
top5_accuracy: 0.90096
```

El fine-tuning no supero al entrenamiento de cabeza. Por eso se eligio `best_head_model.keras`.

### 3.5 B2 y B3

B2 y B3 intentaron mejorar errores puntuales agregando imagenes de especies confundidas.

Resultado:

```text
B2 no reemplazo a B1 porque bajo metricas internas.
B3 mejoro algunos casos externos, pero aumento incertidumbre y no fue suficientemente estable.
```

Conclusion de esta fase:

```text
No convenia mejorar solo unos pocos casos.
Hacia falta una mejora de escala general.
```

### 3.6 B4 Scale

B4 fue la mejora mas importante del proyecto. Se amplio el dataset de entrenamiento de forma mas general.

Dataset activo:

```text
Snake Species Pilot 51 Targeted B4 Scale
train: 11104
val: 1932
test: 1252
clases: 51
```

Modelo:

```text
EfficientNetV2B1
imagen: 260x260
salida: 51 especies
```

Metricas:

```text
accuracy: 0.63898
macro_f1: 0.63459
weighted_f1: 0.63978
top3_accuracy: 0.85783
top5_accuracy: 0.91374
```

B4 quedo como modelo principal porque mejora frente a B1 y mantiene mejor seguridad global que B5.

### 3.7 B5

B5 fue un intento de modelo mas fuerte:

```text
EfficientNetV2B3
imagen: 300x300
dataset: B4 Scale
```

Metricas:

```text
accuracy: 0.64696
top3_accuracy: 0.86342
top5_accuracy: 0.90974
```

B5 parecia mejor por accuracy, pero en el holdout completo produjo falsos seguros. Eso significa que algunas serpientes venenosas podian terminar como no venenosas. Por seguridad, B5 no se usa como modelo principal.

## 4. Arquitectura usada

La arquitectura activa es EfficientNetV2B1.

Componentes:

```text
Entrada: imagen RGB
Tamano: 260x260
Base convolucional: EfficientNetV2B1 sin capa superior
Pooling: GlobalAveragePooling2D
Regularizacion: Dropout
Capa final: Dense con softmax
Salida: 51 probabilidades
```

La salida softmax entrega una probabilidad para cada especie. La especie con mayor probabilidad es el top-1. Tambien se revisan top-3 y top-5.

## 5. Transfer learning

El modelo no se entrena desde cero. Se usa transferencia de aprendizaje:

```text
1. EfficientNetV2B1 ya viene preentrenada con ImageNet.
2. Se quita la cabeza original de ImageNet.
3. Se agrega una cabeza nueva de 51 especies.
4. Primero se congela la base.
5. Se entrena la cabeza nueva.
6. Se prueba fine-tuning suave.
7. Se conserva el modelo que mejor valida.
```

Entrenar la cabeza significa entrenar solo las capas nuevas que convierten las caracteristicas generales de imagen en especies de serpiente.

Fine-tuning significa descongelar una parte final de la red base y ajustarla con learning rate muy bajo. En varios experimentos no mejoro, por eso no se promovio si empeoraba validacion.

## 6. Entrenamiento

El entrenamiento se hizo en Google Colab con GPU.

Datos:

```text
train: aprende
val: decide cuando guardar el mejor modelo
test: evalua el resultado final
```

Callbacks usados:

```text
ModelCheckpoint: guarda el mejor modelo por val_loss.
EarlyStopping: detiene si no mejora.
ReduceLROnPlateau: baja el learning rate si se estanca.
```

Metricas:

```text
accuracy: especie top-1 correcta.
top3_accuracy: especie correcta dentro de las 3 primeras.
top5_accuracy: especie correcta dentro de las 5 primeras.
macro_f1: promedio F1 por clase sin favorecer clases grandes.
weighted_f1: F1 ponderado por cantidad de imagenes.
loss: error de entrenamiento/validacion.
```

## 7. Clasificacion por especie

El modelo recibe una imagen y devuelve un vector de 51 probabilidades.

Ejemplo conceptual:

```text
Agkistrodon contortrix: 0.62
Lampropeltis triangulum: 0.11
Crotalus ruber: 0.07
...
```

El sistema selecciona una especie principal, pero tambien conserva el top-5. El top-5 es clave para seguridad, porque puede revelar que hay una especie venenosa plausible aunque la top-1 sea no venenosa.

## 8. Clasificacion venenosa/no venenosa

La clasificacion de riesgo se obtiene de dos fuentes.

Primera fuente: metadata de especie.

```text
species_pilot51_b4_scale_metadata.csv
```

Cada especie tiene un campo de riesgo:

```text
Venomous
Non Venomous
```

Segunda fuente: modelo binario preventivo.

```text
outputs_targeted_efficientnetb0_lowlr
```

Este modelo binario no manda solo. Sirve como alerta preventiva cuando la salida por especie parece no venenosa pero hay riesgo visual.

## 9. Regla integrada de seguridad

La regla final prioriza seguridad.

Casos:

```text
Si especie seleccionada es Venomous:
    resultado = Venenosa

Si especie seleccionada es Non Venomous pero el top-5 contiene una venenosa relevante:
    resultado = No concluyente

Si el modelo binario alerta Venomous y la especie no venenosa no es suficientemente confiable:
    resultado = No concluyente

Si la confianza de especie es baja:
    resultado = No concluyente

Si especie, top-5 y binario son consistentes con Non Venomous:
    resultado = No venenosa
```

Umbrales activos:

```text
SPECIES_CONFIDENCE_FLOOR = 0.35
NON_VENOMOUS_SAFE_CONFIDENCE = 0.50
NON_VENOMOUS_TOP5_MIN = 4
NON_VENOMOUS_BINARY_ALERT_MAX = 0.45
NON_VENOMOUS_TOP5_VENOMOUS_ALERT = 0.07
```

El objetivo no es maximizar respuestas seguras, sino evitar falsas seguridades.

## 10. Uso de pais o region

El pais o region es opcional.

Si el usuario no conoce el pais, el sistema trabaja solo con imagen.

Si el usuario da pais, se aplica un prior geografico suave:

```text
GEO_PRIOR_ALPHA = 0.75
GEO_PRIOR_TOP_N = 10
```

El sistema revisa las 10 especies visualmente mas probables y reordena usando frecuencia geografica del manifest. No permite que el pais escoja una especie visualmente imposible.

Esto mejora cuando el pais es correcto y controla el riesgo cuando el pais es incorrecto.

## 11. Funcionamiento del codigo

Archivos principales:

```text
app_streamlit.py
api_mobile.py
src/mobile_inference.py
mobile_app/App.js
```

`app_streamlit.py`:

```text
Carga modelos.
Carga metadata.
Recibe imagen.
Aplica clasificacion por especie.
Aplica pais opcional.
Aplica modelo binario.
Integra decision.
Muestra resultado y top-5.
```

`src/mobile_inference.py`:

```text
Contiene la misma logica para la app movil.
Expone una funcion predict_image_bytes.
Devuelve decision, especie, binario, top candidatos y nota de seguridad.
```

`api_mobile.py`:

```text
Expone /health.
Expone /predict.
Recibe imagen y pais opcional.
Llama a predict_image_bytes.
Devuelve JSON a la app movil.
```

`mobile_app/App.js`:

```text
Permite tomar foto o elegir de galeria.
Permite seleccionar pais o region.
Envia la imagen a la API.
Muestra decision, especie, confianza, region, modelo y top-5.
```

## 12. Decision final

La decision final del proyecto es:

```text
Usar B4 EfficientNetV2B1 como modelo principal.
Mantener B5 solo como candidato experimental.
Usar pais como apoyo opcional, no como verdad absoluta.
Mantener salida No concluyente cuando exista duda.
```

Esta decision es la mas segura para cliente porque evita promover un modelo que, aunque tenga mas accuracy, puede generar falsos seguros.

## 13. Limitaciones

Limitaciones actuales:

```text
1. Solo reconoce las 51 especies entrenadas.
2. Si la imagen es borrosa o parcial, puede responder No concluyente.
3. Si la especie no esta en el dataset, el modelo la forzara a una clase conocida.
4. El pais ayuda, pero no garantiza identificacion correcta.
5. No reemplaza expertos ni atencion medica.
```

## 14. Mejoras futuras

Mejoras recomendadas:

```text
1. Aumentar dataset con mas especies y mas imagenes por especie.
2. Crear una arquitectura multi-salida: especie + riesgo.
3. Calibrar probabilidades con temperature scaling.
4. Medir rendimiento por pais.
5. Agregar deteccion de "fuera de distribucion" para especies no conocidas.
6. Evaluar Grad-CAM para explicar que zona de la imagen usa el modelo.
```
