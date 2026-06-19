# Informe tecnico del proyecto

## 1. Objetivo general

El objetivo del proyecto es construir un sistema de vision por computador capaz de analizar una imagen de una serpiente, proponer la especie mas probable y entregar una clasificacion de riesgo para el usuario: `Venenosa`, `No venenosa` o `No concluyente`.

La idea central no fue dejar el sistema como una clasificacion binaria simple. En lugar de entrenar solamente un modelo que diga `venenosa` o `no venenosa`, el proyecto evoluciono hacia una clasificacion por especie. Esa decision permite una respuesta mas completa y mas explicable.

Flujo final:

```text
imagen del usuario
-> clasificacion por especie
-> consulta de metadata biologica
-> clasificacion de riesgo
-> ajuste opcional con pais o region
-> verificacion preventiva con modelo binario
-> decision final segura
```

La salida final no pretende reemplazar a un experto. La aplicacion esta disenada para orientar y prevenir. Por eso, si el sistema encuentra duda o conflicto entre senales, responde `No concluyente` en vez de dar una falsa seguridad.

## 2. Problema trabajado

Identificar serpientes visualmente es un problema dificil porque muchas especies se parecen entre si. Algunas especies no venenosas tienen patrones parecidos a especies venenosas, y algunas fotos tomadas por usuarios pueden tener problemas de calidad: baja resolucion, mala iluminacion, serpiente parcialmente oculta, fondo complejo o angulos poco claros.

Por esa razon, el proyecto no se evaluo solamente con accuracy. El criterio mas importante fue la seguridad. El error mas peligroso es:

```text
serpiente venenosa -> sistema dice no venenosa
```

Ese tipo de error se llama en el proyecto `falso seguro`. Aunque un modelo tenga mejor exactitud general, no debe ser promovido si aumenta los falsos seguros.

## 3. Cambio de clasificacion binaria a clasificacion por especie

### 3.1 Clasificacion binaria inicial

La clasificacion binaria usa dos clases:

```text
Venomous
Non Venomous
```

Ventaja:

```text
Es mas simple de entrenar y de explicar.
```

Problema:

```text
No indica que especie cree estar viendo.
No permite revisar top-5 de especies.
No permite usar informacion geografica por especie.
No ayuda a entender los errores biologicos del modelo.
```

### 3.2 Clasificacion por especie

La clasificacion por especie convierte el problema en multiclase. En el modelo activo hay 51 salidas, una por especie.

Ejemplo:

```text
018_Agkistrodon_contortrix
065_Bogertophis_subocularis
185_Crotalus_ruber
...
```

Ventajas:

```text
Permite mostrar nombre comun y nombre cientifico.
Permite convertir especie a riesgo usando metadata.
Permite revisar si una especie venenosa aparece en el top-5.
Permite usar pais o region como apoyo.
Permite hacer analisis de errores por especie.
```

Esta fue la ruta recomendada para continuar el proyecto porque produce una salida mas informativa y permite tomar decisiones de seguridad mas controladas.

## 4. Dataset y organizacion de datos

El dataset activo se conserva en:

```text
data/active/Snake Species Pilot 51 Targeted B4 Scale
```

Esta carpeta tiene tres divisiones:

```text
train/
val/
test/
```

Definicion de cada division:

```text
train:
    Conjunto usado para que el modelo aprenda. Las imagenes de train modifican los pesos de la red neuronal.

val:
    Conjunto usado durante el entrenamiento para revisar si el modelo esta generalizando. No entrena directamente, pero ayuda a decidir cuando guardar el mejor modelo.

test:
    Conjunto reservado para evaluar el resultado final. No debe usarse para ajustar decisiones durante el entrenamiento.
```

Conteo del dataset B4 Scale:

```text
train: 11104 imagenes
val: 1932 imagenes
test: 1252 imagenes
clases: 51 especies
```

Cada clase corresponde a una carpeta de especie. El nombre de la carpeta conserva un identificador numerico y el nombre cientifico:

```text
018_Agkistrodon_contortrix
065_Bogertophis_subocularis
```

Esto permite que Keras lea automaticamente las clases con `image_dataset_from_directory`.

## 5. Metadata usada por el sistema

La red neuronal predice una clase. Pero para mostrar informacion util, el sistema necesita metadata.

Archivos principales:

```text
data/metadata/species_pilot51_b4_scale_metadata.csv
data/metadata/species_pilot51_b4_scale_manifest.csv
```

`species_pilot51_b4_scale_metadata.csv` contiene informacion por especie:

```text
class_folder:
    Nombre de la carpeta/clase usada por el modelo.

scientific_name:
    Nombre cientifico de la especie.

common_name:
    Nombre comun cuando esta disponible.

venom_status:
    Clasificacion de riesgo: Venomous o Non Venomous.

countries / continents:
    Distribucion geografica asociada a la especie.
```

`species_pilot51_b4_scale_manifest.csv` contiene informacion por imagen:

```text
split:
    Indica si la imagen pertenece a train, val, test u holdout.

class_folder:
    Clase real de la imagen.

country / continent:
    Informacion geografica usada para calcular priors.

source:
    Fuente o procedencia de la imagen cuando esta disponible.
```

La metadata es clave porque el modelo no aprende directamente conceptos como `nombre comun` o `pais`. El modelo solo devuelve probabilidades por clase; el resto se interpreta con estos archivos.

## 6. Evolucion de los modelos

### 6.1 Modelos iniciales

Al inicio se probaron modelos mas simples y pilotos con menos especies. Estos experimentos validaron el flujo, pero no fueron suficientes para el modelo final.

Se probaron:

```text
MobileNetV2:
    Red liviana, util como primer baseline.

EfficientNetV2B0:
    Red mas fuerte que MobileNetV2, usada como primer modelo fuerte por especie.
```

### 6.2 B0

B0 uso EfficientNetV2B0 con imagenes de 224x224.

Resultados:

```text
accuracy: 0.58546
macro_f1: 0.57688
top3_accuracy: 0.82428
top5_accuracy: 0.89137
```

Interpretacion:

```text
El modelo acertaba la especie top-1 en cerca del 58.5% de los casos.
La especie correcta aparecia dentro del top-5 en cerca del 89.1% de los casos.
```

B0 fue importante porque mostro que el top-5 era util para seguridad. Aunque el top-1 fallara, muchas veces la especie correcta aparecia entre las primeras candidatas.

### 6.3 B1

B1 uso EfficientNetV2B1 con imagenes de 260x260.

Resultados:

```text
accuracy: 0.62620
macro_f1: 0.62802
top3_accuracy: 0.83866
top5_accuracy: 0.90096
```

Mejora frente a B0:

```text
Subio accuracy.
Subio macro F1.
Mantuvo buen top-5.
```

En B1 se probo fine-tuning, pero no supero al modelo entrenado solo en la cabeza. Por eso se conservo el mejor modelo de cabeza.

### 6.4 B2 y B3

B2 y B3 fueron intentos de mejorar agregando imagenes de especies especificas que el modelo confundia.

Conclusion:

```text
B2:
    No reemplazo al modelo principal porque bajo metricas internas.

B3:
    Mejoro algunos casos externos, pero no fue suficientemente estable.
```

Aprendizaje:

```text
No era suficiente mejorar unos pocos casos manualmente.
Hacia falta una mejora mas general del dataset.
```

### 6.5 B4 Scale

B4 fue la mejora mas importante. Se amplio el dataset de entrenamiento de forma mas general y se mantuvieron splits de validacion y prueba para comparar.

Modelo:

```text
EfficientNetV2B1
imagen 260x260
51 especies
```

Resultados:

```text
accuracy: 0.63898
macro_f1: 0.63459
weighted_f1: 0.63978
top3_accuracy: 0.85783
top5_accuracy: 0.91374
support: 1252 imagenes
```

Interpretacion:

```text
accuracy:
    El modelo acierta la especie top-1 en cerca del 63.9% del test.

top5_accuracy:
    La especie correcta aparece dentro de las 5 primeras predicciones en cerca del 91.4%.

macro_f1:
    Mide rendimiento promedio por especie sin favorecer especies con mas imagenes.
```

B4 quedo como modelo principal porque tuvo el mejor equilibrio entre rendimiento y seguridad.

### 6.6 B5

B5 uso una arquitectura mas grande:

```text
EfficientNetV2B3
imagen 300x300
dataset B4 Scale
```

Resultados:

```text
accuracy: 0.64696
top3_accuracy: 0.86342
top5_accuracy: 0.90974
```

A primera vista B5 parece mejor porque tiene mayor accuracy. Sin embargo, en la prueba grande de seguridad produjo falsos seguros. Es decir, algunas serpientes venenosas podian terminar clasificadas como no venenosas.

Decision:

```text
B5 queda como candidato experimental.
B4 queda como modelo activo para cliente.
```

## 7. Arquitectura usada

La arquitectura activa es una red neuronal convolucional basada en EfficientNetV2B1.

Resumen de componentes:

```text
Entrada: imagen RGB
Tamano: 260x260
Base convolucional: EfficientNetV2B1 sin capa superior
Pooling: GlobalAveragePooling2D
Regularizacion: Dropout
Capa final: Dense con softmax
Salida: 51 probabilidades
```

Definicion de cada componente:

```text
Entrada: imagen RGB
    La imagen se representa con tres canales de color: rojo, verde y azul. Cada pixel tiene informacion de color en esos tres canales. Este formato es el estandar para modelos de vision.

Tamano: 260x260
    Todas las imagenes se redimensionan a 260 pixeles de ancho por 260 pixeles de alto. La red necesita que todas las entradas tengan el mismo tamano para procesarlas en lotes.

Base convolucional: EfficientNetV2B1 sin capa superior
    Es la parte principal de la red que extrae caracteristicas visuales. Detecta patrones como bordes, colores, textura, escamas, forma del cuerpo y combinaciones complejas de rasgos. "Sin capa superior" significa que se quita la clasificacion original de ImageNet para reemplazarla por una cabeza nueva adaptada a serpientes.

Pooling: GlobalAveragePooling2D
    Convierte los mapas de caracteristicas de la red en un vector compacto. En lugar de conservar toda la ubicacion espacial, resume cada mapa promediando sus valores. Esto reduce parametros y ayuda a conectar la base convolucional con la capa de clasificacion.

Regularizacion: Dropout
    Durante entrenamiento apaga aleatoriamente una parte de las neuronas. Esto obliga al modelo a no depender demasiado de una sola senal visual y ayuda a reducir sobreajuste.

Capa final: Dense con softmax
    Es una capa totalmente conectada que produce una puntuacion por cada especie. Softmax convierte esas puntuaciones en probabilidades que suman 1.

Salida: 51 probabilidades
    El modelo devuelve una probabilidad para cada una de las 51 especies. La especie con mayor probabilidad es el top-1, pero tambien se revisa top-3 y top-5.
```

Forma conceptual:

```text
imagen 260x260x3
-> EfficientNetV2B1
-> vector de caracteristicas
-> dropout
-> dense softmax
-> [p1, p2, p3, ..., p51]
```

## 8. Por que EfficientNetV2B1

EfficientNetV2B1 fue elegida porque ofrece buen balance entre precision y costo computacional.

Comparacion conceptual:

```text
MobileNetV2:
    Mas liviana, pero menor capacidad para distinguir especies visualmente parecidas.

EfficientNetV2B0:
    Mejor que MobileNetV2, pero con rendimiento limitado en este dataset.

EfficientNetV2B1:
    Mejor equilibrio entre capacidad, tamano de imagen y tiempo de entrenamiento.

EfficientNetV2B3:
    Mas grande y con mayor capacidad, pero en este proyecto no fue mas segura.
```

La decision final no fue escoger el modelo mas grande, sino el que produjo el perfil mas seguro.

## 9. Transfer learning

El proyecto usa transferencia de aprendizaje.

Definicion:

```text
Transfer learning consiste en usar una red ya entrenada en un dataset grande y adaptarla a una tarea nueva.
```

En este caso:

```text
EfficientNetV2B1 viene preentrenada en ImageNet.
ImageNet tiene millones de imagenes generales.
La red ya aprendio patrones visuales basicos y complejos.
El proyecto reutiliza esas caracteristicas para clasificar serpientes.
```

Proceso usado:

```text
1. Cargar EfficientNetV2B1 sin la cabeza original.
2. Congelar la base preentrenada.
3. Agregar una cabeza nueva para 51 especies.
4. Entrenar la cabeza con el dataset de serpientes.
5. Evaluar en validacion.
6. Probar fine-tuning suave.
7. Promover solo el modelo que mejora sin afectar seguridad.
```

Entrenar la cabeza significa que solo aprenden las capas nuevas agregadas al final. La base EfficientNet queda congelada y actua como extractor de caracteristicas.

Fine-tuning significa descongelar parte de la base y ajustar sus pesos con un learning rate muy bajo. Es una tecnica util, pero puede empeorar si el dataset no es suficientemente grande o si el modelo se sobreajusta. Por eso no se promovio automaticamente.

## 10. Aumento de datos

Durante el entrenamiento se aplican transformaciones aleatorias a las imagenes.

Ejemplos:

```text
RandomFlip:
    Invierte horizontalmente algunas imagenes. Ayuda porque una serpiente puede aparecer mirando a cualquier lado.

RandomRotation:
    Rota ligeramente la imagen. Ayuda a que el modelo no dependa de una orientacion fija.

RandomZoom:
    Acerca o aleja la imagen. Ayuda cuando la serpiente aparece mas cerca o mas lejos.

RandomContrast:
    Cambia el contraste. Ayuda ante variaciones de luz, sombras o camaras diferentes.
```

Objetivo:

```text
Hacer que el modelo generalice mejor y no memorice imagenes exactas.
```

## 11. Entrenamiento

El entrenamiento se realizo principalmente en Google Colab con GPU.

Por que Colab:

```text
Entrenar redes convolucionales con miles de imagenes es lento en CPU.
La GPU acelera las operaciones matriciales y convolucionales.
Colab permite usar GPU sin depender del computador local.
```

Datos usados durante entrenamiento:

```text
train:
    El modelo aprende de estas imagenes. Aqui se actualizan los pesos de la red.

val:
    Se usa para monitorear rendimiento mientras entrena. Si val_loss mejora, se guarda el modelo.

test:
    Se usa al final para medir rendimiento real. No se debe usar para tomar decisiones durante entrenamiento.
```

Callbacks usados:

```text
ModelCheckpoint:
    Guarda el mejor modelo durante entrenamiento. En este proyecto se monitorea val_loss, porque interesa que el modelo generalice y no solo memorice train.

EarlyStopping:
    Detiene el entrenamiento si el modelo deja de mejorar. Evita gastar tiempo y reduce sobreajuste.

ReduceLROnPlateau:
    Baja el learning rate cuando la validacion se estanca. Esto permite ajustes mas finos al final del entrenamiento.
```

Optimizador:

```text
AdamW:
    Variante de Adam que agrega weight decay. Ayuda a regularizar los pesos y puede mejorar generalizacion.
```

Funcion de perdida:

```text
CategoricalCrossentropy:
    Se usa cuando hay varias clases y las etiquetas estan en formato one-hot.

Label smoothing:
    Evita que el modelo sea excesivamente confiado. En vez de entrenar con 100% para la clase correcta, suaviza un poco la etiqueta.
```

## 12. Metricas usadas

Metricas principales:

```text
accuracy:
    Porcentaje de imagenes donde la especie top-1 fue correcta.

top3_accuracy:
    Porcentaje de imagenes donde la especie correcta aparece dentro de las 3 primeras predicciones.

top5_accuracy:
    Porcentaje de imagenes donde la especie correcta aparece dentro de las 5 primeras predicciones.

macro_f1:
    Promedio del F1 de todas las especies, dando el mismo peso a cada especie. Es importante cuando hay clases con diferente cantidad de imagenes.

weighted_f1:
    Promedio F1 ponderado por cantidad de imagenes por especie.

loss:
    Error numerico que optimiza el modelo. Menor loss normalmente significa mejor ajuste, pero debe revisarse junto con validacion.
```

Por que top-5 es importante:

```text
En identificacion de especies, el top-1 puede fallar entre especies parecidas.
Si la especie correcta aparece en top-5, la app puede mostrar alternativas y detectar riesgo.
El top-5 ayuda a evitar falsos seguros cuando una especie venenosa aparece como candidata relevante.
```

## 13. Clasificacion por especie

La prediccion por especie funciona asi:

```text
1. Se carga la imagen.
2. Se convierte a RGB.
3. Se redimensiona a 260x260.
4. Se transforma en arreglo numerico.
5. El modelo devuelve 51 probabilidades.
6. Se ordenan de mayor a menor.
7. Se selecciona top-1 y se conserva top-5.
```

Ejemplo conceptual:

```text
Agkistrodon contortrix: 0.62
Lampropeltis triangulum: 0.11
Crotalus ruber: 0.07
Bogertophis subocularis: 0.04
Natrix maura: 0.03
```

Interpretacion:

```text
Top-1:
    Agkistrodon contortrix.

Top-5:
    Las cinco especies mas probables segun el modelo.

Riesgo:
    Se consulta en metadata usando la especie seleccionada.
```

## 14. Clasificacion venenosa/no venenosa

La clasificacion de riesgo no depende solamente de una red binaria. Se obtiene principalmente desde la especie.

Proceso:

```text
1. El modelo predice especie.
2. Se busca la especie en metadata.
3. La metadata indica si esa especie es Venomous o Non Venomous.
4. Se consulta el top-5 para revisar si hay especies venenosas relevantes.
5. Se consulta el modelo binario preventivo como respaldo.
6. Se aplica la regla integrada.
```

Esto es mas explicable que una salida binaria directa porque se puede justificar la decision:

```text
"El modelo predijo esta especie, esta especie esta marcada como venenosa/no venenosa, y estas fueron las candidatas cercanas."
```

## 15. Modelo binario preventivo

El modelo binario se conserva como respaldo:

```text
models/experiments/outputs_targeted_efficientnetb0_lowlr/best_model.keras
```

Su funcion no es reemplazar al modelo por especie. Su funcion es alertar cuando hay senales visuales de riesgo.

Ejemplo:

```text
Modelo por especie:
    Top-1 no venenosa con confianza media.

Modelo binario:
    Probabilidad alta de venenosa.

Decision final:
    No concluyente.
```

Esto reduce el riesgo de decir `No venenosa` cuando el sistema no tiene evidencia suficiente.

## 16. Regla integrada de seguridad

La regla integrada combina:

```text
riesgo de especie seleccionada
confianza de especie
top-5 de especies candidatas
probabilidad del modelo binario
pais o region si existe
```

Regla conceptual:

```text
Si la especie seleccionada es Venomous:
    responder Venenosa.

Si la especie seleccionada es Non Venomous pero aparece una Venomous relevante en top-5:
    responder No concluyente.

Si el binario alerta Venomous y la especie no venenosa no es suficientemente fuerte:
    responder No concluyente.

Si la confianza de especie es baja:
    responder No concluyente.

Si todo coincide en bajo riesgo:
    responder No venenosa.
```

Umbrales activos:

```text
SPECIES_CONFIDENCE_FLOOR = 0.35
    Confianza minima para aceptar una especie. Si baja de este valor, se considera insegura.

NON_VENOMOUS_SAFE_CONFIDENCE = 0.50
    Confianza minima para permitir una respuesta no venenosa cuando hay conflicto con el binario.

NON_VENOMOUS_TOP5_MIN = 4
    Cantidad minima de candidatas no venenosas en top-5 para reforzar una decision no venenosa.

NON_VENOMOUS_BINARY_ALERT_MAX = 0.45
    Limite de alerta binaria venenosa para permitir bajo riesgo.

NON_VENOMOUS_TOP5_VENOMOUS_ALERT = 0.07
    Si una especie venenosa aparece en top-5 con confianza igual o superior a este valor, se bloquea la respuesta no venenosa.
```

Razon de esta regla:

```text
La salida No venenosa solo debe aparecer cuando varias evidencias coinciden.
Si hay duda razonable, la salida correcta para seguridad es No concluyente.
```

## 17. Uso de pais o region

El pais o region es opcional. El sistema puede funcionar sin pais.

Cuando el usuario indica pais, el sistema no filtra de forma dura. Usa un prior geografico suave.

Parametros:

```text
GEO_PRIOR_ALPHA = 0.75
GEO_PRIOR_TOP_N = 10
```

Definiciones:

```text
GEO_PRIOR_TOP_N:
    Solo se consideran las 10 especies visualmente mas probables. Esto evita que el pais fuerce una especie visualmente mala.

GEO_PRIOR_ALPHA:
    Controla que tanto peso tiene la informacion geografica frente a la probabilidad visual.
```

Funcionamiento:

```text
1. El modelo predice probabilidades por imagen.
2. Se toman las 10 especies mas probables visualmente.
3. Se revisa la frecuencia geografica de esas especies en el manifest.
4. Se reordena suavemente la seleccion.
5. Se conserva la regla de seguridad top-5.
```

Por que no se usa filtro duro:

```text
El usuario puede equivocarse de pais.
La imagen puede venir de internet o de otro lugar.
La distribucion geografica puede estar incompleta.
Un filtro duro podria descartar la especie correcta.
```

Por eso el pais ayuda, pero no manda solo.

## 18. Funcionamiento del codigo

### 18.1 app_streamlit.py

`app_streamlit.py` implementa la app web local.

Responsabilidades:

```text
1. Mostrar interfaz de carga de imagen.
2. Permitir usar pais o region.
3. Cargar modelo B4 y modelo binario.
4. Ejecutar inferencia por especie.
5. Aplicar prior geografico si corresponde.
6. Ejecutar respaldo binario.
7. Aplicar regla integrada.
8. Mostrar resultado y top-5.
```

Es la app recomendada para pruebas, sustentacion y demostracion rapida.

### 18.2 src/mobile_inference.py

`src/mobile_inference.py` contiene la logica central para la app movil.

Funciones importantes:

```text
ensure_assets_exist:
    Verifica que existan modelos, clases y metadata.

predict_species:
    Ejecuta el modelo por especie y arma top-5.

predict_binary:
    Ejecuta el modelo binario preventivo.

integrated_decision:
    Aplica la politica de seguridad.

predict_image_bytes:
    Funcion principal que recibe bytes de imagen y devuelve el resultado completo.
```

### 18.3 api_mobile.py

`api_mobile.py` expone el backend con FastAPI.

Endpoints:

```text
GET /health:
    Confirma que la API esta activa.

POST /predict:
    Recibe imagen y pais opcional.
    Devuelve decision, especie, top-5, binario y nota de seguridad.
```

La app movil no carga el modelo directamente; envia la imagen al backend.

### 18.4 mobile_app/App.js

`mobile_app/App.js` implementa la interfaz movil en Expo/React Native.

Funciones de usuario:

```text
Tomar foto.
Elegir imagen de galeria.
Seleccionar pais o region.
Enviar imagen a la API.
Mostrar resultado.
Mostrar top-5.
Mostrar nota de seguridad.
```

La app movil muestra tambien que la politica activa es:

```text
B4 safety-first
```

## 19. Decision final del proyecto

Decision:

```text
Modelo principal: B4 EfficientNetV2B1
Modelo experimental: B5 EfficientNetV2B3 300
Politica activa: B4 safety-first
```

Razon:

```text
B4 ofrece mejor perfil de seguridad.
B5 tiene mayor accuracy, pero produjo falsos seguros en pruebas completas.
Para cliente, seguridad pesa mas que accuracy.
```

El sistema final se comporta de forma conservadora:

```text
Si esta seguro de venenosa -> Venenosa.
Si esta seguro de no venenosa -> No venenosa.
Si hay duda -> No concluyente.
```

## 20. Limitaciones actuales

Limitaciones:

```text
1. Solo reconoce especies dentro de las 51 clases entrenadas.
2. Puede confundirse con especies no incluidas en el dataset.
3. Imagenes borrosas o parciales reducen la confianza.
4. El pais ayuda, pero no garantiza identificacion correcta.
5. La metadata geografica puede estar incompleta.
6. El sistema no reemplaza expertos ni atencion medica.
```

La salida `No concluyente` es una respuesta esperada en casos de baja calidad o conflicto. No significa que la app fallo; significa que la politica de seguridad evito una afirmacion riesgosa.

## 21. Mejoras futuras

Mejoras recomendadas:

```text
1. Ampliar dataset con mas especies y mas paises.
2. Aumentar imagenes de especies venenosas que se parecen a no venenosas.
3. Entrenar un modelo multi-salida: especie + riesgo.
4. Calibrar probabilidades con temperature scaling.
5. Agregar deteccion de imagen fuera de distribucion.
6. Evaluar Grad-CAM para explicar visualmente que zonas usa el modelo.
7. Construir una prueba externa permanente por pais.
8. Usar Git LFS o almacenamiento externo para modelos grandes si el proyecto crece.
```

## 22. Conclusiones

El proyecto avanzo desde una clasificacion binaria inicial hacia un sistema mas completo por especie. La arquitectura final usa EfficientNetV2B1 con transferencia de aprendizaje, metadata biologica, prior geografico suave y una regla de seguridad que evita falsos seguros.

La decision de conservar B4 como modelo principal no se baso solo en accuracy. Se baso en el criterio mas importante del proyecto: proteger al usuario cuando existe posibilidad de riesgo.
