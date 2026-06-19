import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras


IMG_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description="Predice la clase de una imagen de serpiente.")
    parser.add_argument("--model", required=True, help="Ruta del modelo .keras.")
    parser.add_argument("--classes", required=True, help="Ruta de class_names.json.")
    parser.add_argument("--image", required=True, help="Ruta de la imagen.")
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--positive-class", default="Venomous")
    parser.add_argument("--positive-threshold", type=float, default=0.50)
    return parser.parse_args()


def load_image(image_path):
    image = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    array = keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def main():
    args = parse_args()
    model = keras.models.load_model(args.model)

    with open(args.classes, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    predictions = model.predict(load_image(args.image), verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    positive_index = [name.lower().strip() for name in class_names].index(
        args.positive_class.lower().strip()
    )
    positive_probability = float(predictions[positive_index])

    print(f"Clase predicha: {class_names[predicted_index]}")
    print(f"Confianza: {confidence:.2%}")
    print(f"Probabilidad {args.positive_class}: {positive_probability:.2%}")

    if positive_probability >= args.positive_threshold:
        print(f"Resultado final: Posible {args.positive_class}")
        print("Recomendacion: tratar como riesgo hasta revision experta.")
    elif confidence < args.confidence_threshold:
        print("Resultado final: No concluyente")
        print("Recomendacion: no descartar riesgo venenoso sin revision experta.")
    else:
        print(f"Resultado final: {class_names[predicted_index]}")


if __name__ == "__main__":
    main()
