import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input


IMG_SIZE = (224, 224)
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena un clasificador de serpientes venenosas y no venenosas."
    )
    parser.add_argument("--data-dir", required=True, help="Carpeta con train, val y test.")
    parser.add_argument(
        "--output-dir",
        default="models/experiments/outputs",
        help="Carpeta de resultados.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--fine-tune-at", type=int, default=100)
    parser.add_argument(
        "--architecture",
        choices=["mobilenetv2", "efficientnetb0"],
        default="mobilenetv2",
        help="Arquitectura base de transfer learning.",
    )
    parser.add_argument("--critical-class", default="Venomous")
    parser.add_argument("--critical-class-weight-multiplier", type=float, default=1.0)
    return parser.parse_args()


def load_split(data_dir, split, batch_size, shuffle):
    return keras.utils.image_dataset_from_directory(
        Path(data_dir) / split,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle,
        seed=SEED,
    )


def load_datasets(data_dir, batch_size):
    train_ds = load_split(data_dir, "train", batch_size, shuffle=True)
    val_ds = load_split(data_dir, "val", batch_size, shuffle=False)
    test_ds = load_split(data_dir, "test", batch_size, shuffle=False)

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def create_base_model(architecture):
    if architecture == "mobilenetv2":
        return (
            MobileNetV2(
                input_shape=IMG_SIZE + (3,),
                include_top=False,
                weights="imagenet",
            ),
            mobilenet_preprocess_input,
        )

    if architecture == "efficientnetb0":
        return (
            EfficientNetB0(
                input_shape=IMG_SIZE + (3,),
                include_top=False,
                weights="imagenet",
            ),
            efficientnet_preprocess_input,
        )

    raise ValueError(f"Arquitectura no soportada: {architecture}")


def build_model(num_classes, learning_rate, architecture):
    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=SEED),
            keras.layers.RandomRotation(0.08, seed=SEED),
            keras.layers.RandomZoom(0.12, seed=SEED),
            keras.layers.RandomTranslation(0.08, 0.08, seed=SEED),
        ],
        name="augmentation",
    )

    base_model, preprocess_input = create_base_model(architecture)
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model, base_model


def normalize_class_name(class_name):
    return class_name.lower().replace("_", " ").strip()


def get_class_weights(data_dir, class_names, critical_class, critical_class_weight_multiplier):
    labels = []
    train_dir = Path(data_dir) / "train"
    for index, class_name in enumerate(class_names):
        class_dir = train_dir / class_name
        labels.extend([index] * len(list(class_dir.glob("*"))))

    if not labels:
        return None

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=np.array(labels),
    )
    class_weights = {index: float(weight) for index, weight in enumerate(weights)}

    critical_class = normalize_class_name(critical_class)
    for index, class_name in enumerate(class_names):
        if normalize_class_name(class_name) == critical_class:
            class_weights[index] *= critical_class_weight_multiplier

    return class_weights


def make_callbacks(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
        ),
    ]


def fine_tune(model, base_model, fine_tune_at, learning_rate):
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def collect_predictions(model, dataset):
    y_true = []
    y_pred = []
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    return np.array(y_true), np.array(y_pred)


def save_plots(history, output_dir):
    for metric in ["accuracy", "loss"]:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=160)
        plt.close()


def save_evaluation(model, test_ds, class_names, output_dir):
    y_true, y_pred = collect_predictions(model, test_ds)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    with open(output_dir / "confusion_matrix.json", "w", encoding="utf-8") as file:
        json.dump(matrix.tolist(), file, indent=2)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Prediccion")
    plt.ylabel("Etiqueta real")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)

    output_dir = Path(args.output_dir)
    train_ds, val_ds, test_ds, class_names = load_datasets(args.data_dir, args.batch_size)
    class_weights = get_class_weights(
        args.data_dir,
        class_names,
        args.critical_class,
        args.critical_class_weight_multiplier,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "data_dir": args.data_dir,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "fine_tune_epochs": args.fine_tune_epochs,
                "learning_rate": args.learning_rate,
                "fine_tune_learning_rate": args.fine_tune_learning_rate,
                "fine_tune_at": args.fine_tune_at,
                "architecture": args.architecture,
                "critical_class": args.critical_class,
                "critical_class_weight_multiplier": args.critical_class_weight_multiplier,
                "class_names": class_names,
                "class_weights": class_weights,
            },
            file,
            indent=2,
        )
    print("Class weights:", class_weights)
    print("Architecture:", args.architecture)

    model, base_model = build_model(len(class_names), args.learning_rate, args.architecture)
    callbacks = make_callbacks(output_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    if args.fine_tune_epochs > 0:
        fine_tune(model, base_model, args.fine_tune_at, args.fine_tune_learning_rate)
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
        )
        for key, values in fine_tune_history.history.items():
            history.history.setdefault(key, []).extend(values)

    model.save(output_dir / "final_model.keras")
    with open(output_dir / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    save_plots(history, output_dir)
    save_evaluation(model, test_ds, class_names, output_dir)


if __name__ == "__main__":
    main()
