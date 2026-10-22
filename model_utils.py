# Обучение и использование модели оценки стоимости недвижимости

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "realty_data.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"
UNKNOWN_VALUE = "Не указано"

NUMERIC_FEATURES = ["total_square", "rooms", "floor", "lat", "lon"]
CATEGORICAL_FEATURES = ["city", "settlement", "district", "area"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "price"


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Приводит признаки к одному формату для обучения и предсказания."""
    data = frame.copy()

    for column in FEATURE_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    data = data[FEATURE_COLUMNS].copy()

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        values = data[column].fillna(UNKNOWN_VALUE).astype(str).str.strip()
        data[column] = values.replace("", UNKNOWN_VALUE)

    return data


def load_dataset(data_path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Читает CSV и удаляет строки с невалидной ценой или площадью."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Не найден файл с данными: {data_path}. "
            "Положите realty_data.csv рядом."
        )

    frame = pd.read_csv(data_path, low_memory=False)
    required = {TARGET_COLUMN, "total_square"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "В CSV отсутствуют обязательные столбцы: " + ", ".join(sorted(missing))
        )

    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame["total_square"] = pd.to_numeric(frame["total_square"], errors="coerce")
    valid = (
        frame[TARGET_COLUMN].notna()
        & np.isfinite(frame[TARGET_COLUMN])
        & (frame[TARGET_COLUMN] > 0)
        & frame["total_square"].notna()
        & np.isfinite(frame["total_square"])
        & (frame["total_square"] > 0)
    )
    frame = frame.loc[valid].copy()

    X = prepare_features(frame)
    y = frame[TARGET_COLUMN].astype(float)
    return X, y


def create_pipeline() -> TransformedTargetRegressor:
    # Создаёт pipeline с обработкой пропусков, категорий и логарифмом цены
    size_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "log",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=UNKNOWN_VALUE)),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("size", size_pipeline, ["total_square"]),
            ("numeric", numeric_pipeline, ["rooms", "floor", "lat", "lon"]),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    regressor = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=20.0, solver="lsqr")),
        ]
    )

    # Логарифмирование делает модель менее чувствительной к очень дорогим объектам
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def train_and_save(
    data_path: Path | str = DATA_PATH,
    model_path: Path | str = MODEL_PATH,
    metrics_path: Path | str = METRICS_PATH,
) -> dict[str, Any]:
    # Оценивает качество на holdout-выборке, обучает итоговую модель и сохраняет её
    X, y = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    validation_model = create_pipeline()
    validation_model.fit(X_train, y_train)
    predictions = np.maximum(validation_model.predict(X_test), 0)

    metrics: dict[str, Any] = {
        "rows_used": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae_rub": float(mean_absolute_error(y_test, predictions)),
        "rmse_rub": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "test_size": 0.2,
        "random_state": 42,
    }

    # После оценки переобучаем модель на всех доступных корректных строках
    final_model = create_pipeline()
    final_model.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metrics


def load_or_train_model() -> Any:
    # Загружает сохранённую модель или обучает её, если файл ещё не создан
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)


def predict_price(features: dict[str, Any] | pd.DataFrame, model: Any | None = None) -> float:
    """Возвращает прогноз цены в рублях для одного объекта недвижимости."""
    if isinstance(features, dict):
        input_frame = pd.DataFrame([features])
    else:
        input_frame = features.copy()

    input_frame = prepare_features(input_frame)
    active_model = model if model is not None else load_or_train_model()
    prediction = float(active_model.predict(input_frame)[0])
    return max(prediction, 0.0)


def read_metrics(metrics_path: Path | str = METRICS_PATH) -> dict[str, Any] | None:
    # Загружает метрики обучения, если они уже рассчитаны
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
