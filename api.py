# FastAPI API для прогноза стоимости недвижимости

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from model_utils import load_or_train_model, predict_price

app = FastAPI(
    title="Realty Price Prediction API",
    description=(
        "API прогнозирования стоимости недвижимости. "
    ),
    version="1.0.0",
)


class RealtyFeatures(BaseModel):
    total_square: float = Field(gt=0, description="Общая площадь объекта в м²")
    rooms: float | None = Field(default=None, ge=0, description="Количество комнат")
    floor: float | None = Field(default=None, ge=0, description="Этаж")
    lat: float | None = Field(default=None, ge=-90, le=90, description="Широта")
    lon: float | None = Field(default=None, ge=-180, le=180, description="Долгота")
    city: str | None = Field(default="Москва", description="Город")
    settlement: str | None = Field(default=None, description="Населённый пункт")
    district: str | None = Field(default=None, description="Район")
    area: str | None = Field(default=None, description="Округ или территория")


@lru_cache(maxsize=1)
def get_model() -> Any:
    # Загружает уже обученную модель или обучает её при первом прогнозе
    return load_or_train_model()


def make_prediction(features: dict[str, Any]) -> dict[str, Any]:
    try:
        price = predict_price(features, model=get_model())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось получить прогноз: {exc}") from exc

    return {"predicted_price": round(price, 2), "currency": "RUB"}


@app.get("/health", tags=["Health"], summary="Liveness-проба")
def health() -> dict[str, str]:
    # Проверяет, что API-процесс отвечает
    return {"status": "ok"}


@app.get("/predict_get", tags=["Prediction"], summary="Прогноз стоимости через GET")
def predict_get(
    total_square: float = Query(gt=0, description="Общая площадь объекта в м²"),
    rooms: float | None = None,
    floor: float | None = None,
    lat: float | None = None,
    lon: float | None = None,
    city: str | None = "Москва",
    settlement: str | None = None,
    district: str | None = None,
    area: str | None = None,
) -> dict[str, Any]:
    # Принимает признаки объекта в query-параметрах и возвращает прогноз в рублях
    return make_prediction({
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        "lat": lat,
        "lon": lon,
        "city": city,
        "settlement": settlement,
        "district": district,
        "area": area,
    })


@app.post("/predict_post", tags=["Prediction"], summary="Прогноз стоимости через POST")
def predict_post(features: RealtyFeatures) -> dict[str, Any]:
    # Принимает признаки объекта в JSON запросе и возвращает прогноз в рублях
    return make_prediction(features.model_dump())
