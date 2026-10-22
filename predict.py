# Пример получения прогноза цены

from model_utils import load_or_train_model, predict_price


def main() -> None:
    model = load_or_train_model()

    example = {
        "total_square": 60.0,
        "rooms": 2,
        "floor": 5,
        "lat": 55.7455,
        "lon": 37.5694,
        "city": "Москва",
        "settlement": "Не указано",
        "district": "Раменки район",
        "area": "Не указано",
    }

    price = predict_price(example, model=model)
    print(f"Прогнозируемая стоимость: {price:,.0f} руб.".replace(",", " "))


if __name__ == "__main__":
    main()
