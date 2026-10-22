# Запуск обучения модели и вывод метрик на тестовой выборке

from model_utils import train_and_save


def main() -> None:
    metrics = train_and_save()

    print("Модель обучена и сохранена в model.joblib")
    print(f"Использовано объектов: {metrics['rows_used']:,}".replace(",", " "))
    print(f"Строк в обучающей выборке: {metrics['train_rows']:,}".replace(",", " "))
    print(f"Строк в тестовой выборке: {metrics['test_rows']:,}".replace(",", " "))
    print(f"MAE: {metrics['mae_rub']:,.0f} руб.".replace(",", " "))
    print(f"RMSE: {metrics['rmse_rub']:,.0f} руб.".replace(",", " "))
    print(f"R²: {metrics['r2']:.3f}")
    print("Метрики сохранены в metrics.json")


if __name__ == "__main__":
    main()
