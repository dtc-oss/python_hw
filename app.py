# Streamlit-интерфейс для прогнозирования стоимости недвижимости

from __future__ import annotations

import pandas as pd
import streamlit as st

from model_utils import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    UNKNOWN_VALUE,
    load_or_train_model,
    predict_price,
    read_metrics,
)

st.set_page_config(
    page_title="Оценка стоимости недвижимости",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def get_model():
    # При первом запуске модель автоматически обучается и сохраняется на диск.
    return load_or_train_model()


@st.cache_data
def get_category_options() -> dict[str, list[str]]:
    # Получает варианты выбора из нужных столбцов CSV, не загружая описания
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден файл с данными: {DATA_PATH}")

    raw = pd.read_csv(DATA_PATH, usecols=CATEGORICAL_FEATURES, low_memory=False)
    options_by_feature: dict[str, list[str]] = {}

    for column in CATEGORICAL_FEATURES:
        values = (
            raw[column]
            .fillna(UNKNOWN_VALUE)
            .astype(str)
            .str.strip()
            .replace("", UNKNOWN_VALUE)
        )
        popular_values = values.value_counts().index.tolist()
        options_by_feature[column] = [UNKNOWN_VALUE] + [
            value for value in popular_values if value != UNKNOWN_VALUE
        ]

    return options_by_feature


st.title("Прогноз стоимости недвижимости")
st.write(
    "Укажите характеристики объекта. "
)
st.caption(
    "Датасет realty_data.csv. "
)

if not DATA_PATH.exists():
    st.error(
        "Не найден realty_data.csv. Поместите его рядом, в ту же папку, "
    )
    st.stop()

try:
    category_options = get_category_options()
    with st.spinner("Загрузка модели (при первом запуске она будет обучена)..."):
        model = get_model()
except Exception as error:
    st.error(f"Не удалось подготовить модель: {error}")
    st.stop()

metrics = read_metrics()
if metrics:
    mae_text = f"{metrics['mae_rub']:,.0f}".replace(",", " ")
    r2_text = f"{metrics['r2']:.3f}"
    st.caption(
        f"Контрольное качество модели: MAE = {mae_text} руб.; R² = {r2_text}. "
        "Метрики рассчитаны на отложенных 20% данных."
    )

st.subheader("Характеристики объекта")

with st.form("property_form"):
    left, right = st.columns(2)

    with left:
        total_square = st.number_input(
            "Общая площадь, м²",
            min_value=1.0,
            max_value=3000.0,
            value=60.0,
            step=1.0,
            help="Площадь квартиры или другого объекта недвижимости.",
        )
        rooms_choice = st.selectbox(
            "Количество комнат",
            options=[0] + list(range(1, 16)),
            index=2,
            format_func=lambda value: "Не указано" if value == 0 else str(value),
        )
        floor = st.number_input(
            "Этаж",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
        )
        city_options = category_options["city"]
        city_default = city_options.index("Москва") if "Москва" in city_options else 0
        city = st.selectbox("Город", options=city_options, index=city_default)
        district = st.selectbox(
            "Район",
            options=category_options["district"],
            index=0,
        )

    with right:
        lat = st.number_input(
            "Широта",
            min_value=50.0,
            max_value=60.0,
            value=55.7455,
            step=0.001,
            format="%.6f",
            help="Координата объекта.",
        )
        lon = st.number_input(
            "Долгота",
            min_value=30.0,
            max_value=45.0,
            value=37.5694,
            step=0.001,
            format="%.6f",
            help="Координата объекта.",
        )
        settlement = st.selectbox(
            "Населённый пункт",
            options=category_options["settlement"],
            index=0,
        )
        area = st.selectbox(
            "Микрорайон / жилой комплекс",
            options=category_options["area"],
            index=0,
        )

    submitted = st.form_submit_button(
        "Рассчитать стоимость",
        type="primary",
        use_container_width=True,
    )

if submitted:
    property_features = {
        "total_square": total_square,
        "rooms": None if rooms_choice == 0 else rooms_choice,
        "floor": floor,
        "lat": lat,
        "lon": lon,
        "city": city,
        "settlement": settlement,
        "district": district,
        "area": area,
    }

    try:
        predicted_price = predict_price(property_features, model=model)
        price_text = f"{predicted_price:,.0f}".replace(",", " ")
        st.metric(label="Прогнозируемая стоимость", value=f"{price_text} ₽")
        st.caption(
            "Это оценка машинной модели. "
        )
    except Exception as error:
        st.error(f"Не удалось рассчитать прогноз: {error}")

with st.expander("Как устроена модель"):
    st.write(
        "Используется Ridge-регрессия с логарифмированием цены, стандартизацией "
        "числовых признаков и One-Hot Encoding категориальных признаков. "
        "Пропущенные значения обрабатываются внутри pipeline. "
        "При обучении 20% строк выделяется для проверки качества; затем итоговая "
        "модель обучается на всех корректных строках датасета."
    )
