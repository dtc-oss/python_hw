# Прогнозный тест запустит обучение, если model.joblib отсутствует
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_predict_get():
    response = client.get('/predict_get', params={
        'total_square': 45,
        'rooms': 2,
        'floor': 5,
        'city': 'Москва',
        'district': 'Тверской район',
    })
    assert response.status_code == 200
    assert response.json()['predicted_price'] > 0


def test_predict_post():
    response = client.post('/predict_post', json={
        'total_square': 45,
        'rooms': 2,
        'floor': 5,
        'city': 'Москва',
        'district': 'Тверской район',
    })
    assert response.status_code == 200
    assert response.json()['predicted_price'] > 0
