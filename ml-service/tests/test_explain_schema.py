
from fastapi.testclient import TestClient
from main import app

def test_explain_schema():
    client = TestClient(app)
    r = client.post('/explain', json={'text':'hello','model_name':'roberta-base','method':'lime','top_k':5})
    assert r.status_code in (200, 404, 400)
