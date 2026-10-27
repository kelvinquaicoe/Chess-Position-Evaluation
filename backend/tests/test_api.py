import pytest
from app import app, encode_board, evaluate_fen
import chess
import numpy as np

@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client

def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"

def test_start_position_returns_baseline_or_model(client):
    response = client.post("/api/evaluate", json={"fen": chess.STARTING_FEN})
    assert response.status_code == 200
    assert response.json["evaluation_source"] in {"cnn", "material_baseline"}
    assert -1 <= response.json["score"] <= 1

def test_missing_fen_is_rejected(client):
    assert client.post("/api/evaluate", json={}).status_code == 400

def test_invalid_fen_is_rejected(client):
    assert client.post("/api/evaluate", json={"fen": "not a FEN"}).status_code == 422

def test_board_encoding_shape():
    encoded = encode_board(chess.Board())
    assert encoded.shape == (8, 8, 12)
    assert np.count_nonzero(encoded) == 32

def test_starting_position_has_balanced_material():
    result = evaluate_fen(chess.STARTING_FEN)
    assert abs(result["score"]) < 1e-5
