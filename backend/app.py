"""Flask API for chess-position evaluation."""
from __future__ import annotations
import math
import os
from pathlib import Path
import numpy as np
import chess
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

MODEL_PATH = Path(os.environ.get("CHESS_MODEL_PATH", Path(__file__).parent / "models" / "position_cnn.keras"))
PIECE_VALUES = {
    chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.25,
    chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0,
}
_model = None
_model_checked = False

def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board as an 8x8x12 one-hot tensor, indexed rank 8 to rank 1."""
    encoded = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        channel = (piece.piece_type - 1) + (6 if piece.color == chess.BLACK else 0)
        encoded[row, col, channel] = 1.0
    return encoded

def material_score(board: chess.Board) -> float:
    """Simple material heuristic normalized to [-1, 1]; not an engine evaluation."""
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return float(np.tanh(score / 12.0))

def load_model():
    global _model, _model_checked
    if not _model_checked:
        _model_checked = True
        if MODEL_PATH.exists():
            try:
                from tensorflow import keras
                _model = keras.models.load_model(MODEL_PATH)
            except Exception as exc:
                app.logger.warning("Could not load model at %s: %s", MODEL_PATH, exc)
    return _model

def evaluate_fen(fen: str) -> dict:
    try:
        board = chess.Board(fen)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid FEN string") from exc
    if not board.is_valid():
        raise ValueError("FEN describes an invalid chess position")
    model = load_model()
    if model is not None:
        prediction = float(np.asarray(model.predict(encode_board(board)[None, ...], verbose=0)).reshape(-1)[0])
        score = float(np.clip(prediction, -1.0, 1.0))
        source = "cnn"
    else:
        score = material_score(board)
        source = "material_baseline"
    # This monotonic mapping is illustrative, not a calibrated chess win probability.
    white_win_estimate = 1.0 / (1.0 + math.exp(-4.0 * score))
    return {
        "fen": board.fen(),
        "score": round(score, 5),
        "white_win_estimate": round(white_win_estimate, 5),
        "black_win_estimate": round(1.0 - white_win_estimate, 5),
        "side_to_move": "white" if board.turn else "black",
        "evaluation_source": source,
        "probability_note": "Heuristic estimate; not calibrated against game outcomes.",
    }

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": load_model() is not None})

@app.post("/api/evaluate")
def evaluate():
    payload = request.get_json(silent=True) or {}
    fen = payload.get("fen")
    if not isinstance(fen, str) or not fen.strip():
        return jsonify({"error": "Request body must include a non-empty 'fen' string."}), 400
    try:
        return jsonify(evaluate_fen(fen))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
