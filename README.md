# Chess Position Evaluation

An interactive chess-position analysis app with a React + TypeScript frontend and a Flask API. The backend includes a TensorFlow/Keras convolutional neural network (CNN) training pipeline for predicting a position score from board features.

## Features

- Enter a position using Forsyth–Edwards Notation (FEN).
- View the position on an interactive chessboard.
- Request an evaluation from the Flask API.
- Train a CNN from a labeled CSV dataset.
- Use a transparent material-based baseline when no trained model is available.
- Keep model evaluation reproducible with a held-out test set and recorded mean squared error (MSE).

## Tech stack

- Frontend: React, TypeScript, Vite, `react-chessboard`, `chess.js`
- Backend: Python, Flask, Flask-CORS
- ML: TensorFlow / Keras, NumPy, pandas, scikit-learn

## Quick start

### 1. Start the API

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The API runs at http://127.0.0.1:5000. Check `/api/health` to confirm it is responding.

### 2. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (usually http://localhost:5173).

## Evaluate a position through the API

```bash
curl -X POST http://127.0.0.1:5000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'
```

The response includes the evaluation score, a white-win probability estimate, and whether the value came from the trained model or the baseline. The baseline is a simple material heuristic, not engine-strength analysis.

## Train the CNN

The training pipeline expects a CSV with a `fen` column and a numeric `target` column. Targets should be consistently normalized to the range [-1, 1], where -1 favors Black, 0 is balanced, and +1 favors White. Prefer labels from a chess engine or a documented source rather than material-only labels if the goal is meaningful position evaluation.

```bash
cd backend
python train.py --data ../data/positions.csv --epochs 20
```

The script writes the best model to `backend/models/position_cnn.keras` and prints held-out test MSE. Do not claim a dataset size or error rate unless it is supported by the dataset and training output.

### Dataset format

```csv
fen,target
"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",0.0
```

A single labeled row is only an example of the schema, not a useful training dataset. You must provide a real, sufficiently large dataset to train a meaningful model.

## Model design

The board encoder maps each square to a 12-channel piece representation (six piece types for each color), yielding an 8 × 8 × 12 tensor. The CNN predicts a scalar score in [-1, 1]. The training script uses a held-out test split and mean squared error (MSE).

## Limitations

- Without a trained model file, the API uses a material-count baseline; it does not run Stockfish.
- The predicted win probability is a heuristic mapping of the score, not a calibrated probability unless calibration has been measured on independent data.
- The CNN's performance depends on the quality and size of the supplied labels.
- This project is a learning prototype, not a replacement for a full chess engine.

## Roadmap

- Add a documented dataset acquisition/generation workflow.
- Compare the CNN with a material baseline and a chess engine.
- Add calibration plots and confidence analysis.
- Publish benchmark results and a model card after training is reproducible.
