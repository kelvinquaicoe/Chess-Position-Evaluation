import { useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";

const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:5000";
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
type Evaluation = {
  fen: string; score: number; white_win_estimate: number; black_win_estimate: number;
  side_to_move: string; evaluation_source: string; probability_note: string;
};
export default function App() {
  const [fen, setFen] = useState(START_FEN);
  const [fenInput, setFenInput] = useState(START_FEN);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [boardVersion, setBoardVersion] = useState(0);

  const board = new Chess(fen);
  const gameOver = board.isGameOver();

  function updatePosition(nextFen: string) {
    try {
      const next = new Chess(nextFen);
      setFen(next.fen()); setFenInput(next.fen()); setEvaluation(null);
      setError(""); setNotice(""); setBoardVersion(v => v + 1);
    } catch {
      setError("That position is invalid. Check the FEN and try again.");
    }
  }

  function onPieceDrop({ sourceSquare, targetSquare }: { sourceSquare: string; targetSquare: string | null }) {
    if (!targetSquare) return false;
    const game = new Chess(fen);
    const move = game.move({ from: sourceSquare, to: targetSquare, promotion: "q" });
    if (!move) return false;
    setFen(game.fen()); setFenInput(game.fen()); setEvaluation(null);
    setError(""); setNotice(""); setBoardVersion(v => v + 1);
    return true;
  }

  async function evaluate() {
    setBusy(true); setError(""); setNotice("");
    try {
      const response = await fetch(`${API}/api/evaluate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error ?? "Evaluation request failed");
      setEvaluation(data); setNotice("Position evaluated successfully.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not reach the evaluation API.");
    } finally { setBusy(false); }
  }

  function applyFen() {
    try {
      const parsed = new Chess(fenInput);
      setFen(parsed.fen()); setFenInput(parsed.fen()); setEvaluation(null);
      setError(""); setNotice("Position loaded from FEN."); setBoardVersion(v => v + 1);
    } catch { setError("Could not load that FEN. Make sure it describes a valid position."); }
  }

  function reset() {
    setFen(START_FEN); setFenInput(START_FEN); setEvaluation(null);
    setError(""); setNotice(""); setBoardVersion(v => v + 1);
  }

  function makeMove(move: string) {
    const game = new Chess(fen);
    try {
      game.move(move);
      setFen(game.fen()); setFenInput(game.fen()); setEvaluation(null);
      setError(""); setNotice(""); setBoardVersion(v => v + 1);
    } catch { /* invalid move: leave the current position unchanged */ }
  }

  return <main className="shell">
    <header className="topbar"><a className="brand" href="#"><span className="brand-icon">♞</span><span>POSITION<span className="brand-light">LAB</span></span></a><span className="top-tag"><i /> CHESS ML / EXPERIMENTAL</span></header>
    <section className="intro"><p className="eyebrow">BOARD STATE → MODEL INFERENCE</p><h1>Read the position.<br/><em>Understand the balance.</em></h1><p className="intro-copy">Explore a position on the board and request an evaluation from a CNN-backed API. Drag pieces or paste a FEN to start.</p></section>
    <div className="workspace">
      <section className="board-panel">
        <div className="section-head"><div><span className="section-index">01 / POSITION</span><h2>Analysis board</h2></div><button className="text-button" onClick={reset}>Reset ↺</button></div>
        <div className="board-wrap"><Chessboard key={boardVersion} options={{position: fen, onPieceDrop, allowDragging: !gameOver, boardOrientation: "white", darkSquareStyle: {backgroundColor:"#5b776b"}, lightSquareStyle:{backgroundColor:"#e7e4d8"}, animationDuration:150, showNotation:true}} /></div>
        <div className="board-caption"><span><i className="dot white-dot" /> White</span><span><i className="dot black-dot" /> Black</span><span className="turn">To move: <b>{board.turn() === "w" ? "White" : "Black"}</b></span></div>
        <div className="fen-box"><label htmlFor="fen">FEN position</label><textarea id="fen" rows={3} value={fenInput} onChange={e => setFenInput(e.target.value)} spellCheck={false}/><button className="outline-button" onClick={applyFen}>Load position</button></div>
        <div className="move-tools"><button onClick={() => makeMove("e4")}>Try 1. e4</button><button onClick={() => makeMove("d4")}>Try 1. d4</button><button onClick={() => makeMove("Nf3")}>Try 1. Nf3</button></div>
      </section>
      <section className="result-panel">
        <div className="section-head"><div><span className="section-index">02 / INFERENCE</span><h2>Position evaluation</h2></div><span className="model-pill">CNN + baseline</span></div>
        <div className="score-card">
          <div className="score-label">POSITION SCORE <span>−1.0 to +1.0</span></div>
          <div className="score-value">{evaluation ? (evaluation.score > 0 ? "+" : "") + evaluation.score.toFixed(3) : "— — —"}</div>
          <div className="score-track"><div className="score-mid"/><div className="score-marker" style={{left: `${evaluation ? (evaluation.score + 1) * 50 : 50}%`}}/></div>
          <div className="track-labels"><span>Black advantage</span><span>Balanced</span><span>White advantage</span></div>
          <p className="score-explain">{evaluation ? "Score returned by the current model or baseline." : "Evaluate the current board to see its score."}</p>
        </div>
        <button className="evaluate-button" onClick={() => void evaluate()} disabled={busy}>{busy ? <><span className="spinner"/> Evaluating position…</> : <>Evaluate position <span>↗</span></>}</button>
        {error && <div className="message error" role="alert">{error}</div>}
        {notice && !error && <div className="message success" role="status">{notice}</div>}
        <div className="probability-card"><div className="prob-head"><h3>Outcome estimate</h3><span>HEURISTIC</span></div>
          <p className="fine-print">Illustrative score-to-probability mapping, not a calibrated prediction of game outcomes.</p>
          <div className="prob-row"><div className="prob-label"><span className="piece-symbol">♔</span> White</div><strong>{evaluation ? (evaluation.white_win_estimate * 100).toFixed(1) + "%" : "—"}</strong></div>
          <div className="prob-bar"><div className="white-fill" style={{width:`${evaluation ? evaluation.white_win_estimate*100 : 50}%`}}/></div>
          <div className="prob-row"><div className="prob-label"><span className="piece-symbol black-piece">♚</span> Black</div><strong>{evaluation ? (evaluation.black_win_estimate * 100).toFixed(1) + "%" : "—"}</strong></div>
          <div className="prob-bar dark"><div className="black-fill" style={{width:`${evaluation ? evaluation.black_win_estimate*100 : 50}%`}}/></div>
        </div>
        <div className="details"><h3>Inference details</h3><div><span>Side to move</span><strong>{evaluation ? evaluation.side_to_move : "—"}</strong></div><div><span>Evaluation source</span><strong>{evaluation ? evaluation.evaluation_source === "cnn" ? "Trained CNN" : "Material baseline" : "Awaiting input"}</strong></div><div><span>Position state</span><strong>{gameOver ? "Game over" : "Playable"}</strong></div></div>
      </section>
    </div>
    <section className="method"><span className="method-mark">i</span><div><strong>Research note</strong><p>When no trained model is present, the API falls back to material counting. A CNN prediction is only as meaningful as its training labels; use held-out evaluation results before making accuracy claims.</p></div></section>
    <footer><span>POSITIONLAB / CHESS POSITION EVALUATION</span><span>Python · Flask · TensorFlow · React</span></footer>
  </main>;
}