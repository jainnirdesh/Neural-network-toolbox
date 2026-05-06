import { useState, useRef, useEffect } from 'react';
import { motion } from 'motion/react';
import { Palette, RotateCcw, Brain, CheckCircle2, Eraser, PenTool } from 'lucide-react';
import { cn } from '../../lib/utils';
import { GoogleGenAI } from "@google/genai";

function getAiClient() {
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("Missing Gemini API key. Set VITE_GEMINI_API_KEY in your .env file.");
  }
  return new GoogleGenAI({ apiKey });
}

export default function HopfieldModule() {
  const [grid, setGrid] = useState<number[]>(new Array(100).fill(0));
  const [isDrawing, setIsDrawing] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // We'll simulate a 10x10 grid for drawing
  const GRID_SIZE = 10;

  const toggleCell = (index: number) => {
    const newGrid = [...grid];
    newGrid[index] = grid[index] === 0 ? 1 : 0;
    setGrid(newGrid);
    setResult(null);
    setError(null);
  };

  const clear = () => {
    setGrid(new Array(100).fill(0));
    setResult(null);
    setError(null);
  };

  const recognizeLocally = () => {
    const rowCounts = Array.from({ length: GRID_SIZE }, (_, row) =>
      grid.slice(row * GRID_SIZE, row * GRID_SIZE + GRID_SIZE).reduce((sum, cell) => sum + cell, 0)
    );
    const colCounts = Array.from({ length: GRID_SIZE }, (_, col) =>
      grid.reduce((sum, cell, index) => sum + (index % GRID_SIZE === col ? cell : 0), 0)
    );

    const middleRow = rowCounts[4] + rowCounts[5];
    const leftCol = colCounts[0] + colCounts[1];
    const rightCol = colCounts[4] + colCounts[5];

    if (leftCol >= 6 && rightCol >= 6 && middleRow >= 4) {
      return 'H';
    }

    const centerCol = colCounts[4] + colCounts[5];
    const topRow = rowCounts[0] + rowCounts[1];
    const bottomRow = rowCounts[8] + rowCounts[9];

    if (centerCol >= 6 && (topRow >= 4 || bottomRow >= 4)) {
      return 'I';
    }

    return '?';
  };

  const recognize = async () => {
    setIsProcessing(true);
    setError(null);
    
    try {
      const ai = getAiClient();
      if (!ai) {
        const char = recognizeLocally();
        setResult(char === '?' ? null : char);
        return;
      }

      const gridString = grid
        .reduce<string[]>((rows, cell, index) => {
          const rowIndex = Math.floor(index / GRID_SIZE);
          rows[rowIndex] = (rows[rowIndex] || '') + (cell === 1 ? 'X' : '.');
          return rows;
        }, [])
        .join('\n');

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [
          {
            role: 'user',
            parts: [
              { text: `Analyze this 10x10 drawing grid where 'X' is a marked pixel and '.' is empty space. Identify which English ALPHABET (one of A-Z) is most clearly drawn. Return ONLY the single uppercase character. If it doesn't look like an alphabet, return '?'.\n\nGrid:\n${gridString}` }
            ]
          }
        ]
      });

      const resultText = response.text || "?";
      const char = resultText.trim().charAt(0) || '?';
      setResult(char === '?' ? null : char as any);
    } catch (err) {
      console.error(err);
      const char = recognizeLocally();
      if (char === '?') {
        setError('Convergence could not run with Gemini, and the local recognizer could not confidently match the pattern.');
      } else {
        setResult(char);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-8 pb-20">
      <div className="grid lg:grid-cols-[1fr_380px] gap-8">
        <div className="space-y-6">
           <div className="p-8 bg-white border border-slate-200 rounded-2xl shadow-sm space-y-6">
              <h3 className="text-sm font-black uppercase tracking-widest text-slate-400 flex items-center gap-3">
                 <Palette className="w-5 h-5 text-blue-600" />
                 State Surface (CAM)
              </h3>

              <div 
                onMouseLeave={() => setIsDrawing(false)}
                className="grid grid-cols-10 gap-1 w-full max-w-[320px] mx-auto aspect-square p-2 bg-slate-50 rounded-xl border border-slate-200 shadow-inner"
              >
                {grid.map((cell, i) => (
                  <div
                    key={i}
                    onMouseDown={() => { setIsDrawing(true); toggleCell(i); }}
                    onMouseEnter={() => isDrawing && toggleCell(i)}
                    onMouseUp={() => setIsDrawing(false)}
                    className={cn(
                      "aspect-square rounded-[1px] transition-colors cursor-crosshair",
                      cell === 1 ? "bg-slate-900 shadow-sm" : "bg-white border-[0.5px] border-slate-100 hover:bg-blue-50"
                    )}
                  />
                ))}
              </div>

              <div className="flex gap-3 max-w-[320px] mx-auto w-full">
                 <button
                   onClick={recognize}
                   disabled={isProcessing}
                   className="flex-1 h-12 rounded-xl bg-blue-600 text-white font-black text-[10px] uppercase tracking-widest flex items-center justify-center gap-2 shadow-lg shadow-blue-500/20 hover:bg-blue-700 transition-all disabled:opacity-50"
                 >
                   {isProcessing ? 'Syncing...' : 'Converge State'}
                 </button>
                 <button
                   onClick={clear}
                   className="w-12 h-12 rounded-xl border border-slate-200 text-slate-400 hover:text-rose-500 transition-all flex items-center justify-center"
                  aria-label="Clear grid"
                 >
                    <RotateCcw className="w-4 h-4" />
                 </button>
              </div>

              {error && (
                <div className="flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-rose-700">
                  <CheckCircle2 className="mt-0.5 w-4 h-4 flex-shrink-0" />
                  <p className="text-[11px] font-medium leading-relaxed">{error}</p>
                </div>
              )}
           </div>
        </div>

        <aside className="space-y-6">
           <div className={cn(
             "h-[280px] rounded-2xl border flex flex-col items-center justify-center text-center transition-all",
             result ? "bg-slate-900 border-slate-800 shadow-2xl" : "bg-white border-slate-200"
           )}>
              {result ? (
                <motion.div initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}>
                   <p className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-500 mb-2">Stable Pattern</p>
                   <span className="text-9xl font-black text-white italic tracking-tighter">{result}</span>
                   <div className="mt-4 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-500/10 text-[9px] font-black uppercase tracking-widest text-blue-400 border border-blue-500/20">
                      <CheckCircle2 className="w-3 h-3" />
                      Convergent Equilibrium
                   </div>
                </motion.div>
              ) : (
                <div className="space-y-2 text-slate-300">
                   <PenTool className="w-12 h-12 mx-auto stroke-[1.5]" />
                   <p className="text-[10px] font-black uppercase tracking-widest">Draw Pattern Data</p>
                </div>
              )}
           </div>

           <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm space-y-4">
              <h4 className="flex items-center gap-2 font-black text-[10px] uppercase tracking-widest text-slate-400">
                <Brain className="w-4 h-4 text-blue-600" />
                Energy Function Logic
              </h4>
              <p className="text-[11px] text-slate-500 leading-relaxed font-medium">
                Hopfield Networks are symmetrical recurrent systems that act as content-addressable memory. The network converges to a <strong>stable energy minimum</strong> from noisy signal input.
              </p>
              <div className="p-3 bg-slate-900 rounded-xl font-mono text-[10px] text-blue-300 border border-slate-800">
                E = -1/2 ΣΣ w_ij s_i s_j
              </div>
           </div>
        </aside>
      </div>
    </div>
  );
}
