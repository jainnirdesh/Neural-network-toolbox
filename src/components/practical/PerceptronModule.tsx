import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Settings, Play, RefreshCcw, Cpu, Layers, Zap } from 'lucide-react';
import { cn } from '../../lib/utils';

export default function PerceptronModule() {
  const [type, setType] = useState<'SLP' | 'MLP'>('SLP');
  const [inputs, setInputs] = useState<[number, number]>([0, 0]);
  const [weights, setWeights] = useState<[number, number]>([0.5, 0.5]);
  const [bias, setBias] = useState(-0.7);
  const [output, setOutput] = useState(0);

  // Simple step activation function
  const stepFunction = (x: number) => (x >= 0 ? 1 : 0);
  // Sigmoid for MLP representation
  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

  useEffect(() => {
    if (type === 'SLP') {
      const sum = inputs[0] * weights[0] + inputs[1] * weights[1] + bias;
      setOutput(stepFunction(sum));
    } else {
      // Simulated XOR-like MLP logic
      // Hidden layer nodes
      const h1 = sigmoid(inputs[0] * 20 + inputs[1] * -20 - 10);
      const h2 = sigmoid(inputs[0] * -20 + inputs[1] * 20 - 10);
      // Output node
      const z = sigmoid(h1 * 20 + h2 * 20 - 10);
      setOutput(z > 0.5 ? 1 : 0);
    }
  }, [inputs, weights, bias, type]);

  const setGate = (gate: 'AND' | 'OR' | 'NAND' | 'XOR') => {
    if (gate === 'AND') {
      setType('SLP');
      setWeights([0.5, 0.5]);
      setBias(-0.7);
    } else if (gate === 'OR') {
      setType('SLP');
      setWeights([0.5, 0.5]);
      setBias(-0.2);
    } else if (gate === 'NAND') {
      setType('SLP');
      setWeights([-0.5, -0.5]);
      setBias(0.7);
    } else if (gate === 'XOR') {
      setType('MLP');
    }
  };

  return (
    <div className="space-y-8 pb-12">
      <div className="flex gap-2 p-1.5 bg-slate-200 rounded-xl w-fit">
        <button
          onClick={() => setType('SLP')}
          className={cn(
            "px-6 py-2 rounded-lg text-xs font-bold transition-all uppercase tracking-widest",
            type === 'SLP' ? "bg-white text-blue-600 shadow-sm" : "text-slate-500 hover:text-slate-700"
          )}
        >
          Single Layer
        </button>
        <button
          onClick={() => setType('MLP')}
          className={cn(
            "px-6 py-2 rounded-lg text-xs font-bold transition-all uppercase tracking-widest",
            type === 'MLP' ? "bg-white text-blue-600 shadow-sm" : "text-slate-500 hover:text-slate-700"
          )}
        >
          Multi Layer
        </button>
      </div>

      <div className="grid lg:grid-cols-[1fr_380px] gap-8 items-start">
        <div className="space-y-6">
          {/* Visualization Area */}
          <section className="relative aspect-video rounded-2xl bg-white border border-slate-200 flex items-center justify-center overflow-hidden p-8 shadow-sm">
            <div className="absolute inset-0 bg-slate-50/50 opacity-10" />
            
            <div className="relative flex items-center justify-between w-full max-w-2xl px-12">
              {/* Inputs */}
              <div className="space-y-20">
                {[0, 1].map((i) => (
                  <div key={i} className="relative flex items-center gap-4 group">
                    <button
                      onClick={() => {
                        const newInputs = [...inputs] as [number, number];
                        newInputs[i] = inputs[i] === 0 ? 1 : 0;
                        setInputs(newInputs);
                      }}
                      className={cn(
                        "w-12 h-12 rounded-xl flex items-center justify-center font-bold text-lg border-2 transition-all cursor-pointer select-none relative z-10",
                        inputs[i] === 1 
                          ? "bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-500/20" 
                          : "bg-slate-50 border-slate-200 text-slate-400"
                      )}
                    >
                      {inputs[i]}
                    </button>
                    <div className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">X{i+1}</div>
                  </div>
                ))}
              </div>

              {/* Hidden Layer (only for MLP) */}
              {type === 'MLP' && (
                <div className="flex flex-col gap-12">
                   {[0, 1].map((i) => (
                     <div key={i} className="w-10 h-10 rounded-full border-2 border-slate-200 bg-white flex items-center justify-center text-[8px] font-bold text-slate-400 shadow-sm relative z-10">
                        H{i+1}
                     </div>
                   ))}
                </div>
              )}

              {/* Node (Output Layer Node) */}
              <div className="relative">
                <motion.div 
                  animate={{ scale: output === 1 ? [1, 1.05, 1] : 1 }}
                  className={cn(
                    "w-24 h-24 rounded-full border-4 flex items-center justify-center transition-all duration-300 relative z-10",
                    output === 1 ? "border-blue-600 bg-blue-50" : "border-slate-200 bg-white"
                  )}
                >
                  <div className="text-center">
                    <div className="text-[9px] uppercase font-black text-blue-600 mb-1">Result</div>
                    <div className="text-3xl font-black text-slate-800">{output}</div>
                  </div>
                </motion.div>
                {type === 'SLP' && (
                  <div className="absolute -top-10 left-1/2 -translate-x-1/2 px-3 py-1 rounded bg-slate-800 text-[9px] font-black text-white uppercase tracking-widest whitespace-nowrap">
                    Bias {bias.toFixed(1)}
                  </div>
                )}
              </div>

              {/* Lines Visualization */}
              <div className="absolute inset-0 pointer-events-none opacity-20">
                <svg width="100%" height="100%" className="w-full h-full">
                  {type === 'SLP' ? (
                    <>
                      <line x1="20%" y1="35%" x2="50%" y2="50%" stroke="currentColor" strokeWidth="2" className="text-slate-400" />
                      <line x1="20%" y1="65%" x2="50%" y2="50%" stroke="currentColor" strokeWidth="2" className="text-slate-400" />
                    </>
                  ) : (
                    <>
                      <line x1="20%" y1="35%" x2="40%" y2="35%" stroke="currentColor" strokeWidth="2" className="text-slate-400" />
                      <line x1="20%" y1="35%" x2="40%" y2="65%" stroke="currentColor" strokeWidth="2" className="text-slate-400" />
                      <line x1="20%" y1="65%" x2="40%" y2="35%" stroke="currentColor" strokeWidth="2" className="text-slate-400" />
                      <line x1="20%" y1="65%" x2="40%" y2="65%" stroke="currentColor" strokeWidth="2" className="text-slate-400" />
                      
                      <line x1="45%" y1="35%" x2="65%" y2="50%" stroke="currentColor" strokeWidth="2" className="text-slate-400" />
                      <line x1="45%" y1="65%" x2="65%" y2="50%" stroke="currentColor" strokeWidth="2" className="text-slate-400" />
                    </>
                  )}
                  <line x1="75%" y1="50%" x2="90%" y2="50%" stroke="currentColor" strokeWidth="2" strokeDasharray="4" className="text-blue-500" />
                </svg>
              </div>

              {/* Final Output */}
              <div className="flex flex-col items-center gap-4">
                <div className={cn(
                  "w-14 h-14 rounded-2xl flex items-center justify-center border-2 transition-all duration-500",
                  output === 1 ? "bg-emerald-600 border-emerald-500 shadow-lg shadow-emerald-500/20" : "bg-slate-50 border-slate-200 text-slate-300"
                )}>
                  {output === 1 ? <Zap className="w-8 h-8 text-white" /> : <RefreshCcw className="w-6 h-6" />}
                </div>
                <div className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-400">Class</div>
              </div>
            </div>
          </section>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-sm">
              <h4 className="text-[10px] uppercase font-black tracking-widest text-slate-400 mb-4 flex items-center gap-2">
                <Cpu className="w-3 h-3" />
                Logic Gate Presets
              </h4>
              <div className="flex gap-2">
                {['AND', 'OR', 'NAND', 'XOR'].map((gate) => (
                  <button
                    key={gate}
                    onClick={() => setGate(gate as any)}
                    className="flex-1 py-2 text-[10px] font-black bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition-colors uppercase"
                  >
                    {gate}
                  </button>
                ))}
              </div>
            </div>
            <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg">
              <h4 className="text-[10px] uppercase font-black tracking-widest text-slate-500 mb-4">Transfer Function</h4>
              <div className="p-3 rounded-lg bg-slate-950 font-mono text-[11px] text-blue-400">
                {type === 'SLP' ? 'f(Σ w·x + b) = 1 if Σ ≥ 0' : 'Sigmoid(Σ Hidden(x)) > 0.5'}
              </div>
            </div>
          </div>
        </div>

        {/* Panel */}
        <aside className="p-6 rounded-2xl bg-white border border-slate-200 shadow-sm space-y-6">
          <div className="flex items-center gap-2 mb-6">
            <Settings className="w-4 h-4 text-blue-600" />
            <h3 className="font-bold text-sm uppercase tracking-tight text-slate-800">Parameters</h3>
          </div>

          {[
            { label: 'Weight 1', val: weights[0], set: (v: number) => setWeights([v, weights[1]]) },
            { label: 'Weight 2', val: weights[1], set: (v: number) => setWeights([weights[0], v]) },
            { label: 'Network Bias', val: bias, set: setBias },
          ].map((slider, i) => (
            <div key={i} className="space-y-3">
              <div className="flex justify-between items-end">
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{slider.label}</label>
                <span className="text-[11px] font-bold text-blue-600">{slider.val.toFixed(2)}</span>
              </div>
              <input 
                type="range" min="-1" max="1" step="0.1" 
                value={slider.val} 
                onChange={(e) => slider.set(parseFloat(e.target.value))}
                className="w-full accent-blue-600 h-1.5 bg-slate-100 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          ))}

          <div className="pt-6 border-t border-slate-100">
            <p className="text-[10px] text-slate-400 italic font-medium leading-relaxed">
              * The decision boundary adjusts dynamically based on the synaptic weights provided in the training parameters.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
