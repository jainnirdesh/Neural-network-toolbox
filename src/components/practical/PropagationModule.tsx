import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Play, RotateCcw, Activity, ArrowRight, ArrowLeft, Target } from 'lucide-react';
import { cn } from '../../lib/utils';

export default function PropagationModule() {
  const [step, setStep] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0.5);
  
  // Weights and parameters
  const [w, setW] = useState(0.8);
  const [learningRate, setLearningRate] = useState(0.1);
  const target = 0.5;
  const input = 1.0;

  useEffect(() => {
    let interval: any;
    if (isTraining) {
      interval = setInterval(() => {
        setEpoch(e => e + 1);
        
        const prediction = w * input;
        const currentLoss = Math.pow(prediction - target, 2);
        const gradient = 2 * (prediction - target) * input;
        
        setW(prevW => prevW - learningRate * gradient);
        setLoss(currentLoss);
        
        if (currentLoss < 0.001) setIsTraining(false);
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isTraining, w, learningRate]);

  const reset = () => {
    setW(0.8);
    setEpoch(0);
    setLoss(0.5);
    setIsTraining(false);
  };

  const stepForward = () => {
    const prediction = w * input;
    const currentLoss = Math.pow(prediction - target, 2);
    setLoss(currentLoss);
    // Move particles visually or update epoch
    setEpoch(e => e + 1);
  };

  const stepBackward = () => {
    const prediction = w * input;
    const gradient = 2 * (prediction - target) * input;
    setW(prevW => prevW - learningRate * gradient);
    const currentLoss = Math.pow(w * input - target, 2);
    setLoss(currentLoss);
  };

  return (
    <div className="space-y-12 pb-20">
      <div className="grid lg:grid-cols-[1fr_350px] gap-8">
        <div className="space-y-8">
          {/* Visual Simulation */}
          <div className="p-8 rounded-2xl bg-white border border-slate-200 shadow-sm space-y-12">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm uppercase tracking-widest text-slate-400 flex items-center gap-3">
                <Activity className="w-5 h-5 text-blue-600" />
                Training Process
              </h3>
              <div className="flex gap-2">
                <button 
                  onClick={stepForward}
                  disabled={isTraining}
                  className="px-4 py-2 rounded-lg bg-blue-50 text-blue-600 text-[9px] font-black uppercase tracking-widest border border-blue-100 hover:bg-blue-100 transition-colors disabled:opacity-50"
                  title="Compute Predicted Output"
                >
                  Forward Step
                </button>
                <button 
                  onClick={stepBackward}
                  disabled={isTraining}
                  className="px-4 py-2 rounded-lg bg-rose-50 text-rose-600 text-[9px] font-black uppercase tracking-widest border border-rose-100 hover:bg-rose-100 transition-colors disabled:opacity-50"
                  title="Apply Gradient Update"
                >
                  Backward Step
                </button>
                <div className="w-px h-8 bg-slate-100 mx-1" />
                <button 
                  onClick={() => setIsTraining(!isTraining)}
                  className={cn(
                    "flex items-center gap-2 px-6 py-2 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all shadow-sm",
                    isTraining ? "bg-rose-600 text-white border border-rose-500" : "bg-blue-600 text-white border border-blue-500"
                  )}
                >
                  <Play className={cn("w-3 h-3", isTraining && "fill-current")} />
                  {isTraining ? 'Halt Loop' : 'Auto Train'}
                </button>
                <button 
                  onClick={reset}
                  className="p-2 rounded-lg bg-slate-100 border border-slate-200 text-slate-500 hover:text-slate-900"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Path Visualization */}
            <div className="relative h-64 border-b border-slate-100 flex items-end justify-around px-12 pb-8">
              <div className="absolute inset-0 flex items-center justify-center opacity-5 pointer-events-none">
                 <svg width="100%" height="100%" viewBox="0 0 400 200">
                    <path d="M 50 150 Q 200 10 350 150" fill="none" stroke="#2563eb" strokeWidth="2" strokeDasharray="5,5" />
                 </svg>
              </div>

              {/* Input Node */}
              <div className="flex flex-col items-center gap-4 relative z-10">
                <div className="w-14 h-14 rounded-xl bg-slate-50 border-2 border-slate-200 flex items-center justify-center font-bold text-slate-700">
                  {input}
                </div>
                <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Inp Matrix</span>
              </div>

              {/* Weight Line & Particles */}
              <div className="relative flex-1 mx-8 h-1 bg-slate-100 mt-[-28px] self-center rounded-full overflow-hidden">
                 {isTraining && (
                   <>
                     {/* Forward Particles */}
                     <motion.div 
                       animate={{ x: ['-20%', '120%'] }}
                       transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
                       className="absolute top-0 w-4 h-full bg-blue-400/50 blur-[2px]"
                     />
                     {/* Backward Particles (red) */}
                     <motion.div 
                       animate={{ x: ['120%', '-20%'] }}
                       transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
                       className="absolute bottom-0 w-6 h-full bg-rose-400/50 blur-[3px]"
                     />
                   </>
                 )}
                 <motion.div 
                   animate={{ scale: isTraining ? [1, 1.1, 1] : 1 }}
                   className="absolute left-1/2 -translate-x-1/2 -translate-y-8 px-3 py-1 rounded bg-slate-900 text-white text-[9px] font-black uppercase tracking-widest z-20"
                 >
                   w: {w.toFixed(3)}
                 </motion.div>
              </div>

              {/* Prediction Node */}
              <div className="flex flex-col items-center gap-4 relative z-10">
                 <motion.div 
                   animate={{ 
                     borderColor: loss < 0.01 ? '#10b981' : '#2563eb',
                     backgroundColor: loss < 0.01 ? 'rgba(16,185,129,0.05)' : 'rgba(37,99,235,0.05)'
                   }}
                   className="w-20 h-20 rounded-full border-4 flex items-center justify-center font-black text-lg transition-colors text-slate-800"
                 >
                   {(w * input).toFixed(2)}
                 </motion.div>
                 <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Pred Output</span>
              </div>

              {/* Target Indicator */}
              <div className="absolute right-12 top-0 flex flex-col items-center gap-1">
                 <Target className="w-5 h-5 text-rose-500 animate-pulse" />
                 <span className="text-[9px] font-black text-rose-500 uppercase tracking-widest">Optimum {target}</span>
              </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-3 gap-4">
               <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                  <p className="text-[9px] uppercase text-slate-400 font-black mb-2 tracking-[0.15em]">Iter Epoch</p>
                  <p className="text-xl font-mono font-bold text-slate-800">{epoch}</p>
               </div>
               <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                  <p className="text-[9px] uppercase text-slate-400 font-black mb-2 tracking-[0.15em]">Gradient MSE</p>
                  <p className={cn("text-xl font-mono font-bold transition-colors", loss < 0.01 ? "text-emerald-600" : "text-rose-600")}>
                    {loss.toFixed(6)}
                  </p>
               </div>
               <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                  <p className="text-[9px] uppercase text-slate-400 font-black mb-2 tracking-[0.15em]">Synaptic Δ</p>
                  <p className="text-xl font-mono font-bold text-blue-600">
                    {(2 * (w * input - target) * input).toFixed(4)}
                  </p>
               </div>
            </div>
          </div>

          {/* Education Blocks */}
          <div className="grid md:grid-cols-2 gap-8">
             <div className="space-y-4">
                <h4 className="flex items-center gap-2 font-black text-[10px] uppercase tracking-widest text-blue-600">
                  <ArrowRight className="w-3 h-3" />
                  Forward Prop.
                </h4>
                <p className="text-[11px] text-slate-500 leading-relaxed font-medium">
                  Linear composition of inputs with weights plus bias, mapped through a non-linear activation function to produce a tentative prediction.
                </p>
             </div>
             <div className="space-y-4">
                <h4 className="flex items-center gap-2 font-black text-[10px] uppercase tracking-widest text-rose-600">
                  <ArrowLeft className="w-3 h-3" />
                  Backprop Engine
                </h4>
                <p className="text-[11px] text-slate-500 leading-relaxed font-medium">
                  Recursive calculation of partial derivatives to determine how each internal parameter impacts the global error threshold.
                </p>
             </div>
          </div>
        </div>

        {/* Sidebar Controls */}
        <aside className="p-6 rounded-2xl bg-white border border-slate-200 shadow-sm h-fit space-y-8 sticky top-24">
           <div className="space-y-6">
              <div className="space-y-4">
                <div className="flex justify-between">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Rate (η)</label>
                  <span className="text-xs font-bold text-blue-600">{learningRate}</span>
                </div>
                <div className="flex gap-2">
                   {[0.01, 0.1, 0.5].map(lr => (
                     <button 
                       key={lr}
                       onClick={() => setLearningRate(lr)}
                       className={cn(
                         "flex-1 py-1.5 rounded text-[10px] font-black border transition-all uppercase tracking-widest",
                         learningRate === lr ? "bg-slate-900 border-slate-800 text-white" : "bg-slate-50 border-slate-200 text-slate-400"
                       )}
                     >
                       {lr}
                     </button>
                   ))}
                </div>
              </div>

              <div className="p-4 rounded-xl bg-blue-50/50 border border-blue-100 text-[10px] text-slate-500 leading-relaxed font-medium">
                Adjusting the learning rate alters the convergence delta of the optimizer.
              </div>
           </div>

           <div className="pt-6 border-t border-slate-100 space-y-4">
              <h4 className="font-black text-[10px] uppercase tracking-widest text-slate-400">Optimization Status</h4>
              <div className="space-y-2">
                 <div className="flex justify-between text-[11px]">
                   <span className="text-slate-500 font-medium">Protocol</span>
                   <span className="text-blue-600 font-bold">SGD Optimizer</span>
                 </div>
                 <div className="flex justify-between text-[11px]">
                   <span className="text-slate-500 font-medium">Convergence’</span>
                   <span className="text-blue-600 font-bold">{loss < 0.01 ? 'STABLE' : 'CALCULATING'}</span>
                 </div>
              </div>
           </div>
        </aside>
      </div>
    </div>
  );
}
