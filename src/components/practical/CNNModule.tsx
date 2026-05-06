import { useState, useRef, ChangeEvent } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Upload, ImageIcon, Search, Brain, CheckCircle2, AlertCircle, Loader2, Play, Target, Layers, Settings, Activity, RefreshCcw, Circle } from 'lucide-react';
import { cn } from '../../lib/utils';
import { classifyImage } from '../../services/geminiService';

export default function CNNModule() {
  const [image, setImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<{ prediction: string; confidence: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target?.result as string);
        setResult(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyze = async () => {
    if (!image) return;
    setIsAnalyzing(true);
    setError(null);
    try {
      const data = await classifyImage(image);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to analyze image. Please check your API usage.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-8 pb-20">
      <div className="grid lg:grid-cols-[1fr_380px] gap-8">
        <div className="space-y-6">
          {/* Main Stage */}
          <section 
            className={cn(
              "bg-white rounded-2xl border border-slate-200 shadow-sm p-8 min-h-[450px] relative flex flex-col items-center justify-center overflow-hidden transition-all",
              !image && "hover:border-blue-300 hover:bg-blue-50/10 cursor-pointer"
            )}
            onClick={() => !image && !isAnalyzing && fileInputRef.current?.click()}
          >
             {!image ? (
               <div className="text-center space-y-6">
                  <div className="w-24 h-24 rounded-3xl bg-blue-50 flex items-center justify-center text-blue-600 mx-auto relative group">
                     <ImageIcon className="w-10 h-10 group-hover:scale-110 transition-transform" />
                     <div className="absolute inset-0 rounded-3xl border-2 border-dashed border-blue-200 scale-125 opacity-50" />
                  </div>
                  <div className="space-y-1">
                    <h3 className="text-slate-800 font-bold">ResNet-50 Classifier</h3>
                    <p className="text-[10px] text-slate-500 font-medium tracking-tight">Identify cats, dogs, and thousands of other objects with high precision.</p>
                  </div>
                  <button className="px-8 py-3 bg-slate-900 text-white rounded-xl text-[10px] font-black uppercase tracking-widest cursor-pointer hover:bg-slate-800 transition-colors shadow-lg shadow-slate-200 inline-flex items-center gap-2">
                    <Upload className="w-3 h-3" />
                    Load Image Data
                  </button>
               </div>
             ) : (
               <div className="w-full space-y-8">
                 <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                       <div className="p-2 bg-blue-50 rounded-lg text-blue-600">
                          <Target className="w-4 h-4" />
                       </div>
                       <h3 className="font-bold text-sm text-slate-800 uppercase tracking-tight">Processing Pipeline</h3>
                    </div>
                    <button 
                      onClick={(e) => { e.stopPropagation(); setImage(null); setResult(null); }}
                      className="text-[10px] font-black text-slate-400 hover:text-rose-500 uppercase tracking-widest underline underline-offset-4"
                    >
                      Reset Stream
                    </button>
                 </div>

                 <div className="grid md:grid-cols-2 gap-8 items-center">
                   <div className="relative group overflow-hidden rounded-2xl bg-slate-50 border border-slate-200 shadow-sm">
                      <img src={image} alt="Input" className="w-full aspect-square object-cover" />
                      {isAnalyzing && (
                        <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm flex flex-col items-center justify-center gap-4">
                           <Loader2 className="w-10 h-10 text-white animate-spin" />
                           <span className="text-[10px] font-black text-white uppercase tracking-[0.2em]">Mapping Layers...</span>
                        </div>
                      )}
                      {/* Scanning Line */}
                      {isAnalyzing && (
                        <motion.div 
                          animate={{ top: ['0%', '100%', '0%'] }}
                          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
                          className="absolute left-0 right-0 h-0.5 bg-blue-400 shadow-[0_0_15px_#2563eb] z-10"
                        />
                      )}
                   </div>

                   <div className="space-y-6">
                      <div className="space-y-4">
                         <button 
                           onClick={analyze}
                           disabled={isAnalyzing}
                           className="w-full py-4 bg-blue-600 text-white rounded-xl text-xs font-black uppercase tracking-widest hover:bg-blue-700 transition-all shadow-lg shadow-blue-500/20 flex items-center justify-center gap-2 disabled:opacity-50"
                         >
                           {isAnalyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                           {isAnalyzing ? 'Processing...' : 'Run Inference'}
                         </button>
                         
                         {result && (
                           <motion.div 
                             initial={{ opacity: 0, y: 10 }}
                             animate={{ opacity: 1, y: 0 }}
                             className="p-5 bg-white border border-blue-100 rounded-xl shadow-blue-500/5 shadow-xl ring-1 ring-blue-50"
                           >
                             <div className="flex justify-between items-center mb-3">
                               <p className="text-[10px] font-black text-blue-600 uppercase tracking-widest">Prediction</p>
                               <span className="text-xs font-bold text-slate-800 capitalize">{result.prediction}</span>
                             </div>
                             <div className="space-y-2">
                               <div className="flex justify-between text-[10px] font-bold">
                                 <span className="text-slate-400 uppercase">Confidence Score</span>
                                 <span className="text-slate-800">{(result.confidence * 100).toFixed(1)}%</span>
                               </div>
                               <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                                  <motion.div 
                                    initial={{ width: 0 }}
                                    animate={{ width: `${result.confidence * 100}%` }}
                                    className="h-full bg-blue-600"
                                  />
                               </div>
                             </div>
                           </motion.div>
                         )}
                      </div>

                      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                         <div className="flex items-center gap-3 mb-2">
                            <Layers className="w-4 h-4 text-blue-400" />
                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Arch Signature</span>
                         </div>
                         <p className="text-[11px] font-mono text-blue-300">
                            ResNet-50 v2 // Weights: IMAGENET1K
                         </p>
                      </div>
                   </div>
                 </div>
               </div>
             )}
            <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleFileUpload} />
          </section>
          
          <div className="grid grid-cols-3 gap-4">
             {[
               { icon: Target, label: 'Conv2D', desc: 'Feature Extraction' },
               { icon: RefreshCcw, label: 'MaxPool', desc: 'Sub-sampling' },
               { icon: Circle, label: 'Softmax', desc: 'Prob. Mapping' }
             ].map((layer, i) => (
               <div key={i} className="p-4 bg-white border border-slate-200 rounded-xl shadow-sm">
                 <div className="p-2.5 bg-slate-50 text-slate-400 w-fit rounded-lg mb-3">
                   <layer.icon className="w-4 h-4" />
                 </div>
                 <p className="text-xs font-bold text-slate-800">{layer.label}</p>
                 <p className="text-[10px] text-slate-400 font-medium">{layer.desc}</p>
               </div>
             ))}
          </div>
        </div>

        {/* Sidebar */}
        <aside className="space-y-6">
           <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm">
              <div className="flex items-center gap-2 mb-6">
                 <Settings className="w-4 h-4 text-blue-600" />
                 <h4 className="font-bold text-sm tracking-tight text-slate-800 uppercase">Engine Configuration</h4>
              </div>
              
              <div className="space-y-6">
                <div className="space-y-2">
                   <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Kernel Stride</label>
                   <select className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs font-bold text-slate-700 outline-none">
                      <option>1x1 Stride</option>
                      <option>2x2 Stride (Dense)</option>
                   </select>
                </div>
                <div className="space-y-2">
                   <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Pooling Factor</label>
                   <input type="range" className="w-full accent-blue-600 h-1 bg-slate-100" />
                </div>
              </div>
           </div>

           <div className="p-6 bg-slate-900 rounded-2xl border border-slate-800 text-white flex items-center justify-between shadow-lg">
              <div>
                <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest mb-1">Latency (avg)</p>
                <p className="text-xl font-bold tracking-tighter font-mono">0.14<span className="text-blue-400 ml-1 uppercase text-[10px]">ms</span></p>
              </div>
              <Activity className="w-10 h-10 text-emerald-500/20" />
           </div>

           {error && (
             <div className="p-4 bg-rose-50 border border-rose-100 rounded-xl text-rose-600 text-xs font-bold flex gap-2">
               <AlertCircle className="w-4 h-4 flex-shrink-0" />
               <p>{error}</p>
             </div>
           )}

           <div className="p-6 bg-slate-50 border border-slate-200 rounded-2xl">
              <p className="text-[10px] text-slate-500 leading-relaxed font-semibold italic">
                CNNs utilize shared-weights architecture and translation-invariant characteristics to recognize structural patterns within local receptive fields.
              </p>
           </div>
        </aside>
      </div>
    </div>
  );
}
