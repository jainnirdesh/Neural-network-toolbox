import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Send, MessageSquare, Quote, Brain, Activity, Loader2, Frown, Smile, Meh } from 'lucide-react';
import { cn } from '../../lib/utils';
import { classifySentiment } from '../../services/geminiService';

export default function RNNModule() {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<{ sentiment: 'positive' | 'negative' | 'neutral'; score: number; reason: string } | null>(null);

  const analyze = async () => {
    if (!text.trim()) return;
    setIsAnalyzing(true);
    try {
      const data = await classifySentiment(text);
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return <Smile className="w-10 h-10 text-emerald-600" />;
      case 'negative': return <Frown className="w-10 h-10 text-rose-600" />;
      default: return <Meh className="w-10 h-10 text-amber-600" />;
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-20">
      <div className="text-center space-y-4">
         <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-600 text-[9px] font-black uppercase tracking-widest leading-none">
            Sequence Modeling Practical
         </div>
         <h2 className="text-3xl font-bold tracking-tight text-slate-800">RNN-LSTM <span className="text-slate-400">Engine</span></h2>
         <p className="text-slate-500 text-sm max-w-xl mx-auto font-medium">
           Analyze temporal relationships in linguistic sequences using Recurrent principles. Input review data below for emotional sentiment scoring.
         </p>
      </div>

      <div className="space-y-4">
        <div className="relative">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Input review data... (e.g., 'The performance was outstandingly efficient.')"
            className="w-full h-44 p-6 rounded-2xl bg-white border border-slate-200 text-sm font-medium text-slate-800 placeholder:text-slate-300 focus:outline-none focus:ring-4 focus:ring-blue-50 transition-all resize-none shadow-sm"
          />
          <div className="absolute top-4 right-6 opacity-5 pointer-events-none">
             <Quote className="w-10 h-10" />
          </div>
        </div>

        <div className="flex items-center gap-3">
           <button
             onClick={analyze}
             disabled={!text.trim() || isAnalyzing}
             className={cn(
               "flex-1 h-12 rounded-xl font-black text-[10px] uppercase tracking-widest flex items-center justify-center gap-2 transition-all",
               text.trim() && !isAnalyzing 
                 ? "bg-blue-600 text-white shadow-lg shadow-blue-500/20 hover:bg-blue-700" 
                 : "bg-slate-200 text-slate-400"
             )}
           >
             {isAnalyzing ? (
               <>
                 <Loader2 className="w-3 h-3 animate-spin" />
                 States Processing...
               </>
             ) : (
               <>
                 <Send className="w-3 h-3" />
                 Analyze Sentiment
               </>
             )}
           </button>
           <button 
             onClick={() => { setText(''); setResult(null); }}
             className="h-12 px-6 rounded-xl border border-slate-200 text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-slate-900 transition-all"
           >
             Reset
           </button>
        </div>
      </div>

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid md:grid-cols-[160px_1fr] gap-6 items-center p-6 rounded-2xl border border-slate-200 bg-white shadow-xl shadow-blue-500/5"
          >
             <div className="flex flex-col items-center gap-3 text-center border-r border-slate-50 pr-6">
                <div className="w-16 h-16 rounded-3xl bg-slate-50 flex items-center justify-center border border-slate-100 shadow-inner group overflow-hidden">
                   {getSentimentIcon(result.sentiment)}
                </div>
                <div>
                   <p className="text-[9px] font-black uppercase tracking-widest text-slate-400">Result</p>
                   <p className={cn(
                     "text-sm font-black capitalize",
                     result.sentiment === 'positive' ? 'text-emerald-600' : result.sentiment === 'negative' ? 'text-rose-600' : 'text-amber-600'
                   )}>
                     {result.sentiment}
                   </p>
                </div>
             </div>

             <div className="space-y-4">
                <div>
                   <h4 className="flex items-center gap-2 font-black text-[10px] uppercase tracking-widest text-blue-600 mb-2">
                     <Brain className="w-3 h-3" />
                     LSTM Logic Interpretation
                   </h4>
                   <p className="text-slate-600 italic text-[11px] leading-relaxed font-medium">
                     "{result.reason}"
                   </p>
                </div>

                <div className="space-y-2">
                   <div className="flex justify-between items-end text-[9px] font-black uppercase tracking-widest text-slate-400">
                     <span>Synaptic Intensity</span>
                     <span className="text-blue-600">{Math.round(result.score * 100)}%</span>
                   </div>
                   <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${result.score * 100}%` }}
                        className="h-full bg-blue-600"
                      />
                   </div>
                </div>

                <div className="flex items-center gap-6 pt-2">
                   <div className="flex items-center gap-2 text-[9px] font-black text-slate-400 uppercase tracking-tight">
                      <Activity className="w-3 h-3 text-emerald-500" />
                      Protocol: <span className="text-slate-800">SEQ 1.0</span>
                   </div>
                   <div className="flex items-center gap-2 text-[9px] font-black text-slate-400 uppercase tracking-tight">
                      <Brain className="w-3 h-3 text-emerald-500" />
                      Gates: <span className="text-slate-800">FORGET-ENABLED</span>
                   </div>
                </div>
             </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-3">
         <h4 className="font-black text-[10px] uppercase tracking-widest text-slate-500">Academic Notice: RNN-LSTM Hierarchy</h4>
         <p className="text-[11px] text-slate-400 leading-relaxed font-medium">
           Traditional systems assume independent inputs. RNNs use <strong>hidden states</strong> to carry information over time. <strong>LSTMs (Long Short-Term Memory)</strong> specifically address the vanishing gradient paradox by utilizing specific gated mechanisms to modulate information flow across long sequence indices.
         </p>
      </div>
    </div>
  );
}
