import { motion } from 'motion/react';
import { Network, Zap, Target, Cpu, Brain, Database, ArrowRight } from 'lucide-react';
import { STUDENT_INFO } from '../../constants';

export default function Dashboard({ setActiveTab }: { setActiveTab: (tab: any) => void }) {
  const cards = [
    { id: 'perceptron', title: 'Perceptron Models', icon: Cpu, color: 'text-cyan-700', bg: 'bg-cyan-100', module: '01', desc: 'Single-layer & Multi-layer Perceptron (MLP) configuration for linear and non-linear classification tasks.' },
    { id: 'propagation', title: 'Optimization / BP', icon: Zap, color: 'text-emerald-700', bg: 'bg-emerald-100', module: '02', desc: 'Back-propagation visualization and weight optimization through standard gradient descent.' },
    { id: 'cnn', title: 'CNN Image Classifier', icon: Target, color: 'text-indigo-700', bg: 'bg-indigo-100', module: '03', desc: 'Layer Map: [Conv] -> [Pool] -> [FC] classifier for real-world categorical image data.' },
    { id: 'rnn', title: 'RNN-LSTM Sentiment', icon: Network, color: 'text-fuchsia-700', bg: 'bg-fuchsia-100', module: '04', desc: 'Sequence modeling engine for linguistic pattern recognition and emotional sentiment scoring.' },
    { id: 'opencv', title: 'OpenCV Face Detection', icon: Database, color: 'text-rose-700', bg: 'bg-rose-100', module: '05', desc: 'Haar Cascade Classifier stream processing for identification of facial silhouettes.' },
    { id: 'hopfield', title: 'Hopfield Recognizer', icon: Brain, color: 'text-amber-700', bg: 'bg-amber-100', module: '06', desc: 'Associative content-addressable memory network for pattern storage and retrieval.' },
  ];

  return (
    <div className="space-y-8 pb-8">
      {/* Header Info Banner */}
      <section className="group relative overflow-hidden rounded-3xl border border-slate-800 bg-slate-950 p-8 shadow-2xl">
         <div className="relative z-10 flex flex-col justify-between gap-8 lg:flex-row lg:items-center">
            <div className="space-y-4">
              <div className="inline-flex items-center rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1 text-[10px] font-black uppercase tracking-[0.2em] text-cyan-200">
                Neural Network Practical Console
              </div>
              <h2 className="text-3xl font-black tracking-tight text-white md:text-4xl">Neural Intelligence</h2>
              <p className="max-w-lg text-sm font-medium leading-relaxed text-slate-300">
                The toolbox uses advanced synaptic weighting systems to simulate high-level cognitive pattern recognition. Explore fundamental architectures through interactive practical simulations.
              </p>
              <div className="flex items-center gap-6 pt-2">
                <div className="flex flex-col">
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Developer</span>
                  <span className="text-xs font-bold text-cyan-300">{STUDENT_INFO.name}</span>
                </div>
                <div className="h-8 w-px bg-slate-700" />
                <div className="flex flex-col">
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Registration</span>
                  <span className="text-xs font-bold text-slate-200">{STUDENT_INFO.rollNo}</span>
                </div>
              </div>
            </div>
            
            {/* Neural SVG Decoration */}
            <svg className="opacity-25 transition-opacity group-hover:opacity-40" width="220" height="120" viewBox="0 0 220 120">
              <circle cx="30" cy="60" r="4" fill="#3b82f6" />
              <circle cx="90" cy="24" r="4" fill="#3b82f6" />
              <circle cx="90" cy="96" r="4" fill="#3b82f6" />
              <circle cx="150" cy="24" r="4" fill="#3b82f6" />
              <circle cx="150" cy="96" r="4" fill="#3b82f6" />
              <circle cx="205" cy="60" r="6" fill="#f59e0b" />
              <line x1="30" y1="60" x2="90" y2="24" stroke="#475569" strokeWidth="1" />
              <line x1="30" y1="60" x2="90" y2="96" stroke="#475569" strokeWidth="1" />
              <line x1="90" y1="24" x2="150" y2="24" stroke="#475569" strokeWidth="1" />
              <line x1="90" y1="24" x2="150" y2="96" stroke="#475569" strokeWidth="1" />
              <line x1="90" y1="96" x2="150" y2="24" stroke="#475569" strokeWidth="1" />
              <line x1="90" y1="96" x2="150" y2="96" stroke="#475569" strokeWidth="1" />
              <line x1="150" y1="24" x2="205" y2="60" stroke="#475569" strokeWidth="1" />
              <line x1="150" y1="96" x2="205" y2="60" stroke="#475569" strokeWidth="1" />
            </svg>
         </div>
         <div className="absolute -right-8 -top-8 h-64 w-64 rounded-full bg-cyan-500/10 blur-[110px]" />
         <div className="absolute -bottom-10 left-20 h-56 w-56 rounded-full bg-amber-500/10 blur-[110px]" />
      </section>

      {/* Grid Section */}
      <section className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {cards.map((card, i) => (
          <motion.button
            key={card.id}
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06, type: 'spring', stiffness: 120 }}
            onClick={() => setActiveTab(card.id)}
            className="group flex flex-col rounded-2xl border border-slate-200 bg-white/95 p-5 text-left shadow-md shadow-slate-200/50 transition-all hover:-translate-y-0.5 hover:border-cyan-300 hover:shadow-xl hover:shadow-cyan-100/60 active:scale-[0.985]"
          >
            <div className="flex justify-between items-start mb-4">
              <span className={`rounded-xl p-2.5 ${card.bg} ${card.color} transition-transform group-hover:scale-110`}>
                <card.icon className="w-5 h-5" />
              </span>
              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Module {card.module}</span>
            </div>
            <h3 className="mb-2 text-sm font-black uppercase tracking-wide text-slate-800">{card.title}</h3>
            <p className="mb-6 text-[11px] font-medium leading-relaxed text-slate-500">
              {card.desc}
            </p>
            <div className="mt-auto flex items-center justify-between border-t border-slate-100 pt-4">
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-cyan-700 opacity-0 transition-opacity group-hover:opacity-100">Launch Module</span>
              <ArrowRight className="h-4 w-4 text-slate-300 transition-colors group-hover:text-cyan-600" />
            </div>
          </motion.button>
        ))}
      </section>
    </div>
  );
}
