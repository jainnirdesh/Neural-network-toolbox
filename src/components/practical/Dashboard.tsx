import { motion } from 'motion/react';
import {
  ArrowRight,
  Atom,
  BookMarked,
  Brain,
  Cpu,
  Database,
  Layers3,
  Network,
  Scan,
  Sparkles,
  Target,
  Zap,
} from 'lucide-react';
import { STUDENT_INFO } from '../../constants';

export default function Dashboard({ setActiveTab }: { setActiveTab: (tab: any) => void }) {
  const cards = [
    {
      id: 'perceptron',
      title: 'Perceptron Models',
      icon: Cpu,
      accent: 'from-cyan-300/20 to-sky-400/20',
      ring: 'border-cyan-300/20',
      module: '01',
      desc: 'Single-layer and multi-layer perceptron experiments for linear decision boundaries and classification rules.',
    },
    {
      id: 'propagation',
      title: 'Propagation Flow',
      icon: Zap,
      accent: 'from-emerald-300/20 to-lime-400/20',
      ring: 'border-emerald-300/20',
      module: '02',
      desc: 'Forward pass, backpropagation, and gradient descent visualized as a compact optimization loop.',
    },
    {
      id: 'cnn',
      title: 'CNN Image Classifier',
      icon: Target,
      accent: 'from-indigo-300/20 to-violet-400/20',
      ring: 'border-indigo-300/20',
      module: '03',
      desc: 'A layered vision pipeline for feature extraction, pooling, and image classification workflows.',
    },
    {
      id: 'rnn',
      title: 'RNN-LSTM Sentiment',
      icon: Network,
      accent: 'from-fuchsia-300/20 to-pink-400/20',
      ring: 'border-fuchsia-300/20',
      module: '04',
      desc: 'Sequence modeling for text sentiment analysis with recurrent memory and temporal context.',
    },
    {
      id: 'opencv',
      title: 'OpenCV Face Detection',
      icon: Scan,
      accent: 'from-rose-300/20 to-orange-400/20',
      ring: 'border-rose-300/20',
      module: '05',
      desc: 'Face detection and counting backed by practical computer-vision utilities.',
    },
    {
      id: 'hopfield',
      title: 'Hopfield Recognizer',
      icon: Brain,
      accent: 'from-amber-300/20 to-yellow-400/20',
      ring: 'border-amber-300/20',
      module: '06',
      desc: 'Associative memory and pattern retrieval for character recognition experiments.',
    },
  ];

  return (
    <div className="space-y-8 pb-10">
      <section className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-white/[0.05] p-6 shadow-2xl shadow-black/20 md:p-8">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_rgba(34,211,238,0.16),_transparent_25%),radial-gradient(circle_at_bottom_left,_rgba(250,204,21,0.14),_transparent_28%)]" />
        <div className="relative grid gap-8 lg:grid-cols-[1.25fr_0.75fr] lg:items-end">
          <div className="space-y-5">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.24em] text-cyan-100">
              <Sparkles className="h-3.5 w-3.5" />
              Neural practical studio
            </div>
            <h2 className="max-w-2xl text-4xl font-black tracking-tight text-white md:text-6xl">
              Learn by switching between live network experiences.
            </h2>
            <p className="max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
              This workspace brings the experiments into one polished interface: concept summaries, interactive modules, and quick access to each practical.
            </p>
            <div className="flex flex-wrap gap-3 pt-2 text-sm text-slate-300">
              <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2">Perceptron to Hopfield</div>
              <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2">Vision, text, and memory</div>
              <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2">Built for local execution</div>
            </div>
          </div>

          <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/70 p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">Developer</p>
                <p className="mt-2 text-xl font-semibold text-white">{STUDENT_INFO.name}</p>
              </div>
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-cyan-300/20 bg-cyan-300/10 text-cyan-200">
                <BookMarked className="h-5 w-5" />
              </div>
            </div>
            <div className="mt-6 space-y-3">
              <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                <p className="text-[10px] uppercase tracking-[0.24em] text-slate-500">Registration</p>
                <p className="mt-1 text-sm font-semibold text-slate-100">{STUDENT_INFO.rollNo}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                <p className="text-[10px] uppercase tracking-[0.24em] text-slate-500">Course</p>
                <p className="mt-1 text-sm font-semibold text-slate-100">{STUDENT_INFO.course}</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
        {cards.map((card, i) => (
          <motion.button
            key={card.id}
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06, type: 'spring', stiffness: 120 }}
            onClick={() => setActiveTab(card.id)}
            className="group flex flex-col rounded-[1.75rem] border border-white/10 bg-white/[0.05] p-5 text-left shadow-lg shadow-black/10 transition-all duration-200 hover:-translate-y-1 hover:border-cyan-300/20 hover:bg-white/[0.075] active:scale-[0.985]"
          >
            <div className="mb-5 flex items-start justify-between">
              <span className={`rounded-2xl border ${card.ring} bg-gradient-to-br ${card.accent} p-3 text-cyan-100 transition-transform group-hover:scale-105`}>
                <card.icon className="h-5 w-5" />
              </span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-400">
                Module {card.module}
              </span>
            </div>
            <h3 className="text-lg font-semibold tracking-tight text-white">{card.title}</h3>
            <p className="mt-3 text-sm leading-6 text-slate-400">
              {card.desc}
            </p>
            <div className="mt-6 flex items-center justify-between border-t border-white/10 pt-4">
              <span className="text-[10px] font-semibold uppercase tracking-[0.24em] text-cyan-100 opacity-0 transition-opacity group-hover:opacity-100">
                Open module
              </span>
              <ArrowRight className="h-4 w-4 text-slate-500 transition-transform group-hover:translate-x-1 group-hover:text-cyan-200" />
            </div>
          </motion.button>
        ))}
      </section>

      <section className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/55 p-6">
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
            <Layers3 className="h-4 w-4 text-amber-300" />
            Workflow snapshot
          </div>
          <div className="mt-5 grid gap-3 md:grid-cols-3">
            {[
              { title: 'Study the concept', body: 'Read the compact explanation and identify the network role.' },
              { title: 'Run the experiment', body: 'Open the module, adjust parameters, and inspect the output.' },
              { title: 'Review the result', body: 'Compare how each architecture behaves across tasks.' },
            ].map((step, index) => (
              <div key={step.title} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">Step 0{index + 1}</p>
                <p className="mt-3 text-base font-semibold text-white">{step.title}</p>
                <p className="mt-2 text-sm leading-6 text-slate-400">{step.body}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-[1.75rem] border border-white/10 bg-white/[0.05] p-6">
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">
            <Atom className="h-4 w-4 text-cyan-300" />
            What’s included
          </div>
          <div className="mt-5 space-y-3">
            {[
              'Perceptron and propagation controls',
              'CNN image classification workflow',
              'RNN sentiment analysis panel',
              'OpenCV detection utilities',
              'Hopfield associative memory demo',
            ].map((item) => (
              <div key={item} className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/55 px-4 py-3">
                <span className="text-sm text-slate-200">{item}</span>
                <ArrowRight className="h-4 w-4 text-slate-500" />
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
