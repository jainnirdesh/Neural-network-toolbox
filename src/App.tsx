
import { useState, type ComponentType } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import {
  ArrowRight,
  Brain,
  BookOpen,
  CheckCircle2,
  Cpu,
  Grid2x2,
  Home,
  Image as ImageIcon,
  Layers3,
  MessageSquare,
  Scan,
  Sparkles,
  User,
} from 'lucide-react';
import { cn } from './lib/utils';
import { STUDENT_INFO } from './constants';

// Internal Components
import Dashboard from './components/practical/Dashboard';
import PerceptronModule from './components/practical/PerceptronModule';
import PropagationModule from './components/practical/PropagationModule';
import CNNModule from './components/practical/CNNModule';
import RNNModule from './components/practical/RNNModule';
import OpenCVModule from './components/practical/OpenCVModule';
import HopfieldModule from './components/practical/HopfieldModule';

type ModuleId = 'home' | 'perceptron' | 'propagation' | 'cnn' | 'rnn' | 'opencv' | 'hopfield';

interface NavItem {
  id: ModuleId;
  label: string;
  icon: ComponentType<{ className?: string }>;
  description: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'home', label: 'Overview', icon: Home, description: 'Project hub and student profile' },
  { id: 'perceptron', label: 'Perceptron', icon: Cpu, description: 'Linear classifiers and decision rules' },
  { id: 'propagation', label: 'Propagation', icon: ArrowRight, description: 'Forward pass and optimization flow' },
  { id: 'cnn', label: 'CNN Classifier', icon: ImageIcon, description: 'Visual pattern recognition lab' },
  { id: 'rnn', label: 'RNN (LSTM)', icon: MessageSquare, description: 'Sequence modelling and sentiment' },
  { id: 'opencv', label: 'OpenCV Detection', icon: Scan, description: 'Face detection and counting' },
  { id: 'hopfield', label: 'Hopfield Network', icon: Brain, description: 'Associative memory and recall' },
];

const quickStats = [
  { label: 'Modules', value: '6 practicals', icon: Grid2x2 },
  { label: 'Mode', value: 'Interactive lab', icon: Sparkles },
  { label: 'Status', value: 'Ready to explore', icon: CheckCircle2 },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<ModuleId>('home');
  const activeModule = NAV_ITEMS.find((i) => i.id === activeTab);

  return (
    <div className="relative min-h-screen overflow-hidden text-slate-100 selection:bg-cyan-300/40 selection:text-slate-950">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(250,204,21,0.16),_transparent_30%),radial-gradient(circle_at_80%_10%,_rgba(34,211,238,0.22),_transparent_28%),linear-gradient(180deg,_#061018_0%,_#0b1320_48%,_#070b12_100%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.035)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.035)_1px,transparent_1px)] bg-[size:64px_64px] opacity-30" />
      <div className="pointer-events-none absolute left-[-8rem] top-32 h-72 w-72 rounded-full bg-cyan-400/10 blur-3xl" />
      <div className="pointer-events-none absolute bottom-[-8rem] right-[-6rem] h-96 w-96 rounded-full bg-amber-400/10 blur-3xl" />

      <div className="relative mx-auto flex min-h-screen max-w-[1600px] flex-col gap-4 p-4 lg:flex-row lg:p-6">
        <aside className="flex w-full flex-col overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/80 shadow-2xl shadow-black/30 backdrop-blur-xl lg:w-[20rem] xl:w-[22rem]">
          <div className="border-b border-white/10 p-5">
            <div className="inline-flex items-center gap-3 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.24em] text-cyan-200">
              <Brain className="h-3.5 w-3.5" />
              Neural Network Toolbox
            </div>
            <div className="mt-5 space-y-3">
              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-400">Academic console</p>
              <h1 className="max-w-xs text-3xl font-black leading-none tracking-tight text-white md:text-4xl">
                Explore every practical from one workspace.
              </h1>
              <p className="max-w-sm text-sm leading-6 text-slate-400">
                A focused lab interface for perceptron models, propagation, vision, sequence learning, and memory networks.
              </p>
            </div>

            <div className="mt-6 grid grid-cols-3 gap-2">
              {quickStats.map((stat) => {
                const Icon = stat.icon;
                return (
                  <div key={stat.label} className="rounded-2xl border border-white/8 bg-white/5 p-3">
                    <Icon className="h-4 w-4 text-cyan-300" />
                    <p className="mt-3 text-[10px] uppercase tracking-[0.24em] text-slate-500">{stat.label}</p>
                    <p className="mt-1 text-xs font-semibold text-slate-100">{stat.value}</p>
                  </div>
                );
              })}
            </div>
          </div>

          <nav className="custom-scrollbar flex-1 space-y-2 overflow-y-auto p-3">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;

              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={cn(
                    'group flex w-full items-center gap-3 rounded-2xl border px-4 py-3 text-left transition-all duration-200',
                    isActive
                      ? 'border-cyan-300/30 bg-cyan-300/10 text-white shadow-lg shadow-cyan-900/20'
                      : 'border-transparent bg-white/0 text-slate-400 hover:border-white/10 hover:bg-white/5 hover:text-slate-100'
                  )}
                >
                  <span
                    className={cn(
                      'flex h-10 w-10 items-center justify-center rounded-xl border transition-colors',
                      isActive ? 'border-cyan-300/30 bg-cyan-300/15 text-cyan-200' : 'border-white/10 bg-white/5 text-slate-300'
                    )}
                  >
                    <Icon className="h-4.5 w-4.5" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-semibold tracking-tight">{item.label}</span>
                    <span className="block truncate text-[11px] leading-5 text-slate-500 group-hover:text-slate-400">{item.description}</span>
                  </span>
                  <ArrowRight className={cn('h-4 w-4 transition-transform', isActive ? 'translate-x-0 text-cyan-300' : '-translate-x-1 text-slate-600 group-hover:translate-x-0')} />
                </button>
              );
            })}
          </nav>

          <div className="border-t border-white/10 p-4">
            <div className="rounded-[1.5rem] border border-white/10 bg-white/5 p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-500">Developer</p>
                  <p className="mt-2 text-base font-semibold text-white">{STUDENT_INFO.name}</p>
                  <p className="mt-1 text-sm text-slate-400">Roll no. {STUDENT_INFO.rollNo}</p>
                </div>
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-cyan-300/20 bg-cyan-300/10 text-cyan-200">
                  <User className="h-5 w-5" />
                </div>
              </div>
              <div className="mt-4 flex items-center gap-2 rounded-2xl border border-emerald-400/20 bg-emerald-400/10 px-3 py-2 text-[11px] font-medium text-emerald-200">
                <Sparkles className="h-4 w-4" />
                Live practical workspace ready
              </div>
            </div>
          </div>
        </aside>

        <main className="relative flex min-h-[calc(100vh-2rem)] flex-1 flex-col overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/55 shadow-2xl shadow-black/20 backdrop-blur-xl lg:min-h-[calc(100vh-3rem)]">
          <header className="flex flex-col gap-4 border-b border-white/10 px-5 py-5 md:flex-row md:items-center md:justify-between md:px-8">
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-300">
                <BookOpen className="h-3.5 w-3.5 text-amber-300" />
                {activeModule?.label ?? 'Overview'}
              </div>
              <h2 className="text-2xl font-black tracking-tight text-white md:text-4xl">
                {activeTab === 'home' ? 'Learning dashboard' : activeModule?.description}
              </h2>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-[11px] font-medium text-slate-300">
                Session 2023-24
              </div>
              <div className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-cyan-200">
                Interactive lab
              </div>
            </div>
          </header>

          <div className="custom-scrollbar flex-1 overflow-y-auto px-4 py-4 md:px-8 md:py-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 16, filter: 'blur(4px)' }}
                animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                exit={{ opacity: 0, y: -10, filter: 'blur(4px)' }}
                transition={{ duration: 0.22, ease: 'easeOut' }}
                className="mx-auto max-w-7xl"
              >
                {activeTab === 'home' && <Dashboard setActiveTab={setActiveTab} />}
                {activeTab === 'perceptron' && <PerceptronModule />}
                {activeTab === 'propagation' && <PropagationModule />}
                {activeTab === 'cnn' && <CNNModule />}
                {activeTab === 'rnn' && <RNNModule />}
                {activeTab === 'opencv' && <OpenCVModule />}
                {activeTab === 'hopfield' && <HopfieldModule />}
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
}

