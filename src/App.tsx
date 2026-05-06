
import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Brain, 
  Cpu, 
  Image as ImageIcon, 
  MessageSquare, 
  User, 
  Scan, 
  PenTool,
  Home,
  ChevronRight,
  Sparkles
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
  icon: any;
  description: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'home', label: 'Dashboard', icon: Home, description: 'Overview and Student Info' },
  { id: 'perceptron', label: 'Perceptron', icon: Cpu, description: 'Single & Multi-layer' },
  { id: 'propagation', label: 'Propagation', icon: MoveRightIcon, description: 'Forward / Backward / Gradient Descent' },
  { id: 'cnn', label: 'CNN Classifier', icon: ImageIcon, description: 'Cat vs Dog Image Recognition' },
  { id: 'rnn', label: 'RNN (LSTM)', icon: MessageSquare, description: 'Text Sentiment Analysis' },
  { id: 'opencv', label: 'OpenCV Detection', icon: Scan, description: 'Face Detection & Counting' },
  { id: 'hopfield', label: 'Hopfield Network', icon: PenTool, description: 'Character Recognition' },
];

function MoveRightIcon(props: any) {
  return (
    <div className="flex items-center">
      <ChevronRight {...props} />
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState<ModuleId>('home');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const activeModule = NAV_ITEMS.find((i) => i.id === activeTab);

  return (
    <div className="relative min-h-screen bg-slate-100 text-slate-800 font-sans selection:bg-cyan-300/40 selection:text-slate-900">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-32 left-1/2 h-80 w-80 -translate-x-1/2 rounded-full bg-cyan-300/35 blur-3xl" />
        <div className="absolute bottom-0 right-10 h-72 w-72 rounded-full bg-amber-200/45 blur-3xl" />
      </div>

      <div className="relative flex min-h-screen gap-2 p-2 md:h-screen md:gap-4 md:p-4">
        {/* Sidebar */}
        <aside 
          className={cn(
            "relative z-40 flex shrink-0 flex-col overflow-hidden rounded-2xl border border-slate-800/80 bg-slate-950/95 text-slate-300 shadow-2xl transition-all duration-300",
            isSidebarOpen ? "w-72" : "w-20"
          )}
        >
          <div className="flex h-20 items-center gap-3 border-b border-slate-800 px-5">
            <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 shadow-lg shadow-cyan-600/20">
              <Brain className="h-5 w-5 text-white" />
            </div>
            {isSidebarOpen && (
              <div className="flex flex-col">
                <span className="leading-none text-lg font-black tracking-tight text-white">NEUROLAB</span>
                <span className="mt-1 text-[9px] uppercase tracking-[0.24em] text-cyan-300/60">v1.0 toolbox</span>
              </div>
            )}
          </div>

          <nav className="custom-scrollbar flex-1 space-y-1 overflow-y-auto px-3 py-4">
            {isSidebarOpen && <div className="px-3 py-2 text-[10px] font-black uppercase tracking-[0.2em] text-slate-600">Foundations</div>}
            {NAV_ITEMS.slice(0, 3).map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={cn(
                    "group relative flex w-full items-center rounded-xl px-3 py-2.5 transition-all duration-200",
                    isActive 
                      ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-blue-900/40" 
                      : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                  )}
                >
                  <Icon className={cn("h-4 w-4 flex-shrink-0", isActive && "text-white")} />
                  {isSidebarOpen && (
                    <div className="ml-3 flex min-w-0 flex-col items-start text-left">
                      <span className="truncate text-xs font-bold uppercase tracking-wide">{item.label}</span>
                    </div>
                  )}
                </button>
              );
            })}

            {isSidebarOpen && <div className="px-3 py-4 text-[10px] font-black uppercase tracking-[0.2em] text-slate-600">Advanced</div>}
            {NAV_ITEMS.slice(3).map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={cn(
                    "group relative flex w-full items-center rounded-xl px-3 py-2.5 transition-all duration-200",
                    isActive 
                      ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-blue-900/40" 
                      : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                  )}
                >
                  <Icon className={cn("h-4 w-4 flex-shrink-0", isActive && "text-white")} />
                  {isSidebarOpen && (
                    <div className="ml-3 flex min-w-0 flex-col items-start text-left">
                      <span className="truncate text-xs font-bold uppercase tracking-wide">{item.label}</span>
                    </div>
                  )}
                </button>
              );
            })}
          </nav>

          <div className="border-t border-slate-800 bg-slate-950/90 p-4">
            {isSidebarOpen && (
              <div className="mb-4 rounded-xl border border-slate-800 bg-slate-900/80 p-3">
                <p className="mb-1 text-[9px] font-black uppercase tracking-[0.2em] text-slate-500">Developer</p>
                <p className="text-xs font-bold tracking-tight text-cyan-300">{STUDENT_INFO.name}</p>
                <p className="text-[10px] text-slate-400">Registration: {STUDENT_INFO.rollNo}</p>
              </div>
            )}
            <button 
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="flex h-9 w-full items-center justify-center rounded-xl border border-slate-800 text-slate-500 transition-colors hover:bg-slate-900"
            >
              <ChevronRight className={cn("h-5 w-5 transition-transform", isSidebarOpen && "rotate-180")} />
            </button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="relative flex flex-1 flex-col overflow-hidden rounded-2xl border border-slate-200/80 bg-white/80 shadow-xl backdrop-blur-md">
          <header className="sticky top-0 z-30 flex min-h-16 items-center justify-between border-b border-slate-200/80 bg-white/70 px-4 backdrop-blur-md md:px-8">
            <div className="min-w-0 space-y-1">
              <h1 className="flex items-center gap-2 truncate text-sm font-black uppercase tracking-tight text-slate-800">
                <span className="truncate">{activeModule?.label}</span>
                <span className="rounded-full bg-cyan-50 px-2 py-0.5 text-[9px] font-black uppercase tracking-[0.2em] text-cyan-700">Interactive practical</span>
              </h1>
              <p className="hidden truncate text-[11px] font-medium text-slate-500 md:block">{activeModule?.description}</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="hidden sm:flex flex-col items-end">
                <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">NeuroLab Academic Portal</span>
                <span className="text-[10px] font-bold uppercase tracking-widest text-cyan-700">Session 2023-24</span>
              </div>
              <div className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-slate-50 shadow-sm">
                <User className="h-4 w-4 text-slate-400" />
              </div>
              <div className="hidden h-9 items-center gap-2 rounded-full border border-cyan-100 bg-cyan-50 px-3 text-[10px] font-black uppercase tracking-[0.2em] text-cyan-700 lg:flex">
                <Sparkles className="h-3 w-3" />
                Live Lab
              </div>
            </div>
          </header>

          <div className="custom-scrollbar flex-1 overflow-y-auto p-4 md:p-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                transition={{ duration: 0.15 }}
                className="mx-auto max-w-6xl"
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

