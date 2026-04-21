import { useMemo, useRef, useState } from "react";
import { Play, Sparkles } from "lucide-react";

declare global {
  interface Window {
    SpeechRecognition?: new () => {
      continuous: boolean;
      interimResults: boolean;
      lang: string;
      onstart: (() => void) | null;
      onresult: ((event: any) => void) | null;
      onerror: ((event: any) => void) | null;
      onend: (() => void) | null;
      start: () => void;
      stop: () => void;
    };
    webkitSpeechRecognition?: new () => {
      continuous: boolean;
      interimResults: boolean;
      lang: string;
      onstart: (() => void) | null;
      onresult: ((event: any) => void) | null;
      onerror: ((event: any) => void) | null;
      onend: (() => void) | null;
      start: () => void;
      stop: () => void;
    };
  }
}

type AppMessage = { role: "user" | "assistant"; text: string };

const recommendationItems = [
  { name: "Spotify - Discover Weekly", domain: "Music", goal: "Explore" },
  { name: "YouTube - Suggested Videos", domain: "Video", goal: "Explore" },
  { name: "Netflix - Personalized Rows", domain: "Movies", goal: "Watch" },
  { name: "Amazon - Product Picks", domain: "Shopping", goal: "Buy" },
  { name: "Coursera - Course Suggestions", domain: "Learning", goal: "Learn" },
  { name: "Duolingo - Practice Plan", domain: "Learning", goal: "Practice" },
  { name: "Medium - Reading Feed", domain: "Articles", goal: "Read" },
  { name: "Swiggy - Meal Suggestions", domain: "Food", goal: "Order" },
];

function clampNeuronValue(value: string, min: number, max: number, fallback: number): number {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function getAssistantReply(input: string): string {
  const text = input.toLowerCase();

  if (text.includes("hello") || text.includes("hi")) {
    return "Hello! I can help with neural networks, training setup, and model debugging.";
  }
  if (text.includes("learning rate")) {
    return "Learning rate controls how big each update step is. Start small (0.001 to 0.01) for stable training.";
  }
  if (text.includes("overfit") || text.includes("overfitting")) {
    return "To reduce overfitting, try dropout, early stopping, and more validation data.";
  }
  if (text.includes("hidden") || text.includes("layer")) {
    return "Hidden layers help learn complex patterns. More layers can increase power, but also training complexity.";
  }
  if (text.includes("recommend") || text.includes("application")) {
    return "Neural networks are widely used in speech recognition, recommendation systems, and AI assistants.";
  }

  return "I understood your question. For this demo, ask about learning rate, hidden layers, or overfitting for targeted guidance.";
}

export function NeuralNetworkVisualizerPage() {
  const [inputNeurons, setInputNeurons] = useState(4);
  const [hiddenLayer1, setHiddenLayer1] = useState(4);
  const [hiddenLayer2, setHiddenLayer2] = useState(4);
  const [hiddenLayer3, setHiddenLayer3] = useState(3);
  const [outputNeurons, setOutputNeurons] = useState(2);
  const [isAnimating, setIsAnimating] = useState(false);
  const [activeConnections, setActiveConnections] = useState(0);

  const [isListening, setIsListening] = useState(false);
  const [speechText, setSpeechText] = useState("");
  const [speechStatus, setSpeechStatus] = useState("Click start to capture voice.");
  const recognitionRef = useRef<null | { stop: () => void }>(null);

  const [domain, setDomain] = useState("Learning");
  const [goal, setGoal] = useState("Learn");

  const [assistantInput, setAssistantInput] = useState("");
  const [assistantMessages, setAssistantMessages] = useState<AppMessage[]>([
    {
      role: "assistant",
      text: "Hi, I am your AI assistant demo. Ask me about learning rate, layers, or overfitting.",
    },
  ]);

  const handleAnimate = () => {
    const total = totalConnections;
    if (total <= 0) return;

    setIsAnimating(true);
    setActiveConnections(0);

    let step = 0;
    const interval = setInterval(() => {
      step += Math.max(1, Math.ceil(total / 28));
      if (step >= total) {
        setActiveConnections(total);
        setIsAnimating(false);
        clearInterval(interval);
        return;
      }
      setActiveConnections(step);
    }, 90);
  };

  const renderLayer = (neurons: number, layerIndex: number, label: string) => {
    const totalLayers = 5;
    const layerX = (layerIndex / (totalLayers - 1)) * 100;

    return (
      <div
        className="absolute flex flex-col justify-center gap-8"
        style={{
          left: `${layerX}%`,
          top: "50%",
          transform: "translate(-50%, -50%)",
        }}
      >
        <div className="text-xs text-slate-500 text-center mb-2">{label}</div>
        {Array.from({ length: neurons }).map((_, i) => (
          <div
            key={i}
            className={`w-12 h-12 rounded-full border-2 flex items-center justify-center transition-all duration-300 ${
              isAnimating
                ? "bg-gradient-to-br from-indigo-500 to-purple-600 border-indigo-400 shadow-lg shadow-indigo-500/50 animate-pulse"
                : "bg-slate-800 border-slate-600"
            }`}
          >
            <div className="w-2 h-2 rounded-full bg-white"></div>
          </div>
        ))}
      </div>
    );
  };

  const renderConnections = () => {
    const layers = [inputNeurons, hiddenLayer1, hiddenLayer2, hiddenLayer3, outputNeurons];
    const connections = [];
    const totalLayers = 5;

    for (let l = 0; l < layers.length - 1; l++) {
      const fromNeurons = layers[l];
      const toNeurons = layers[l + 1];

      for (let i = 0; i < fromNeurons; i++) {
        for (let j = 0; j < toNeurons; j++) {
          const x1 = (l / (totalLayers - 1)) * 100;
          const x2 = ((l + 1) / (totalLayers - 1)) * 100;

          connections.push(
            <line
              key={`${l}-${i}-${j}`}
              x1={`${x1}%`}
              y1={`${(i + 1) * (100 / (fromNeurons + 1))}%`}
              x2={`${x2}%`}
              y2={`${(j + 1) * (100 / (toNeurons + 1))}%`}
              className={`transition-all duration-300 ${
                connections.length + 1 <= activeConnections
                  ? "stroke-indigo-400 opacity-90"
                  : "stroke-slate-700 opacity-30"
              }`}
              strokeWidth={connections.length + 1 <= activeConnections ? "2" : "1"}
            />
          );
        }
      }
    }

    return connections;
  };

  const totalConnections =
    inputNeurons * hiddenLayer1 +
    hiddenLayer1 * hiddenLayer2 +
    hiddenLayer2 * hiddenLayer3 +
    hiddenLayer3 * outputNeurons;

  const numberInputClass =
    "w-full bg-slate-950 border border-slate-700 text-white rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500";

  const isSpeechSupported =
    typeof window !== "undefined" &&
    (typeof window.SpeechRecognition !== "undefined" || typeof window.webkitSpeechRecognition !== "undefined");

  const startSpeechRecognition = () => {
    if (!isSpeechSupported || isListening) {
      return;
    }

    const SpeechCtor = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!SpeechCtor) {
      return;
    }

    const recognition = new SpeechCtor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      setIsListening(true);
      setSpeechStatus("Listening... speak now.");
    };

    recognition.onresult = (event: any) => {
      let transcript = "";
      for (let i = 0; i < event.results.length; i += 1) {
        transcript += event.results[i][0].transcript;
      }
      setSpeechText(transcript.trim());
    };

    recognition.onerror = (event: any) => {
      setSpeechStatus(`Speech error: ${event.error}`);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      setSpeechStatus("Stopped. You can start again.");
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  const stopSpeechRecognition = () => {
    recognitionRef.current?.stop();
  };

  const recommendations = useMemo(() => {
    const scored = recommendationItems
      .map((item) => {
        let score = 0;
        if (item.domain === domain) score += 0.6;
        if (item.goal === goal) score += 0.4;
        return { ...item, score };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    return scored;
  }, [domain, goal]);

  const sendAssistantMessage = () => {
    const trimmed = assistantInput.trim();
    if (!trimmed) {
      return;
    }

    const reply = getAssistantReply(trimmed);
    setAssistantMessages((prev) => [
      ...prev,
      { role: "user", text: trimmed },
      { role: "assistant", text: reply },
    ]);
    setAssistantInput("");
  };

  return (
    <div className="min-h-full bg-slate-950 p-12">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-3">Neural Network Visualizer</h1>
        <p className="text-slate-400">Understand how neural networks process data</p>
      </div>

      <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 mb-8">
        <h2 className="text-xl font-bold text-white mb-4">What is a Neural Network?</h2>
        <div className="space-y-3 text-slate-300 leading-relaxed">
          <p>Neural networks are inspired by the way the human brain processes information.</p>
          <p>They consist of connected layers: Input, Hidden, and Output.</p>
          <p>They learn patterns from data and use those patterns to make predictions.</p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-8">
        {/* Left Panel - Controls */}
        <div className="space-y-6">
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Controls</h2>

            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Input Neurons: <span className="text-orange-400">{inputNeurons}</span>
              </label>
              <input
                type="number"
                min="1"
                max="10"
                step="1"
                value={inputNeurons}
                onChange={(e) => setInputNeurons(clampNeuronValue(e.target.value, 1, 10, inputNeurons))}
                className={numberInputClass}
              />
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Hidden Layer 1: <span className="text-orange-400">{hiddenLayer1}</span>
              </label>
              <input
                type="number"
                min="1"
                max="10"
                step="1"
                value={hiddenLayer1}
                onChange={(e) => setHiddenLayer1(clampNeuronValue(e.target.value, 1, 10, hiddenLayer1))}
                className={numberInputClass}
              />
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Hidden Layer 2: <span className="text-orange-400">{hiddenLayer2}</span>
              </label>
              <input
                type="number"
                min="1"
                max="10"
                step="1"
                value={hiddenLayer2}
                onChange={(e) => setHiddenLayer2(clampNeuronValue(e.target.value, 1, 10, hiddenLayer2))}
                className={numberInputClass}
              />
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Hidden Layer 3: <span className="text-orange-400">{hiddenLayer3}</span>
              </label>
              <input
                type="number"
                min="1"
                max="10"
                step="1"
                value={hiddenLayer3}
                onChange={(e) => setHiddenLayer3(clampNeuronValue(e.target.value, 1, 10, hiddenLayer3))}
                className={numberInputClass}
              />
            </div>

            <div className="mb-8">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Output Neurons: <span className="text-orange-400">{outputNeurons}</span>
              </label>
              <input
                type="number"
                min="1"
                max="5"
                step="1"
                value={outputNeurons}
                onChange={(e) => setOutputNeurons(clampNeuronValue(e.target.value, 1, 5, outputNeurons))}
                className={numberInputClass}
              />
            </div>

            <button
              onClick={handleAnimate}
              disabled={isAnimating}
              className="w-full px-6 py-4 bg-gradient-to-r from-orange-500 to-pink-600 text-white rounded-xl font-semibold shadow-lg shadow-orange-500/50 hover:shadow-orange-500/70 hover:scale-[1.02] transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
            >
              {isAnimating ? (
                <>
                  <Sparkles className="w-5 h-5 animate-spin" />
                  Animating...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Animate Network
                </>
              )}
            </button>
          </div>

          {/* Network Stats */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Network Stats</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Total Layers</span>
                <span className="text-xl font-bold text-orange-400">5</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Total Neurons</span>
                <span className="text-xl font-bold text-orange-400">
                  {inputNeurons + hiddenLayer1 + hiddenLayer2 + hiddenLayer3 + outputNeurons}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Parameters</span>
                <span className="text-xl font-bold text-orange-400">
                  {totalConnections}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Center and Right - Canvas */}
        <div className="col-span-2">
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 h-full">
            <h2 className="text-xl font-bold text-white mb-6">Network Visualization</h2>
            <div className="relative w-full h-[600px] bg-slate-950/50 rounded-xl border border-slate-800 overflow-hidden">
              <svg className="absolute inset-0 w-full h-full">
                {renderConnections()}
              </svg>
              {renderLayer(inputNeurons, 0, "Input")}
              {renderLayer(hiddenLayer1, 1, "Hidden 1")}
              {renderLayer(hiddenLayer2, 2, "Hidden 2")}
              {renderLayer(hiddenLayer3, 3, "Hidden 3")}
              {renderLayer(outputNeurons, 4, "Output")}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-8">
        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 lg:col-span-2">
          <h2 className="text-xl font-bold text-white mb-4">How it Works</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              "Input",
              "Weighted Sum",
              "Activation",
              "Output",
            ].map((step, i) => (
              <div key={step} className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
                <p className="text-xs uppercase tracking-wide text-indigo-300 mb-2">Step {i + 1}</p>
                <p className="text-white font-medium">{step}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
          <h2 className="text-xl font-bold text-white mb-4">Key Terms</h2>
          <div className="space-y-3 text-slate-300 text-sm">
            <p><span className="text-white font-semibold">Neuron:</span> Basic processing unit that receives and sends signals.</p>
            <p><span className="text-white font-semibold">Weight:</span> Connection strength between neurons.</p>
            <p><span className="text-white font-semibold">Bias:</span> Extra value added to adjust neuron output.</p>
            <p><span className="text-white font-semibold">Activation Function:</span> Rule that decides how strongly a neuron fires.</p>
          </div>
        </div>
      </div>

      <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 mt-8">
        <h2 className="text-xl font-bold text-white mb-4">Real-world Applications</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {["Speech Recognition", "Recommendation Systems", "AI Assistants"].map((item) => (
            <div key={item} className="bg-slate-950/60 border border-slate-800 rounded-xl p-5">
              <p className="text-white font-medium">{item}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8 mt-8">
        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
          <h3 className="text-lg font-bold text-white mb-4">Speech Recognition Demo</h3>
          <p className="text-slate-400 text-sm mb-4">Browser speech-to-text using Web Speech API.</p>

          <div className="flex gap-3 mb-4">
            <button
              onClick={startSpeechRecognition}
              disabled={!isSpeechSupported || isListening}
              className="px-4 py-2 rounded-lg bg-indigo-600 text-white disabled:opacity-50"
            >
              Start
            </button>
            <button
              onClick={stopSpeechRecognition}
              disabled={!isListening}
              className="px-4 py-2 rounded-lg bg-slate-700 text-white disabled:opacity-50"
            >
              Stop
            </button>
          </div>

          {!isSpeechSupported && (
            <p className="text-amber-300 text-sm mb-3">Your browser does not support Web Speech API.</p>
          )}

          <p className="text-xs text-slate-400 mb-3">Status: {speechStatus}</p>
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 min-h-[120px] text-slate-200 text-sm">
            {speechText || "Transcript will appear here..."}
          </div>
        </div>

        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
          <h3 className="text-lg font-bold text-white mb-4">Recommendation System Demo</h3>
          <p className="text-slate-400 text-sm mb-4">Simple ranking based on user preference vectors.</p>

          <label className="block text-sm text-slate-300 mb-2">Preferred Domain</label>
          <select
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            className="w-full mb-4 bg-slate-950 border border-slate-700 text-white rounded-lg px-3 py-2"
          >
            {["Learning", "Music", "Video", "Movies", "Shopping", "Articles", "Food"].map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>

          <label className="block text-sm text-slate-300 mb-2">Current Goal</label>
          <select
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            className="w-full mb-4 bg-slate-950 border border-slate-700 text-white rounded-lg px-3 py-2"
          >
            {["Learn", "Explore", "Watch", "Buy", "Practice", "Read", "Order"].map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>

          <div className="space-y-2">
            {recommendations.map((item, idx) => (
              <div key={item.name} className="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
                <p className="text-white text-sm font-medium">
                  {idx + 1}. {item.name}
                </p>
                <p className="text-xs text-slate-400">Match score: {(item.score * 100).toFixed(0)}%</p>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
          <h3 className="text-lg font-bold text-white mb-4">AI Assistant Demo</h3>
          <p className="text-slate-400 text-sm mb-4">Rule-based assistant to simulate Q&A behavior.</p>

          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 h-[200px] overflow-auto space-y-2 mb-3">
            {assistantMessages.map((msg, idx) => (
              <div
                key={`${msg.role}-${idx}`}
                className={`text-sm p-2 rounded-lg ${
                  msg.role === "assistant"
                    ? "bg-indigo-900/30 border border-indigo-800 text-indigo-100"
                    : "bg-slate-800 border border-slate-700 text-white"
                }`}
              >
                <span className="font-semibold mr-2">{msg.role === "assistant" ? "Assistant:" : "You:"}</span>
                <span>{msg.text}</span>
              </div>
            ))}
          </div>

          <div className="flex gap-2">
            <input
              value={assistantInput}
              onChange={(e) => setAssistantInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  sendAssistantMessage();
                }
              }}
              placeholder="Ask about learning rate, layers, overfitting..."
              className="flex-1 bg-slate-950 border border-slate-700 text-white rounded-lg px-3 py-2"
            />
            <button
              onClick={sendAssistantMessage}
              className="px-4 py-2 rounded-lg bg-indigo-600 text-white"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
