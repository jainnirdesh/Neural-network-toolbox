import { useState } from "react";
import { Play, Sparkles, Info, Mail, ShieldCheck, CreditCard, ImageIcon, CheckCircle2, ArrowRight } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export function ClassificationPage() {
  const [learningRate, setLearningRate] = useState(0.01);
  const [epochs, setEpochs] = useState(50);
  const [hiddenLayers, setHiddenLayers] = useState(3);
  const [dataset, setDataset] = useState("iris");
  const [optimizer, setOptimizer] = useState("adam");
  const [isTraining, setIsTraining] = useState(false);

  const predictions = [
    { label: "Class A (Setosa)", probability: 0.82 },
    { label: "Class B (Versicolor)", probability: 0.12 },
    { label: "Class C (Virginica)", probability: 0.06 },
  ];

  const lossData = [
    { epoch: 0, loss: 2.3, accuracy: 0.33 },
    { epoch: 10, loss: 1.8, accuracy: 0.52 },
    { epoch: 20, loss: 1.2, accuracy: 0.68 },
    { epoch: 30, loss: 0.7, accuracy: 0.81 },
    { epoch: 40, loss: 0.4, accuracy: 0.89 },
    { epoch: 50, loss: 0.2, accuracy: 0.94 },
  ];

  const handleTrain = () => {
    setIsTraining(true);
    setTimeout(() => setIsTraining(false), 2000);
  };

  const useCases = [
    { icon: Mail, title: "Spam Detection", description: "Filter unwanted emails automatically" },
    { icon: ShieldCheck, title: "Disease Prediction", description: "Classify medical conditions from symptoms" },
    { icon: CreditCard, title: "Fraud Detection", description: "Identify suspicious transactions in real-time" },
    { icon: ImageIcon, title: "Image Classification", description: "Categorize images by content and context" },
  ];

  const workflowSteps = [
    { step: 1, title: "Input Features", description: "Feed your data into the network" },
    { step: 2, title: "Apply Weights", description: "Multiply inputs by learned parameters" },
    { step: 3, title: "Activation Functions", description: "Process data through non-linear transformations" },
    { step: 4, title: "Output Probabilities", description: "Generate class predictions" },
    { step: 5, title: "Backpropagation", description: "Update weights to minimize error" },
  ];

  return (
    <div className="min-h-full bg-slate-950 p-12">
      {/* Header */}
      <div className="mb-10">
        <h1 className="text-4xl font-bold text-white mb-3">Classification Model</h1>
        <p className="text-lg text-slate-400">Train and understand classification using neural networks</p>
      </div>

      {/* Educational Card - What is Classification? */}
      <div className="mb-10 bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/30 rounded-2xl p-8">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0">
            <Info className="w-6 h-6 text-white" />
          </div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-white mb-4">What is Classification?</h2>
            <p className="text-slate-300 mb-4 leading-relaxed">
              Classification is a supervised machine learning task where the model learns to categorize inputs into predefined classes.
              For example, an email can be classified as either <span className="text-indigo-400 font-semibold">Spam</span> or <span className="text-emerald-400 font-semibold">Not Spam</span>.
            </p>
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-indigo-400 flex-shrink-0 mt-0.5" />
                <p className="text-slate-300 text-sm">Neural networks learn patterns from labeled training data</p>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-indigo-400 flex-shrink-0 mt-0.5" />
                <p className="text-slate-300 text-sm">Multiple layers extract increasingly complex features</p>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-indigo-400 flex-shrink-0 mt-0.5" />
                <p className="text-slate-300 text-sm">Output layer produces probability for each class</p>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-indigo-400 flex-shrink-0 mt-0.5" />
                <p className="text-slate-300 text-sm">Model improves through iterative training and optimization</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Grid - Interactive Model */}
      <div className="grid grid-cols-2 gap-8 mb-10">
        {/* Left Panel - Controls */}
        <div className="space-y-6">
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Model Configuration</h2>

            {/* Learning Rate Slider */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Learning Rate: <span className="text-indigo-400">{learningRate}</span>
              </label>
              <input
                type="range"
                min="0.001"
                max="0.1"
                step="0.001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
              />
            </div>

            {/* Epochs Slider */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Epochs: <span className="text-indigo-400">{epochs}</span>
              </label>
              <input
                type="range"
                min="10"
                max="200"
                step="10"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
              />
            </div>

            {/* Hidden Layers Slider */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Hidden Layers: <span className="text-indigo-400">{hiddenLayers}</span>
              </label>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={hiddenLayers}
                onChange={(e) => setHiddenLayers(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
              />
            </div>

            {/* Dataset Dropdown */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">Dataset</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="iris">Iris Dataset</option>
                <option value="mnist">MNIST Digits</option>
                <option value="cifar10">CIFAR-10</option>
                <option value="custom">Custom Dataset</option>
              </select>
            </div>

            {/* Optimizer Dropdown */}
            <div className="mb-8">
              <label className="block text-sm font-medium text-slate-300 mb-3">Optimizer</label>
              <select
                value={optimizer}
                onChange={(e) => setOptimizer(e.target.value)}
                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="adam">Adam</option>
                <option value="sgd">SGD</option>
                <option value="rmsprop">RMSprop</option>
                <option value="adagrad">Adagrad</option>
              </select>
            </div>

            {/* Train Button */}
            <button
              onClick={handleTrain}
              disabled={isTraining}
              className="w-full px-6 py-4 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-xl font-semibold shadow-lg shadow-indigo-500/50 hover:shadow-indigo-500/70 hover:scale-[1.02] transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
            >
              {isTraining ? (
                <>
                  <Sparkles className="w-5 h-5 animate-spin" />
                  Training Model...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Run Training
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Panel - Results */}
        <div className="space-y-6">
          {/* Predictions */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Predictions</h2>
            <div className="space-y-4">
              {predictions.map((pred, idx) => (
                <div key={idx}>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium text-slate-300">{pred.label}</span>
                    <span className="text-sm font-bold text-indigo-400">{(pred.probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full transition-all duration-500 shadow-lg shadow-indigo-500/50"
                      style={{ width: `${pred.probability * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Loss Chart */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Training Loss</h2>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="epoch" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" }}
                  labelStyle={{ color: "#e2e8f0" }}
                />
                <Line type="monotone" dataKey="loss" stroke="#6366f1" strokeWidth={3} dot={{ fill: "#6366f1" }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Accuracy Chart */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Model Accuracy</h2>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="epoch" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" }}
                  labelStyle={{ color: "#e2e8f0" }}
                />
                <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={3} dot={{ fill: "#10b981" }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="mb-10">
        <h2 className="text-2xl font-bold text-white mb-6">How the Model Works</h2>
        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
          <div className="flex items-center gap-4 overflow-x-auto pb-4">
            {workflowSteps.map((step, idx) => (
              <div key={idx} className="flex items-center gap-4 flex-shrink-0">
                <div className="text-center min-w-[200px]">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center mx-auto mb-3 shadow-lg shadow-indigo-500/50">
                    <span className="text-white font-bold text-lg">{step.step}</span>
                  </div>
                  <h3 className="text-white font-semibold mb-2">{step.title}</h3>
                  <p className="text-sm text-slate-400">{step.description}</p>
                </div>
                {idx < workflowSteps.length - 1 && (
                  <ArrowRight className="w-6 h-6 text-slate-600 flex-shrink-0" />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Real-world Applications */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-6">Real-world Applications</h2>
        <div className="grid grid-cols-4 gap-6">
          {useCases.map((useCase, idx) => {
            const Icon = useCase.icon;
            return (
              <div
                key={idx}
                className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-6 hover:border-indigo-500/50 hover:shadow-lg hover:shadow-indigo-500/10 transition-all duration-300 group"
              >
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-600/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Icon className="w-6 h-6 text-indigo-400" />
                </div>
                <h3 className="text-white font-semibold mb-2">{useCase.title}</h3>
                <p className="text-sm text-slate-400">{useCase.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
