import { Brain, TrendingUp, Image, Network, Users, Target, Zap, Activity } from "lucide-react";
import { Link } from "react-router-dom";

export function HomePage() {
  const stats = [
    { label: "Models Trained", value: "1,247", icon: Brain, color: "from-indigo-500 to-purple-600" },
    { label: "Accuracy", value: "94.8%", icon: Target, color: "from-emerald-500 to-teal-600" },
    { label: "Active Users", value: "12,543", icon: Users, color: "from-blue-500 to-cyan-600" },
    { label: "Experiments Run", value: "3,891", icon: Zap, color: "from-orange-500 to-pink-600" },
  ];

  const features = [
    {
      title: "Classification",
      description: "Train powerful classification models with custom datasets and hyperparameters",
      icon: Brain,
      path: "/classification",
      color: "from-indigo-500 to-purple-600",
    },
    {
      title: "Regression",
      description: "Build regression models with advanced curve fitting and regularization techniques",
      icon: TrendingUp,
      path: "/regression",
      color: "from-emerald-500 to-teal-600",
    },
    {
      title: "Image Recognition",
      description: "Deploy state-of-the-art image recognition models with real-time predictions",
      icon: Image,
      path: "/image-recognition",
      color: "from-blue-500 to-cyan-600",
    },
    {
      title: "Neural Network Visualizer",
      description: "Visualize and animate neural network architectures in real-time",
      icon: Network,
      path: "/neural-network",
      color: "from-orange-500 to-pink-600",
    },
  ];

  return (
    <div className="min-h-full bg-slate-950">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/20 via-purple-500/20 to-pink-500/20"></div>
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSAxMCAwIEwgMCAwIDAgMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjAzKSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')] opacity-40"></div>

        <div className="relative px-12 py-24">
          <div className="max-w-4xl">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-500/10 border border-indigo-500/20 mb-6">
              <Activity className="w-4 h-4 text-indigo-400" />
              <span className="text-sm text-indigo-300">Advanced AI Research Platform</span>
            </div>
            <h1 className="text-6xl font-bold text-white mb-6 bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              NeuroVerse AI Lab
            </h1>
            <p className="text-2xl text-slate-300 mb-8">
              Interactive Neural Network Learning Platform
            </p>
            <p className="text-lg text-slate-400 max-w-2xl">
              Build, train, and deploy cutting-edge machine learning models with our comprehensive suite of AI tools. From classification to neural network visualization, unlock the power of artificial intelligence.
            </p>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="px-12 py-12">
        <div className="grid grid-cols-4 gap-6">
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <div
                key={index}
                className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-6 hover:border-slate-700 transition-all duration-300 hover:shadow-xl hover:shadow-indigo-500/10"
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${stat.color} flex items-center justify-center mb-4`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
                <div className="text-sm text-slate-400">{stat.label}</div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Features Section */}
      <section className="px-12 py-12">
        <h2 className="text-3xl font-bold text-white mb-8">AI Capabilities</h2>
        <div className="grid grid-cols-2 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Link
                key={index}
                to={feature.path}
                className="group bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 hover:border-slate-700 transition-all duration-300 hover:shadow-2xl hover:shadow-indigo-500/20 hover:-translate-y-1"
              >
                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <Icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3 group-hover:text-indigo-400 transition-colors">
                  {feature.title}
                </h3>
                <p className="text-slate-400 leading-relaxed">
                  {feature.description}
                </p>
              </Link>
            );
          })}
        </div>
      </section>
    </div>
  );
}
