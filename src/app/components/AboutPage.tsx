import { Brain, Zap, Shield, Users } from "lucide-react";

export function AboutPage() {
  const features = [
    {
      icon: Brain,
      title: "Advanced AI Models",
      description: "State-of-the-art machine learning algorithms and neural network architectures",
    },
    {
      icon: Zap,
      title: "Real-time Training",
      description: "Interactive model training with live feedback and visualization",
    },
    {
      icon: Shield,
      title: "Enterprise Security",
      description: "Bank-level encryption and secure data processing for all your experiments",
    },
    {
      icon: Users,
      title: "Collaborative Platform",
      description: "Share models, datasets, and insights with your team seamlessly",
    },
  ];

  return (
    <div className="min-h-full bg-slate-950 p-12">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-16">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 mb-6">
            <Brain className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-5xl font-bold text-white mb-4">About NeuroVerse AI Lab</h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            The most advanced interactive platform for machine learning research and education
          </p>
        </div>

        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-12 mb-12">
          <h2 className="text-2xl font-bold text-white mb-6">Our Mission</h2>
          <p className="text-slate-300 leading-relaxed mb-4">
            NeuroVerse AI Lab is dedicated to democratizing artificial intelligence by providing an intuitive,
            powerful platform for researchers, students, and professionals to explore the frontiers of machine learning.
          </p>
          <p className="text-slate-300 leading-relaxed">
            We believe that AI education should be accessible, interactive, and inspiring. Our platform combines
            cutting-edge technology with user-friendly design to create an unparalleled learning experience.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-6 mb-12">
          {features.map((feature, idx) => {
            const Icon = feature.icon;
            return (
              <div
                key={idx}
                className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 hover:border-slate-700 transition-all duration-300"
              >
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center mb-4">
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">{feature.title}</h3>
                <p className="text-slate-400">{feature.description}</p>
              </div>
            );
          })}
        </div>

        <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl p-12 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">Join Our Community</h2>
          <p className="text-indigo-100 mb-8 max-w-2xl mx-auto">
            Connect with thousands of AI researchers and enthusiasts. Share your discoveries, learn from others,
            and push the boundaries of what's possible with artificial intelligence.
          </p>
          <button className="px-8 py-4 bg-white text-indigo-600 rounded-xl font-semibold shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-200">
            Get Started Free
          </button>
        </div>
      </div>
    </div>
  );
}
