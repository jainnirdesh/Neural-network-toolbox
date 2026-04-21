import { useMemo, useState } from "react";
import { Play } from "lucide-react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function RegressionPage() {
  const [degree, setDegree] = useState(2);
  const [learningRate, setLearningRate] = useState(0.03);
  const [epochs, setEpochs] = useState(60);
  const [dataset, setDataset] = useState("Housing");
  const [runCount, setRunCount] = useState(0);

  const [areaSqft, setAreaSqft] = useState(1500);
  const [bedrooms, setBedrooms] = useState(3);
  const [locationScore, setLocationScore] = useState(7);

  const [stockCurrent, setStockCurrent] = useState(120);
  const [stockTrend, setStockTrend] = useState(0.02);
  const [stockDays, setStockDays] = useState(30);

  const [lastMonthSales, setLastMonthSales] = useState(25000);
  const [marketingSpend, setMarketingSpend] = useState(5000);
  const [seasonFactor, setSeasonFactor] = useState(1.0);

  const [yesterdayTemp, setYesterdayTemp] = useState(30);
  const [humidity, setHumidity] = useState(60);
  const [windSpeed, setWindSpeed] = useState(12);

  const housePrice = useMemo(() => {
    const base = areaSqft * 190 + bedrooms * 12000;
    const locationBoost = 1 + locationScore * 0.045;
    return base * locationBoost;
  }, [areaSqft, bedrooms, locationScore]);

  const stockForecast = useMemo(() => {
    return stockCurrent * Math.pow(1 + stockTrend, stockDays / 30);
  }, [stockCurrent, stockTrend, stockDays]);

  const salesForecast = useMemo(() => {
    const marketingImpact = marketingSpend * 1.6;
    return (lastMonthSales + marketingImpact) * seasonFactor;
  }, [lastMonthSales, marketingSpend, seasonFactor]);

  const predictedTemp = useMemo(() => {
    return yesterdayTemp + (humidity - 50) * 0.04 - windSpeed * 0.12;
  }, [yesterdayTemp, humidity, windSpeed]);

  const chartData = useMemo(() => {
    const isHousing = dataset === "Housing";
    const points = [];

    const seededNoise = (x: number, seed: number) => {
      const value = Math.sin(x * 12.9898 + seed * 78.233) * 43758.5453;
      return (value - Math.floor(value)) * 2 - 1;
    };

    for (let i = 0; i <= 40; i += 1) {
      const x = i / 4;
      const base = isHousing ? 2.1 * x + 5 : 0.75 * x * x + 2.8;
      const noise = seededNoise(x, runCount + (isHousing ? 4 : 8)) * (isHousing ? 2.3 : 3.1);
      const actual = base + noise;

      let fitted = 0;
      if (isHousing) {
        fitted = 4.8 + 2.15 * x;
        if (degree >= 2) fitted += 0.03 * x * x;
        if (degree >= 3) fitted += 0.002 * x * x * x;
      } else {
        fitted = 3 + 0.55 * x;
        if (degree >= 2) fitted += 0.72 * x * x;
        if (degree >= 3) fitted += 0.01 * x * x * x;
        if (degree >= 4) fitted -= 0.0007 * x * x * x * x;
        if (degree >= 5) fitted += 0.00002 * x * x * x * x * x;
      }

      points.push({ x: Number(x.toFixed(2)), actual: Number(actual.toFixed(2)), fitted: Number(fitted.toFixed(2)) });
    }

    return points;
  }, [dataset, degree, runCount]);

  const handleRun = () => {
    setRunCount((prev) => prev + 1);
  };

  return (
    <div className="min-h-full bg-slate-950 p-12">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-3">Regression Model</h1>
        <p className="text-slate-400">Predict continuous values using neural networks</p>
      </div>

      <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 mb-8">
        <h2 className="text-xl font-bold text-white mb-4">What is Regression?</h2>
        <ul className="list-disc pl-5 text-slate-300 space-y-2">
          <li>Regression predicts continuous numerical values.</li>
          <li>Example: House price prediction.</li>
          <li>Neural networks learn relationships between variables.</li>
        </ul>
      </div>

      <div className="grid grid-cols-2 gap-8">
        <div className="space-y-6">
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Interactive UI</h2>

            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Degree (1-5): <span className="text-emerald-400">{degree}</span>
              </label>
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={degree}
                onChange={(e) => setDegree(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Learning Rate: <span className="text-emerald-400">{learningRate.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min="0.001"
                max="1"
                step="0.01"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Epochs: <span className="text-emerald-400">{epochs}</span>
              </label>
              <input
                type="range"
                min="10"
                max="100"
                step="1"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
            </div>

            <div className="mb-8">
              <label className="block text-sm font-medium text-slate-300 mb-3">Dataset</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
              >
                <option value="Housing">Housing</option>
                <option value="Student Scores">Student Scores</option>
              </select>
            </div>

            <button
              onClick={handleRun}
              className="w-full px-6 py-4 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-xl font-semibold shadow-lg shadow-emerald-500/50 hover:shadow-emerald-500/70 hover:scale-[1.02] transition-all duration-200 flex items-center justify-center gap-3"
            >
              <Play className="w-5 h-5" />
              Run Training
            </button>

            <p className="text-xs text-slate-500 mt-3">
              Dummy {dataset} dataset visualization | lr={learningRate.toFixed(3)}, epochs={epochs}
            </p>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Graph / Output</h2>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="x" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" }}
                  labelStyle={{ color: "#e2e8f0" }}
                />
                <Scatter dataKey="actual" fill="#60a5fa" name="Data points" />
                <Line
                  type="monotone"
                  dataKey="fitted"
                  stroke="#f97316"
                  strokeWidth={2.5}
                  dot={false}
                  name="Fitted curve"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 mt-8">
        <h2 className="text-xl font-bold text-white mb-4">How it Works</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {["Input Variables", "Weights", "Activation", "Continuous Output"].map((step, idx) => (
            <div key={step} className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
              <p className="text-xs uppercase tracking-wide text-emerald-300 mb-1">Step {idx + 1}</p>
              <p className="text-white font-medium">{step}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 mt-8">
        <h2 className="text-xl font-bold text-white mb-4">Real-world Applications</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {["House price prediction", "Stock forecasting", "Sales prediction", "Temperature prediction"].map((item) => (
            <div key={item} className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
              <p className="text-white font-medium">{item}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 mt-8">
        <h2 className="text-xl font-bold text-white mb-4">Real-world Applications Lab</h2>
        <p className="text-slate-400 text-sm mb-6">Interactive regression demos with live prediction outputs.</p>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-5">
            <h3 className="text-white font-semibold mb-4">House Price Prediction</h3>
            <div className="space-y-3 text-sm">
              <label className="block text-slate-300">
                Area (sqft): {areaSqft}
                <input type="range" min="500" max="5000" step="50" value={areaSqft} onChange={(e) => setAreaSqft(parseInt(e.target.value))} className="w-full mt-2 accent-emerald-500" />
              </label>
              <label className="block text-slate-300">
                Bedrooms: {bedrooms}
                <input type="range" min="1" max="6" step="1" value={bedrooms} onChange={(e) => setBedrooms(parseInt(e.target.value))} className="w-full mt-2 accent-emerald-500" />
              </label>
              <label className="block text-slate-300">
                Location Score: {locationScore}/10
                <input type="range" min="1" max="10" step="1" value={locationScore} onChange={(e) => setLocationScore(parseInt(e.target.value))} className="w-full mt-2 accent-emerald-500" />
              </label>
            </div>
            <p className="text-emerald-300 font-bold mt-4">Predicted Price: ${housePrice.toFixed(0)}</p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-5">
            <h3 className="text-white font-semibold mb-4">Stock Forecasting</h3>
            <div className="space-y-3 text-sm">
              <label className="block text-slate-300">
                Current Price ($)
                <input type="number" value={stockCurrent} onChange={(e) => setStockCurrent(parseFloat(e.target.value || "0"))} className="w-full mt-2 px-3 py-2 bg-slate-900 border border-slate-700 rounded text-white" />
              </label>
              <label className="block text-slate-300">
                Monthly Trend ({(stockTrend * 100).toFixed(1)}%)
                <input type="range" min="-0.2" max="0.2" step="0.01" value={stockTrend} onChange={(e) => setStockTrend(parseFloat(e.target.value))} className="w-full mt-2 accent-emerald-500" />
              </label>
              <label className="block text-slate-300">
                Forecast Days: {stockDays}
                <input type="range" min="7" max="180" step="1" value={stockDays} onChange={(e) => setStockDays(parseInt(e.target.value))} className="w-full mt-2 accent-emerald-500" />
              </label>
            </div>
            <p className="text-emerald-300 font-bold mt-4">Forecast Price: ${stockForecast.toFixed(2)}</p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-5">
            <h3 className="text-white font-semibold mb-4">Sales Prediction</h3>
            <div className="space-y-3 text-sm">
              <label className="block text-slate-300">
                Last Month Sales ($)
                <input type="number" value={lastMonthSales} onChange={(e) => setLastMonthSales(parseFloat(e.target.value || "0"))} className="w-full mt-2 px-3 py-2 bg-slate-900 border border-slate-700 rounded text-white" />
              </label>
              <label className="block text-slate-300">
                Marketing Spend ($)
                <input type="number" value={marketingSpend} onChange={(e) => setMarketingSpend(parseFloat(e.target.value || "0"))} className="w-full mt-2 px-3 py-2 bg-slate-900 border border-slate-700 rounded text-white" />
              </label>
              <label className="block text-slate-300">
                Season Factor: {seasonFactor.toFixed(2)}
                <input type="range" min="0.7" max="1.3" step="0.01" value={seasonFactor} onChange={(e) => setSeasonFactor(parseFloat(e.target.value))} className="w-full mt-2 accent-emerald-500" />
              </label>
            </div>
            <p className="text-emerald-300 font-bold mt-4">Predicted Next Month Sales: ${salesForecast.toFixed(0)}</p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-5">
            <h3 className="text-white font-semibold mb-4">Temperature Prediction</h3>
            <div className="space-y-3 text-sm">
              <label className="block text-slate-300">
                Yesterday Temp (°C)
                <input type="number" value={yesterdayTemp} onChange={(e) => setYesterdayTemp(parseFloat(e.target.value || "0"))} className="w-full mt-2 px-3 py-2 bg-slate-900 border border-slate-700 rounded text-white" />
              </label>
              <label className="block text-slate-300">
                Humidity: {humidity}%
                <input type="range" min="10" max="100" step="1" value={humidity} onChange={(e) => setHumidity(parseInt(e.target.value))} className="w-full mt-2 accent-emerald-500" />
              </label>
              <label className="block text-slate-300">
                Wind Speed: {windSpeed} km/h
                <input type="range" min="0" max="50" step="1" value={windSpeed} onChange={(e) => setWindSpeed(parseInt(e.target.value))} className="w-full mt-2 accent-emerald-500" />
              </label>
            </div>
            <p className="text-emerald-300 font-bold mt-4">Predicted Today Temp: {predictedTemp.toFixed(1)}°C</p>
          </div>
        </div>
      </div>
    </div>
  );
}
