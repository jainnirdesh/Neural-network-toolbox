import { useEffect, useRef, useState } from "react";
import { Upload, Image as ImageIcon } from "lucide-react";

type Prediction = {
  label: string;
  confidence: number;
};

export function ImageRecognitionPage() {
  const [model, setModel] = useState("CNN");
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [status, setStatus] = useState("Upload an image and run classification.");
  const [isRunning, setIsRunning] = useState(false);

  const imageRef = useRef<HTMLImageElement | null>(null);
  const modelRef = useRef<any>(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [faceStatus, setFaceStatus] = useState("Camera not started.");
  const [faceCount, setFaceCount] = useState(0);
  const [securityStatus, setSecurityStatus] = useState("Monitoring stopped.");
  const [motionScore, setMotionScore] = useState(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const analysisCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const faceModelRef = useRef<any>(null);
  const previousFrameRef = useRef<ImageData | null>(null);
  const faceIntervalRef = useRef<number | null>(null);
  const motionIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (uploadedImage) {
        URL.revokeObjectURL(uploadedImage);
      }

      if (faceIntervalRef.current) {
        window.clearInterval(faceIntervalRef.current);
      }
      if (motionIntervalRef.current) {
        window.clearInterval(motionIntervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [uploadedImage]);

  const onUpload: React.ChangeEventHandler<HTMLInputElement> = (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (uploadedImage) {
      URL.revokeObjectURL(uploadedImage);
    }

    const objectUrl = URL.createObjectURL(file);
    setUploadedImage(objectUrl);
    setPredictions([]);
    setStatus("Image uploaded. Click Run Classification.");
  };

  const buildSimulatedPredictions = (): Prediction[] => {
    return [
      { label: "Pattern A", confidence: 0.62 },
      { label: "Pattern B", confidence: 0.24 },
      { label: "Pattern C", confidence: 0.14 },
    ];
  };

  const runClassification = async () => {
    if (!uploadedImage || !imageRef.current) {
      setStatus("Please upload an image first.");
      return;
    }

    setIsRunning(true);

    try {
      if (model === "Pretrained") {
        setStatus("Loading pretrained model...");
        const tf = await import("@tensorflow/tfjs");
        const mobilenet = await import("@tensorflow-models/mobilenet");

        await tf.ready();

        if (!modelRef.current) {
          modelRef.current = await mobilenet.load({ version: 2, alpha: 1.0 });
        }

        setStatus("Running inference...");
        const results = await modelRef.current.classify(imageRef.current, 3);
        const top3: Prediction[] = results.map((item: { className: string; probability: number }) => ({
          label: item.className,
          confidence: item.probability,
        }));

        setPredictions(top3);
        setStatus("Pretrained inference complete.");
      } else {
        const simulated = buildSimulatedPredictions();
        setPredictions(simulated);
        setStatus("CNN mode is simulated in this demo (no local training yet).");
      }
    } catch (error) {
      setStatus("Could not classify image. Try another image or refresh the page.");
      setPredictions([]);
    } finally {
      setIsRunning(false);
    }
  };

  const stopRealWorldLab = () => {
    if (faceIntervalRef.current) {
      window.clearInterval(faceIntervalRef.current);
      faceIntervalRef.current = null;
    }
    if (motionIntervalRef.current) {
      window.clearInterval(motionIntervalRef.current);
      motionIntervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    previousFrameRef.current = null;
    setCameraActive(false);
    setFaceStatus("Camera stopped.");
    setSecurityStatus("Monitoring stopped.");
    setFaceCount(0);
    setMotionScore(0);
  };

  const startFaceDetectionLoop = async () => {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    const tf = await import("@tensorflow/tfjs");
    const blazeface = await import("@tensorflow-models/blazeface");
    await tf.ready();

    if (!faceModelRef.current) {
      faceModelRef.current = await blazeface.load();
    }

    if (faceIntervalRef.current) {
      window.clearInterval(faceIntervalRef.current);
    }

    faceIntervalRef.current = window.setInterval(async () => {
      if (!videoRef.current || videoRef.current.readyState < 2 || !faceModelRef.current) {
        return;
      }

      try {
        const faces = await faceModelRef.current.estimateFaces(videoRef.current, false);
        setFaceCount(faces.length);
        if (faces.length > 0) {
          setFaceStatus(`${faces.length} face(s) detected`);
        } else {
          setFaceStatus("No face detected");
        }
      } catch {
        setFaceStatus("Face detection temporarily unavailable.");
      }
    }, 700);
  };

  const startSecurityMotionLoop = () => {
    if (motionIntervalRef.current) {
      window.clearInterval(motionIntervalRef.current);
    }

    motionIntervalRef.current = window.setInterval(() => {
      const video = videoRef.current;
      const canvas = analysisCanvasRef.current;
      if (!video || !canvas || video.readyState < 2) {
        return;
      }

      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) {
        return;
      }

      const width = 160;
      const height = 120;
      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(video, 0, 0, width, height);
      const frame = ctx.getImageData(0, 0, width, height);

      if (previousFrameRef.current) {
        const prev = previousFrameRef.current.data;
        const cur = frame.data;
        let sum = 0;
        let samples = 0;

        for (let i = 0; i < cur.length; i += 16) {
          const dr = Math.abs(cur[i] - prev[i]);
          const dg = Math.abs(cur[i + 1] - prev[i + 1]);
          const db = Math.abs(cur[i + 2] - prev[i + 2]);
          sum += (dr + dg + db) / 3;
          samples += 1;
        }

        const avgDiff = samples > 0 ? sum / samples : 0;
        const normalized = Math.min(1, avgDiff / 60);
        setMotionScore(normalized);

        if (normalized > 0.2) {
          setSecurityStatus("Motion alert: activity detected");
        } else {
          setSecurityStatus("Area stable: low activity");
        }
      }

      previousFrameRef.current = frame;
    }, 350);
  };

  const startRealWorldLab = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) {
        return;
      }

      video.srcObject = stream;
      await video.play();

      setCameraActive(true);
      setFaceStatus("Starting face detection...");
      setSecurityStatus("Starting security monitor...");

      await startFaceDetectionLoop();
      startSecurityMotionLoop();
    } catch {
      setCameraActive(false);
      setFaceStatus("Camera access denied or unavailable.");
      setSecurityStatus("Could not start monitoring.");
    }
  };

  return (
    <div className="min-h-full bg-slate-950 p-12">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-3">Image Recognition</h1>
        <p className="text-slate-400">Classify images using neural networks</p>
      </div>

      <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 mb-8">
        <h2 className="text-xl font-bold text-white mb-4">What is Image Recognition?</h2>
        <div className="space-y-3 text-slate-300 leading-relaxed">
          <p>Image recognition means identifying objects in images.</p>
          <p>It commonly uses Convolutional Neural Networks (CNN).</p>
          <p>Example: detecting handwritten digits or animals in a photo.</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-8">
        {/* Left Panel - Upload & Controls */}
        <div className="space-y-6">
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Upload Section</h2>

            <label
              htmlFor="image-upload-input"
              className="block border-2 border-dashed border-slate-700 rounded-xl p-12 text-center hover:border-blue-500 hover:bg-slate-800/50 transition-all cursor-pointer"
            >
              <Upload className="w-12 h-12 text-slate-500 mx-auto mb-4" />
              <p className="text-slate-300 font-medium mb-2">Drag and drop image upload</p>
              <p className="text-sm text-slate-500">Supports JPG, PNG, JPEG</p>
            </label>
            <input
              id="image-upload-input"
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              onChange={onUpload}
              className="hidden"
            />
          </div>

          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Controls</h2>

            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">Model Selector</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="CNN">CNN</option>
                <option value="Pretrained">Pretrained</option>
              </select>
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Confidence Threshold: <span className="text-blue-400">{(confidenceThreshold * 100).toFixed(0)}%</span>
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
            </div>

            <button
              onClick={runClassification}
              disabled={isRunning}
              className="w-full px-6 py-4 bg-gradient-to-r from-blue-500 to-cyan-600 text-white rounded-xl font-semibold shadow-lg shadow-blue-500/50 hover:shadow-blue-500/70 hover:scale-[1.02] transition-all duration-200 flex items-center justify-center gap-3 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              <ImageIcon className="w-5 h-5" />
              {isRunning ? "Classifying..." : "Run Classification"}
            </button>

            <p className="text-xs text-slate-500 mt-3">
              {model === "Pretrained"
                ? "Pretrained mode uses MobileNet for real predictions."
                : "CNN mode uses dummy predictions until a trainable model is integrated."}
            </p>
            <p className="text-xs text-slate-400 mt-2">Status: {status}</p>
          </div>
        </div>

        {/* Right Panel - Results */}
        <div className="space-y-6">
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Output</h2>
            {uploadedImage ? (
              <div className="aspect-square bg-slate-800 rounded-xl overflow-hidden">
                <img
                  ref={imageRef}
                  src={uploadedImage}
                  alt="Uploaded image"
                  className="w-full h-full object-cover"
                />
              </div>
            ) : (
              <div className="aspect-square bg-slate-800 rounded-xl flex items-center justify-center">
                <div className="text-center">
                  <ImageIcon className="w-16 h-16 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-500">No image uploaded yet</p>
                </div>
              </div>
            )}
          </div>

          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Top-3 Predictions</h2>
            <div className="space-y-4">
              {predictions.length === 0 ? (
                <p className="text-slate-500 text-sm">No predictions yet. Upload image and run classification.</p>
              ) : (
                predictions.map((pred, idx) => (
                  <div key={idx}>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium text-slate-300">
                        {idx + 1}. {pred.label}
                      </span>
                      <span className="text-sm font-bold text-blue-400">
                        {(pred.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-cyan-600 rounded-full transition-all duration-500 shadow-lg shadow-blue-500/50"
                        style={{ width: `${pred.confidence * 100}%` }}
                      ></div>
                    </div>
                    {pred.confidence < confidenceThreshold && (
                      <p className="text-xs text-slate-500 mt-1">Below threshold ({confidenceThreshold.toFixed(2)})</p>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h2 className="text-xl font-bold text-white mb-6">Simulated Heatmap (Grad-CAM style)</h2>
            <div className="relative aspect-square rounded-xl overflow-hidden border border-slate-800">
              {uploadedImage ? (
                <img src={uploadedImage} alt="Heatmap base" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full bg-slate-900"></div>
              )}
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_35%_35%,rgba(59,130,246,0.55),transparent_35%),radial-gradient(circle_at_65%_65%,rgba(236,72,153,0.45),transparent_30%),radial-gradient(circle_at_55%_30%,rgba(34,211,238,0.4),transparent_25%)]"></div>
              <div className="absolute bottom-3 left-3 text-xs text-white/90 bg-black/40 px-2 py-1 rounded">
                High-attention regions
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 mt-8">
        <h2 className="text-xl font-bold text-white mb-4">How it Works</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {["Image", "Filters", "Feature Maps", "Classification"].map((step, idx) => (
            <div key={step} className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
              <p className="text-xs uppercase tracking-wide text-blue-300 mb-1">Step {idx + 1}</p>
              <p className="text-white font-medium">{step}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 mt-8">
        <h2 className="text-xl font-bold text-white mb-4">Real-world Applications</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {["Face recognition", "Medical imaging", "Self-driving cars", "Security systems"].map((item) => (
            <div key={item} className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
              <p className="text-white font-medium">{item}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 mt-8">
        <h2 className="text-xl font-bold text-white mb-4">Real-world Applications Lab</h2>
        <p className="text-slate-400 text-sm mb-6">
          Run live demos for Face Recognition (face presence) and Security Systems (motion alert).
        </p>

        <div className="flex flex-wrap gap-3 mb-6">
          <button
            onClick={startRealWorldLab}
            disabled={cameraActive}
            className="px-4 py-2 rounded-lg bg-emerald-600 text-white disabled:opacity-60 disabled:cursor-not-allowed"
          >
            Start Live Demo
          </button>
          <button
            onClick={stopRealWorldLab}
            disabled={!cameraActive}
            className="px-4 py-2 rounded-lg bg-slate-700 text-white disabled:opacity-60 disabled:cursor-not-allowed"
          >
            Stop Demo
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-950/60 border border-slate-800 rounded-xl p-4">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full aspect-video rounded-lg bg-slate-900 object-cover"
            />
            <canvas ref={analysisCanvasRef} className="hidden" />
          </div>

          <div className="space-y-4">
            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
              <p className="text-xs uppercase tracking-wide text-cyan-300 mb-1">Face Recognition</p>
              <p className="text-white font-medium">{faceStatus}</p>
              <p className="text-slate-400 text-sm mt-1">Detected Faces: {faceCount}</p>
            </div>

            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
              <p className="text-xs uppercase tracking-wide text-amber-300 mb-1">Security System</p>
              <p className="text-white font-medium">{securityStatus}</p>
              <p className="text-slate-400 text-sm mt-1">Motion Score: {(motionScore * 100).toFixed(1)}%</p>
              <div className="w-full h-2 bg-slate-800 rounded-full mt-3 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 via-yellow-500 to-red-500 transition-all duration-200"
                  style={{ width: `${Math.min(100, motionScore * 100)}%` }}
                ></div>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
              <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Note</p>
              <p className="text-slate-300 text-sm">
                This is an on-device demo. Face module detects presence/count, and security module detects motion.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
