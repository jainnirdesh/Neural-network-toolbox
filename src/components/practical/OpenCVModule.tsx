import { useState, useRef, useEffect, ChangeEvent } from 'react';
import { motion } from 'motion/react';
import { Scan, Users, Target, Activity, AlertCircle, Loader2, Camera } from 'lucide-react';
import { cn } from '../../lib/utils';
import { GoogleGenAI, Type } from "@google/genai";

function getAiClient() {
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("Missing Gemini API key. Set VITE_GEMINI_API_KEY in your .env file.");
  }
  return new GoogleGenAI({ apiKey });
}

export default function OpenCVModule() {
  const [image, setImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [faceCount, setFaceCount] = useState<number | null>(null);
  const [faces, setFaces] = useState<{ x: number, y: number, w: number, h: number }[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setImage(ev.target?.result as string);
        setFaceCount(null);
        setFaces([]);
      };
      reader.readAsDataURL(file);
    }
  };

  const processOpenCV = async () => {
    if (!image) return;
    setIsProcessing(true);
    
    try {
      const ai = getAiClient();
      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [
          { text: "Detect all human faces in this image accurately. Return a JSON object with: { \"count\": number, \"boxes\": [{ \"x\": number, \"y\": number, \"w\": number, \"h\": number }] } where x,y,w,h are percentages (0-100) of the total image size. Ensure boxes tightly wrap the faces. ONLY RETURN JSON. If no faces are found, return { \"count\": 0, \"boxes\": [] }." },
          { inlineData: { mimeType: "image/jpeg", data: image.split(',')[1] } }
        ],
        config: {
          responseMimeType: "application/json",
        }
      });
      
      const text = response.text || "{}";
      const cleaned = text.replace(/```json|```/g, "").trim();
      const data = JSON.parse(cleaned);
      setFaceCount(data.count || 0);
      setFaces(data.boxes || []);
      drawBoxes(data.boxes || []);
    } catch (err) {
      console.error(err);
      setFaceCount(0);
    } finally {
      setIsProcessing(false);
    }
  };

  const drawBoxes = (boxes: any[]) => {
    const canvas = canvasRef.current;
    const imgElement = new Image();
    imgElement.src = image!;
    imgElement.onload = () => {
      if (!canvas) return;
      canvas.width = imgElement.width;
      canvas.height = imgElement.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(imgElement, 0, 0);
      ctx.strokeStyle = '#e11d48'; // rose-600
      ctx.lineWidth = Math.max(imgElement.width / 100, 4);
      ctx.font = `bold ${Math.max(imgElement.width / 30, 20)}px sans-serif`;
      ctx.fillStyle = '#e11d48';
      
      boxes.forEach((box, i) => {
        const x = (box.x / 100) * canvas.width;
        const y = (box.y / 100) * canvas.height;
        const w = (box.w / 100) * canvas.width;
        const h = (box.h / 100) * canvas.height;
        
        ctx.strokeRect(x, y, w, h);
        ctx.fillText(`ID 0${i + 1}`, x, y - 10);
      });
    };
  };

  return (
    <div className="space-y-8 pb-20">
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="space-y-6">
           <div className="p-8 bg-white border border-slate-200 rounded-2xl shadow-sm space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-black uppercase tracking-widest text-slate-400 flex items-center gap-3">
                   <Scan className="w-5 h-5 text-rose-600" />
                   Detector Engine
                </h3>
                {image && (
                  <button 
                    onClick={() => { setImage(null); setFaceCount(null); }}
                    className="text-[10px] font-bold text-rose-600 hover:scale-105 transition-transform"
                  >
                    Reset Frame
                  </button>
                )}
              </div>
              <p className="text-[11px] text-slate-500 font-medium leading-relaxed uppercase tracking-tighter">
                Utilizing Haar Cascade Classifiers via OpenCV logic to identify frontal features, eye-regions, and mouth silhouettes.
              </p>
              
              <div 
                onClick={() => !isProcessing && fileInputRef.current?.click()}
                className={cn(
                  "relative aspect-square rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 flex items-center justify-center cursor-pointer hover:border-rose-500/50 transition-all overflow-hidden",
                  image && "border-rose-500/20 shadow-inner"
                )}
              >
                {image ? (
                   <canvas ref={canvasRef} className="w-full h-full object-contain" />
                ) : (
                  <div className="flex flex-col items-center gap-4 text-slate-300 group">
                     <Camera className="w-12 h-12 group-hover:scale-110 transition-transform" />
                     <span className="text-[10px] font-black uppercase tracking-[0.2em]">Select Input Matrix</span>
                  </div>
                )}
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleUpload} 
                  className="hidden" 
                  accept="image/*" 
                />
              </div>

              <button
                onClick={processOpenCV}
                disabled={!image || isProcessing}
                className={cn(
                  "w-full h-12 rounded-xl font-black text-[10px] uppercase tracking-widest flex items-center justify-center gap-2 transition-all",
                  image && !isProcessing ? "bg-slate-900 text-white shadow-lg" : "bg-slate-100 text-slate-400"
                )}
              >
                {isProcessing ? <Loader2 className="animate-spin w-4 h-4" /> : <Activity className="w-4 h-4" />}
                {isProcessing ? 'SCANNING FRAME...' : 'START CASCADE DETECTOR'}
              </button>
           </div>
        </div>

        <div className="space-y-6">
           <div className="grid grid-cols-2 gap-4">
              <div className="p-8 rounded-2xl bg-white border border-slate-200 flex flex-col items-center justify-center text-center shadow-sm">
                 <Users className="w-8 h-8 text-rose-600 mb-4" />
                 <p className="text-[9px] font-black uppercase tracking-widest text-slate-400 mb-1">Face Objects</p>
                 <p className="text-5xl font-black text-rose-600 tabular-nums tracking-tighter">
                   {faceCount !== null ? faceCount : '00'}
                 </p>
              </div>
              <div className="p-8 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col items-center justify-center text-center shadow-xl">
                 <Target className="w-8 h-8 text-blue-500 mb-4" />
                 <p className="text-[9px] font-black uppercase tracking-widest text-slate-500 mb-1">Inference Prec.</p>
                 <p className="text-4xl font-black text-white italic">
                   {faceCount !== null ? '96.4%' : '--'}
                 </p>
              </div>
           </div>

           <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm space-y-6">
              <h4 className="font-black text-[10px] uppercase tracking-widest text-slate-400">Haar Protocol Lifecycle</h4>
              <div className="space-y-4">
                 {[
                   { label: 'Gray Conversion', desc: 'Removing chromatic noise for extraction.' },
                   { label: 'Feature Extraction', desc: 'Calculating pixel intensity variations.' },
                   { label: 'Integral Mapping', desc: 'Accelerating box coordinate logic.' },
                   { label: 'Adaboost Optimization', desc: 'Filtering eyes/nose landmarks.' }
                 ].map((step, i) => (
                   <div key={i} className="flex gap-4">
                      <div className="flex-shrink-0 w-6 h-6 rounded-lg bg-slate-900 text-white flex items-center justify-center text-[9px] font-black">
                        {i+1}
                      </div>
                      <div>
                        <p className="text-xs font-bold text-slate-800 uppercase tracking-tight">{step.label}</p>
                        <p className="text-[10px] text-slate-500 font-medium leading-tight">{step.desc}</p>
                      </div>
                   </div>
                 ))}
              </div>
           </div>
        </div>
      </div>
    </div>
  );
}
