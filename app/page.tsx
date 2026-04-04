"use client";

import { useState } from "react";
import ImageUpload from "@/components/ImageUpload";
import ResultDisplay from "@/components/ResultDisplay";

interface PredictionResult {
  label: "REAL" | "FAKE";
  confidence: number;
  fake_prob: number;
  real_prob: number;
  heatmap?: string;
  frames_analyzed?: number;
  frame_results?: { frame: number; time_sec: number; fake_prob: number }[];
}

type Status = "idle" | "loading" | "done" | "error";

export default function Home() {
  const [preview, setPreview] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<"image" | "video" | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState("");

  async function handleMediaSelected(file: File, previewUrl: string) {
    const isVideo = file.type.startsWith("video/");
    setPreview(previewUrl);
    setMediaType(isVideo ? "video" : "image");
    setResult(null);
    setErrorMsg("");
    setStatus("loading");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const endpoint = isVideo ? "/api/predict_video" : "/api/predict";
      const res = await fetch(endpoint, { method: "POST", body: formData });
      const text = await res.text();

      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error("Server error — please try a different file");
      }

      if (!res.ok) {
        throw new Error(data.detail || "Prediction failed");
      }

      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Something went wrong");
      setStatus("error");
    }
  }

  function reset() {
    setPreview(null);
    setMediaType(null);
    setResult(null);
    setStatus("idle");
    setErrorMsg("");
  }

  const showHeatmap = status === "done" && result?.heatmap;

  return (
    <div className="min-h-screen flex flex-col items-center p-6">
      {/* Header */}
      <header className="w-full max-w-[1024px] mx-auto pt-12 pb-8 text-center">
        <h1 className="font-headline text-4xl italic tracking-tight text-on-surface drop-shadow-sm">
          Deepfake Detector
        </h1>
        <p className="font-label uppercase tracking-[0.05rem] text-[10px] text-outline-variant mt-2">
          Upload a facial image or video to detect AI manipulation.
        </p>
      </header>

      {/* Main Dossier */}
      <main className="w-full max-w-[1024px] flex flex-col items-center space-y-12">
        {/* Media Box */}
        <div className="w-full max-w-xl aspect-[4/3] relative group bg-surface-container rounded-xl overflow-hidden shadow-2xl transition-all duration-500">
          <div className="absolute inset-0 border-2 border-dashed border-outline-variant/30 m-4 rounded-lg flex flex-col items-center justify-center overflow-hidden">
            {status === "idle" && (
              <ImageUpload onMediaSelected={handleMediaSelected} />
            )}

            {(status === "loading" || status === "done") && preview && (
              <div className="relative w-full h-full">
                {showHeatmap ? (
                  <img
                    src={result!.heatmap}
                    alt="forensic analysis heatmap"
                    className="w-full h-full object-cover"
                  />
                ) : mediaType === "video" ? (
                  <video
                    src={preview}
                    className={`w-full h-full object-cover ${status === "loading" ? "grayscale opacity-60" : ""}`}
                    muted
                    loop
                    autoPlay
                    playsInline
                  />
                ) : (
                  <img
                    src={preview}
                    alt="forensic facial analysis"
                    className={`w-full h-full object-cover ${status === "loading" ? "grayscale opacity-60" : ""}`}
                  />
                )}

                {status === "loading" && (
                  <div className="absolute inset-0 bg-surface-container/40 backdrop-blur-md flex flex-col items-center justify-center">
                    <div className="relative">
                      <div className="absolute -inset-4 border border-secondary/30 rounded-full animate-ping" />
                      <span className="material-symbols-outlined text-secondary text-5xl">
                        {mediaType === "video" ? "video_search" : "face_unlock"}
                      </span>
                    </div>
                    <p className="mt-6 text-xs font-label uppercase tracking-widest text-on-surface-variant font-medium">
                      {mediaType === "video" ? "Analyzing Frames..." : "Analyzing Artifacts..."}
                    </p>
                    <div className="w-48 h-1 bg-surface-container-lowest mt-4 rounded-full overflow-hidden">
                      <div className="h-full bg-secondary w-3/4 rounded-full opacity-80 animate-pulse" />
                    </div>
                  </div>
                )}
              </div>
            )}

            {status === "error" && (
              <div className="flex flex-col items-center space-y-3 px-6 text-center">
                <span className="material-symbols-outlined text-error text-4xl">
                  error_outline
                </span>
                <p className="text-xs font-label uppercase tracking-widest text-error">
                  {errorMsg}
                </p>
                <button
                  onClick={reset}
                  className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant hover:text-on-surface border border-outline-variant px-4 py-2 rounded-lg transition-colors"
                >
                  Try again
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Result */}
        {status === "done" && result && (
          <ResultDisplay result={result} onReset={reset} />
        )}
      </main>

      {/* Footer */}
      <footer className="mt-auto pt-12 pb-6 opacity-30 flex flex-col items-center space-y-2">
        <div className="flex items-center space-x-4 text-[10px] font-label uppercase tracking-tighter text-on-surface-variant">
          <span>Model: EfficientNet-B4</span>
          <span className="w-1 h-1 bg-on-surface-variant rounded-full" />
          <span>Dataset: FaceForensics++</span>
          <span className="w-1 h-1 bg-on-surface-variant rounded-full" />
          <span>No data stored</span>
        </div>
        <p className="text-[9px] font-body text-on-surface-variant">
          Computer Vision Course Project, IIIT-Delhi
        </p>
        <p className="text-[9px] font-label uppercase tracking-widest text-on-surface-variant">
          Team Vincenzo
        </p>
      </footer>
    </div>
  );
}
