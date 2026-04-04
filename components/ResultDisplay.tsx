"use client";

import { useState } from "react";

interface PredictionResult {
  label: "REAL" | "FAKE";
  confidence: number;
  fake_prob: number;
  real_prob: number;
  heatmap?: string;
  frames_analyzed?: number;
}

interface AdversarialResult {
  original_label: "REAL" | "FAKE";
  original_fake_prob: number;
  adversarial_label: "REAL" | "FAKE";
  adversarial_fake_prob: number;
  adversarial_image: string;
  epsilon: number;
  fooled: boolean;
}

interface Props {
  result: PredictionResult;
  originalFile: File | null;
  onReset: () => void;
}

export default function ResultDisplay({ result, originalFile, onReset }: Props) {
  const isFake = result.label === "FAKE";
  const pct = (result.confidence * 100).toFixed(1);
  const color = isFake ? "#ff7b70" : "#6daa8b";

  const [epsilon, setEpsilon] = useState(0.02);
  const [advResult, setAdvResult] = useState<AdversarialResult | null>(null);
  const [advStatus, setAdvStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [advError, setAdvError] = useState("");

  async function runAdversarial() {
    if (!originalFile) return;
    setAdvStatus("loading");
    setAdvResult(null);
    setAdvError("");

    try {
      const formData = new FormData();
      formData.append("file", originalFile);
      formData.append("epsilon", String(epsilon));

      const res = await fetch("/api/adversarial", { method: "POST", body: formData });
      const text = await res.text();
      let data;
      try { data = JSON.parse(text); } catch { throw new Error("Server error"); }
      if (!res.ok) throw new Error(data.detail || "Attack failed");
      setAdvResult(data);
      setAdvStatus("done");
    } catch (err) {
      setAdvError(err instanceof Error ? err.message : "Something went wrong");
      setAdvStatus("error");
    }
  }

  return (
    <div className="flex flex-col items-center text-center space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700 w-full max-w-xl">
      {/* Verdict */}
      <div className="space-y-1">
        <div className="flex items-center justify-center space-x-3">
          <span
            className="material-symbols-outlined text-4xl leading-none -translate-y-1"
            style={{ fontVariationSettings: "'FILL' 1", color }}
          >
            {isFake ? "warning" : "verified"}
          </span>
          <h2 className="text-5xl font-headline italic tracking-tight leading-none" style={{ color }}>
            {result.label}
          </h2>
        </div>

        <div className="flex flex-col items-center pt-2">
          <p className="text-xs font-label uppercase tracking-[0.1em] text-on-surface-variant/70">
            Forensic Confidence
          </p>
          <p className="text-2xl font-body font-light text-on-surface">{pct}%</p>
        </div>
      </div>

      {/* Confidence meter */}
      <div className="w-64 h-1.5 bg-secondary-container rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            backgroundColor: color,
            boxShadow: `0 0 8px ${color}80`,
          }}
        />
      </div>

      {/* Video frame info */}
      {result.frames_analyzed !== undefined && (
        <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant/50">
          {result.frames_analyzed} frame{result.frames_analyzed !== 1 ? "s" : ""} analyzed
        </p>
      )}

      {/* Adversarial robustness section — images only */}
      {result.frames_analyzed === undefined && (
        <div className="w-full border border-outline-variant/30 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-center space-x-2">
            <span className="material-symbols-outlined text-sm text-on-surface-variant">security</span>
            <p className="text-xs font-label uppercase tracking-widest text-on-surface-variant">
              Robustness Test
            </p>
          </div>

          {/* Epsilon slider */}
          <div className="flex flex-col items-center space-y-1">
            <div className="flex items-center justify-between w-full max-w-xs">
              <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant/60">
                Perturbation strength (ε)
              </p>
              <p className="text-[10px] font-body text-on-surface-variant">{epsilon.toFixed(3)}</p>
            </div>
            <input
              type="range"
              min={0.001}
              max={0.1}
              step={0.001}
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              className="w-full max-w-xs accent-secondary"
            />
            <div className="flex justify-between w-full max-w-xs">
              <span className="text-[9px] font-label text-outline-variant uppercase">Subtle</span>
              <span className="text-[9px] font-label text-outline-variant uppercase">Strong</span>
            </div>
          </div>

          {/* Attack button */}
          {advStatus !== "done" && (
            <button
              onClick={runAdversarial}
              disabled={advStatus === "loading"}
              className="px-6 py-2.5 border border-outline-variant hover:border-on-surface-variant hover:bg-surface-container-high transition-all duration-300 rounded-lg flex items-center space-x-2 group disabled:opacity-40 mx-auto"
            >
              {advStatus === "loading" ? (
                <>
                  <span className="material-symbols-outlined text-sm text-on-surface-variant animate-spin">refresh</span>
                  <span className="text-xs font-label uppercase tracking-widest text-on-surface-variant">Attacking...</span>
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-sm text-on-surface-variant group-hover:text-on-surface">bolt</span>
                  <span className="text-xs font-label uppercase tracking-widest text-on-surface-variant group-hover:text-on-surface">
                    Run FGSM Attack
                  </span>
                </>
              )}
            </button>
          )}

          {advStatus === "error" && (
            <p className="text-[10px] font-label uppercase tracking-widest text-error">{advError}</p>
          )}

          {/* Adversarial result */}
          {advStatus === "done" && advResult && (
            <div className="space-y-4 animate-in fade-in duration-500">
              {/* Fooled / Not fooled banner */}
              <div
                className="flex items-center justify-center space-x-2 py-2 px-4 rounded-lg"
                style={{
                  backgroundColor: advResult.fooled ? "rgba(255,123,112,0.1)" : "rgba(109,170,139,0.1)",
                  border: `1px solid ${advResult.fooled ? "#ff7b7040" : "#6daa8b40"}`,
                }}
              >
                <span
                  className="material-symbols-outlined text-sm"
                  style={{ color: advResult.fooled ? "#ff7b70" : "#6daa8b", fontVariationSettings: "'FILL' 1" }}
                >
                  {advResult.fooled ? "crisis_alert" : "shield"}
                </span>
                <p
                  className="text-xs font-label uppercase tracking-widest"
                  style={{ color: advResult.fooled ? "#ff7b70" : "#6daa8b" }}
                >
                  {advResult.fooled ? "Model fooled" : "Model held firm"}
                </p>
              </div>

              {/* Before / After */}
              <div className="grid grid-cols-2 gap-3 text-center">
                <div className="space-y-2">
                  <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant/60">Original</p>
                  <p className="text-lg font-headline italic" style={{ color: advResult.original_label === "FAKE" ? "#ff7b70" : "#6daa8b" }}>
                    {advResult.original_label}
                  </p>
                  <p className="text-xs font-body text-on-surface-variant">
                    {(advResult.original_fake_prob * 100).toFixed(1)}% fake
                  </p>
                </div>
                <div className="space-y-2">
                  <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant/60">After Attack (ε={advResult.epsilon})</p>
                  <p className="text-lg font-headline italic" style={{ color: advResult.adversarial_label === "FAKE" ? "#ff7b70" : "#6daa8b" }}>
                    {advResult.adversarial_label}
                  </p>
                  <p className="text-xs font-body text-on-surface-variant">
                    {(advResult.adversarial_fake_prob * 100).toFixed(1)}% fake
                  </p>
                </div>
              </div>

              {/* Perturbed image */}
              <div className="space-y-1">
                <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant/60">
                  Perturbed image (imperceptible to humans)
                </p>
                <img
                  src={advResult.adversarial_image}
                  alt="adversarial"
                  className="w-32 h-32 object-cover rounded-lg mx-auto border border-outline-variant/20"
                />
              </div>

              <button
                onClick={() => { setAdvStatus("idle"); setAdvResult(null); }}
                className="text-[10px] font-label uppercase tracking-widest text-outline-variant hover:text-on-surface-variant transition-colors"
              >
                Try different ε
              </button>
            </div>
          )}
        </div>
      )}

      {/* Reset button */}
      <button
        onClick={onReset}
        className="mt-4 px-8 py-3 border border-outline-variant hover:border-on-surface-variant hover:bg-surface-container-high transition-all duration-300 rounded-lg flex items-center space-x-3 group"
      >
        <span className="material-symbols-outlined text-sm text-on-surface-variant group-hover:text-on-surface group-hover:rotate-180 transition-all duration-500">
          refresh
        </span>
        <span className="text-xs font-label uppercase tracking-widest font-semibold text-on-surface-variant group-hover:text-on-surface">
          Analyze another file
        </span>
      </button>
    </div>
  );
}
