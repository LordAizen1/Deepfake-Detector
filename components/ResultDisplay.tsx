"use client";

interface PredictionResult {
  label: "REAL" | "FAKE";
  confidence: number;
  fake_prob: number;
  real_prob: number;
  heatmap?: string;
}

interface Props {
  result: PredictionResult;
  onReset: () => void;
}

export default function ResultDisplay({ result, onReset }: Props) {
  const isFake = result.label === "FAKE";
  const pct = (result.confidence * 100).toFixed(1);
  const color = isFake ? "tertiary" : "secondary";

  return (
    <div className="flex flex-col items-center text-center space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Verdict */}
      <div className="space-y-1">
        <div className="flex items-center justify-center space-x-3">
          <span
            className="material-symbols-outlined text-4xl leading-none -translate-y-1"
            style={{
              fontVariationSettings: "'FILL' 1",
              color: isFake ? "var(--tw-color-tertiary, #ff7b70)" : "#6daa8b",
            }}
          >
            {isFake ? "warning" : "verified"}
          </span>
          <h2
            className="text-5xl font-headline italic tracking-tight leading-none"
            style={{ color: isFake ? "#ff7b70" : "#6daa8b" }}
          >
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
            backgroundColor: isFake ? "#ff7b70" : "#6daa8b",
            boxShadow: isFake
              ? "0 0 8px rgba(255,123,112,0.5)"
              : "0 0 8px rgba(109,170,139,0.5)",
          }}
        />
      </div>

      {/* Reset button */}
      <button
        onClick={onReset}
        className="mt-4 px-8 py-3 border border-outline-variant hover:border-on-surface-variant hover:bg-surface-container-high transition-all duration-300 rounded-lg flex items-center space-x-3 group"
      >
        <span className="material-symbols-outlined text-sm text-on-surface-variant group-hover:text-on-surface group-hover:rotate-180 transition-all duration-500">
          refresh
        </span>
        <span className="text-xs font-label uppercase tracking-widest font-semibold text-on-surface-variant group-hover:text-on-surface">
          Upload another image
        </span>
      </button>
    </div>
  );
}
