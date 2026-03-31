"use client";
import { useEffect, useState } from "react";
import { useInView } from "react-intersection-observer";

interface Segment {
  label: string;
  bpw: number;
  pct: string;
  color: string;
}

const segments: Segment[] = [
  { label: "Pass 1 indices", bpw: 4.0, pct: "57%", color: "#58a6ff" },
  { label: "Pass 2 indices", bpw: 2.0, pct: "28.5%", color: "#d2a8ff" },
  { label: "Pass 1 norms", bpw: 0.25, pct: "3.6%", color: "#7ee787" },
  { label: "Pass 2 norms", bpw: 0.25, pct: "3.6%", color: "#3fb950" },
  { label: "Non-quantized", bpw: 0.52, pct: "7.3%", color: "#ffa657" },
];

const totalBpw = segments.reduce((s, seg) => s + seg.bpw, 0);

export function BPWBreakdownViz() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.3 });
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let start: number;
    const tick = (now: number) => {
      if (!start) start = now;
      const p = Math.min((now - start) / 2000, 1);
      setProgress(p);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [inView]);

  const ease = (t: number) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t);
  const p = ease(progress);

  // Compute cumulative offsets
  let cumulative = 0;
  const bars = segments.map((seg) => {
    const startPct = (cumulative / totalBpw) * 100;
    const widthPct = (seg.bpw / totalBpw) * 100;
    cumulative += seg.bpw;
    return { ...seg, startPct, widthPct };
  });

  return (
    <div ref={ref} className="space-y-4">
      {/* Header */}
      <div className="flex items-baseline justify-between text-sm">
        <span className="text-txt-2">Qwen3.5-0.8B · 4+2 residual · d=128</span>
        <span className="text-txt font-semibold">
          ~{totalBpw.toFixed(2)} BPW
        </span>
      </div>

      {/* Stacked bar */}
      <div className="relative h-12 rounded-lg overflow-hidden bg-bg-3 border border-border">
        {bars.map((bar, i) => {
          // Each segment animates in sequentially
          const segStart = i / bars.length;
          const segEnd = (i + 1) / bars.length;
          const localP = Math.max(
            0,
            Math.min(1, (p - segStart) / (segEnd - segStart))
          );
          const displayWidth = bar.widthPct * localP;

          return (
            <div
              key={bar.label}
              className="absolute top-0 h-full flex items-center justify-center transition-none"
              style={{
                left: `${bar.startPct}%`,
                width: `${displayWidth}%`,
                backgroundColor: bar.color,
                opacity: localP > 0 ? 0.85 : 0,
              }}
            >
              {displayWidth > 8 && (
                <span className="text-[10px] font-bold text-bg truncate px-1">
                  {bar.bpw}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Legend rows */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-2 text-xs">
        {bars.map((bar) => (
          <div key={bar.label} className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-sm shrink-0"
              style={{ background: bar.color }}
            />
            <span className="text-txt-2 truncate">
              {bar.label}{" "}
              <span className="text-txt font-medium">
                {bar.bpw} ({bar.pct})
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
