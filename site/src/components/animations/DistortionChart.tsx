"use client";
import { useEffect, useRef, useState } from "react";
import { useInView } from "react-intersection-observer";

const lloydMaxData = [
  { bits: 1, Db: 0.3634 },
  { bits: 2, Db: 0.1175 },
  { bits: 3, Db: 0.03454 },
  { bits: 4, Db: 0.009497 },
  { bits: 5, Db: 0.002499 },
];

function shannonBound(R: number): number {
  return Math.pow(2, -2 * R);
}

const LM_COLOR = "#58a6ff";
const SHANNON_COLOR = "#7ee787";
const GRID_COLOR = "#30363d";
const LABEL_COLOR = "#8b949e";

export function DistortionChart() {
  const svgRef = useRef<SVGSVGElement>(null);
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

  const width = 600;
  const height = 340;
  const pad = { top: 30, right: 40, bottom: 50, left: 70 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  // Log scale: y maps Db values (range ~0.001 to 0.5)
  const yMin = Math.log10(0.001);
  const yMax = Math.log10(0.5);
  const xMin = 0.5;
  const xMax = 5.5;

  const toX = (bits: number) => pad.left + ((bits - xMin) / (xMax - xMin)) * plotW;
  const toY = (db: number) => {
    const logVal = Math.log10(Math.max(db, 1e-6));
    return pad.top + ((yMax - logVal) / (yMax - yMin)) * plotH;
  };

  // Shannon bound line (smooth curve)
  const shannonPoints: string[] = [];
  for (let b = 0.5; b <= 5.5; b += 0.1) {
    const d = shannonBound(b);
    shannonPoints.push(`${toX(b)},${toY(d)}`);
  }
  const shannonPath = `M${shannonPoints.join("L")}`;

  // Lloyd-Max line
  const lmPoints = lloydMaxData.map((d) => `${toX(d.bits)},${toY(d.Db)}`);
  const lmPath = `M${lmPoints.join("L")}`;

  // Grid lines (log scale)
  const yTicks = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3];
  const xTicks = [1, 2, 3, 4, 5];

  // Gap annotation at 4 bits
  const gapX = toX(4);
  const gapYLm = toY(0.009497);
  const gapYSh = toY(shannonBound(4));

  // Ease function
  const ease = (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
  const p = ease(progress);

  return (
    <div ref={ref}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full max-w-[600px] mx-auto"
        style={{ overflow: "visible" }}
      >
        {/* Grid lines */}
        {yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            <line
              x1={pad.left}
              y1={toY(tick)}
              x2={width - pad.right}
              y2={toY(tick)}
              stroke={GRID_COLOR}
              strokeWidth={0.5}
              opacity={0.5}
            />
            <text
              x={pad.left - 10}
              y={toY(tick) + 4}
              textAnchor="end"
              fill={LABEL_COLOR}
              fontSize={11}
            >
              {tick >= 0.01 ? tick.toString() : tick.toExponential(0)}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <g key={`x-${tick}`}>
            <line
              x1={toX(tick)}
              y1={pad.top}
              x2={toX(tick)}
              y2={height - pad.bottom}
              stroke={GRID_COLOR}
              strokeWidth={0.5}
              opacity={0.3}
            />
            <text
              x={toX(tick)}
              y={height - pad.bottom + 20}
              textAnchor="middle"
              fill={LABEL_COLOR}
              fontSize={12}
            >
              {tick}
            </text>
          </g>
        ))}

        {/* Axis labels */}
        <text
          x={width / 2}
          y={height - 5}
          textAnchor="middle"
          fill={LABEL_COLOR}
          fontSize={12}
        >
          Bits (b)
        </text>
        <text
          x={15}
          y={height / 2}
          textAnchor="middle"
          fill={LABEL_COLOR}
          fontSize={12}
          transform={`rotate(-90, 15, ${height / 2})`}
        >
          Distortion (Dᵦ)
        </text>

        {/* Shannon bound (dashed) */}
        <path
          d={shannonPath}
          fill="none"
          stroke={SHANNON_COLOR}
          strokeWidth={2}
          strokeDasharray="6 4"
          opacity={p}
          pathLength={1}
          strokeDashoffset={1 - p}
          style={{ transition: "none" }}
        />

        {/* Lloyd-Max line */}
        <path
          d={lmPath}
          fill="none"
          stroke={LM_COLOR}
          strokeWidth={2}
          opacity={p}
          pathLength={1}
          strokeDashoffset={0}
        />

        {/* Lloyd-Max data points */}
        {lloydMaxData.map((d, i) => {
          const pointProgress = Math.max(
            0,
            Math.min(1, (p - i * 0.15) / 0.3)
          );
          return (
            <circle
              key={d.bits}
              cx={toX(d.bits)}
              cy={toY(d.Db)}
              r={5 * pointProgress}
              fill={LM_COLOR}
              opacity={pointProgress}
            />
          );
        })}

        {/* Gap annotation at 4 bits */}
        {p > 0.7 && (
          <g opacity={Math.min((p - 0.7) / 0.3, 1)}>
            <line
              x1={gapX + 12}
              y1={gapYLm}
              x2={gapX + 12}
              y2={gapYSh}
              stroke="#ffa657"
              strokeWidth={1.5}
              markerStart="url(#arrowUp)"
              markerEnd="url(#arrowDown)"
            />
            <rect
              x={gapX + 20}
              y={(gapYLm + gapYSh) / 2 - 12}
              width={58}
              height={24}
              rx={4}
              fill="#161b22"
              stroke="#ffa657"
              strokeWidth={1}
            />
            <text
              x={gapX + 49}
              y={(gapYLm + gapYSh) / 2 + 4}
              textAnchor="middle"
              fill="#ffa657"
              fontSize={12}
              fontWeight="bold"
            >
              2.43×
            </text>
          </g>
        )}

        {/* Arrow markers */}
        <defs>
          <marker
            id="arrowUp"
            markerWidth={6}
            markerHeight={6}
            refX={3}
            refY={3}
            orient="auto"
          >
            <path d="M0,6 L3,0 L6,6" fill="none" stroke="#ffa657" strokeWidth={1} />
          </marker>
          <marker
            id="arrowDown"
            markerWidth={6}
            markerHeight={6}
            refX={3}
            refY={3}
            orient="auto"
          >
            <path d="M0,0 L3,6 L6,0" fill="none" stroke="#ffa657" strokeWidth={1} />
          </marker>
        </defs>
      </svg>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-3 text-xs text-txt-2">
        <span className="flex items-center gap-1.5">
          <span
            className="w-3 h-0.5 inline-block rounded"
            style={{ background: LM_COLOR }}
          />
          Lloyd-Max D<sub>b</sub>
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="w-3 h-0.5 inline-block rounded border-b border-dashed"
            style={{ borderColor: SHANNON_COLOR }}
          />
          Shannon D*(R) = 2<sup>−2R</sup>
        </span>
      </div>
    </div>
  );
}
