"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { KernelFusionViz } from "@/components/animations/KernelFusionViz";
import { Reveal } from "@/components/Reveal";
import { AnimatedCounter } from "@/components/AnimatedCounter";

export default function FusedKernelsPage() {
  return (
    <TechniqueLayout
      title="Fused GPU Kernels"
      subtitle="CuTile and Triton kernels fuse unpack → lookup → matmul → rescale into a single launch. The 64-byte codebook lives in registers."
      color="#7ee787"
      icon="🚀"
      prev={{ href: "/techniques/packing/", label: "4-bit Packing" }}
      next={{ href: "/techniques/qjl/", label: "QJL" }}
    >
      <Section title="The Problem: Intermediate Materialization">
        <p className="text-txt-2 leading-relaxed">
          The naive dequantization pipeline creates multiple intermediate tensors, each requiring
          a separate kernel launch with a global memory round-trip. For large models, this
          intermediate materialization dominates both latency and memory.
        </p>
      </Section>

      <Section title="Naive vs Fused: Side by Side">
        <KernelFusionViz />
      </Section>

      <Section title="Kernel Algorithm">
        <p className="text-txt-2 leading-relaxed mb-6">
          Each thread block computes a tile of the output. The codebook (16 × 4 bytes = 64 bytes)
          fits entirely in registers or L1 cache, making the lookup essentially free:
        </p>
        <div className="space-y-2">
          {[
            { step: "1", label: "Load", detail: "packed uint8 bytes from global memory", badge: "Global → Registers" },
            { step: "2", label: "Unpack", detail: "nibbles via bitwise ops (& 0x0F, >> 4)", badge: "In Registers" },
            { step: "3", label: "Lookup", detail: "codebook values (64 bytes in shared memory)", badge: "Shared Memory" },
            { step: "4", label: "MMA", detail: "tensor core multiply-accumulate (TF32/FP16)", badge: "Tensor Cores" },
            { step: "5", label: "Rescale", detail: "by pre-computed norms / √d", badge: "In Registers" },
            { step: "6", label: "Store", detail: "final result to global memory", badge: "Registers → Global" },
          ].map((s, i) => (
            <Reveal key={s.step} delay={0.08 * i}>
              <div className="flex items-center gap-4 bg-bg-2 border border-border rounded-xl p-4">
                <div className="w-8 h-8 rounded-full bg-accent-green/10 flex items-center justify-center text-accent-green font-bold text-sm shrink-0">
                  {s.step}
                </div>
                <div className="flex-1">
                  <span className="font-semibold text-sm">{s.label}</span>{" "}
                  <span className="text-sm text-txt-2">{s.detail}</span>
                </div>
                <span className="text-[10px] px-2 py-1 rounded bg-accent-green/10 text-accent-green whitespace-nowrap">
                  {s.badge}
                </span>
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      <Section title="Execution Paths">
        <div className="flex items-center gap-4 flex-wrap">
          {[
            { name: "CuTile", desc: "Fastest (CUDA 13.1+, Ampere+)", color: "#7ee787" },
            { name: "Triton", desc: "Portable (Triton 3.0+)", color: "#d2a8ff" },
            { name: "PyTorch", desc: "Fallback (no deps)", color: "#8b949e" },
          ].map((p, i) => (
            <Reveal key={p.name} delay={0.15 * i} className="flex items-center gap-3">
              <div className="bg-bg-2 border border-border rounded-xl px-5 py-3 text-center">
                <div className="font-semibold text-sm" style={{ color: p.color }}>
                  {p.name}
                </div>
                <div className="text-[10px] text-txt-2">{p.desc}</div>
              </div>
              {i < 2 && <span className="text-accent text-lg">→</span>}
            </Reveal>
          ))}
        </div>
      </Section>

      <Section title="Performance Impact">
        <div className="grid sm:grid-cols-3 gap-4">
          {[
            { label: "CuTile Speedup", value: "3.98×", color: "#7ee787", sub: "vs PyTorch (Qwen3.5-4B)" },
            { label: "Memory Reduction", value: "5.7×", color: "#58a6ff", sub: "CuTile vs PyTorch" },
            { label: "Codebook in Cache", value: "64", color: "#d2a8ff", sub: "bytes (fits in L1)" },
          ].map((b) => (
            <div key={b.label} className="bg-bg-2 border border-border rounded-xl p-5 text-center">
              <div className="text-2xl font-extrabold" style={{ color: b.color }}>
                {b.value}
              </div>
              <div className="text-sm font-semibold mt-1">{b.label}</div>
              <div className="text-xs text-txt-2">{b.sub}</div>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Implementation">
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm space-y-2">
          <div>
            <span className="text-accent-purple">cutile_kernels.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">cutile_fused_matmul()</span>
          </div>
          <div>
            <span className="text-accent-purple">triton_kernels.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">triton_fused_matmul()</span>
          </div>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
