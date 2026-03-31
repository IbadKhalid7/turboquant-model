"use client";
import { motion } from "framer-motion";
import Link from "next/link";
import { Reveal } from "@/components/Reveal";
import { AnimatedCounter } from "@/components/AnimatedCounter";

const techniques = [
  {
    href: "/techniques/lloyd-max/",
    icon: "📊",
    title: "Lloyd-Max Quantization",
    desc: "Optimal scalar quantizer for Gaussian distributions — 16 centroids, 64 bytes, shared globally.",
    color: "#58a6ff",
  },
  {
    href: "/techniques/rotation/",
    icon: "🔄",
    title: "Random Rotation",
    desc: "Orthogonal rotation decorrelates weights and Gaussianizes coordinates via the central limit theorem.",
    color: "#7ee787",
  },
  {
    href: "/techniques/walsh-hadamard/",
    icon: "⚡",
    title: "Walsh-Hadamard Transform",
    desc: "O(d log d) butterfly rotation with O(d) storage — faster and leaner than QR decomposition.",
    color: "#d2a8ff",
  },
  {
    href: "/techniques/residual/",
    icon: "🎯",
    title: "Residual Quantization",
    desc: "Multi-pass quantization captures progressively finer detail. 4+4 bits → KLD 0.002 (near-lossless).",
    color: "#ffa657",
  },
  {
    href: "/techniques/packing/",
    icon: "📦",
    title: "4-bit Packing",
    desc: "Two 4-bit indices per uint8 byte — halving storage with near-free bitwise pack/unpack.",
    color: "#58a6ff",
  },
  {
    href: "/techniques/fused-kernels/",
    icon: "🚀",
    title: "Fused GPU Kernels",
    desc: "CuTile & Triton fuse unpack→lookup→matmul→rescale into one launch. 4× speedup, 5.7× memory savings.",
    color: "#7ee787",
  },
  {
    href: "/techniques/entropy-codec/",
    icon: "🗜️",
    title: "Entropy Coding (rANS)",
    desc: "Compress indices below nominal bit-width via rANS — saves ~0.24 BPW by exploiting non-uniform Gaussian bin probabilities.",
    color: "#f778ba",
  },
  {
    href: "/techniques/norm-codec/",
    icon: "📐",
    title: "Norm Compression",
    desc: "Rank-1 SVD + int8 residual factorizes the norm tensor α_{m,g} — reducing norm overhead from 0.25 to ~0.06 BPW.",
    color: "#79c0ff",
  },
  {
    href: "/techniques/qjl/",
    icon: "�",
    title: "QJL (Quantized JL)",
    desc: "1-bit random projection for unbiased inner product estimation — elegant for KV-cache, but not used in this project.",
    color: "#d2a8ff",
  },
];

const particles = [
  { left: "15%", top: "25%", delay: 0, color: "#58a6ff" },
  { left: "20%", top: "30%", delay: 1, color: "#7ee787" },
  { left: "70%", top: "20%", delay: 2, color: "#d2a8ff" },
  { left: "85%", top: "60%", delay: 3, color: "#58a6ff" },
  { left: "10%", top: "70%", delay: 4, color: "#ffa657" },
  { left: "50%", top: "80%", delay: 2.5, color: "#7ee787" },
];

export default function HomePage() {
  return (
    <>
      {/* ── HERO ── */}
      <section className="min-h-screen flex items-center justify-center text-center relative pt-20 overflow-hidden">
        {/* Particles */}
        <div className="absolute inset-0 pointer-events-none">
          {particles.map((p, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 rounded-full"
              style={{ left: p.left, top: p.top, background: p.color }}
              animate={{ y: [0, -12, 0] }}
              transition={{ duration: 6, delay: p.delay, repeat: Infinity, ease: "easeInOut" }}
            />
          ))}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-accent/[.06] rounded-full blur-3xl" />
        </div>

        <div className="relative max-w-3xl mx-auto px-6">
          <motion.h1
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-5xl md:text-7xl font-extrabold mb-4"
          >
            <span className="bg-gradient-to-r from-accent via-accent-green to-accent-purple bg-clip-text text-transparent">
              TurboQuant
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.15 }}
            className="text-lg md:text-xl text-txt-2 mb-8 max-w-2xl mx-auto"
          >
            Applying the TurboQuant paper&apos;s rotation + Lloyd-Max framework to{" "}
            <strong className="text-txt">offline LLM weight compression</strong> — replacing
            QJL with multi-pass residual quantization and fused GPU kernels.
          </motion.p>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="flex justify-center gap-10 flex-wrap mb-10"
          >
            <Stat value={4} suffix="" label="bits per weight" />
            <Stat value={4} suffix="×" label="compression ratio" />
            <Stat value={2.7} suffix="×" decimals={1} label="near optimal MSE" />
            <Stat value={3.2} suffix="×" decimals={1} label="memory savings" />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.45 }}
            className="flex gap-3 justify-center"
          >
            <Link
              href="/quantize-pipeline/"
              className="px-7 py-3 rounded-lg font-semibold bg-accent text-bg hover:bg-accent/80 transition-all hover:-translate-y-0.5 hover:shadow-lg hover:shadow-accent/25"
            >
              See How It Works
            </Link>
            <a
              href="https://arxiv.org/abs/2504.19874"
              target="_blank"
              rel="noopener noreferrer"
              className="px-7 py-3 rounded-lg font-semibold border border-border text-txt hover:border-accent transition-colors"
            >
              Read the Paper
            </a>
          </motion.div>
        </div>
      </section>

      {/* ── PAPER vs PROJECT ── */}
      <section className="py-24 bg-bg-2">
        <div className="max-w-5xl mx-auto px-6">
          <Reveal>
            <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-accent to-accent-purple bg-clip-text text-transparent">
              Paper → Practice
            </h2>
            <p className="text-txt-2 mb-10 max-w-2xl">
              This project takes the core ideas from the TurboQuant paper and applies them to
              offline LLM weight compression — a different use case than the paper&apos;s
              primary focus.
            </p>
          </Reveal>

          <div className="grid md:grid-cols-2 gap-6">
            <Reveal delay={0.1}>
              <div className="bg-bg-3 border border-border rounded-2xl p-6 h-full">
                <h3 className="font-semibold text-accent-purple mb-3">📄 The Paper (Zandieh et al.)</h3>
                <p className="text-sm text-txt-2 leading-relaxed">
                  Introduces <strong className="text-txt">online vector quantization</strong> with
                  near-optimal distortion rate. Primary contribution is{" "}
                  <strong className="text-accent-purple">TurboQuant<sub>prod</sub></strong> — an
                  unbiased inner product estimator combining Lloyd-Max quantization with a 1-bit
                  QJL correction for KV-cache attention.
                </p>
                <div className="flex flex-wrap gap-1.5 mt-4">
                  {["Online estimation", "QJL 1-bit correction", "Unbiased dot products", "KV-cache focus"].map(
                    (t) => (
                      <span key={t} className="text-[11px] px-2.5 py-1 rounded-full border border-accent-purple/30 text-accent-purple">
                        {t}
                      </span>
                    )
                  )}
                </div>
              </div>
            </Reveal>
            <Reveal delay={0.2}>
              <div className="bg-bg-3 border border-accent-green/30 rounded-2xl p-6 h-full">
                <h3 className="font-semibold text-accent-green mb-3">⚡ This Project</h3>
                <p className="text-sm text-txt-2 leading-relaxed">
                  Takes the paper&apos;s <strong className="text-txt">rotation + Lloyd-Max</strong> foundation
                  and applies it to <strong className="text-accent-green">offline weight compression</strong>.
                  Replaces QJL with multi-pass{" "}
                  <strong className="text-accent-green">residual quantization</strong> using full Lloyd-Max
                  codebooks, and adds <strong className="text-accent-green">fused GPU kernels</strong> for
                  production inference.
                </p>
                <div className="flex flex-wrap gap-1.5 mt-4">
                  {["Offline weight compression", "No QJL", "Residual quantization", "Fused CuTile/Triton kernels"].map(
                    (t) => (
                      <span key={t} className="text-[11px] px-2.5 py-1 rounded-full border border-accent-green/30 text-accent-green">
                        {t}
                      </span>
                    )
                  )}
                </div>
              </div>
            </Reveal>
          </div>

          {/* Key differences */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-10">
            {[
              { icon: "🚫", title: "No QJL", desc: "QJL's 1-bit unbiased estimator designed for streaming KV-cache is unnecessary for offline weight compression." },
              { icon: "🎯", title: "Residual Instead", desc: "Full Lloyd-Max passes on the residual error. 4+4 bits achieves KLD 0.002 — strictly dominating 1-bit QJL." },
              { icon: "🔄", title: "Rotate Input", desc: "Pre-rotate the activation (B×d, cheap) instead of inverse-rotating the weight matrix (M×N, expensive)." },
              { icon: "🚀", title: "Fused Kernels", desc: "CuTile/Triton fuse unpack→lookup→matmul→rescale. 64-byte codebook in registers, zero round-trips." },
            ].map((d, i) => (
              <Reveal key={d.title} delay={0.1 * i}>
                <div className="bg-bg border border-border rounded-xl p-5 h-full">
                  <div className="text-xl mb-2">{d.icon}</div>
                  <h4 className="font-semibold text-sm mb-1">{d.title}</h4>
                  <p className="text-xs text-txt-2 leading-relaxed">{d.desc}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ── TECHNIQUES GRID ── */}
      <section className="py-24">
        <div className="max-w-5xl mx-auto px-6">
          <Reveal>
            <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-accent to-accent-green bg-clip-text text-transparent">
              Core Techniques
            </h2>
            <p className="text-txt-2 mb-10 max-w-2xl">
              Eight techniques combine to achieve near-information-theoretic-optimal weight compression.
              Click any card to explore in depth.
            </p>
          </Reveal>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {techniques.map((t, i) => (
              <Reveal key={t.href} delay={0.08 * i}>
                <Link href={t.href} className="group block">
                  <div className="bg-bg-2 border border-border rounded-2xl p-6 h-full relative overflow-hidden transition-all duration-300 group-hover:border-accent group-hover:-translate-y-1 group-hover:shadow-lg group-hover:shadow-accent/10">
                    <div
                      className="absolute top-0 left-0 right-0 h-[3px] origin-left scale-x-0 group-hover:scale-x-100 transition-transform duration-300"
                      style={{ background: `linear-gradient(90deg, ${t.color}, transparent)` }}
                    />
                    <div
                      className="w-12 h-12 rounded-xl flex items-center justify-center text-xl mb-4"
                      style={{ background: `${t.color}15` }}
                    >
                      {t.icon}
                    </div>
                    <h3 className="font-semibold mb-2">{t.title}</h3>
                    <p className="text-sm text-txt-2 leading-relaxed">{t.desc}</p>
                  </div>
                </Link>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ── PIPELINE LINKS ── */}
      <section className="py-24 bg-bg-2">
        <div className="max-w-5xl mx-auto px-6">
          <Reveal>
            <h2 className="text-3xl font-bold mb-10 bg-gradient-to-r from-accent-orange to-accent bg-clip-text text-transparent">
              Pipelines
            </h2>
          </Reveal>
          <div className="grid md:grid-cols-3 gap-6">
            <Reveal delay={0.1}>
              <Link href="/quantize-pipeline/" className="group block">
                <div className="bg-bg border border-border rounded-2xl p-8 transition-all duration-300 group-hover:border-accent-orange group-hover:-translate-y-1">
                  <h3 className="text-xl font-bold mb-3">Quantization Pipeline</h3>
                  <p className="text-txt-2 text-sm leading-relaxed">
                    How TurboQuant compresses model weights from bf16/fp32 down to 4-bit packed indices.
                    Normalize → Rotate → Scale → Quantize → Pack.
                  </p>
                  <div className="flex items-center gap-2 mt-4 text-sm text-accent">
                    <span>Explore</span>
                    <span className="group-hover:translate-x-1 transition-transform">→</span>
                  </div>
                </div>
              </Link>
            </Reveal>
            <Reveal delay={0.2}>
              <Link href="/dequantize-pipeline/" className="group block">
                <div className="bg-bg border border-border rounded-2xl p-8 transition-all duration-300 group-hover:border-accent-green group-hover:-translate-y-1">
                  <h3 className="text-xl font-bold mb-3">Dequantization Pipeline</h3>
                  <p className="text-txt-2 text-sm leading-relaxed">
                    On-the-fly inference: rotate the input, not the weight. Fused kernel implementations
                    with CuTile, Triton, and PyTorch fallback.
                  </p>
                  <div className="flex items-center gap-2 mt-4 text-sm text-accent-green">
                    <span>Explore</span>
                    <span className="group-hover:translate-x-1 transition-transform">→</span>
                  </div>
                </div>
              </Link>
            </Reveal>
            <Reveal delay={0.3}>
              <Link href="/formulation/" className="group block">
                <div className="bg-bg border border-border rounded-2xl p-8 transition-all duration-300 group-hover:border-accent-purple group-hover:-translate-y-1">
                  <h3 className="text-xl font-bold mb-3">Quantization Formulation</h3>
                  <p className="text-txt-2 text-sm leading-relaxed">
                    Mathematical foundations: MSE analysis, near-optimality argument, BPW budget,
                    and research directions for pushing compression further.
                  </p>
                  <div className="flex items-center gap-2 mt-4 text-sm text-accent-purple">
                    <span>Explore</span>
                    <span className="group-hover:translate-x-1 transition-transform">→</span>
                  </div>
                </div>
              </Link>
            </Reveal>
          </div>
        </div>
      </section>

      {/* ── BENCHMARKS PREVIEW ── */}
      <section className="py-24">
        <div className="max-w-5xl mx-auto px-6">
          <Reveal>
            <h2 className="text-3xl font-bold mb-8 bg-gradient-to-r from-accent-green to-accent bg-clip-text text-transparent">
              Benchmark Highlights
            </h2>
          </Reveal>
          <div className="grid sm:grid-cols-3 gap-5">
            {[
              { label: "4+4 Residual PPL", value: "14.28", sub: "vs 14.29 baseline", color: "#7ee787" },
              { label: "KL Divergence", value: "0.002", sub: "nats (near-lossless)", color: "#58a6ff" },
              { label: "CuTile Speedup", value: "3.98×", sub: "vs PyTorch fallback", color: "#d2a8ff" },
            ].map((b, i) => (
              <Reveal key={b.label} delay={0.1 * i}>
                <div className="bg-bg-2 border border-border rounded-2xl p-6 text-center">
                  <div className="text-3xl font-extrabold mb-1" style={{ color: b.color }}>
                    {b.value}
                  </div>
                  <div className="font-semibold text-sm mb-1">{b.label}</div>
                  <div className="text-xs text-txt-2">{b.sub}</div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}

function Stat({
  value,
  suffix,
  decimals = 0,
  label,
}: {
  value: number;
  suffix: string;
  decimals?: number;
  label: string;
}) {
  return (
    <div className="text-center">
      <div className="text-4xl font-extrabold bg-gradient-to-r from-accent to-accent-green bg-clip-text text-transparent">
        <AnimatedCounter target={value} decimals={decimals} suffix={suffix} />
      </div>
      <div className="text-sm text-txt-2">{label}</div>
    </div>
  );
}
