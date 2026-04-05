"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { Reveal } from "@/components/Reveal";
import Link from "next/link";

export default function BlockwiseCalibrationPage() {
  return (
    <TechniqueLayout
      title="Block-wise Calibration"
      subtitle="Fine-tune per-row norms through each transformer block to minimize end-to-end reconstruction error — recovering ~14% of the quantization gap."
      color="#f0883e"
      icon="🎯"
      prev={{ href: "/techniques/norm-compression/", label: "Norm Compression" }}
      next={{ href: "/techniques/qjl/", label: "QJL" }}
    >
      {/* ─── Why not per-layer? ─── */}
      <Section title="Why Per-Layer Calibration Fails">
        <p className="text-txt-2 leading-relaxed mb-4">
          After quantization, per-row norms{" "}
          <Math expr="\alpha_m = \|W_m\|_2" /> are computed analytically.
          While optimal in a <strong className="text-txt">per-layer MSE</strong> sense,
          they don&apos;t account for <strong className="text-txt">error propagation</strong> —
          the output error of block <Math expr="\ell" /> becomes the input error for block{" "}
          <Math expr="\ell + 1" />.
        </p>
        <Reveal>
          <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-5 mb-4">
            <h4 className="font-semibold text-sm text-red-400 mb-2">Per-layer calibration result</h4>
            <p className="text-xs text-txt-2 leading-relaxed">
              On Qwen3.5-0.8B-Base, per-layer calibration <strong className="text-red-400">degraded</strong> end-to-end quality:
              PPL increased by +0.0656 and KLD by +0.005 — even though per-layer MSE improved for every layer.
              Locally optimal norms can amplify errors when composed through the network.
            </p>
          </div>
        </Reveal>
      </Section>

      {/* ─── The Objective ─── */}
      <Section title="Block-wise Objective">
        <p className="text-txt-2 leading-relaxed mb-4">
          For each transformer block <Math expr="\mathcal{B}_\ell" />, given the quantized model&apos;s
          actual input at that block, optimize all norm vectors jointly:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-4">
          <Math
            expr="\min_{\{\alpha^{(i)}_\ell\}} \; \underbrace{\|h_\ell^{\text{fp}} - h_\ell^{\text{tq}}\|^2}_{\text{MSE}} + \lambda \left( \underbrace{1 - \cos(h^{\text{fp}}, h^{\text{tq}})}_{\text{angular}} + \underbrace{D_{\text{KL}}(\sigma(h^{\text{fp}}) \| \sigma(h^{\text{tq}}))}_{\text{distributional}} \right)"
            display
          />
        </div>
        <div className="grid sm:grid-cols-3 gap-4 mb-4">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-4 text-center">
              <div className="text-lg mb-1">📐</div>
              <h4 className="font-semibold text-xs mb-1">MSE</h4>
              <p className="text-[11px] text-txt-2">Matches output magnitude — the primary reconstruction signal.</p>
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <div className="bg-bg-2 border border-border rounded-xl p-4 text-center">
              <div className="text-lg mb-1">🧭</div>
              <h4 className="font-semibold text-xs mb-1">Angular</h4>
              <p className="text-[11px] text-txt-2">Preserves direction — critical for attention dot products.</p>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="bg-bg-2 border border-border rounded-xl p-4 text-center">
              <div className="text-lg mb-1">📊</div>
              <h4 className="font-semibold text-xs mb-1">KLD</h4>
              <p className="text-[11px] text-txt-2">Preserves distribution shape across the feature dimension.</p>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ─── Parameterization ─── */}
      <Section title="Exponential Parameterization">
        <p className="text-txt-2 leading-relaxed mb-4">
          Instead of optimizing <Math expr="\alpha_m" /> directly, we use a multiplicative correction:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-4">
          <Math
            expr="\hat{\alpha}_m = \alpha_m \cdot \exp(\beta_m), \quad \beta_m \in \mathbb{R}, \quad \beta_m^{(0)} = 0"
            display
          />
        </div>
        <div className="grid sm:grid-cols-3 gap-4">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-4">
              <h4 className="font-semibold text-xs mb-1 text-accent-green">Identity at init</h4>
              <p className="text-[11px] text-txt-2"><Math expr="\exp(0) = 1" />, so the initial solution is the analytical norm.</p>
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <div className="bg-bg-2 border border-border rounded-xl p-4">
              <h4 className="font-semibold text-xs mb-1 text-accent-green">Always positive</h4>
              <p className="text-[11px] text-txt-2"><Math expr="\exp(\beta) > 0" /> for all <Math expr="\beta" />, ensuring valid norms.</p>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="bg-bg-2 border border-border rounded-xl p-4">
              <h4 className="font-semibold text-xs mb-1 text-accent-green">Smooth landscape</h4>
              <p className="text-[11px] text-txt-2">Multiplicative perturbation avoids scale sensitivity near zero.</p>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ─── Algorithm ─── */}
      <Section title="Algorithm">
        <div className="space-y-3">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0 bg-accent/10 text-accent">1</div>
                <div>
                  <h4 className="font-semibold text-sm mb-1">Pre-capture FP targets</h4>
                  <p className="text-xs text-txt-2 leading-relaxed">
                    Run one full-precision forward pass, capturing all <Math expr="L" /> block outputs.
                    Then offload the FP model to CPU to free GPU memory.
                  </p>
                </div>
              </div>
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0 bg-accent-green/10 text-accent-green">2</div>
                <div>
                  <h4 className="font-semibold text-sm mb-1">Sequential block optimization</h4>
                  <p className="text-xs text-txt-2 leading-relaxed">
                    For each block <Math expr="\ell = 1, \ldots, L" />: disable fused kernels,
                    create learnable <Math expr="\beta" /> parameters, run AdamW for{" "}
                    <Math expr="T" /> iterations against the FP target, fold the optimal correction
                    into the norms, then restore fused kernels.
                  </p>
                </div>
              </div>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0 bg-accent-orange/10 text-accent-orange">3</div>
                <div>
                  <h4 className="font-semibold text-sm mb-1">Propagate through calibrated blocks</h4>
                  <p className="text-xs text-txt-2 leading-relaxed">
                    After calibrating block <Math expr="\ell" />, run a forward pass through the
                    now-calibrated model to capture the actual input for block{" "}
                    <Math expr="\ell + 1" />. This ensures each block sees the correct error landscape.
                  </p>
                </div>
              </div>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ─── Results ─── */}
      <Section title="Results (Qwen3.5-0.8B-Base)">
        <div className="overflow-x-auto mb-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-[#f0883e] text-xs uppercase">Method</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">PPL</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">ΔPPL</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">KLD</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">ΔKLD</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Time</th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              <tr className="border-b border-border">
                <td className="py-3 px-4">Analytical norms</td>
                <td className="py-3 px-4 font-mono">13.9564</td>
                <td className="py-3 px-4 font-mono text-txt-2">—</td>
                <td className="py-3 px-4 font-mono">0.1301</td>
                <td className="py-3 px-4 font-mono text-txt-2">—</td>
                <td className="py-3 px-4 font-mono text-txt-2">—</td>
              </tr>
              <tr className="border-b border-border bg-red-500/5">
                <td className="py-3 px-4 text-red-400">Per-layer cal</td>
                <td className="py-3 px-4 font-mono">14.0220</td>
                <td className="py-3 px-4 font-mono text-red-400">+0.066</td>
                <td className="py-3 px-4 font-mono">0.1352</td>
                <td className="py-3 px-4 font-mono text-red-400">+0.005</td>
                <td className="py-3 px-4 font-mono">~35 min</td>
              </tr>
              <tr className="border-b border-border bg-[#f0883e]/5">
                <td className="py-3 px-4 text-[#f0883e] font-semibold">Blockwise (4s / 50i) ✨</td>
                <td className="py-3 px-4 font-mono font-semibold">13.6971</td>
                <td className="py-3 px-4 font-mono text-accent-green font-semibold">−0.259</td>
                <td className="py-3 px-4 font-mono font-semibold">0.1170</td>
                <td className="py-3 px-4 font-mono text-accent-green font-semibold">−0.013</td>
                <td className="py-3 px-4 font-mono font-semibold">12.9 min</td>
              </tr>
              <tr className="border-b border-border bg-[#f0883e]/3">
                <td className="py-3 px-4 text-[#f0883e]">Blockwise (16s / 200i)</td>
                <td className="py-3 px-4 font-mono">13.7079</td>
                <td className="py-3 px-4 font-mono text-accent-green">−0.249</td>
                <td className="py-3 px-4 font-mono">0.1165</td>
                <td className="py-3 px-4 font-mono text-accent-green">−0.014</td>
                <td className="py-3 px-4 font-mono">50.7 min</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-txt-2 leading-relaxed text-sm mb-2">
          bf16 baseline PPL: <strong className="text-txt">12.1303</strong>.
          The quantization gap is <Math expr="13.9564 - 12.1303 = 1.826" />. Blockwise calibration
          recovers <strong className="text-txt">14.2%</strong> of that gap.
        </p>
        <Reveal>
          <div className="bg-accent-green/5 border border-accent-green/20 rounded-xl p-5">
            <h4 className="font-semibold text-sm text-accent-green mb-2">4 samples, 50 iters is optimal</h4>
            <p className="text-xs text-txt-2 leading-relaxed">
              Surprisingly, fewer samples avoid overfitting to the calibration set while still capturing
              the block-level error structure. The 4× faster configuration achieves equal or
              slightly better quality than the heavy setting.
            </p>
          </div>
        </Reveal>
      </Section>

      {/* ─── When to use ─── */}
      <Section title="When to Use">
        <div className="grid sm:grid-cols-2 gap-4">
          <Reveal>
            <div className="bg-accent-green/5 border border-accent-green/20 rounded-xl p-5">
              <div className="text-xl mb-2">✅</div>
              <h4 className="font-semibold text-sm mb-1">4-bit single-pass</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Significant PPL and KLD improvement. Adds ~13 min for 0.8B models.
                Enabled with <code className="text-accent text-[11px] bg-accent/10 px-1 rounded">--calibrate</code>.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-5">
              <div className="text-xl mb-2">⏭️</div>
              <h4 className="font-semibold text-sm mb-1">4+4 residual</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Already near-perfect (<Math expr="\cos \approx 1.000" />). Calibration has zero
                effect on PPL — skip it and save hours.
              </p>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ─── Relationship to other techniques ─── */}
      <Section title="Relationship to Other Techniques">
        <div className="grid sm:grid-cols-2 gap-4">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">📐</div>
              <h4 className="font-semibold text-sm mb-1">
                <Link href="/techniques/norm-compression/" className="text-accent hover:underline">
                  Norm Compression
                </Link>
              </h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Calibration modifies the norm values; norm compression reduces their storage.
                Apply calibration first, then compress the calibrated norms.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">🧮</div>
              <h4 className="font-semibold text-sm mb-1">
                <Link href="/techniques/lloyd-max/" className="text-accent hover:underline">
                  Lloyd-Max Codebook
                </Link>
              </h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Calibration only adjusts norms — the codebook and quantized indices remain fixed.
                This makes it a lightweight post-quantization step.
              </p>
            </div>
          </Reveal>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
