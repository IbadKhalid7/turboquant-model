"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { Reveal } from "@/components/Reveal";
import Link from "next/link";

export default function NormCodecPage() {
  return (
    <TechniqueLayout
      title="Norm Compression"
      subtitle="Rank-1 SVD factorization with int8 residual reduces norm storage by ~4× — targeting the second-largest BPW term."
      color="#79c0ff"
      icon="📐"
      prev={{ href: "/techniques/entropy-codec/", label: "Entropy Coding" }}
      next={{ href: "/techniques/qjl/", label: "QJL" }}
    >
      {/* ─── Which part of the BPW budget ─── */}
      <Section title="Where It Fits: The Norm Term">
        <p className="text-txt-2 leading-relaxed mb-4">
          In the{" "}
          <Link href="/formulation/" className="text-accent hover:underline">
            quantization formulation
          </Link>
          , the <strong className="text-txt">norm tensor</strong>{" "}
          <Math expr="\alpha_{m,g} \in \mathbb{R}^{M \times G}" /> is
          the second-largest storage component after quantized indices:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="\text{BPW} \approx b + \underbrace{\frac{32 \cdot n_{\text{norm}}}{d}}_{\text{norm overhead (target)}} + \text{BPW}_{\text{non-quant}}"
            display
          />
        </div>
        <p className="text-txt-2 leading-relaxed">
          At <Math expr="d = 128" /> with float32 norms, each pass contributes{" "}
          <Math expr="32/128 = 0.25" /> BPW. With two residual passes, this becomes{" "}
          <strong className="text-txt">0.50 BPW</strong> — a significant fraction of the total
          budget that norm compression directly reduces.
        </p>
      </Section>

      {/* ─── The Idea ─── */}
      <Section title="Rank-1 Factorization">
        <p className="text-txt-2 leading-relaxed mb-4">
          The norm tensor has strong low-rank structure: rows of the same layer tend to have
          similar magnitude patterns across groups. This motivates a rank-1 SVD approximation
          with a small int8 correction:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="\alpha_{m,g} \approx \beta_m \cdot \gamma_g \cdot (1 + \varepsilon_{m,g})"
            display
          />
        </div>
        <div className="grid sm:grid-cols-3 gap-4">
          <Reveal>
            <div className="bg-bg-3 border border-border rounded-xl p-5 text-center">
              <div className="text-2xl mb-2 text-[#79c0ff]">β<sub>m</sub></div>
              <div className="text-xs text-txt-2">Row scale (float16)</div>
              <div className="text-xs text-txt-2 mt-1">Per-row magnitude</div>
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <div className="bg-bg-3 border border-border rounded-xl p-5 text-center">
              <div className="text-2xl mb-2 text-accent-green">γ<sub>g</sub></div>
              <div className="text-xs text-txt-2">Group scale (float16)</div>
              <div className="text-xs text-txt-2 mt-1">Per-group pattern</div>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="bg-bg-3 border border-border rounded-xl p-5 text-center">
              <div className="text-2xl mb-2 text-accent-orange">ε<sub>m,g</sub></div>
              <div className="text-xs text-txt-2">Residual (int8)</div>
              <div className="text-xs text-txt-2 mt-1">Fractional correction</div>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ─── Steps ─── */}
      <Section title="How It Works">
        <div className="space-y-4">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0 bg-[#79c0ff]/10 text-[#79c0ff]">1</div>
                <div>
                  <h4 className="font-semibold text-sm mb-1">SVD of the norm matrix</h4>
                  <p className="text-xs text-txt-2 leading-relaxed">
                    Compute <Math expr="\alpha = U \Sigma V^T" /> and take the first singular vector:{" "}
                    <Math expr="\beta_m = \sigma_1 u_{1,m}" />, <Math expr="\gamma_g = v_{1,g}" />.
                  </p>
                </div>
              </div>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0 bg-accent-green/10 text-accent-green">2</div>
                <div>
                  <h4 className="font-semibold text-sm mb-1">Compute fractional residual</h4>
                  <div className="bg-bg-3 rounded-lg p-3 text-center mt-2">
                    <Math expr="\varepsilon_{m,g} = \frac{\alpha_{m,g}}{\beta_m \gamma_g} - 1" display />
                  </div>
                </div>
              </div>
            </div>
          </Reveal>
          <Reveal delay={0.2}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0 bg-accent-orange/10 text-accent-orange">3</div>
                <div>
                  <h4 className="font-semibold text-sm mb-1">Quantize residual to int8</h4>
                  <p className="text-xs text-txt-2 leading-relaxed">
                    Symmetric quantization: <Math expr="s = \max|\varepsilon| / 127" />, then{" "}
                    <Math expr="\hat{\varepsilon} = \text{round}(\varepsilon / s)" />.
                    Typically <Math expr="|\varepsilon| < 0.5\%" /> of the norm value.
                  </p>
                </div>
              </div>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ─── Storage comparison ─── */}
      <Section title="Storage Comparison">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-[#79c0ff] text-xs uppercase">Method</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Components</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">BPW (d=128)</th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              <tr className="border-b border-border">
                <td className="py-3 px-4">float32 (baseline)</td>
                <td className="py-3 px-4"><Math expr="M \cdot G \cdot 32" /> bits</td>
                <td className="py-3 px-4 font-mono">0.250</td>
              </tr>
              <tr className="border-b border-border">
                <td className="py-3 px-4">float16</td>
                <td className="py-3 px-4"><Math expr="M \cdot G \cdot 16" /> bits</td>
                <td className="py-3 px-4 font-mono">0.125</td>
              </tr>
              <tr className="border-b border-border bg-[#79c0ff]/5">
                <td className="py-3 px-4 text-[#79c0ff] font-semibold">Factored int8 ✨</td>
                <td className="py-3 px-4"><Math expr="M \cdot 16 + G \cdot 16 + M \cdot G \cdot 8 + 32" /></td>
                <td className="py-3 px-4 font-mono text-accent-green font-semibold">~0.063</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-xs text-txt-2 mt-2">
          The factored representation achieves <strong className="font-semibold text-txt">~4× compression</strong> vs
          float32 norms, saving <strong className="font-semibold text-txt">~0.19 BPW per pass</strong>.
        </p>
      </Section>

      {/* ─── Reconstruction ─── */}
      <Section title="Reconstruction">
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-4">
          <Math
            expr="\hat{\alpha}_{m,g} = \beta_m \cdot \gamma_g \cdot (1 + s \cdot \hat{\varepsilon}_{m,g})"
            display
          />
        </div>
        <p className="text-txt-2 leading-relaxed text-sm">
          The reconstruction error is bounded by the int8 quantization granularity:{" "}
          <Math expr="|\alpha - \hat{\alpha}| \leq \beta_m \gamma_g \cdot s" />, which is
          typically less than 0.5% of the norm value.
        </p>
      </Section>

      {/* ─── Relationship to other techniques ─── */}
      <Section title="Relationship to Other Techniques">
        <div className="grid sm:grid-cols-2 gap-4">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">🔄</div>
              <h4 className="font-semibold text-sm mb-1">Row Normalization (Step 1)</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                The norm tensor is produced during the normalization step of the quantization
                pipeline. This codec compresses that output.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">🎯</div>
              <h4 className="font-semibold text-sm mb-1">Residual Quantization</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Each residual pass produces its own norm tensor. Factorization is especially
                beneficial here since residual norms are highly structured.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">🗜️</div>
              <h4 className="font-semibold text-sm mb-1">Entropy Coding</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Entropy coding compresses the index tensor; norm factorization compresses the
                norm tensor. Together they address both major storage components.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.15}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">📐</div>
              <h4 className="font-semibold text-sm mb-1">BPW Budget</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                At <Math expr="d = 128" />, switching from float32 to factored int8 saves ~0.19 BPW
                per pass — directly reducing the norm overhead in the formulation.
              </p>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ─── Implementation ─── */}
      <Section title="Implementation">
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm space-y-2">
          <div>
            <span className="text-accent-purple">norm_compression.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">factorize_norms()</span>{" "}
            <span className="text-txt-2">(rank-1 SVD + int8 residual)</span>
          </div>
          <div>
            <span className="text-accent-purple">norm_compression.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">reconstruct_norms()</span>{" "}
            <span className="text-txt-2">(reconstruct from factored form)</span>
          </div>
          <div>
            <span className="text-accent-purple">norm_compression.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">norm_bpw()</span>{" "}
            <span className="text-txt-2">(compute BPW overhead for any method)</span>
          </div>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
