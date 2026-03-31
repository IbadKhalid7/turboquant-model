"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { Reveal } from "@/components/Reveal";

export default function QjlPage() {
  return (
    <TechniqueLayout
      title="Quantized Johnson-Lindenstrauss (QJL)"
      subtitle="A 1-bit random projection technique for unbiased inner product estimation — elegant for KV-cache attention, but not the right tool for offline weight compression."
      color="#d2a8ff"
      icon="📐"
      prev={{ href: "/techniques/norm-codec/", label: "Norm Compression" }}
      next={{ href: "/quantize-pipeline/", label: "Quantize Pipeline" }}
    >
      {/* ── THE TECHNIQUE ── */}

      <Section title="The Johnson-Lindenstrauss Lemma">
        <p className="text-txt-2 leading-relaxed mb-4">
          The JL lemma (1984) states that any set of <Math expr="n" /> points in high-dimensional
          space can be embedded into <Math expr="O(\log n / \varepsilon^2)" /> dimensions while
          preserving all pairwise distances within a factor of <Math expr="1 \pm \varepsilon" />.
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-4">
          <Math
            expr="(1 - \varepsilon)\|u - v\|^2 \;\leq\; \|f(u) - f(v)\|^2 \;\leq\; (1 + \varepsilon)\|u - v\|^2"
            display
          />
        </div>
        <p className="text-txt-2 leading-relaxed">
          The projection is a random linear map — a matrix with i.i.d. Gaussian or
          sub-Gaussian entries, scaled appropriately. This is the theoretical foundation
          behind QJL.
        </p>
      </Section>

      <Section title="How QJL Works">
        <p className="text-txt-2 leading-relaxed mb-4">
          QJL (Zandieh et al., 2024) takes the JL idea further: instead of storing the full
          projected coordinates, it keeps only the <strong className="text-txt">sign</strong> — just
          1 bit per projection. Given <Math expr="m" /> random directions{" "}
          <Math expr="r_1, \ldots, r_m" />, the inner product estimator is:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="\hat{\langle q, k \rangle} = \frac{\|q\| \cdot \|k\|}{m} \sum_{i=1}^{m} \text{sign}(\langle r_i, q \rangle) \cdot \text{sign}(\langle r_i, k \rangle)"
            display
          />
        </div>

        <div className="bg-bg-2 border border-border rounded-xl p-6 space-y-4">
          <h4 className="font-semibold text-sm text-accent-purple">Key Properties</h4>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-bg-3 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">🎯</div>
              <div className="text-sm font-semibold mb-1">Unbiased</div>
              <div className="text-xs text-txt-2">
                <Math expr="\mathbb{E}[\hat{\langle q,k \rangle}] = \langle q,k \rangle" />
              </div>
            </div>
            <div className="bg-bg-3 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">💾</div>
              <div className="text-sm font-semibold mb-1">1 bit per projection</div>
              <div className="text-xs text-txt-2">Store only sign(⟨r<sub>i</sub>, v⟩)</div>
            </div>
            <div className="bg-bg-3 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">⚡</div>
              <div className="text-sm font-semibold mb-1">Zero decode overhead</div>
              <div className="text-xs text-txt-2">Sign comparisons via bitwise XOR + popcount</div>
            </div>
          </div>
        </div>
      </Section>

      <Section title="QJL in the TurboQuant Paper">
        <p className="text-txt-2 leading-relaxed mb-4">
          The paper defines <strong className="text-accent-purple">TurboQuant<sub>prod</sub></strong>,
          which combines standard TurboQuant with a QJL correction for an unbiased inner product
          estimator:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 space-y-3">
          <div className="flex gap-3 items-start">
            <span className="w-6 h-6 rounded-full bg-accent-purple/10 text-accent-purple text-xs font-bold flex items-center justify-center shrink-0">1</span>
            <span className="text-sm text-txt-2">Quantize the vector using TurboQuant (rotation + Lloyd-Max) → <Math expr="\tilde{k}" /></span>
          </div>
          <div className="flex gap-3 items-start">
            <span className="w-6 h-6 rounded-full bg-accent-purple/10 text-accent-purple text-xs font-bold flex items-center justify-center shrink-0">2</span>
            <span className="text-sm text-txt-2">Compute residual <Math expr="e = k - \tilde{k}" /></span>
          </div>
          <div className="flex gap-3 items-start">
            <span className="w-6 h-6 rounded-full bg-accent-purple/10 text-accent-purple text-xs font-bold flex items-center justify-center shrink-0">3</span>
            <span className="text-sm text-txt-2">Apply <strong className="text-accent-purple">1-bit QJL</strong> to <Math expr="e" /> for an unbiased correction: <Math expr="\hat{\langle q,k \rangle} = \langle q, \tilde{k} \rangle + \widehat{\langle q, e \rangle}_{\text{QJL}}" /></span>
          </div>
        </div>
        <p className="text-txt-2 leading-relaxed mt-4">
          This makes the overall estimator <strong className="text-txt">unbiased</strong> — critical
          for KV-cache attention where you quantize keys once and query with many different
          vectors over the sequence lifetime.
        </p>
      </Section>

      {/* ── WHY NOT IN THIS PROJECT ── */}

      <Section title="Why This Project Doesn't Use QJL">
        <p className="text-txt-2 leading-relaxed mb-6">
          QJL is designed for a fundamentally different use case. Here are the four reasons
          we chose multi-pass residual quantization instead.
        </p>
        <div className="space-y-4">
          <Reveal>
            <ReasonCard num="1" title="Different Problem: Online vs Offline" color="#f85149">
              QJL is designed for <strong className="text-txt">online inner product estimation</strong> — quantize
              once, query many times with different vectors. Weight quantization is{" "}
              <strong className="text-txt">offline</strong>: we compress <Math expr="W" /> once and compute{" "}
              <Math expr="y = xW^T" /> repeatedly. We want minimum reconstruction error{" "}
              <Math expr="\|W - \tilde{W}\|" />, not an unbiased dot-product estimator.
            </ReasonCard>
          </Reveal>

          <Reveal delay={0.1}>
            <ReasonCard num="2" title="Unbiasedness Is Unnecessary for Weights" color="#ffa657">
              A small deterministic bias from MSE-optimal quantization is absorbed by layer norms,
              residual connections, and softmax normalization. An unbiased but{" "}
              <strong className="text-txt">high-variance</strong> estimator (QJL at 1 bit) introduces
              stochastic noise that changes every forward pass — worse for stable inference.
            </ReasonCard>
          </Reveal>

          <Reveal delay={0.2}>
            <ReasonCard num="3" title="Residual Quantization Strictly Dominates" color="#7ee787">
              <p>
                QJL uses <strong className="text-txt">1 bit</strong> (random sign projection) for the
                residual correction. Our residual pass uses <Math expr="b_2" /> bits with a full
                Lloyd-Max codebook + independent rotation — capturing far more residual information.
              </p>
              <div className="mt-3 grid grid-cols-2 gap-3">
                <div className="bg-bg-3 rounded-lg p-3 text-center">
                  <div className="text-sm font-bold" style={{ color: "#f85149" }}>QJL correction</div>
                  <div className="text-xs text-txt-2 mt-1">1 bit per weight</div>
                  <div className="text-xs text-txt-2">Random sign only</div>
                </div>
                <div className="bg-bg-3 rounded-lg p-3 text-center">
                  <div className="text-sm font-bold text-accent-green">Residual TQ</div>
                  <div className="text-xs text-txt-2 mt-1">4 bits per weight</div>
                  <div className="text-xs text-txt-2">Full Lloyd-Max codebook</div>
                </div>
              </div>
              <p className="mt-3">
                At 4+4 total bits, residual TurboQuant achieves KL divergence of only{" "}
                <strong className="text-accent-green">0.002 nats</strong> (practically lossless). A 1-bit QJL
                correction cannot compete.
              </p>
            </ReasonCard>
          </Reveal>

          <Reveal delay={0.3}>
            <ReasonCard num="4" title="QJL Requires the Query at Runtime" color="#d2a8ff">
              The QJL correction term depends on the input activation <Math expr="x" />, making it
              incompatible with offline weight compression. You&apos;d need to recompute corrections per
              forward pass — defeating the purpose of weight-only quantization.
            </ReasonCard>
          </Reveal>
        </div>
      </Section>

      <Section title="Visual Comparison">
        <div className="grid md:grid-cols-2 gap-6">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-6 relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-[3px] bg-gradient-to-r from-accent-purple/70 to-transparent" />
              <h4 className="font-semibold text-sm mb-4 text-accent-purple">
                TurboQuant<sub>prod</sub> (Paper)
              </h4>
              <div className="space-y-2 text-xs text-txt-2">
                <div className="bg-bg-3 rounded-lg p-2.5 flex items-center gap-2">
                  <span className="text-accent-purple">Pass 1:</span> Lloyd-Max quantize (b₁ bits)
                </div>
                <div className="text-center text-accent text-sm">+</div>
                <div className="bg-bg-3 rounded-lg p-2.5 flex items-center gap-2">
                  <span style={{ color: "#f85149" }}>Pass 2:</span> QJL 1-bit sign projection on residual
                </div>
                <div className="text-center text-accent text-sm">↓</div>
                <div className="bg-bg-3 rounded-lg p-2.5">
                  <strong className="text-txt">Unbiased</strong> inner product estimator.
                  Needs query x at runtime.
                </div>
              </div>
            </div>
          </Reveal>

          <Reveal delay={0.1}>
            <div className="bg-bg-2 border border-accent-green/30 rounded-xl p-6 relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-[3px] bg-gradient-to-r from-accent-green/70 to-transparent" />
              <h4 className="text-accent-green font-semibold text-sm mb-4">
                This Project (Residual TQ)
              </h4>
              <div className="space-y-2 text-xs text-txt-2">
                <div className="bg-bg-3 rounded-lg p-2.5 flex items-center gap-2">
                  <span className="text-accent-green">Pass 1:</span> Full TQ: rotate + Lloyd-Max (4 bits)
                </div>
                <div className="text-center text-accent text-sm">+</div>
                <div className="bg-bg-3 rounded-lg p-2.5 flex items-center gap-2">
                  <span className="text-accent-green">Pass 2:</span> Full TQ on residual (4 bits, new codebook)
                </div>
                <div className="text-center text-accent text-sm">↓</div>
                <div className="bg-bg-3 rounded-lg p-2.5">
                  <strong className="text-accent-green">Near-lossless</strong> weight compression.
                  Offline, no runtime dependency.
                </div>
              </div>
            </div>
          </Reveal>
        </div>
      </Section>

      <Section title="Summary">
        <div className="bg-bg-2 border border-accent-green/20 rounded-xl p-6">
          <p className="text-txt-2 leading-relaxed">
            QJL is an elegant technique rooted in the JL lemma — perfect for
            streaming / KV-cache inner product preservation with 1-bit signed projections.
            For <strong className="text-txt">offline weight compression</strong>, multi-pass residual
            quantization with optimal scalar codebooks is the natural and superior choice — achieving
            practically lossless results at 4+4 bits with no runtime overhead.
          </p>
        </div>
      </Section>

      <Section title="References">
        <div className="space-y-2 text-sm text-txt-2">
          <p>
            <strong className="text-txt">QJL:</strong> Zandieh et al., &quot;QJL: 1-Bit Quantized JL
            Transform for KV Cache Quantization with Zero Overhead,&quot; 2024.
          </p>
          <p>
            <strong className="text-txt">Johnson-Lindenstrauss:</strong> W. Johnson &amp; J. Lindenstrauss,
            &quot;Extensions of Lipschitz mappings into a Hilbert space,&quot; Contemporary Mathematics, 1984.
          </p>
          <p>
            <strong className="text-txt">TurboQuant:</strong> Zandieh et al., &quot;TurboQuant: Online Vector
            Quantization with Near-optimal Distortion Rate,&quot; arXiv:2504.19874, 2025.
          </p>
        </div>
      </Section>
    </TechniqueLayout>
  );
}

function ReasonCard({
  num,
  title,
  color,
  children,
}: {
  num: string;
  title: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-bg-2 border border-border rounded-xl p-6">
      <div className="flex items-start gap-4">
        <div
          className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold shrink-0"
          style={{ background: `${color}15`, color }}
        >
          {num}
        </div>
        <div>
          <h3 className="font-semibold mb-2">{title}</h3>
          <div className="text-sm text-txt-2 leading-relaxed">{children}</div>
        </div>
      </div>
    </div>
  );
}
