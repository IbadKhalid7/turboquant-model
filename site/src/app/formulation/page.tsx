"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { Reveal } from "@/components/Reveal";
import { DistortionChart } from "@/components/animations/DistortionChart";
import { BPWBreakdownViz } from "@/components/animations/BPWBreakdownViz";

/* ── Step Card (mirrors quantize-pipeline pattern) ── */
function StepCard({
  num,
  title,
  color,
  equations,
  children,
}: {
  num: string;
  title: string;
  color: string;
  equations: string[];
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
        <div className="flex-1">
          <h3 className="font-semibold mb-2">{title}</h3>
          <p className="text-sm text-txt-2 leading-relaxed mb-3">{children}</p>
          {equations.map((eq, i) => (
            <div key={i} className="bg-bg-3 rounded-lg p-3 text-center mt-2">
              <Math expr={eq} display />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── Research Direction Card ── */
function ResearchCard({
  icon,
  title,
  saving,
  children,
}: {
  icon: string;
  title: string;
  saving: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-bg-2 border border-border rounded-xl p-5">
      <div className="flex items-start gap-3">
        <span className="text-xl shrink-0">{icon}</span>
        <div>
          <div className="font-semibold text-sm mb-1">{title}</div>
          <span className="inline-block text-xs bg-accent-green/10 text-accent-green rounded px-2 py-0.5 mb-2">
            {saving}
          </span>
          <p className="text-xs text-txt-2 leading-relaxed">{children}</p>
        </div>
      </div>
    </div>
  );
}

export default function FormulationPage() {
  return (
    <TechniqueLayout
      title="Quantization Formulation"
      subtitle="Mathematical foundations: from problem statement to near-optimal compression in b bits per weight."
      color="#d2a8ff"
      icon="📐"
      prev={{ href: "/dequantize-pipeline/", label: "Dequantize Pipeline" }}
    >
      {/* ─── 1. Problem Statement ─── */}
      <Section title="Problem Statement">
        <p className="text-txt-2 leading-relaxed mb-6">
          Given a pre-trained weight matrix{" "}
          <Math expr="W \in \mathbb{R}^{M \times N}" />, find a compressed
          representation <Math expr="\hat{W}" /> that minimizes mean squared
          reconstruction error, subject to using only{" "}
          <Math expr="b" /> bits per element plus a small side-information
          budget (norms, codebook, seed).
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center">
          <Math
            expr="\min_{\hat{W}} \;\; \frac{1}{MN} \|W - \hat{W}\|_F^2"
            display
          />
        </div>
      </Section>

      {/* ─── 2. Notation ─── */}
      <Section title="Notation">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-accent-purple text-xs uppercase">
                  Symbol
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  Meaning
                </th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              {[
                ["W \\in \\mathbb{R}^{M \\times N}", "Full-precision weight matrix (M = out, N = in)"],
                ["b", "Bit-width of the quantizer (L = 2^b levels)"],
                ["d", "Group size (columns processed together)"],
                ["G = N/d", "Number of groups"],
                ["\\Pi_g \\in \\mathbb{R}^{d \\times d}", "Random orthogonal rotation for group g"],
                ["\\alpha_{m,g}", "Row norm of group g, row m"],
                ["Q_b : \\mathbb{R} \\to \\{c_1,\\ldots,c_L\\}", "Scalar quantizer mapping to L centroids"],
                ["\\{c_\\ell\\}_{\\ell=1}^L", "Lloyd-Max codebook (centroids)"],
                ["\\{t_\\ell\\}_{\\ell=0}^L", "Decision boundaries (t_0 = -\\infty, t_L = +\\infty)"],
              ].map(([sym, meaning], i) => (
                <tr key={i} className="border-b border-border">
                  <td className="py-3 px-4">
                    <Math expr={sym} />
                  </td>
                  <td className="py-3 px-4">{meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* ─── 3. Single-Pass Pipeline ─── */}
      <Section title="Single-Pass Pipeline">
        <p className="text-txt-2 leading-relaxed mb-6">
          For each group <Math expr="g" /> and each row <Math expr="m" />, the
          pipeline proceeds in five steps. Columns are partitioned into{" "}
          <Math expr="G = N/d" /> groups, and each group is quantized
          independently.
        </p>
        <div className="space-y-6">
          <Reveal>
            <StepCard
              num="1"
              title="Row Normalization"
              color="#58a6ff"
              equations={[
                "\\alpha_{m,g} = \\|W^{(g)}_m\\|_2, \\qquad \\bar{W}^{(g)}_m = \\frac{W^{(g)}_m}{\\alpha_{m,g}}",
              ]}
            >
              Extract the row norm and normalize. After this step,{" "}
              <Math expr="\|\bar{W}^{(g)}_m\|_2 = 1" /> and each component has
              expected magnitude <Math expr="1/\sqrt{d}" />.
            </StepCard>
          </Reveal>
          <Reveal delay={0.1}>
            <StepCard
              num="2"
              title="Random Rotation"
              color="#7ee787"
              equations={[
                "Y^{(g)}_m = \\bar{W}^{(g)}_m \\cdot \\Pi_g^T \\in \\mathbb{R}^d",
              ]}
            >
              Apply a random orthogonal transform from the Haar measure on{" "}
              <Math expr="\mathcal{O}(d)" />. Because <Math expr="\Pi_g" /> is
              orthogonal, the norm is preserved and each component satisfies{" "}
              <Math expr="Y^{(g)}_{m,k} \sim \text{approx.}\; \mathcal{N}(0, 1/d)" />.
            </StepCard>
          </Reveal>
          <Reveal delay={0.2}>
            <StepCard
              num="3"
              title="Variance Normalization"
              color="#d2a8ff"
              equations={[
                "Z^{(g)}_{m,k} = \\sqrt{d} \\;\\cdot\\; Y^{(g)}_{m,k}",
              ]}
            >
              Scale to unit variance so each scalar satisfies{" "}
              <Math expr="Z^{(g)}_{m,k} \sim \text{approx.}\; \mathcal{N}(0,1)" /> —
              exactly matching the Lloyd-Max codebook distribution.
            </StepCard>
          </Reveal>
          <Reveal delay={0.3}>
            <StepCard
              num="4"
              title="Lloyd-Max Quantization"
              color="#ffa657"
              equations={[
                "\\hat{Z}^{(g)}_{m,k} = Q_b(Z^{(g)}_{m,k}) = c_\\ell \\quad \\text{where } \\ell = \\arg\\min_j |Z^{(g)}_{m,k} - c_j|",
              ]}
            >
              Each scalar is independently quantized using the optimal{" "}
              <Math expr="\mathcal{N}(0,1)" /> boundaries. At 4 bits: 16
              centroids, 15 decision boundaries.
            </StepCard>
          </Reveal>
          <Reveal delay={0.4}>
            <StepCard
              num="5"
              title="Reconstruction"
              color="#58a6ff"
              equations={[
                "\\hat{W}^{(g)}_m = \\frac{\\alpha_{m,g}}{\\sqrt{d}} \\cdot \\hat{Z}^{(g)}_m \\cdot \\Pi_g",
              ]}
            >
              Undo the rotation and rescale by the stored norm to obtain the
              quantized approximation in the original coordinate space.
            </StepCard>
          </Reveal>
        </div>
      </Section>

      {/* ─── 4. MSE Analysis ─── */}
      <Section title="MSE Analysis">
        <p className="text-txt-2 leading-relaxed mb-4">
          Because orthogonal rotation preserves the Frobenius norm, the
          per-element reconstruction error factors cleanly:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="\mathbb{E}\!\left[(W^{(g)}_{m,k} - \hat{W}^{(g)}_{m,k})^2\right] = \frac{\alpha_{m,g}^2}{d} \cdot D_b"
            display
          />
        </div>
        <p className="text-txt-2 leading-relaxed mb-4">
          where <Math expr="D_b" /> is the distortion of the{" "}
          <Math expr="b" />-bit Lloyd-Max quantizer on{" "}
          <Math expr="\mathcal{N}(0,1)" />. The overall MSE is:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="\text{MSE} = D_b \cdot \overline{\alpha^2}"
            display
          />
        </div>
        <p className="text-txt-2 leading-relaxed mb-4">
          where{" "}
          <Math expr="\overline{\alpha^2} = \frac{1}{MN}\sum_{m,g}\alpha_{m,g}^2" />{" "}
          is the average squared norm per weight element.
        </p>

        {/* Lloyd-Max distortion table */}
        <h3 className="text-lg font-semibold mb-4 mt-8">
          Lloyd-Max Distortion Values
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-accent text-xs uppercase">
                  b (bits)
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  L (levels)
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  D<sub>b</sub>
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  SNR (dB)
                </th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              {[
                [1, 2, "0.3634", "4.40"],
                [2, 4, "0.1175", "9.30"],
                [3, 8, "0.03454", "14.62"],
                [4, 16, "0.009497", "20.22"],
                [5, 32, "0.002499", "26.02"],
              ].map(([b, L, Db, snr]) => (
                <tr
                  key={String(b)}
                  className={`border-b border-border ${b === 4 ? "bg-accent/5" : ""}`}
                >
                  <td className="py-3 px-4 font-mono text-accent">{b}</td>
                  <td className="py-3 px-4">{L}</td>
                  <td className="py-3 px-4 font-mono">{Db}</td>
                  <td className="py-3 px-4">{snr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-txt-2 mt-2">
          Each additional bit roughly halves the distortion (~6 dB improvement).
        </p>

        {/* Distortion chart visualization */}
        <div className="bg-bg-2 border border-border rounded-2xl p-6 mt-6">
          <DistortionChart />
        </div>
      </Section>

      {/* ─── 5. Near-Optimality ─── */}
      <Section title="Near-Optimality">
        <p className="text-txt-2 leading-relaxed mb-4">
          The Shannon rate-distortion function for{" "}
          <Math expr="\mathcal{N}(0,1)" /> at rate <Math expr="R" /> bits is:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math expr="D^*(R) = 2^{-2R}" display />
        </div>
        <p className="text-txt-2 leading-relaxed mb-4">
          At <Math expr="b = 4" /> bits,{" "}
          <Math expr="D^*(4) = 2^{-8} \approx 0.00391" />. The Lloyd-Max
          quantizer achieves <Math expr="D_4 = 0.00950" />, giving a gap of
          only:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math expr="\frac{D_4}{D^*(4)} \approx 2.43" display />
        </div>

        {/* Callout */}
        <div className="bg-accent-purple/5 border border-accent-purple/20 rounded-xl p-6">
          <h4 className="text-accent-purple font-semibold text-sm mb-2">
            Why rotation makes this possible
          </h4>
          <p className="text-sm text-txt-2 leading-relaxed">
            Without rotation, trained neural network weights are correlated and
            non-Gaussian. Scalar quantization operating per-coordinate leaves
            inter-coordinate redundancy unexploited. The random rotation
            decorrelates coordinates and projects them onto i.i.d. approximate
            Gaussians — reducing the problem to the case where scalar Lloyd-Max
            is near-optimal. The gap is only ~3.9 dB from the theoretical
            optimum and decreases for higher <Math expr="b" />.
          </p>
        </div>
      </Section>

      {/* ─── 6. Residual Quantization ─── */}
      <Section title="Residual Quantization">
        <p className="text-txt-2 leading-relaxed mb-4">
          The single-pass error can be reduced by iteratively quantizing the
          reconstruction residual. Define the residual sequence:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 space-y-2 font-mono text-sm mb-6">
          <div>
            <Math expr="R^{(0)} = W" />
          </div>
          <div>
            <Math expr="\hat{R}^{(k)} = \text{TQ}(R^{(k)}, b_k, s_k)" />
          </div>
          <div>
            <Math expr="R^{(k+1)} = R^{(k)} - \hat{R}^{(k)}" />
          </div>
          <div className="pt-2 border-t border-border">
            <Math expr="\hat{W} = \sum_{k=0}^{P-1} \hat{R}^{(k)}" />
          </div>
        </div>

        <p className="text-txt-2 leading-relaxed mb-4">
          After <Math expr="P" /> passes the total MSE is approximately:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="\text{MSE}_P \approx \overline{\alpha^2} \cdot \prod_{k=0}^{P-1} D_{b_k}"
            display
          />
        </div>

        {/* Effective bit-rate table */}
        <h3 className="text-lg font-semibold mb-4">
          Effective Bit-Rate Configurations
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-accent-orange text-xs uppercase">
                  Config
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  Passes
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  Bits / weight
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  Expected MSE ratio
                </th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              {[
                ["4-bit single", "1", "4", "D_4 \\approx 0.0095"],
                ["4+4 residual", "2", "8", "D_4^2 \\approx 9.0 \\times 10^{-5}"],
                ["4+2 residual", "2", "6", "D_4 \\cdot D_2 \\approx 1.1 \\times 10^{-3}"],
                ["2+2+2+2", "4", "8", "D_2^4 \\approx 1.9 \\times 10^{-4}"],
              ].map(([cfg, passes, bpw, mse], i) => (
                <tr
                  key={i}
                  className={`border-b border-border ${cfg === "4+4 residual" ? "bg-accent-green/5" : ""}`}
                >
                  <td className="py-3 px-4 font-mono text-accent-orange">
                    {cfg}
                  </td>
                  <td className="py-3 px-4">{passes}</td>
                  <td className="py-3 px-4">{bpw}</td>
                  <td className="py-3 px-4">
                    <Math expr={mse} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* ─── 7. Rotation Strategies ─── */}
      <Section title="Rotation Strategies">
        <p className="text-txt-2 leading-relaxed mb-6">
          The choice of rotation seed(s) across residual passes has a
          significant impact on error decorrelation and inference efficiency.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-accent-purple text-xs uppercase">
                  Strategy
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  Seeds
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  Advantage
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  Disadvantage
                </th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              <tr className="border-b border-border">
                <td className="py-3 px-4 font-semibold text-txt">
                  Independent
                </td>
                <td className="py-3 px-4">
                  <Math expr="s_k \neq s_j" />
                </td>
                <td className="py-3 px-4">
                  Errors projected onto different subspaces — maximizes error
                  reduction per pass
                </td>
                <td className="py-3 px-4">
                  Cannot merge passes in the rotated domain
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="py-3 px-4 font-semibold text-txt">Shared</td>
                <td className="py-3 px-4">
                  <Math expr="s_0 = s_1 = \cdots" />
                </td>
                <td className="py-3 px-4">
                  Enables fast-path merge — reduces storage and inference cost
                  to single-pass levels
                </td>
                <td className="py-3 px-4">
                  Correlated errors across passes; reduced benefit per pass
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="py-3 px-4 font-semibold text-txt">
                  Alternating
                </td>
                <td className="py-3 px-4">
                  <Math expr="s_a, s_b" /> (two seeds)
                </td>
                <td className="py-3 px-4">
                  Adjacent passes use different bases — near-independent error
                  reduction with only two rotation matrices
                </td>
                <td className="py-3 px-4">
                  Cannot use the fast-path merge
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      {/* ─── 8. Inference ─── */}
      <Section title="Inference">
        <p className="text-txt-2 leading-relaxed mb-4">
          Instead of materializing the full <Math expr="M \times N" /> weight
          matrix, inference operates in the rotated domain using the{" "}
          <strong className="text-txt">pre-rotate input trick</strong>.
        </p>
        <p className="text-txt-2 leading-relaxed mb-4">
          For input <Math expr="\mathbf{x} \in \mathbb{R}^N" /> and output{" "}
          <Math expr="y_m = W_m \cdot \mathbf{x}" />, substitute the
          reconstruction:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="y_m = \sum_g \frac{\alpha_{m,g}}{\sqrt{d}} \cdot \mathbf{c}[\boldsymbol{\ell}_m] \cdot \underbrace{(\Pi_g \mathbf{x}^{(g)})}_{\mathbf{x}_{\text{rot}}^{(g)}}"
            display
          />
        </div>
        <p className="text-txt-2 leading-relaxed mb-4">
          Pre-rotating the input <Math expr="\mathbf{x}_{\text{rot}}^{(g)} = \Pi_g \mathbf{x}^{(g)}" />{" "}
          reduces the inner product to a lookup + dot product:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="y_m = \sum_g \frac{\alpha_{m,g}}{\sqrt{d}} \sum_{k=1}^d c_{\ell_{m,k}} \cdot x_{\text{rot},k}^{(g)}"
            display
          />
        </div>

        {/* Cost table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-accent text-xs uppercase">
                  Operation
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  Cost
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  Comment
                </th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              <tr className="border-b border-border">
                <td className="py-3 px-4">Input rotation</td>
                <td className="py-3 px-4">
                  <Math expr="O(d^2)" /> or <Math expr="O(d \log d)" />
                </td>
                <td className="py-3 px-4">QR vs Hadamard</td>
              </tr>
              <tr className="border-b border-border">
                <td className="py-3 px-4">Codebook lookup</td>
                <td className="py-3 px-4">
                  <Math expr="O(Md)" />
                </td>
                <td className="py-3 px-4">Index → centroid</td>
              </tr>
              <tr className="border-b border-border">
                <td className="py-3 px-4">Fused dot product</td>
                <td className="py-3 px-4">
                  <Math expr="O(Md)" />
                </td>
                <td className="py-3 px-4">Sum of products</td>
              </tr>
              <tr>
                <td className="py-3 px-4">Norm rescaling</td>
                <td className="py-3 px-4">
                  <Math expr="O(M)" />
                </td>
                <td className="py-3 px-4">
                  Multiply by <Math expr="\alpha/\sqrt{d}" />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="bg-accent/5 border border-accent/20 rounded-xl p-5 mt-6">
          <p className="text-sm text-txt-2 leading-relaxed">
            <strong className="text-accent">Total:</strong>{" "}
            <Math expr="O(MN)" /> per forward pass — same asymptotic cost as
            dense matmul, but with much smaller memory footprint.
          </p>
        </div>
      </Section>

      {/* ─── 9. Compact Summary ─── */}
      <Section title="Compact Summary">
        <p className="text-txt-2 leading-relaxed mb-4">
          The TurboQuant quantization objective can be written compactly as:
        </p>
        <div className="bg-bg-2 border border-accent-purple/20 rounded-xl p-6 text-center mb-6">
          <Math
            expr="\hat{W} = \sum_{k=0}^{P-1} \text{TQ}(R^{(k)}, b_k, s_k)"
            display
          />
        </div>
        <p className="text-txt-2 leading-relaxed mb-4">where the single-pass operator is:</p>
        <div className="bg-bg-2 border border-accent-purple/20 rounded-xl p-6 text-center mb-6">
          <Math
            expr="\text{TQ}(X, b, s)^{(g)}_m = \frac{\|X^{(g)}_m\|_2}{\sqrt{d}} \cdot Q_b\!\left(\sqrt{d} \cdot \frac{X^{(g)}_m}{\|X^{(g)}_m\|_2} \cdot \Pi_g(s)^T\right) \cdot \Pi_g(s)"
            display
          />
        </div>

        {/* Why near-optimal */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {[
            { icon: "🔄", title: "Rotation", desc: "Decorrelates weight coordinates → i.i.d. approximate 𝒩(0, 1/d)" },
            { icon: "📐", title: "Normalization", desc: "Matches the codebook's design distribution → 𝒩(0, 1)" },
            { icon: "📊", title: "Lloyd-Max", desc: "Optimal scalar quantizer for known distributions → minimal D_b" },
            { icon: "🎯", title: "Residual passes", desc: "Exploit remaining structure → multiplicative MSE reduction" },
            { icon: "⚡", title: "On-the-fly dequant", desc: "Preserves the b-bit memory advantage at inference" },
          ].map((item, i) => (
            <Reveal key={i} delay={i * 0.08}>
              <div className="bg-bg-3 border border-border rounded-xl p-4">
                <span className="text-lg">{item.icon}</span>
                <div className="font-semibold text-sm mt-1">{item.title}</div>
                <p className="text-xs text-txt-2 mt-1">{item.desc}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* ─── 10. BPW Analysis ─── */}
      <Section title="BPW Analysis">
        <p className="text-txt-2 leading-relaxed mb-4">
          Beyond MSE, minimizing <strong className="text-txt">bits per weight</strong>{" "}
          (BPW) determines model size on disk and memory footprint. The total
          storage per weight element decomposes as:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="\text{BPW} \approx b_{\text{idx}} + \frac{32 \cdot n_{\text{norm}}}{d} + \text{BPW}_{\text{non-quant}}"
            display
          />
        </div>

        {/* Variables table */}
        <h3 className="text-lg font-semibold mb-4">Variables That Affect BPW</h3>
        <div className="overflow-x-auto mb-8">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-accent-green text-xs uppercase">
                  Variable
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  How it affects BPW
                </th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">
                  Default
                </th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              {[
                ["Index bit-width (b)", "Directly: BPW ∝ b", "4"],
                ["Residual passes (P)", "Total index bits = Σ bₖ", "1 or 2"],
                ["Group size (d)", "Norm overhead = 32/d per norm set", "128"],
                ["Norm precision", "Overhead = p_norm/d (currently 32-bit)", "float32"],
                ["Number of norm sets", "Each pass stores one set of norms", "P"],
                ["Non-quantized layers", "Embeddings, LayerNorm, lm_head at full precision", "model-dep"],
              ].map(([variable, effect, def], i) => (
                <tr key={i} className="border-b border-border">
                  <td className="py-3 px-4 text-txt font-medium">{variable}</td>
                  <td className="py-3 px-4">{effect}</td>
                  <td className="py-3 px-4 font-mono">{def}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* BPW Budget Example */}
        <h3 className="text-lg font-semibold mb-4">
          Budget Example: Qwen3.5-0.8B (4+2 residual)
        </h3>
        <div className="bg-bg-2 border border-border rounded-2xl p-6">
          <BPWBreakdownViz />
        </div>
      </Section>

      {/* ─── 11. Research Directions ─── */}
      <Section title="Research Directions">
        <p className="text-txt-2 leading-relaxed mb-6">
          Seven ideas to push effective BPW lower while preserving quality. Combined potential: with
          fp16 norms, <Math expr="d = 256" />, 3+2 config, and entropy coding, effective BPW
          could reach ~4.5 at quality comparable to current 4+2 (7 BPW).
        </p>
        <div className="grid sm:grid-cols-2 gap-4">
          <Reveal>
            <ResearchCard icon="🔢" title="Reduce Norm Precision" saving="−0.25 BPW">
              Store norms in float16/bfloat16 instead of float32, halving the overhead
              to <Math expr="16/d" /> BPW. Low risk for typical weight magnitudes.
            </ResearchCard>
          </Reveal>
          <Reveal delay={0.05}>
            <ResearchCard icon="📏" title="Larger Group Size" saving="−0.13 BPW / pass">
              Increase <Math expr="d" /> from 128 to 256 or 512. The Gaussian
              approximation improves with larger <Math expr="d" /> (better
              concentration of measure).
            </ResearchCard>
          </Reveal>
          <Reveal delay={0.1}>
            <ResearchCard icon="⬇️" title="Sub-4-bit Primary" saving="−1.0 BPW">
              Use 3-bit for the primary pass and rely on residual passes for
              quality. A 3+2 config achieves 5 BPW with distortion comparable
              to 4-bit single.
            </ResearchCard>
          </Reveal>
          <Reveal delay={0.15}>
            <ResearchCard icon="🎚️" title="Non-Uniform Bit Allocation" saving="−0.5–1.0 BPW">
              Assign higher bits to sensitive layers (attention Q/K) and lower
              bits to less sensitive ones (MLP). Solvable via dynamic
              programming.
            </ResearchCard>
          </Reveal>
          <Reveal delay={0.2}>
            <ResearchCard icon="🗜️" title="Entropy Coding" saving="−0.24 BPW">
              Lloyd-Max indices are non-uniform (inner levels more probable).
              Shannon entropy is ~3.76 bits vs 4 allocated. ANS or Huffman
              coding recovers the gap.
            </ResearchCard>
          </Reveal>
          <Reveal delay={0.25}>
            <ResearchCard icon="📉" title="Delta Norms" saving="−0.19 BPW">
              Residual norms are much smaller. Store them as ratios of pass-1
              norms, quantized to 8-bit.
            </ResearchCard>
          </Reveal>
          <Reveal delay={0.3}>
            <ResearchCard icon="🚫" title="Norm-Free Quantization" saving="variable">
              Absorb the norm into a per-group scaled codebook:{" "}
              <Math expr="c_\ell^{(g)} = \sigma_g \cdot c_\ell" />. Trades
              per-row norms for a single per-group scale factor.
            </ResearchCard>
          </Reveal>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
