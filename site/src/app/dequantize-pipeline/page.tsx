"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { PipelineFlow } from "@/components/animations/PipelineFlow";
import { KernelFusionViz } from "@/components/animations/KernelFusionViz";
import { Reveal } from "@/components/Reveal";

const dequantSteps = [
  { icon: "📥", label: "Input x", detail: "(B, N) activation", color: "#7ee787" },
  { icon: "🔄", label: "Rotate x", detail: "x · Πᵀ (cheap: B×d)", color: "#58a6ff" },
  { icon: "📖", label: "Unpack + Lookup", detail: "uint8 → codebook[idx]", color: "#d2a8ff" },
  { icon: "✖️", label: "Matmul", detail: "x_rot @ W_q.T", color: "#ffa657" },
  { icon: "⚖️", label: "Rescale", detail: "× α / √d", color: "#7ee787" },
];

export default function DequantizePipelinePage() {
  return (
    <TechniqueLayout
      title="Inference Dequantization Pipeline"
      subtitle="Key insight: rotate the input, not the weight. Pre-rotating the activation avoids materializing the full weight matrix."
      color="#7ee787"
      icon="🔓"
      prev={{ href: "/quantize-pipeline/", label: "Quantize Pipeline" }}
      next={{ href: "/formulation/", label: "Quantization Formulation" }}
    >
      <Section title="The Key Insight">
        <div className="bg-bg-2 border border-accent-green/20 rounded-xl p-6">
          <p className="text-txt-2 leading-relaxed">
            Naively, dequantization would reconstruct the full weight matrix{" "}
            <Math expr="\tilde{W} \in \mathbb{R}^{M \times N}" /> and compute{" "}
            <Math expr="y = x\tilde{W}^T" />. Instead, we{" "}
            <strong className="text-accent-green">pre-rotate the activation</strong>:
          </p>
          <div className="mt-4 text-center">
            <Math
              expr="x_{\text{rot}} = x \cdot \Pi^T \qquad \text{then} \qquad \text{output} = x_{\text{rot}} \cdot C[\mathbf{i}]^T \cdot \frac{\alpha}{\sqrt{d}}"
              display
            />
          </div>
          <p className="text-sm text-txt-2 mt-4">
            The rotation is applied to x once per group per layer — a (B, d) matrix multiply
            vs the (M, d) inverse rotation on the weight side.
          </p>
        </div>
      </Section>

      <Section title="Pipeline Overview">
        <PipelineFlow steps={dequantSteps} />
      </Section>

      <Section title="Forward Pass Algorithm">
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm leading-relaxed overflow-x-auto">
          <pre className="text-txt-2">
{`output = zeros(B, M)

for each group g in [0, n_groups):
    x_g   = x[:, g*d : (g+1)*d]           `}<span className="text-txt-2"># (B, d)</span>{`
    x_rot = x_g @ Pi_g.T                  `}<span className="text-txt-2"># (B, d)  rotate input</span>{`
    idx_g = unpack_4bit(packed[..., g])    `}<span className="text-txt-2"># (M, d)  unpack</span>{`
    W_g   = codebook[idx_g]               `}<span className="text-txt-2"># (M, d)  lookup</span>{`
    out_g = x_rot @ W_g.T                 `}<span className="text-txt-2"># (B, M)  matmul</span>{`
    out_g = out_g * (norms_g / sqrt(d))   `}<span className="text-txt-2"># (B, M)  rescale</span>{`
    output += out_g`}
          </pre>
        </div>
      </Section>

      <Section title="Kernel Fusion">
        <p className="text-txt-2 leading-relaxed mb-6">
          Steps 2–5 (unpack → lookup → matmul → rescale) are fused into a single GPU kernel to
          avoid intermediate tensor materialization.
        </p>
        <KernelFusionViz />
      </Section>

      <Section title="Execution Paths">
        <div className="space-y-4">
          {[
            {
              name: "CuTile",
              color: "#7ee787",
              desc: "NVIDIA cuda.tile_experimental API. Shared-memory codebook, FP16/BF16 tensor cores, tile-based prefetching.",
              req: "CUDA 13.1+, Ampere+ (sm80/sm89/sm100+)",
            },
            {
              name: "Triton",
              color: "#d2a8ff",
              desc: "Portable alternative. Autotuned block sizes per problem shape, software pipelining, TF32 tensor cores.",
              req: "Triton ≥ 3.0",
            },
            {
              name: "PyTorch (fallback)",
              color: "#8b949e",
              desc: "Explicit operations: unpack → codebook[indices] → matmul → rescale. Materializes dequantized weight slice.",
              req: "No special dependencies",
            },
          ].map((p, i) => (
            <Reveal key={p.name} delay={0.1 * i}>
              <div className="bg-bg-2 border border-border rounded-xl p-5">
                <div className="flex items-center gap-3 mb-2">
                  <span className="w-3 h-3 rounded-full" style={{ background: p.color }} />
                  <span className="font-semibold" style={{ color: p.color }}>{p.name}</span>
                  <span className="text-[10px] text-txt-2 ml-auto">{p.req}</span>
                </div>
                <p className="text-sm text-txt-2">{p.desc}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      <Section title="Residual Pass Handling">
        <p className="text-txt-2 leading-relaxed mb-4">
          When a layer has residual quantization, the forward method runs _forward_pass twice
          with different packed data and sums the results:
        </p>
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm">
          <pre className="text-txt-2">
{`output  = _forward_pass(x, pass1_data)
output += _forward_pass(x, pass2_data)  `}<span className="text-accent-green"># if residual</span>{`
output += bias                           `}<span className="text-accent-green"># if present</span>
          </pre>
        </div>
      </Section>

      <Section title="Memory Profile">
        <p className="text-txt-2 leading-relaxed mb-4">
          The pipeline never materializes the full M×N weight matrix. Peak additional memory:
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-accent text-xs uppercase">Component</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Size</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Notes</th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              <tr className="border-b border-border"><td className="py-3 px-4">x_rot</td><td className="py-3 px-4">B × d × 4B</td><td className="py-3 px-4">Per group, reused</td></tr>
              <tr className="border-b border-border"><td className="py-3 px-4">W slice (PyTorch only)</td><td className="py-3 px-4">M × d × 4B</td><td className="py-3 px-4">Per group, reused</td></tr>
              <tr className="border-b border-border"><td className="py-3 px-4">Output acc.</td><td className="py-3 px-4">B × M × 4B</td><td className="py-3 px-4">Persistent</td></tr>
              <tr><td className="py-3 px-4">Rotation matrix</td><td className="py-3 px-4">d × d × 4B</td><td className="py-3 px-4">Cached</td></tr>
            </tbody>
          </table>
        </div>
        <p className="text-sm text-txt-2 mt-4">
          With fused kernels, the dequantized weight slice only exists in registers/shared memory
          within the kernel — <strong className="text-accent-green">never written to global memory</strong>.
        </p>
      </Section>

      <Section title="Implementation">
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm space-y-2">
          <div>
            <span className="text-accent-purple">module.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">TurboQuantLinear._forward_pass()</span>,{" "}
            <span className="text-accent">TurboQuantLinear.forward()</span>
          </div>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
