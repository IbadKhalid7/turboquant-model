"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { Reveal } from "@/components/Reveal";
import Link from "next/link";

export default function CpuOffloadPage() {
  return (
    <TechniqueLayout
      title="CPU Offload (Pass 2)"
      subtitle="Pipelined H2D streaming halves the VRAM cost of 4+4 residual — pass 2 lives on CPU, streamed to a shared double-buffered scratch pool via CUDA streams."
      color="#56d364"
      icon="💾"
      prev={{ href: "/techniques/qjl/", label: "QJL" }}
      next={{ href: "/quantize-pipeline/", label: "Quantize Pipeline" }}
    >
      {/* ─── Motivation ─── */}
      <Section title="Why Offload?">
        <p className="text-txt-2 leading-relaxed mb-4">
          In{" "}
          <Link href="/techniques/residual/" className="text-accent hover:underline">
            4+4 residual quantization
          </Link>
          , each layer stores <strong className="text-txt">two full sets</strong> of packed
          indices, norms, and codebook — one per pass. Pass 2 approximately{" "}
          <strong className="text-txt">doubles the VRAM</strong> footprint compared to
          single-pass 4-bit.
        </p>
        <p className="text-txt-2 leading-relaxed mb-4">
          For small-batch inference (batch 1), the GPU is compute-bound — PCIe bandwidth
          sits mostly idle. CPU offload exploits this by streaming pass 2 data from host
          memory to GPU on-demand.
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center">
          <Math
            expr="\text{VRAM saved} = \sum_{i=1}^{L} S_i - 2 \cdot \max_i(S_i) \approx (L - 2) \cdot \bar{S}"
            display
          />
          <p className="text-xs text-txt-2 mt-3">
            For 24 equal-sized layers: <strong className="text-txt">~92% reduction</strong> in
            pass 2 VRAM
          </p>
        </div>
      </Section>

      {/* ─── Architecture ─── */}
      <Section title="Shared Double-Buffered Scratch">
        <p className="text-txt-2 leading-relaxed mb-4">
          Instead of per-layer GPU scratch (which costs the same as just keeping pass 2 on GPU),
          a single <code className="text-accent">SharedScratchPool</code> holds{" "}
          <strong className="text-txt">2 scratch slots</strong> sized to the largest offloaded
          layer. Layers are assigned alternating slots (ping-pong):
        </p>
        <div className="grid sm:grid-cols-2 gap-4 mb-6">
          <Reveal>
            <div className="bg-bg-3 border border-border rounded-xl p-5 text-center">
              <div className="text-2xl mb-2 text-[#56d364]">Slot 0</div>
              <div className="text-xs text-txt-2">Even-indexed layers</div>
              <div className="text-xs text-txt-2 mt-1">Consumed while Slot 1 receives H2D</div>
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <div className="bg-bg-3 border border-border rounded-xl p-5 text-center">
              <div className="text-2xl mb-2 text-accent-orange">Slot 1</div>
              <div className="text-xs text-txt-2">Odd-indexed layers</div>
              <div className="text-xs text-txt-2 mt-1">Consumed while Slot 0 receives H2D</div>
            </div>
          </Reveal>
        </div>
        <p className="text-txt-2 leading-relaxed text-sm">
          Total GPU cost: <Math expr="2 \times \max_i(S_i)" /> — constant regardless of layer
          count. The rest lives in <strong className="text-txt">pinned CPU memory</strong> for
          DMA transfers at full PCIe bandwidth.
        </p>
      </Section>

      {/* ─── Data Layout ─── */}
      <Section title="Data Layout">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-[#56d364] text-xs uppercase">Buffer</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Location</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Scope</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Purpose</th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              <tr className="border-b border-border">
                <td className="py-3 px-4">Pass 1 (indices, norms, codebook)</td>
                <td className="py-3 px-4">GPU</td>
                <td className="py-3 px-4">Per-layer</td>
                <td className="py-3 px-4">Permanent, read by kernels</td>
              </tr>
              <tr className="border-b border-border">
                <td className="py-3 px-4">Pass 2 (pinned copies)</td>
                <td className="py-3 px-4">CPU</td>
                <td className="py-3 px-4">Per-layer</td>
                <td className="py-3 px-4">Source for async H2D</td>
              </tr>
              <tr className="border-b border-border bg-[#56d364]/5">
                <td className="py-3 px-4 text-[#56d364] font-semibold">SharedScratchPool ✨</td>
                <td className="py-3 px-4">GPU</td>
                <td className="py-3 px-4 font-semibold text-txt">Global (shared)</td>
                <td className="py-3 px-4">2 ping-pong scratch slots</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      {/* ─── Timeline ─── */}
      <Section title="Execution Timeline">
        <p className="text-txt-2 leading-relaxed mb-4">
          Two CUDA streams operate concurrently. The copy stream runs{" "}
          <strong className="text-txt">in parallel</strong> with compute on the default stream:
        </p>

        <Reveal>
          <div className="bg-bg-2 border border-border rounded-xl p-5 mb-4">
            <h4 className="font-semibold text-sm mb-3 text-[#56d364]">Single Layer (No Prefetch)</h4>
            <pre className="text-xs text-txt-2 font-mono leading-relaxed overflow-x-auto">
{`Copy stream:  ╠══ H2D pass2 ═══╣
Default:      ╠══ pass1 rot ═══╬══ pass1 kernel ═══╬═ wait ═╬══ pass2 kernel ═══╣
                                                      ↑
                                                 wait_event()`}
            </pre>
          </div>
        </Reveal>

        <Reveal delay={0.1}>
          <div className="bg-bg-2 border border-border rounded-xl p-5 mb-4">
            <h4 className="font-semibold text-sm mb-3 text-accent-orange">With Next-Layer Prefetch</h4>
            <pre className="text-xs text-txt-2 font-mono leading-relaxed overflow-x-auto">
{`Copy stream:  ╠═ H2D L₀ pass2 ═╬═ H2D L₁ pass2 ═╬═ H2D L₂ pass2 ═╣
Default:      ╠═ L₀ p1 ═╬ wait ╬═ L₀ p2 ═╬═ L₁ p1 ═╬ wait ╬═ L₁ p2 ═╣
                          ↑     prefetch L₁→           ↑
                     wait event₀                  wait event₁
                     (free)                       (usually free)`}
            </pre>
          </div>
        </Reveal>

        <Reveal delay={0.2}>
          <div className="bg-bg-2 border border-border rounded-xl p-5">
            <h4 className="font-semibold text-sm mb-3 text-[#d2a8ff]">With Dual-Pass Fused Kernel</h4>
            <pre className="text-xs text-txt-2 font-mono leading-relaxed overflow-x-auto">
{`Copy stream:  ╠═ H2D pass2 ═══════╣
Default:      ╠═ rotations (both) ═╬═ wait ═╬═ dual_fused_kernel ═══╣
                                      ↑
                                 wait_event()`}
            </pre>
          </div>
        </Reveal>
      </Section>

      {/* ─── Latency ─── */}
      <Section title="Latency Impact">
        <p className="text-txt-2 leading-relaxed mb-4">
          For a typical layer (<Math expr="M = N = 2048" />), the pass 2 H2D transfer
          takes <strong className="text-txt">~0.08 ms</strong> on PCIe 4.0 x16. The pass 1
          kernel takes 0.2–0.5 ms, so the copy is{" "}
          <strong className="text-txt">fully hidden</strong>.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-[#56d364] text-xs uppercase">Batch</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Pass 1 Time</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">H2D Time</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Overhead</th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              <tr className="border-b border-border">
                <td className="py-2 px-4 font-mono">1</td>
                <td className="py-2 px-4">0.3 ms</td>
                <td className="py-2 px-4">0.08 ms</td>
                <td className="py-2 px-4 text-accent-green font-semibold">0% (hidden)</td>
              </tr>
              <tr className="border-b border-border">
                <td className="py-2 px-4 font-mono">8</td>
                <td className="py-2 px-4">0.8 ms</td>
                <td className="py-2 px-4">0.08 ms</td>
                <td className="py-2 px-4 text-accent-green font-semibold">0% (hidden)</td>
              </tr>
              <tr className="border-b border-border">
                <td className="py-2 px-4 font-mono">32</td>
                <td className="py-2 px-4">2.5 ms</td>
                <td className="py-2 px-4">0.08 ms</td>
                <td className="py-2 px-4 text-accent-green font-semibold">0% (hidden)</td>
              </tr>
              <tr className="border-b border-border">
                <td className="py-2 px-4 font-mono">128</td>
                <td className="py-2 px-4">9 ms</td>
                <td className="py-2 px-4">0.08 ms</td>
                <td className="py-2 px-4 text-accent-green font-semibold">0% (hidden)</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-xs text-txt-2 mt-2">
          H2D time is constant regardless of batch size — hidden at all practical workloads.
        </p>
      </Section>

      {/* ─── Memory budget ─── */}
      <Section title="Memory Budget">
        <p className="text-txt-2 leading-relaxed mb-4">
          For <Math expr="L" /> equal-sized layers with pass 2 size <Math expr="S" />:
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-[#56d364] text-xs uppercase">Mode</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">GPU (pass 2)</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">CPU</th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              <tr className="border-b border-border">
                <td className="py-3 px-4">Non-offloaded</td>
                <td className="py-3 px-4"><Math expr="L \cdot S" /></td>
                <td className="py-3 px-4">0</td>
              </tr>
              <tr className="border-b border-border bg-[#56d364]/5">
                <td className="py-3 px-4 text-[#56d364] font-semibold">CPU offload ✨</td>
                <td className="py-3 px-4 font-semibold text-txt"><Math expr="2 \cdot S" />{" "}(constant)</td>
                <td className="py-3 px-4"><Math expr="L \cdot S" />{" "}(pinned)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <Reveal delay={0.1}>
          <div className="mt-6">
            <h4 className="font-semibold text-sm mb-3">Qwen3.5-0.8B Example (24 layers)</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b-2 border-border">
                    <th className="text-left py-2 px-4 text-xs uppercase text-txt-2">Mode</th>
                    <th className="text-left py-2 px-4 text-xs uppercase text-txt-2">VRAM (weights)</th>
                    <th className="text-left py-2 px-4 text-xs uppercase text-txt-2">CPU (pinned)</th>
                  </tr>
                </thead>
                <tbody className="text-txt-2">
                  <tr className="border-b border-border">
                    <td className="py-2 px-4">bf16 baseline</td>
                    <td className="py-2 px-4 font-mono">~1.6 GB</td>
                    <td className="py-2 px-4">—</td>
                  </tr>
                  <tr className="border-b border-border">
                    <td className="py-2 px-4">4-bit single</td>
                    <td className="py-2 px-4 font-mono">~0.4 GB</td>
                    <td className="py-2 px-4">—</td>
                  </tr>
                  <tr className="border-b border-border">
                    <td className="py-2 px-4">4+4 residual</td>
                    <td className="py-2 px-4 font-mono">~0.8 GB</td>
                    <td className="py-2 px-4">—</td>
                  </tr>
                  <tr className="border-b border-border bg-[#56d364]/5">
                    <td className="py-2 px-4 text-[#56d364] font-semibold">4+4 CPU offload ✨</td>
                    <td className="py-2 px-4 font-mono font-semibold text-txt">~0.43 GB</td>
                    <td className="py-2 px-4 font-mono">~0.4 GB</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </Reveal>
      </Section>

      {/* ─── Synchronization ─── */}
      <Section title="Synchronization: CUDA Events">
        <p className="text-txt-2 leading-relaxed mb-4">
          The pipeline uses <strong className="text-txt">CUDA events</strong> for device-side
          synchronization — no CPU blocking:
        </p>
        <div className="space-y-3">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0 bg-[#56d364]/10 text-[#56d364]">1</div>
                <div>
                  <h4 className="font-semibold text-sm mb-1">Copy stream records event</h4>
                  <p className="text-xs text-txt-2 leading-relaxed">
                    After H2D copy completes, <code className="text-accent">record_event()</code>{" "}
                    marks the position in the copy stream.
                  </p>
                </div>
              </div>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0 bg-accent-orange/10 text-accent-orange">2</div>
                <div>
                  <h4 className="font-semibold text-sm mb-1">Default stream waits on event</h4>
                  <p className="text-xs text-txt-2 leading-relaxed">
                    <code className="text-accent">wait_event()</code> is non-blocking to the{" "}
                    <strong className="text-txt">CPU</strong> — only the GPU pauses if the copy
                    isn{"'"}t done yet.
                  </p>
                </div>
              </div>
            </div>
          </Reveal>
          <Reveal delay={0.2}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0 bg-[#d2a8ff]/10 text-[#d2a8ff]">3</div>
                <div>
                  <h4 className="font-semibold text-sm mb-1">Prefetch the next layer</h4>
                  <p className="text-xs text-txt-2 leading-relaxed">
                    At the end of each layer{"'"}s forward, start H2D for the{" "}
                    <strong className="text-txt">next</strong> layer onto the copy stream. By the
                    time it{"'"}s needed, the copy is likely complete.
                  </p>
                </div>
              </div>
            </div>
          </Reveal>
        </div>
        <p className="text-txt-2 leading-relaxed mt-4 text-sm">
          This is critical: <code className="text-accent">stream.synchronize()</code> blocks the
          CPU, preventing it from submitting the next kernel.{" "}
          <code className="text-accent">wait_event()</code> keeps the CPU free.
        </p>
      </Section>

      {/* ─── Usage ─── */}
      <Section title="Usage">
        <Reveal>
          <div className="bg-bg-2 border border-border rounded-xl p-5 mb-4">
            <h4 className="font-semibold text-sm mb-3 text-[#56d364]">CLI</h4>
            <pre className="text-xs text-txt-2 font-mono leading-relaxed overflow-x-auto">
{`turboquant quantize \\
    --model Qwen/Qwen3-0.6B \\
    --output ./quantized \\
    --residual-bit-width 4 \\
    --cpu-offload-pass2`}
            </pre>
          </div>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="bg-bg-2 border border-border rounded-xl p-5">
            <h4 className="font-semibold text-sm mb-3 text-accent-orange">Python API</h4>
            <pre className="text-xs text-txt-2 font-mono leading-relaxed overflow-x-auto">
{`from turboquant_model import (
    TurboQuantConfig,
    quantize_model,
    enable_prefetch_chain,
)

config = TurboQuantConfig(
    bit_width=4,
    residual_bit_width=4,
    cpu_offload_pass2=True,
)
model = quantize_model(model, config)
# enable_prefetch_chain() is called automatically`}
            </pre>
          </div>
        </Reveal>
      </Section>

      {/* ─── Related ─── */}
      <Section title="Relationship to Other Techniques">
        <div className="grid sm:grid-cols-2 gap-4">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">🎯</div>
              <h4 className="font-semibold text-sm mb-1">
                <Link href="/techniques/residual/" className="text-accent hover:underline">
                  Residual Quantization
                </Link>
              </h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                CPU offload is specifically designed for 4+4 residual — it offloads the
                pass 2 data while pass 1 remains on GPU.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">⚡</div>
              <h4 className="font-semibold text-sm mb-1">
                <Link href="/techniques/fused-kernels/" className="text-accent hover:underline">
                  Fused Kernels
                </Link>
              </h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                The dual-pass fused kernel integrates with the offload pipeline, consuming
                scratch buffers directly in a single kernel launch.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">📐</div>
              <h4 className="font-semibold text-sm mb-1">
                <Link href="/techniques/norm-compression/" className="text-accent hover:underline">
                  Norm Compression
                </Link>
              </h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Compressed norms reduce the H2D transfer size per layer, making the offload
                pipeline even cheaper.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.15}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">🗜️</div>
              <h4 className="font-semibold text-sm mb-1">
                <Link href="/techniques/entropy-codec/" className="text-accent hover:underline">
                  Entropy Coding
                </Link>
              </h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Entropy-coded indices must be decoded before the kernel can use them — this
                decoding happens before the H2D copy.
              </p>
            </div>
          </Reveal>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
