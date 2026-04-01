"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { Reveal } from "@/components/Reveal";
import Link from "next/link";

export default function EntropyCodecPage() {
  return (
    <TechniqueLayout
      title="Entropy Coding (rANS)"
      subtitle="Compress quantized indices below their nominal bit-width by exploiting non-uniform Gaussian bin probabilities."
      color="#f778ba"
      icon="🗜️"
      prev={{ href: "/techniques/fused-kernels/", label: "Fused GPU Kernels" }}
      next={{ href: "/techniques/norm-compression/", label: "Norm Compression" }}
    >
      {/* ─── Which part of the BPW budget ─── */}
      <Section title="Where It Fits: The Index Term">
        <p className="text-txt-2 leading-relaxed mb-4">
          In the{" "}
          <Link href="/formulation/" className="text-accent hover:underline">
            quantization formulation
          </Link>
          , the dominant term in the storage budget is the{" "}
          <strong className="text-txt">index tensor</strong>{" "}
          <Math expr="\boldsymbol{\ell}_{m,k}" /> at <Math expr="b" /> bits per
          weight:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="\text{BPW} \approx \underbrace{b}_{\text{indices (target)}} + \frac{32}{d} + \text{BPW}_{\text{non-quant}}"
            display
          />
        </div>
        <p className="text-txt-2 leading-relaxed">
          Entropy coding targets the first term. Because Lloyd-Max quantization of{" "}
          <Math expr="\mathcal{N}(0,1)" /> produces <strong className="text-txt">non-uniform</strong>{" "}
          bin probabilities (inner levels are more probable than outer), the Shannon
          entropy <Math expr="H" /> is strictly less than <Math expr="b" />.
        </p>
      </Section>

      {/* ─── Entropy savings table ─── */}
      <Section title="Entropy Gap">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-[#f778ba] text-xs uppercase">b (bits)</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Levels</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">H (bits/sym)</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Saving</th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              {[
                [2, 4, "1.911", "0.089"],
                [3, 8, "2.832", "0.168"],
                [4, 16, "3.764", "0.236"],
                [5, 32, "4.755", "0.245"],
              ].map(([b, L, H, saving]) => (
                <tr
                  key={String(b)}
                  className={`border-b border-border ${b === 4 ? "bg-[#f778ba]/5" : ""}`}
                >
                  <td className="py-3 px-4 font-mono text-[#f778ba]">{b}</td>
                  <td className="py-3 px-4">{L}</td>
                  <td className="py-3 px-4 font-mono">{H}</td>
                  <td className="py-3 px-4 font-mono text-accent-green">−{saving}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-txt-2 mt-2">
          At 4 bits, entropy coding saves <strong className="font-semibold text-txt">~0.24 BPW</strong> — bringing
          the index cost from 4.0 to ~3.76 bits per weight.
        </p>
      </Section>

      {/* ─── How rANS works ─── */}
      <Section title="How rANS Works">
        <p className="text-txt-2 leading-relaxed mb-6">
          <strong className="text-txt">Asymmetric Numeral Systems</strong> (Duda 2009)
          achieve near-entropy-optimal compression with a simple, GPU-friendly decode loop.
          Symbols are split into blocks of <Math expr="B = 4096" /> for independent parallel
          decoding.
        </p>

        <div className="space-y-4">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <h4 className="font-semibold text-sm mb-2 text-[#f778ba]">Encode (sequential per block)</h4>
              <p className="text-sm text-txt-2 leading-relaxed mb-3">
                Process symbols in reverse. For symbol <Math expr="s" /> with frequency{" "}
                <Math expr="f_s" /> and cumulative <Math expr="c_s" />:
              </p>
              <div className="bg-bg-3 rounded-lg p-3 text-center">
                <Math
                  expr="\text{state}' = \left\lfloor \frac{\text{state}}{f_s} \right\rfloor \cdot 2^{P} + (\text{state} \bmod f_s) + c_s"
                  display
                />
              </div>
              <p className="text-xs text-txt-2 mt-2">
                <Math expr="P = 14" /> — frequencies are quantized to sum to{" "}
                <Math expr="2^{14} = 16384" />.
              </p>
            </div>
          </Reveal>

          <Reveal delay={0.1}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <h4 className="font-semibold text-sm mb-2 text-accent-green">Decode (GPU-parallel per block)</h4>
              <p className="text-sm text-txt-2 leading-relaxed mb-3">
                Each block starts from a known 4-byte state. Per symbol:
              </p>
              <div className="space-y-2 text-sm font-mono text-txt-2">
                <div className="flex gap-3">
                  <span className="text-accent shrink-0">1.</span>
                  <span>
                    slot = state &amp; (2<sup>P</sup> − 1)
                  </span>
                </div>
                <div className="flex gap-3">
                  <span className="text-accent shrink-0">2.</span>
                  <span>symbol = LUT[slot] <span className="text-txt-2">// O(1) table lookup</span></span>
                </div>
                <div className="flex gap-3">
                  <span className="text-accent shrink-0">3.</span>
                  <span>
                    state = f<sub>s</sub> × (state ≫ P) + slot − c<sub>s</sub>
                  </span>
                </div>
                <div className="flex gap-3">
                  <span className="text-accent shrink-0">4.</span>
                  <span>renormalize: read bytes while state &lt; 2<sup>16</sup></span>
                </div>
              </div>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ─── Tiny decode tables ─── */}
      <Section title="Decode Table Size">
        <p className="text-txt-2 leading-relaxed mb-4">
          The entire decode table fits comfortably in GPU shared memory or registers:
        </p>
        <div className="grid sm:grid-cols-2 gap-4">
          <Reveal>
            <div className="bg-bg-3 border border-border rounded-xl p-5">
              <h4 className="text-[#f778ba] font-semibold text-sm mb-2">Frequency table</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                <Math expr="L \times 2" /> bytes (uint16). At 4-bit:{" "}
                <strong className="text-txt">32 bytes</strong>.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="bg-bg-3 border border-border rounded-xl p-5">
              <h4 className="text-accent font-semibold text-sm mb-2">Cumulative table</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                <Math expr="(L+1) \times 4" /> bytes (uint32). At 4-bit:{" "}
                <strong className="text-txt">68 bytes</strong>.
              </p>
            </div>
          </Reveal>
        </div>
        <p className="text-xs text-txt-2 mt-3">
          Total: ~100 bytes for 4-bit — derived from the <em>known</em> Gaussian bin
          probabilities, no training data needed.
        </p>
      </Section>

      {/* ─── Relationship to other techniques ─── */}
      <Section title="Relationship to Other Techniques">
        <div className="grid sm:grid-cols-2 gap-4">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">📊</div>
              <h4 className="font-semibold text-sm mb-1">Lloyd-Max</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Entropy coding exploits the non-uniform bin probabilities from optimal Gaussian
                quantization. Uniform quantizers would have <Math expr="H = b" /> (no saving).
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">📦</div>
              <h4 className="font-semibold text-sm mb-1">4-bit Packing</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Packing reduces storage by fitting two indices per byte. Entropy coding goes
                further by exploiting statistical redundancy within those indices.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">🎯</div>
              <h4 className="font-semibold text-sm mb-1">Residual Quantization</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Each residual pass produces its own index tensor — entropy coding applies
                independently to each pass.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.15}>
            <div className="bg-bg-2 border border-border rounded-xl p-5">
              <div className="text-xl mb-2">📐</div>
              <h4 className="font-semibold text-sm mb-1">Norm Compression</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Entropy coding compresses the index tensor; norm factorization compresses the
                norm tensor. Together they address both major storage components.
              </p>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ─── Implementation ─── */}
      <Section title="Implementation">
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm space-y-2">
          <div>
            <span className="text-accent-purple">entropy_codec.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">gaussian_bin_probs()</span>{" "}
            <span className="text-txt-2">(compute bin probabilities from Lloyd-Max)</span>
          </div>
          <div>
            <span className="text-accent-purple">entropy_codec.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">compute_entropy()</span>{" "}
            <span className="text-txt-2">(theoretical entropy lower bound)</span>
          </div>
          <div>
            <span className="text-accent-purple">entropy_codec.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">build_ans_table()</span>{" "}
            <span className="text-txt-2">(frequency + cumulative tables)</span>
          </div>
          <div>
            <span className="text-accent-purple">entropy_codec.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">rANSCodec.encode() / decode()</span>{" "}
            <span className="text-txt-2">(block-parallel rANS)</span>
          </div>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
