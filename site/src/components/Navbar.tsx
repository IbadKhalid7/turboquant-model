"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useRef, useEffect } from "react";

const techniqueLinks = [
  { href: "/techniques/lloyd-max/", label: "Lloyd-Max" },
  { href: "/techniques/rotation/", label: "Rotation" },
  { href: "/techniques/walsh-hadamard/", label: "Hadamard" },
  { href: "/techniques/residual/", label: "Residual" },
  { href: "/techniques/packing/", label: "4-bit Packing" },
  { href: "/techniques/fused-kernels/", label: "Fused Kernels" },
  { href: "/techniques/entropy-codec/", label: "Entropy Coding" },
  { href: "/techniques/norm-compression/", label: "Norm Compression" },
  { href: "/techniques/qjl/", label: "QJL" },
];

const topLinks = [
  { href: "/", label: "Home" },
  { href: "/quantize-pipeline/", label: "Quantize" },
  { href: "/dequantize-pipeline/", label: "Dequantize" },
  { href: "/formulation/", label: "Formulation" },
];

function DropdownMenu() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const timeout = useRef<ReturnType<typeof setTimeout>>();

  const isTechniqueActive = techniqueLinks.some(
    (l) => pathname === l.href || pathname === l.href.slice(0, -1)
  );

  const handleEnter = () => {
    clearTimeout(timeout.current);
    setOpen(true);
  };
  const handleLeave = () => {
    timeout.current = setTimeout(() => setOpen(false), 150);
  };

  useEffect(() => () => clearTimeout(timeout.current), []);

  return (
    <div
      ref={ref}
      className="relative"
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
    >
      <button
        className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center gap-1 ${
          isTechniqueActive
            ? "text-accent bg-accent/10"
            : "text-txt-2 hover:text-accent hover:bg-accent/5"
        }`}
        onClick={() => setOpen((o) => !o)}
      >
        Techniques
        <svg
          className={`w-3 h-3 transition-transform ${open ? "rotate-180" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-1 py-2 bg-bg/95 backdrop-blur-xl border border-border rounded-lg shadow-lg min-w-[180px] z-50">
          {techniqueLinks.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              onClick={() => setOpen(false)}
              className={`block px-4 py-2 text-xs font-medium transition-colors ${
                pathname === l.href || pathname === l.href.slice(0, -1)
                  ? "text-accent bg-accent/10"
                  : "text-txt-2 hover:text-accent hover:bg-accent/5"
              }`}
            >
              {l.label}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-bg/85 backdrop-blur-xl border-b border-border">
      <div className="max-w-7xl mx-auto flex items-center justify-between h-14 px-4">
        <Link href="/" className="font-bold text-lg flex items-center gap-2 group">
          <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" fill="rgba(88,166,255,.2)" />
          </svg>
          <span className="group-hover:text-accent transition-colors">TurboQuant</span>
        </Link>
        <div className="hidden lg:flex items-center gap-1">
          {topLinks.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                pathname === l.href || pathname === l.href.slice(0, -1)
                  ? "text-accent bg-accent/10"
                  : "text-txt-2 hover:text-accent hover:bg-accent/5"
              }`}
            >
              {l.label}
            </Link>
          ))}
          <DropdownMenu />
          <a
            href="https://arxiv.org/abs/2504.19874"
            target="_blank"
            rel="noopener noreferrer"
            className="px-3 py-1.5 rounded-md text-xs font-medium text-txt-2 hover:text-accent transition-colors"
          >
            Paper ↗
          </a>
        </div>
      </div>
    </nav>
  );
}
