"use client";

import { useState } from "react";
import { motion } from "framer-motion";

const EXAMPLES = [
  { label: "Minoxidil", smiles: "CNC1=NC(N)=C(N)N=C1CO" },
  { label: "Caffeine", smiles: "Cn1cnc2c1c(=O)n(C)c(=O)n2C" },
  { label: "Cinnamaldehyde", smiles: "O=CC=CC1=CC=CC=C1" },
  { label: "Hydroquinone", smiles: "Oc1ccc(O)cc1" },
  { label: "Salicylic acid", smiles: "OC(=O)c1ccccc1O" },
];

export default function SmilesInput({
  onSubmit,
  loading,
}: {
  onSubmit: (smiles: string) => void;
  loading: boolean;
}) {
  const [value, setValue] = useState("");

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="rounded-xl2 bg-white border border-line shadow-card p-5"
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (value.trim()) onSubmit(value.trim());
        }}
        className="flex flex-col sm:flex-row gap-3"
      >
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Paste a SMILES string, e.g. CNC1=NC(N)=C(N)N=C1CO"
          spellCheck={false}
          className="flex-1 h-12 px-4 rounded-xl border border-line bg-ivory font-mono text-sm outline-none focus:border-sage focus:ring-2 focus:ring-sage/25 transition"
        />
        <motion.button
          whileTap={{ scale: 0.97 }}
          type="submit"
          disabled={loading || !value.trim()}
          className="h-12 px-8 rounded-xl bg-terracotta text-white font-medium hover:bg-terracotta/90 disabled:opacity-50 transition-colors whitespace-nowrap"
        >
          {loading ? "Screening…" : "Run screening"}
        </motion.button>
      </form>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="text-xs text-ink-muted mr-1">Try:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            onClick={() => {
              setValue(ex.smiles);
              onSubmit(ex.smiles);
            }}
            disabled={loading}
            className="text-xs px-3 py-1.5 rounded-full bg-parchment hover:bg-sage/25 text-ink transition-colors disabled:opacity-50"
          >
            {ex.label}
          </button>
        ))}
      </div>
    </motion.div>
  );
}
