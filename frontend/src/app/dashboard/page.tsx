"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";

import { AuthProvider, TopBar } from "@/components/AuthProvider";
import SmilesInput from "@/components/SmilesInput";
import ResultsPanel from "@/components/ResultsPanel";
import { predict, explain, saveHistory, type PredictResult } from "@/lib/api";

function DashboardInner() {
  const [result, setResult] = useState<PredictResult | null>(null);
  const [narrative, setNarrative] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [explaining, setExplaining] = useState(false);
  const [currentSmiles, setCurrentSmiles] = useState("");

  async function run(smiles: string) {
    setLoading(true);
    setNarrative(null);
    setResult(null);
    setCurrentSmiles(smiles);
    try {
      const r = await predict(smiles);
      if (!r.valid) {
        toast.error(r.error || "Invalid molecule");
        return;
      }
      setResult(r);
      try {
        await saveHistory({
          smiles,
          canonical_smiles: r.canonical_smiles || smiles,
          result: r,
        });
      } catch {
        toast.warning("Saved analysis failed to sync to history");
      }
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function askWhy() {
    if (!result) return;
    setExplaining(true);
    try {
      setNarrative(await explain(currentSmiles, result));
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setExplaining(false);
    }
  }

  return (
    <div className="min-h-screen">
      <TopBar title="Ingredient analyzer" />
      <main className="mx-auto max-w-6xl px-6 py-10">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
        >
          <h1 className="font-display text-3xl md:text-4xl text-forest">
            Analyze an ingredient
          </h1>
          <p className="mt-2 text-ink-muted max-w-2xl">
            Enter a SMILES string to screen a compound across{" "}
            <strong className="text-ink">21 hair-health, toxicity and safety
            endpoints</strong> — with calibrated uncertainty and mechanistic motif
            explanations.
          </p>
        </motion.div>

        <div className="mt-8">
          <SmilesInput onSubmit={run} loading={loading} />
        </div>

        <AnimatePresence mode="wait">
          {loading && (
            <motion.div
              key="skeleton"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mt-8 grid gap-4 md:grid-cols-3"
            >
              {[...Array(3)].map((_, i) => (
                <div key={i} className="h-44 rounded-xl2 bg-parchment animate-pulse" />
              ))}
            </motion.div>
          )}

          {result && !loading && (
            <motion.div
              key={result.canonical_smiles}
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              className="mt-8"
            >
              <ResultsPanel
                result={result}
                narrative={narrative}
                explaining={explaining}
                onAskWhy={askWhy}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <AuthProvider require>
      <DashboardInner />
    </AuthProvider>
  );
}
