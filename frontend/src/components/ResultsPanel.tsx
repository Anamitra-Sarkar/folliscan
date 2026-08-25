"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import type { PredictResult, Prediction } from "@/lib/api";

const GROUP_META = {
  hair: { title: "Hair-health activity", accent: "#1E3A2F", hint: "Higher = more favorable efficacy signal" },
  tox: { title: "Toxicity endpoints", accent: "#8a5a44", hint: "Lower is better — assay-based toxicity calls" },
  safety: { title: "Cosmetic safety", accent: "#C4674F", hint: "Lower hazard probability is safer" },
} as const;

export default function ResultsPanel({
  result,
  narrative,
  explaining,
  onAskWhy,
}: {
  result: PredictResult;
  narrative: string | null;
  explaining: boolean;
  onAskWhy: () => void;
}) {
  const groups = useMemo(() => {
    const g: Record<string, Prediction[]> = { hair: [], tox: [], safety: [] };
    for (const p of result.predictions || []) g[p.group]?.push(p);
    return g;
  }, [result]);

  return (
    <div className="space-y-6">
      {/* header card */}
      <div className="rounded-xl2 bg-white border border-line shadow-card p-6 flex flex-col md:flex-row gap-6">
        <div className="molecule-svg shrink-0 w-full md:w-80 h-52 rounded-xl bg-[#FAF7F2] grid place-items-center overflow-hidden">
          {result.molecule_svg ? (
            <div dangerouslySetInnerHTML={{ __html: result.molecule_svg }} />
          ) : (
            <span className="text-ink-muted text-sm">structure unavailable</span>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <h2 className="font-display text-xl text-forest truncate">
                {result.canonical_smiles}
              </h2>
              <p className="text-xs font-mono text-ink-muted mt-1 break-all">
                input: {result.input_smiles}
              </p>
            </div>
            <span
              className={`shrink-0 text-xs px-3 py-1.5 rounded-full font-medium ${
                result.uncertainty_note === "low"
                  ? "bg-sage/20 text-forest"
                  : result.uncertainty_note === "moderate"
                  ? "bg-amber-100 text-amber-800"
                  : "bg-red-100 text-red-800"
              }`}
              title={`mean epistemic std ${result.mean_epistemic_std}`}
            >
              uncertainty: {result.uncertainty_note}
            </span>
          </div>

          {(result.alerts?.length ?? 0) > 0 && (
            <div className="mt-4 space-y-2">
              {result.alerts?.map((a) => (
                <div
                  key={a.motif_id}
                  className="text-sm bg-terracotta/10 text-terracotta rounded-lg px-3 py-2"
                >
                  ⚠︎ {a.message}
                </div>
              ))}
            </div>
          )}

          <button
            onClick={onAskWhy}
            disabled={explaining}
            className="mt-5 inline-flex items-center gap-2 text-sm font-medium px-4 py-2 rounded-full bg-forest text-ivory hover:bg-forest-deep transition-colors disabled:opacity-60"
          >
            {explaining ? "Thinking…" : "Ask why →"}
          </button>
        </div>
      </div>

      {/* narrative */}
      {narrative && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-xl2 bg-white border border-line shadow-card p-6"
        >
          <h3 className="font-display text-lg text-forest mb-3">Expert readout</h3>
          <pre className="whitespace-pre-wrap font-body text-sm leading-relaxed text-ink">
            {narrative}
          </pre>
        </motion.div>
      )}

      {/* task groups */}
      <div className="grid gap-6 lg:grid-cols-3">
        {(["hair", "tox", "safety"] as const).map((g) => (
          <TaskGroupCard key={g} group={g} title={GROUP_META[g].title}
            accent={GROUP_META[g].accent} hint={GROUP_META[g].hint}
            preds={groups[g]} />
        ))}
      </div>

      {/* motifs + pathways */}
      <div className="grid gap-6 lg:grid-cols-2">
        <MotifCard motifs={result.motifs || []} />
        <PathwayCard pathways={result.pathways || []} />
      </div>
    </div>
  );
}

function TaskGroupCard({
  group,
  title,
  accent,
  hint,
  preds,
}: {
  group: string;
  title: string;
  accent: string;
  hint: string;
  preds: Prediction[];
}) {
  return (
    <div className="rounded-xl2 bg-white border border-line shadow-card p-5">
      <h3 className="font-display text-lg" style={{ color: accent }}>
        {title}
      </h3>
      <p className="text-xs text-ink-muted mt-0.5 mb-4">{hint}</p>
      <div className="space-y-3 max-h-96 overflow-y-auto thin-scroll pr-1">
        {preds.map((p, i) => (
          <Gauge key={p.task_id} pred={p} accent={accent} index={i} invert={group !== "hair"} />
        ))}
      </div>
    </div>
  );
}

function Gauge({
  pred,
  accent,
  invert,
  index,
}: {
  pred: Prediction;
  accent: string;
  invert: boolean;
  index: number;
}) {
  // For tox/safety, color intensity reflects hazard; conformal band shows the
  // statistically calibrated interval around the point estimate.
  const p = pred.probability;
  const [lo, hi] = pred.conformal_set;
  const risky = invert ? p : 0;
  const barColor =
    risky > 0.66 ? "#b0472e" : risky > 0.33 ? "#c98a4b" : accent;

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.03 }}
      title={`${pred.desc || ""} · 95% interval [${lo.toFixed(2)}, ${hi.toFixed(2)}] · std ${pred.std.toFixed(3)}`}
    >
      <div className="flex justify-between items-baseline text-sm">
        <span className="truncate mr-2">{pred.task_id}</span>
        <span className="font-mono text-xs text-ink-muted">{(p * 100).toFixed(1)}%</span>
      </div>
      <div className="relative mt-1 h-2.5 rounded-full bg-parchment overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(hi * 100, 2)}%` }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1], delay: 0.15 + index * 0.03 }}
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ background: `${barColor}33` }}   // conformal upper bound (soft)
        />
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(p * 100, 2)}%` }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ background: barColor }}
        />
      </div>
    </motion.div>
  );
}

function MotifCard({ motifs }: { motifs: PredictResult["motifs"] }) {
  const sorted = [...(motifs || [])]
    .filter((m) => m.severity !== "info" || m.importance > 0)
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10);
  return (
    <div className="rounded-xl2 bg-white border border-line shadow-card p-5">
      <h3 className="font-display text-lg text-forest">Driving substructures</h3>
      <p className="text-xs text-ink-muted mt-0.5 mb-4">
        Mechanistic explanation — matched functional groups ranked by influence on this prediction.
      </p>
      {sorted.length === 0 && (
        <p className="text-sm text-ink-muted">No significant structural motifs detected.</p>
      )}
      <div className="space-y-2">
        {sorted.map((m) => (
          <div key={m.id} className="flex items-center gap-3 text-sm">
            <span
              className={`w-2 h-2 rounded-full shrink-0 ${
                m.severity === "hazard"
                  ? "bg-terracotta"
                  : m.severity === "alert"
                  ? "bg-amber-500"
                  : "bg-sage"
              }`}
            />
            <span className="flex-1 truncate">{m.name}</span>
            <div className="w-24 h-1.5 rounded-full bg-parchment overflow-hidden shrink-0">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${m.importance * 200}%` }}
                transition={{ duration: 0.6 }}
                className="h-full bg-forest/70"
                style={{ maxWidth: "100%" }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PathwayCard({ pathways }: { pathways: PredictResult["pathways"] }) {
  const top = (pathways || []).slice(0, 8);
  return (
    <div className="rounded-xl2 bg-white border border-line shadow-card p-5">
      <h3 className="font-display text-lg text-forest">Pathway relevance</h3>
      <p className="text-xs text-ink-muted mt-0.5 mb-4">
        Attention over known hair-biology &amp; toxicology pathways (Wnt/Shh/BMP cascades, AOP anchors).
      </p>
      <div className="space-y-2">
        {top.map((pw) => (
          <div key={pw.name} className="flex items-center gap-3 text-sm">
            <span className="flex-1 truncate" title={pw.role}>{pw.name}</span>
            <span className="text-[10px] uppercase tracking-wide text-ink-muted w-12">
              {pw.group}
            </span>
            <div className="w-24 h-1.5 rounded-full bg-parchment overflow-hidden shrink-0">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${pw.relevance * 100}%` }}
                transition={{ duration: 0.6 }}
                className="h-full bg-sage"
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
