"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";

import { AuthProvider, TopBar } from "@/components/AuthProvider";
import { listHistory, deleteHistory, type HistoryItem } from "@/lib/api";

function timeAgo(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 3600) return `${Math.max(1, Math.floor(s / 60))} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

function riskOf(item: HistoryItem): number | null {
  const safety = item.result?.predictions?.filter((p) => p.group === "safety") || [];
  if (!safety.length) return null;
  const mean = safety.reduce((a, p) => a + p.probability, 0) / safety.length;
  return mean;
}

function HistoryInner() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [cursor, setCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);

  async function load(append = false) {
    setLoading(true);
    try {
      const r = await listHistory(20, append ? cursor || undefined : undefined);
      setItems((prev) => (append ? [...prev, ...r.items] : r.items));
      setCursor(r.next_cursor);
      setHasMore(Boolean(r.next_cursor));
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let active = true;

    void listHistory(20)
      .then((r) => {
        if (!active) return;
        setItems(r.items);
        setCursor(r.next_cursor);
        setHasMore(Boolean(r.next_cursor));
      })
      .catch((e: unknown) => {
        if (active) toast.error((e as Error).message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  async function remove(id: string) {
    try {
      await deleteHistory(id);
      setItems((prev) => prev.filter((i) => i.id !== id));
      toast.success("Deleted");
    } catch (e) {
      toast.error((e as Error).message);
    }
  }

  return (
    <div className="min-h-screen">
      <TopBar title="Your analyses" />
      <main className="mx-auto max-w-6xl px-6 py-10">
        <h1 className="font-display text-3xl text-forest">Analysis history</h1>
        <p className="mt-2 text-ink-muted">
          Every screening you have run, private to your account.
        </p>

        <div className="mt-8 space-y-3">
          {loading && items.length === 0 && (
            <>
              {[...Array(4)].map((_, i) => (
                <div key={i} className="h-20 rounded-xl2 bg-parchment animate-pulse" />
              ))}
            </>
          )}
          {!loading && items.length === 0 && (
            <div className="rounded-xl2 bg-white border border-line shadow-card p-10 text-center">
              <p className="font-display text-lg text-forest">Nothing here yet</p>
              <p className="text-sm text-ink-muted mt-1">
                Run your first screening from the analyzer.
              </p>
            </div>
          )}

          <AnimatePresence initial={false}>
            {items.map((item) => {
              const risk = riskOf(item);
              return (
                <motion.div
                  key={item.id}
                  layout
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -40 }}
                  whileHover={{ scale: 1.005 }}
                  className="rounded-xl2 bg-white border border-line shadow-card p-4 flex items-center gap-4"
                >
                  <div
                    className={`w-11 h-11 rounded-xl grid place-items-center font-mono text-xs shrink-0 ${
                      risk == null
                        ? "bg-parchment text-ink-muted"
                        : risk > 0.5
                        ? "bg-terracotta/15 text-terracotta"
                        : risk > 0.25
                        ? "bg-amber-100 text-amber-800"
                        : "bg-sage/20 text-forest"
                    }`}
                  >
                    {risk == null ? "–" : `${Math.round(risk * 100)}%`}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-mono text-sm truncate">{item.canonical_smiles}</p>
                    <p className="text-xs text-ink-muted mt-0.5">
                      {timeAgo(item.createdAt)}
                      {" · "}
                      {item.result?.uncertainty_note || "?"} uncertainty
                    </p>
                  </div>
                  <button
                    onClick={() => remove(item.id)}
                    className="text-sm px-3 py-1.5 rounded-full border border-line hover:border-terracotta hover:text-terracotta transition-colors"
                  >
                    Delete
                  </button>
                </motion.div>
              );
            })}
          </AnimatePresence>

          {hasMore && !loading && (
            <button
              onClick={() => load(true)}
              className="w-full py-3 rounded-xl2 border border-dashed border-line text-sm text-ink-muted hover:bg-parchment transition-colors"
            >
              Load more
            </button>
          )}
        </div>
      </main>
    </div>
  );
}

export default function HistoryPage() {
  return (
    <AuthProvider require>
      <HistoryInner />
    </AuthProvider>
  );
}
