import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen overflow-hidden">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-5 py-6 sm:px-8">
        <Link href="/" className="flex items-center gap-3 text-forest" aria-label="FolliScan home">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-forest font-display text-lg text-ivory">F</span>
          <span className="font-display text-xl font-semibold tracking-tight">FolliScan</span>
        </Link>
        <Link
          href="/login"
          className="rounded-full bg-forest px-5 py-2.5 text-sm font-semibold text-ivory shadow-sm transition hover:bg-forest-deep"
        >
          Researcher sign in
        </Link>
      </header>
      <section aria-label="Hero illustration">
        <img
          src="/hero.png"
          alt="Abstract illustration of a hair follicle cross-section transitioning into a molecular graph of genes and metabolites"
          style={{ width: '100%', maxWidth: '1100px', maxHeight: '300px', objectFit: 'cover', display: 'block', margin: '0 auto', borderRadius: 12 }}
          loading="eager"
        />
      </section>

      <section className="mx-auto grid max-w-6xl gap-12 px-5 pb-16 pt-10 sm:px-8 lg:grid-cols-[1.02fr_0.98fr] lg:items-center lg:pb-24 lg:pt-16">
        <div className="max-w-2xl">
          <p className="mb-5 text-xs font-semibold uppercase tracking-[0.18em] text-terracotta">
            Ingredient research workspace
          </p>
          <h1 className="font-display text-5xl leading-[0.95] text-forest sm:text-6xl lg:text-7xl">
            A clearer path through cosmetic ingredient evidence.
          </h1>
          <p className="mt-7 max-w-xl text-lg leading-8 text-ink-muted">
            FolliScan is a private research environment for documenting ingredient questions, reviewing
            evidence, and preserving uncertainty for expert assessment. It is not a diagnostic service or a
            consumer safety recommendation.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-4">
            <Link
              href="/login"
              className="rounded-full bg-forest px-6 py-3.5 text-sm font-semibold text-ivory shadow-card transition hover:bg-forest-deep"
            >
              Enter the protected workspace
            </Link>
            <a href="#status" className="text-sm font-semibold text-forest underline decoration-sage underline-offset-4">
              Read the release status
            </a>
          </div>
          <p className="mt-6 text-sm text-ink-muted">Protected research records · Shared account identity · No patient data</p>
        </div>

        <div className="relative isolate min-h-[380px] overflow-hidden rounded-[2rem] border border-line bg-parchment p-6 shadow-card sm:p-9">
          <div className="absolute -right-16 -top-20 h-64 w-64 rounded-full bg-sage/30 blur-3xl" aria-hidden />
          <div className="absolute -bottom-20 -left-12 h-52 w-52 rounded-full bg-terracotta/15 blur-3xl" aria-hidden />
          <div className="relative flex h-full flex-col justify-between rounded-2xl border border-white/70 bg-white/70 p-6 backdrop-blur-sm sm:p-8">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-muted">Evidence review flow</p>
              <span className="rounded-full border border-line bg-ivory px-3 py-1 text-xs font-medium text-ink-muted">No live result</span>
            </div>
            <div className="my-9 space-y-4">
              {[
                ["01", "Frame a research question"],
                ["02", "Inspect recorded evidence"],
                ["03", "Assess limitations"],
                ["04", "Document expert review"],
              ].map(([step, label]) => (
                <div key={step} className="flex items-center gap-4 rounded-xl border border-line/80 bg-ivory/70 px-4 py-3">
                  <span className="font-display text-xl text-terracotta">{step}</span>
                  <span className="text-sm font-medium text-ink">{label}</span>
                </div>
              ))}
            </div>
            <p id="status" className="rounded-xl bg-forest px-4 py-3 text-sm leading-6 text-ivory">
              <strong>Model release pending.</strong> The deployed research interface does not represent an
              available ingredient prediction while no approved model artifact is loaded.
            </p>
          </div>
        </div>
      </section>

      <section className="border-y border-line bg-white/60">
        <div className="mx-auto grid max-w-6xl gap-8 px-5 py-12 sm:px-8 md:grid-cols-3">
          <div>
            <p className="text-sm font-semibold text-forest">Research-only context</p>
            <p className="mt-2 text-sm leading-6 text-ink-muted">Keep the question, source, limitation, and review trail together.</p>
          </div>
          <div>
            <p className="text-sm font-semibold text-forest">Protected by design</p>
            <p className="mt-2 text-sm leading-6 text-ink-muted">Workspace and history routes remain behind the existing sign-in gate.</p>
          </div>
          <div>
            <p className="text-sm font-semibold text-forest">Release-gated output</p>
            <p className="mt-2 text-sm leading-6 text-ink-muted">No availability claim is made until reviewed artifact evidence is registered.</p>
          </div>
        </div>
      </section>
    </main>
  );
}
