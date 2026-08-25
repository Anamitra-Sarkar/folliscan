"use client";

import { redirect } from "next/navigation";

export default function Home() {
  // Static shell; real auth state is resolved client-side by the login page
  // (Firebase requires a browser). Middleware handles cookie-based routing.
  return (
    <main className="min-h-screen flex items-center justify-center">
      <div className="animate-pulse text-ink-muted font-display text-xl">Folliscan</div>
    </main>
  );
}
