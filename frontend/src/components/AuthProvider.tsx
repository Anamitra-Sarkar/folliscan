"use client";

import { useEffect, useState, createContext, useContext } from "react";
import { onAuthStateChanged, signOut, type User } from "firebase/auth";
import { useRouter } from "next/navigation";
import { getFirebaseAuth, syncSessionCookie } from "@/lib/firebase";
import { setCachedUser } from "@/lib/tokens";
import Link from "next/link";

interface AuthCtx {
  user: User | null;
  loading: boolean;
}

const Ctx = createContext<AuthCtx>({ user: null, loading: true });

export function AuthProvider({
  children,
  require = false,
}: {
  children: React.ReactNode;
  require?: boolean;
}) {
  const [state, setState] = useState<AuthCtx>({ user: null, loading: true });
  const router = useRouter();

  useEffect(() => {
    const unsub = onAuthStateChanged(getFirebaseAuth(), async (u) => {
      setCachedUser(u);
      await syncSessionCookie(u);
      setState({ user: u, loading: false });
    });
    return unsub;
  }, []);

  useEffect(() => {
    if (!state.loading && require && !state.user) {
      router.replace("/login");
    }
  }, [state.loading, state.user, require, router]);

  return <Ctx.Provider value={state}>{children}</Ctx.Provider>;
}

export function useAuth() {
  return useContext(Ctx);
}

export function TopBar({ title }: { title?: string }) {
  const { user } = useAuth();
  const router = useRouter();

  async function handleSignOut() {
    await signOut(getFirebaseAuth());
    await syncSessionCookie(null);
    router.push("/login");
  }

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-ivory/85 backdrop-blur">
      <div className="mx-auto max-w-6xl px-6 h-16 flex items-center justify-between">
        <Link href="/dashboard" className="flex items-center gap-2 group">
          <span className="w-8 h-8 rounded-lg bg-forest text-ivory grid place-items-center font-display text-sm shadow-card transition-transform group-hover:-rotate-6">
            F
          </span>
          <span className="font-display text-lg text-forest">Folliscan</span>
        </Link>
        <nav className="flex items-center gap-1 sm:gap-4">
          {title && (
            <span className="hidden md:block mr-2 text-sm text-ink-muted">{title}</span>
          )}
          <NavLink href="/dashboard">Analyze</NavLink>
          <NavLink href="/history">History</NavLink>
          {user && (
            <button
              onClick={handleSignOut}
              className="ml-2 text-sm px-3 py-1.5 rounded-full border border-line hover:border-terracotta hover:text-terracotta transition-colors"
            >
              Sign out
            </button>
          )}
        </nav>
      </div>
    </header>
  );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="text-sm px-3 py-1.5 rounded-full hover:bg-parchment transition-colors"
    >
      {children}
    </Link>
  );
}
