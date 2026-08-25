"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  updateProfile,
} from "firebase/auth";
import { getFirebaseAuth, syncSessionCookie } from "@/lib/firebase";
import { AuthProvider } from "@/components/AuthProvider";
import { toast } from "sonner";

function friendlyError(code: string): string {
  const map: Record<string, string> = {
    "auth/invalid-email": "That email address doesn't look right.",
    "auth/missing-password": "Please enter your password.",
    "auth/weak-password": "Password should be at least 6 characters.",
    "auth/wrong-password": "Incorrect password. Try again.",
    "auth/user-not-found": "No account with this email yet — create one below.",
    "auth/email-already-in-use":
      "You already have a Folliscan account — sign in instead (works across all our apps).",
    "auth/popup-closed-by-user": "Google sign-in was closed before finishing.",
    "auth/too-many-requests": "Too many attempts. Please wait a moment.",
  };
  return map[code] || "Something went wrong. Please try again.";
}

function LoginForm() {
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const router = useRouter();
  const params = useSearchParams();
  const nextUrl = params.get("next") || "/dashboard";

  async function finish() {
    await syncSessionCookie({ uid: "pending" });
    router.replace(nextUrl);
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!email || !password) return;
    setBusy(true);
    try {
      const auth = getFirebaseAuth();
      if (mode === "signup") {
        const cred = await createUserWithEmailAndPassword(auth, email, password);
        if (name) await updateProfile(cred.user, { displayName: name });
        toast.success("Welcome to Folliscan!");
      } else {
        await signInWithEmailAndPassword(auth, email, password);
        toast.success("Welcome back!");
      }
      await finish();
    } catch (err) {
      toast.error(friendlyError((err as { code?: string })?.code || ""));
    } finally {
      setBusy(false);
    }
  }

  async function google() {
    setBusy(true);
    try {
      await signInWithPopup(getFirebaseAuth(), new GoogleAuthProvider());
      toast.success("Signed in with Google");
      await finish();
    } catch (err) {
      toast.error(friendlyError((err as { code?: string })?.code || ""));
    } finally {
      setBusy(false);
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="w-full max-w-md"
    >
      <div className="rounded-xl2 bg-white shadow-card border border-line p-8">
        <div className="flex items-center gap-3 mb-6">
          <span className="w-11 h-11 rounded-xl bg-forest text-ivory grid place-items-center font-display text-lg">
            F
          </span>
          <div>
            <h1 className="font-display text-2xl text-forest leading-tight">
              {mode === "signin" ? "Welcome back" : "Create your account"}
            </h1>
            <p className="text-sm text-ink-muted">
              Ingredient intelligence for safer hair care.
            </p>
          </div>
        </div>

        <form onSubmit={submit} className="space-y-4">
          {mode === "signup" && (
            <Field label="Name">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Dr. Jane Doe"
                className={inputCls}
              />
            </Field>
          )}
          <Field label="Email">
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@lab.org"
              className={inputCls}
            />
          </Field>
          <Field label="Password">
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className={inputCls}
            />
          </Field>

          <motion.button
            whileTap={{ scale: 0.98 }}
            disabled={busy}
            type="submit"
            className="w-full h-11 rounded-full bg-forest text-ivory font-medium hover:bg-forest-deep disabled:opacity-60 transition-colors"
          >
            {busy ? "One moment…" : mode === "signin" ? "Sign in" : "Create account"}
          </motion.button>
        </form>

        <div className="my-5 flex items-center gap-3 text-xs text-ink-muted">
          <span className="h-px flex-1 bg-line" /> or <span className="h-px flex-1 bg-line" />
        </div>

        <button
          onClick={google}
          disabled={busy}
          className="w-full h-11 rounded-full border border-line bg-white hover:bg-parchment font-medium transition-colors flex items-center justify-center gap-2"
        >
          <svg width="17" height="17" viewBox="0 0 48 48" aria-hidden>
            <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9 3.5l6.7-6.7C35.6 2.4 30.2 0 24 0 14.6 0 6.5 5.4 2.6 13.2l7.8 6C12.3 13.2 17.7 9.5 24 9.5z" />
            <path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.7c-.6 3-2.3 5.5-4.8 7.2l7.5 5.8c4.4-4.1 7.1-10.1 7.1-17.5z" />
            <path fill="#FBBC05" d="M10.4 28.8a14.5 14.5 0 0 1 0-9.6l-7.8-6a24 24 0 0 0 0 21.6l7.8-6z" />
            <path fill="#34A853" d="M24 48c6.2 0 11.4-2 15.4-5.5l-7.5-5.8c-2.1 1.4-4.8 2.3-7.9 2.3-6.3 0-11.7-3.7-13.6-8.7l-7.8 6C6.5 42.6 14.6 48 24 48z" />
          </svg>
          Continue with Google
        </button>

        <p className="mt-6 text-center text-sm text-ink-muted">
          {mode === "signin" ? (
            <>
              New here?{" "}
              <button
                onClick={() => setMode("signup")}
                className="text-terracotta hover:underline"
              >
                Create an account
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button
                onClick={() => setMode("signin")}
                className="text-terracotta hover:underline"
              >
                Sign in
              </button>
            </>
          )}
        </p>
      </div>

      <p className="mt-4 text-center text-xs text-ink-muted max-w-md mx-auto">
        One account works across all of our applications — your identity is shared,
        never duplicated. Predictions are private to your account.
      </p>
    </motion.div>
  );
}

const inputCls =
  "w-full h-11 px-4 rounded-xl border border-line bg-ivory focus:border-sage focus:ring-2 focus:ring-sage/25 outline-none text-sm transition";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="block mb-1.5 text-sm font-medium text-ink">{label}</span>
      {children}
    </label>
  );
}

export default function LoginPage() {
  return (
    <AuthProvider>
      <main className="min-h-screen flex items-center justify-center px-4 py-10">
        <Suspense fallback={null}>
          <LoginForm />
        </Suspense>
      </main>
    </AuthProvider>
  );
}
