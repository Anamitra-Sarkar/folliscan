"use client";

import type { User } from "firebase/auth";

let cachedUser: User | null = null;

export function setCachedUser(u: User | null) {
  cachedUser = u;
}

export async function authHeader(): Promise<Record<string, string>> {
  if (!cachedUser) throw new Error("Not signed in");
  const token = await cachedUser.getIdToken();
  return { Authorization: `Bearer ${token}` };
}
