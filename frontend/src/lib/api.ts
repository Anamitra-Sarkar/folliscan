"use client";

import { authHeader } from "./tokens";

const API_URL = process.env.NEXT_PUBLIC_API_URL!;
const USER_API_URL = process.env.NEXT_PUBLIC_USER_API_URL!;

export interface Prediction {
  task_id: string;
  group: "hair" | "tox" | "safety";
  probability: number;
  std: number;
  conformal_set: [number, number];
  desc?: string;
}

export interface MotifHit {
  id: string;
  name: string;
  smarts: string;
  atom_indices: number[];
  importance: number;
  severity: string;
}

export interface PathwayHit {
  name: string;
  group: string;
  role: string;
  relevance: number;
}

export interface PredictResult {
  valid: boolean;
  input_smiles: string;
  canonical_smiles?: string;
  molecule_svg?: string | null;
  predictions?: Prediction[];
  motifs?: MotifHit[];
  pathways?: PathwayHit[];
  alerts?: { motif_id: string; message: string }[];
  regulatory_flags?: string[];
  uncertainty_note?: "high" | "moderate" | "low";
  mean_epistemic_std?: number;
  error?: string;
}

async function request<T>(url: string, init: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const body = await res.json().catch(() => ({}));
  if (res.status === 401) {
    throw new Error("Your session has expired. Please sign in again.");
  }
  if (!res.ok) {
    throw new Error((body as { detail?: string }).detail || `Request failed (${res.status})`);
  }
  return body as T;
}

export async function predict(smiles: string): Promise<PredictResult> {
  return request(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(await authHeader()) },
    body: JSON.stringify({ smiles }),
  });
}

export async function explain(smiles: string, payload: PredictResult): Promise<string> {
  const r = await request<{ narrative: string }>(`${API_URL}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(await authHeader()) },
    body: JSON.stringify({ smiles, payload }),
  });
  return r.narrative;
}

export async function saveHistory(entry: {
  smiles: string;
  canonical_smiles: string;
  result: PredictResult;
}): Promise<{ id: string }> {
  return request(`${USER_API_URL}/history`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(await authHeader()) },
    body: JSON.stringify(entry),
  });
}

export interface HistoryItem {
  id: string;
  smiles: string;
  canonical_smiles: string;
  result: PredictResult;
  createdAt: string;
}

export async function listHistory(limit = 20, cursor?: string): Promise<{
  items: HistoryItem[];
  next_cursor: string | null;
}> {
  const qs = new URLSearchParams({ limit: String(limit) });
  if (cursor) qs.set("cursor", cursor);
  return request(`${USER_API_URL}/history?${qs}`, {
    method: "GET",
    headers: await authHeader(),
  });
}

export async function deleteHistory(id: string): Promise<void> {
  await request(`${USER_API_URL}/history/${id}`, {
    method: "DELETE",
    headers: await authHeader(),
  });
}

export async function getMe(): Promise<{
  uid: string;
  email: string;
  displayName: string;
  createdAt: string;
}> {
  return request(`${USER_API_URL}/me`, { method: "GET", headers: await authHeader() });
}
