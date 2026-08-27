import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

// This runs inside `npm run build` so deployment cannot ship the former
// indefinite loading shell or imply live model availability.
const sourcePath = fileURLToPath(new URL("../src/app/page.tsx", import.meta.url));
const source = await readFile(sourcePath, "utf8");

const requiredPhrases = [
  "Ingredient research workspace",
  "No live result",
  "Model release pending.",
  "available ingredient prediction",
  'href="/login"',
];

const prohibitedPhrases = [
  "animate-pulse text-ink-muted font-display text-xl\">Folliscan",
  "Live prediction",
  "Diagnosis",
];

for (const phrase of requiredPhrases) {
  if (!source.includes(phrase)) throw new Error(`Public landing boundary is missing required text: ${phrase}`);
}

for (const phrase of prohibitedPhrases) {
  if (source.includes(phrase)) throw new Error(`Public landing boundary contains prohibited text: ${phrase}`);
}

console.log("Public landing release boundary verified.");
