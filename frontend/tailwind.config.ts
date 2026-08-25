import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ivory: "#FAF7F2",
        parchment: "#F3EDE3",
        forest: "#1E3A2F",
        "forest-deep": "#142A22",
        sage: "#8FAE9B",
        terracotta: "#C4674F",
        "terracotta-soft": "#D98B73",
        ink: "#26312C",
        "ink-muted": "#5C6862",
        line: "#E4DCCF",
      },
      fontFamily: {
        display: ["var(--font-fraunces)", "Georgia", "serif"],
        body: ["var(--font-inter)", "system-ui", "sans-serif"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(38,49,44,.06), 0 8px 24px rgba(38,49,44,.08)",
        lift: "0 2px 6px rgba(38,49,44,.08), 0 16px 40px rgba(38,49,44,.14)",
      },
      borderRadius: { xl2: "1.25rem" },
    },
  },
  plugins: [],
};
export default config;
