import type { Metadata } from "next";
import { Fraunces, Inter } from "next/font/google";
import "./globals.css";
import { Toaster } from "sonner";

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
});
const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });

export const metadata: Metadata = {
  title: "Folliscan — Cosmetic ingredient intelligence",
  description:
    "Multi-task graph neural network screening of hair-health activity, toxicity and cosmetic safety, with uncertainty-aware predictions and mechanistic explanations.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${fraunces.variable} ${inter.variable}`}>
      <body>
        {children}
        <Toaster
          position="top-center"
          toastOptions={{
            style: {
              background: "#1E3A2F",
              color: "#FAF7F2",
              borderRadius: "12px",
            },
          }}
        />
      </body>
    </html>
  );
}
