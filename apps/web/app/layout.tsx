import type { Metadata } from "next";
import localFont from "next/font/local";
import { Nav } from "@/components/Nav";
import "./globals.css";

/** Local fonts — avoid next/font/google network fetches that flake in CI/Vercel builds. */
const sora = localFont({
  src: [
    { path: "../fonts/sora-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "../fonts/sora-latin-500-normal.woff2", weight: "500", style: "normal" },
    { path: "../fonts/sora-latin-600-normal.woff2", weight: "600", style: "normal" },
    { path: "../fonts/sora-latin-700-normal.woff2", weight: "700", style: "normal" },
  ],
  variable: "--font-sora",
  display: "swap",
});

const plex = localFont({
  src: [
    { path: "../fonts/ibm-plex-sans-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "../fonts/ibm-plex-sans-latin-500-normal.woff2", weight: "500", style: "normal" },
    { path: "../fonts/ibm-plex-sans-latin-600-normal.woff2", weight: "600", style: "normal" },
  ],
  variable: "--font-plex",
  display: "swap",
});

const plexMono = localFont({
  src: [
    { path: "../fonts/ibm-plex-mono-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "../fonts/ibm-plex-mono-latin-500-normal.woff2", weight: "500", style: "normal" },
  ],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Job Extractor LoRA",
  description:
    "A small fine-tuned model that turns job postings into structured JSON — compared against its base model and Gemini.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sora.variable} ${plex.variable} ${plexMono.variable}`}>
      <body
        style={
          {
            "--font-display": "var(--font-sora)",
            "--font-body": "var(--font-plex)",
            "--font-mono": "var(--font-plex-mono)",
          } as React.CSSProperties
        }
      >
        <div className="min-h-screen">
          <Nav />
          <main className="mx-auto w-full max-w-6xl px-4 pb-16 pt-8 sm:px-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
