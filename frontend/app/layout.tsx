import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-geist-sans" });

export const metadata: Metadata = {
  title: "CoralGuard AI — Marine Ecosystem Monitoring",
  description:
    "AI-powered coral health classification, ocean environment analysis, and autonomous conservation alerts.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans`}>
        <header className="sticky top-0 z-50 border-b border-teal-500/20 bg-ocean-950/90 backdrop-blur-md">
          <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
            <Link href="/" className="text-xl font-bold text-reef-teal">
              CoralGuard AI
            </Link>
            <div className="flex gap-4 text-sm">
              <Link
                href="/dashboard"
                className="text-slate-300 transition hover:text-reef-teal"
              >
                Dashboard
              </Link>
              <Link
                href="/results"
                className="text-slate-300 transition hover:text-reef-teal"
              >
                Results
              </Link>
            </div>
          </nav>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
