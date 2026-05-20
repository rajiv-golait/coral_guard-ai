"use client";

import Link from "next/link";
import { motion } from "framer-motion";

const features = [
  {
    title: "CNN Classification",
    desc: "EfficientNet-B3 fusion model classifies coral as Healthy, Bleached, or Dead.",
  },
  {
    title: "GenAI Reports",
    desc: "Groq LLM generates actionable conservation assessments.",
  },
  {
    title: "Agentic Alerts",
    desc: "Autonomous email & SMS alerts to marine officials when thresholds are met.",
  },
];

function WaveParticles() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden opacity-40">
      {Array.from({ length: 24 }).map((_, i) => (
        <motion.div
          key={i}
          className="absolute h-2 w-2 rounded-full bg-reef-teal/60"
          style={{
            left: `${(i * 17) % 100}%`,
            top: `${(i * 23) % 100}%`,
          }}
          animate={{
            y: [0, -30, 0],
            opacity: [0.2, 0.8, 0.2],
          }}
          transition={{
            duration: 4 + (i % 5),
            repeat: Infinity,
            delay: i * 0.2,
          }}
        />
      ))}
    </div>
  );
}

export default function LandingPage() {
  return (
    <section className="relative min-h-[calc(100vh-73px)] gradient-ocean overflow-hidden">
      <WaveParticles />
      <div
        className="absolute inset-0 bg-cover bg-center opacity-20"
        style={{
          backgroundImage:
            "linear-gradient(180deg, rgba(10,22,40,0.7) 0%, rgba(10,22,40,0.95) 100%), url('https://images.unsplash.com/photo-1583212292454-1fe6229603b6?w=1920&q=80')",
        }}
      />
      <div className="relative z-10 mx-auto flex max-w-4xl flex-col items-center px-6 py-24 text-center sm:py-32">
        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold tracking-tight text-white sm:text-6xl"
        >
          CoralGuard AI
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="mt-4 max-w-2xl text-lg text-slate-300 sm:text-xl"
        >
          Intelligent marine ecosystem monitoring — classify coral health, analyze
          ocean zones, and dispatch autonomous conservation alerts.
        </motion.p>

        <motion.ul
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-12 grid w-full max-w-2xl gap-4 text-left sm:grid-cols-3"
        >
          {features.map((f, i) => (
            <motion.li
              key={f.title}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35 + i * 0.1 }}
              className="glass-panel p-4"
            >
              <h3 className="font-semibold text-reef-teal">{f.title}</h3>
              <p className="mt-1 text-sm text-slate-400">{f.desc}</p>
            </motion.li>
          ))}
        </motion.ul>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.55 }}
          className="mt-14"
        >
          <Link
            href="/dashboard"
            className="inline-flex items-center rounded-full bg-reef-teal px-10 py-4 text-lg font-semibold text-ocean-950 shadow-lg transition hover:bg-teal-400 hover:shadow-reef-teal/30"
          >
            Start Analysis
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
