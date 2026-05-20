"use client";

import type { HealthClass, RiskLevel } from "@/types";
import HealthGauge from "./HealthGauge";

interface ResultCardProps {
  healthClass: HealthClass;
  confidence: number;
  riskLevel: RiskLevel;
}

const STYLES: Record<
  HealthClass,
  { bg: string; border: string; text: string }
> = {
  Healthy: {
    bg: "bg-emerald-500/15",
    border: "border-emerald-500/50",
    text: "text-emerald-400",
  },
  Bleached: {
    bg: "bg-orange-500/15",
    border: "border-orange-500/50",
    text: "text-orange-400",
  },
  Dead: {
    bg: "bg-red-500/15",
    border: "border-red-500/50",
    text: "text-red-400",
  },
};

const RISK_COLORS: Record<RiskLevel, string> = {
  CRITICAL: "bg-red-600 text-white",
  HIGH: "bg-orange-600 text-white",
  MODERATE: "bg-yellow-600 text-ocean-950",
  LOW: "bg-emerald-600 text-white",
};

export default function ResultCard({
  healthClass,
  confidence,
  riskLevel,
}: ResultCardProps) {
  const style = STYLES[healthClass];

  return (
    <div
      className={`glass-panel flex flex-col items-center gap-6 border-2 p-8 sm:flex-row sm:justify-between ${style.bg} ${style.border}`}
    >
      <div className="text-center sm:text-left">
        <p className="text-sm uppercase tracking-wider text-slate-400">
          Classification Result
        </p>
        <h2 className={`mt-2 text-4xl font-bold ${style.text}`}>{healthClass}</h2>
        <span
          className={`mt-4 inline-block rounded-full px-4 py-1 text-sm font-semibold ${RISK_COLORS[riskLevel]}`}
        >
          Risk: {riskLevel}
        </span>
      </div>
      <HealthGauge confidence={confidence} healthClass={healthClass} />
    </div>
  );
}
