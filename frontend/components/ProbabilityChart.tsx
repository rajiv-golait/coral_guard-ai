"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { HealthClass } from "@/types";

interface ProbabilityChartProps {
  probabilities: Record<HealthClass, number>;
}

const BAR_COLORS: Record<HealthClass, string> = {
  Healthy: "#10b981",
  Bleached: "#f97316",
  Dead: "#ef4444",
};

export default function ProbabilityChart({
  probabilities,
}: ProbabilityChartProps) {
  const data = (["Healthy", "Bleached", "Dead"] as HealthClass[]).map(
    (name) => ({
      name,
      probability: Math.round(probabilities[name] * 1000) / 10,
      fill: BAR_COLORS[name],
    })
  );

  return (
    <div className="glass-panel p-6">
      <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-slate-400">
        Class Probabilities
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
          <XAxis type="number" domain={[0, 100]} unit="%" stroke="#94a3b8" />
          <YAxis type="category" dataKey="name" width={80} stroke="#94a3b8" />
          <Tooltip
            contentStyle={{
              background: "#0f2744",
              border: "1px solid #14b8a6",
              borderRadius: "8px",
            }}
            formatter={(value: number) => [`${value}%`, "Probability"]}
          />
          <Bar dataKey="probability" radius={[0, 6, 6, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
