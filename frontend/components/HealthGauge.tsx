"use client";

import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";

interface HealthGaugeProps {
  confidence: number;
  healthClass: string;
}

export default function HealthGauge({ confidence, healthClass }: HealthGaugeProps) {
  const pct = Math.round(confidence * 100);
  const color =
    healthClass === "Healthy"
      ? "#10b981"
      : healthClass === "Bleached"
        ? "#f97316"
        : "#ef4444";

  const data = [
    { name: "confidence", value: pct },
    { name: "remainder", value: 100 - pct },
  ];

  return (
    <div className="relative h-36 w-36">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={48}
            outerRadius={60}
            startAngle={90}
            endAngle={-270}
            dataKey="value"
            stroke="none"
          >
            <Cell fill={color} />
            <Cell fill="#1e293b" />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold" style={{ color }}>
          {pct}%
        </span>
        <span className="text-xs text-slate-400">confidence</span>
      </div>
    </div>
  );
}
