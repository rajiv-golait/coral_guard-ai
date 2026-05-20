"use client";

interface ClusterBadgeProps {
  clusterName: string;
  clusterId: number;
  isAnomaly: boolean;
  thermalStress: number;
  lightIndex: number;
  sstTotal: number;
}

const CLUSTER_ICONS: Record<number, string> = {
  [-1]: "⚠️",
  0: "🌊",
  1: "🌡️",
  2: "🧪",
  3: "🌫️",
};

export default function ClusterBadge({
  clusterName,
  clusterId,
  isAnomaly,
  thermalStress,
  lightIndex,
  sstTotal,
}: ClusterBadgeProps) {
  const icon = CLUSTER_ICONS[clusterId] ?? "🔬";

  return (
    <div className="glass-panel p-6">
      <h3 className="text-sm font-medium uppercase tracking-wider text-slate-400">
        Ocean Environment
      </h3>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <span className="text-3xl">{icon}</span>
        <div>
          <p className="text-xl font-semibold text-reef-teal">{clusterName}</p>
          <p className="text-sm text-slate-400">Cluster ID: {clusterId}</p>
        </div>
        {isAnomaly && (
          <span className="rounded-full bg-red-600/90 px-3 py-1 text-xs font-bold text-white">
            ANOMALY DETECTED
          </span>
        )}
      </div>
      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        <div className="rounded-lg bg-ocean-800/80 p-4">
          <p className="text-xs text-slate-400">Thermal Stress</p>
          <p className="text-lg font-mono text-white">{thermalStress.toFixed(3)}</p>
        </div>
        <div className="rounded-lg bg-ocean-800/80 p-4">
          <p className="text-xs text-slate-400">Light Index</p>
          <p className="text-lg font-mono text-white">{lightIndex.toFixed(3)}</p>
        </div>
        <div className="rounded-lg bg-ocean-800/80 p-4">
          <p className="text-xs text-slate-400">SST Total</p>
          <p className="text-lg font-mono text-white">{sstTotal.toFixed(2)} °C</p>
        </div>
      </div>
    </div>
  );
}
