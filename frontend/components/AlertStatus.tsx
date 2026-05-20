"use client";

import type { AlertResult } from "@/types";

interface AlertStatusProps {
  alert: AlertResult;
  timestamp: string;
}

export default function AlertStatus({ alert, timestamp }: AlertStatusProps) {
  if (!alert.triggered) {
    return (
      <div className="glass-panel border border-slate-600/50 p-6 text-center text-slate-400">
        <p className="text-sm">No alert required — conditions below threshold.</p>
      </div>
    );
  }

  return (
    <div className="glass-panel border-2 border-reef-teal/40 p-6">
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="text-lg font-semibold text-white">Alert Status</h3>
        {alert.alert_type && (
          <span
            className={`rounded-full px-3 py-1 text-xs font-bold ${
              alert.alert_type === "CRITICAL"
                ? "bg-red-600 text-white"
                : "bg-orange-600 text-white"
            }`}
          >
            {alert.alert_type}
          </span>
        )}
      </div>
      <p className="mt-3 text-reef-teal">
        Alert dispatched to marine officials
      </p>
      <p className="mt-1 text-sm text-slate-400">{alert.message}</p>
      <p className="mt-2 text-xs text-slate-500">{timestamp}</p>

      <div className="mt-6 flex flex-wrap gap-6">
        <StatusIndicator label="Email" sent={alert.email_sent} />
        <StatusIndicator
          label="SMS"
          sent={alert.sms_sent}
          note={alert.alert_type !== "CRITICAL" ? "CRITICAL only" : undefined}
        />
      </div>
    </div>
  );
}

function StatusIndicator({
  label,
  sent,
  note,
}: {
  label: string;
  sent: boolean;
  note?: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={`h-3 w-3 rounded-full ${sent ? "bg-emerald-500" : "bg-red-500"}`}
      />
      <span className="text-sm text-slate-300">
        {label}: {sent ? "Sent" : "Not sent"}
        {note ? ` (${note})` : ""}
      </span>
    </div>
  );
}
