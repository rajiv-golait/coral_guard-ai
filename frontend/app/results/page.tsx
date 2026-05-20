"use client";

import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import AlertStatus from "@/components/AlertStatus";
import ClusterBadge from "@/components/ClusterBadge";
import ConservationReportCard from "@/components/ConservationReport";
import ProbabilityChart from "@/components/ProbabilityChart";
import ResultCard from "@/components/ResultCard";
import type { AnalysisSession } from "@/types";

const SESSION_KEY = "coralguard_analysis";

export default function ResultsPage() {
  const router = useRouter();
  const [session, setSession] = useState<AnalysisSession | null>(null);

  useEffect(() => {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) {
      router.replace("/dashboard");
      return;
    }
    try {
      setSession(JSON.parse(raw) as AnalysisSession);
    } catch {
      router.replace("/dashboard");
    }
  }, [router]);

  if (!session) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <div className="h-10 w-10 animate-spin rounded-full border-2 border-reef-teal border-t-transparent" />
      </div>
    );
  }

  const { prediction, report, alert, siteName, imagePreview, timestamp } =
    session;

  const formattedTime = new Date(timestamp).toLocaleString();

  return (
    <div className="mx-auto max-w-5xl space-y-8 px-4 py-8 sm:px-6 sm:py-12">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Analysis Results</h1>
          <p className="text-slate-400">{siteName}</p>
        </div>
        {imagePreview && (
          <div className="relative h-20 w-28 overflow-hidden rounded-lg border border-teal-500/30">
            <Image
              src={imagePreview}
              alt="Analyzed coral"
              fill
              className="object-cover"
              unoptimized
            />
          </div>
        )}
      </div>

      <ResultCard
        healthClass={prediction.health_class}
        confidence={prediction.confidence}
        riskLevel={prediction.risk_level}
      />

      <ProbabilityChart probabilities={prediction.probabilities} />

      <ClusterBadge
        clusterName={prediction.cluster_name}
        clusterId={prediction.cluster_id}
        isAnomaly={prediction.is_anomaly}
        thermalStress={prediction.thermal_stress}
        lightIndex={prediction.light_index}
        sstTotal={prediction.sst_total}
      />

      <ConservationReportCard report={report} />

      <AlertStatus alert={alert} timestamp={formattedTime} />

      <div className="flex justify-center pt-4">
        <Link
          href="/dashboard"
          className="rounded-full border border-reef-teal px-8 py-3 font-medium text-reef-teal transition hover:bg-reef-teal/10"
        >
          Analyze Another Reef
        </Link>
      </div>
    </div>
  );
}
