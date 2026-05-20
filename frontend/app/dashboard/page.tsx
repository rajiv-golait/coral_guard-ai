"use client";

import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import ImageUpload from "@/components/ImageUpload";
import ParameterSliders, {
  DEFAULT_PARAMS,
} from "@/components/ParameterSliders";
import {
  fileToBase64,
  generateReport,
  predictCoral,
  triggerAlert,
} from "@/lib/api";
import type { AnalysisSession, OceanParams } from "@/types";

const LOADING_STEPS = [
  "Preprocessing image...",
  "Running CNN classification...",
  "Analyzing ocean environment...",
  "Generating conservation report...",
  "Checking alert thresholds...",
];

const SESSION_KEY = "coralguard_analysis";

export default function DashboardPage() {
  const router = useRouter();
  const [params, setParams] = useState<OceanParams>(DEFAULT_PARAMS);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [siteName, setSiteName] = useState("Great Barrier Reef Site A");
  const [overrideEmail, setOverrideEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const onImageSelect = useCallback((file: File, previewUrl: string) => {
    setImageFile(file);
    setImagePreview(previewUrl);
    setError(null);
  }, []);

  const runAnalysis = async () => {
    if (!imageFile) {
      setError("Please upload a coral image before analyzing.");
      return;
    }

    setLoading(true);
    setError(null);
    setLoadingStep(0);

    try {
      setLoadingStep(0);
      const prediction = await predictCoral(imageFile, params);

      setLoadingStep(1);
      setLoadingStep(2);

      setLoadingStep(3);
      const report = await generateReport({
        ...prediction,
        latitude: params.Latitude_Degrees,
        longitude: params.Longitude_Degrees,
        depth_m: params.Depth_m,
        turbidity: params.Turbidity,
        ssta: params.SSTA,
        tsa: params.TSA,
        site_name: siteName,
      });

      setLoadingStep(4);
      const imageBase64 = await fileToBase64(imageFile);
      const alert = await triggerAlert({
        health_class: prediction.health_class,
        confidence: prediction.confidence,
        cluster_name: prediction.cluster_name,
        is_anomaly: prediction.is_anomaly,
        risk_level: prediction.risk_level,
        site_name: siteName,
        latitude: params.Latitude_Degrees,
        longitude: params.Longitude_Degrees,
        depth_m: params.Depth_m,
        executive_summary: report.executive_summary,
        immediate_actions: report.immediate_actions,
        override_email: overrideEmail.trim() || undefined,
        image_base64: imageBase64,
        image_filename: imageFile.name || "coral_image.jpg",
      });

      const session: AnalysisSession = {
        prediction,
        report,
        alert,
        params,
        siteName,
        imagePreview,
        timestamp: new Date().toISOString(),
      };

      sessionStorage.setItem(SESSION_KEY, JSON.stringify(session));
      router.push("/results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 sm:py-12">
      <h1 className="text-2xl font-bold text-white sm:text-3xl">
        Reef Analysis Dashboard
      </h1>
      <p className="mt-2 text-slate-400">
        Upload coral imagery and configure oceanographic parameters.
      </p>

      <div className="mt-8 flex flex-col gap-8 lg:flex-row">
        <div className="w-full space-y-6 lg:w-[40%]">
          <div className="glass-panel p-6">
            <ImageUpload onImageSelect={onImageSelect} />
            <div className="mt-6 space-y-4">
              <div>
                <label className="text-sm text-slate-300">Site Name</label>
                <input
                  type="text"
                  value={siteName}
                  onChange={(e) => setSiteName(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-slate-600 bg-ocean-800 px-3 py-2 text-white focus:border-reef-teal focus:outline-none"
                />
              </div>
              <div>
                <label className="text-sm text-slate-300">
                  Alert Email Override (optional)
                </label>
                <input
                  type="email"
                  value={overrideEmail}
                  onChange={(e) => setOverrideEmail(e.target.value)}
                  placeholder="marine.official@agency.gov"
                  className="mt-1 w-full rounded-lg border border-slate-600 bg-ocean-800 px-3 py-2 text-white focus:border-reef-teal focus:outline-none"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="w-full lg:w-[60%]">
          <div className="glass-panel p-6">
            <h2 className="mb-4 text-lg font-semibold text-reef-teal">
              Oceanographic Parameters
            </h2>
            <ParameterSliders
              params={params}
              onChange={setParams}
              disabled={loading}
            />

            {error && (
              <p className="mt-4 rounded-lg bg-red-500/20 px-4 py-2 text-sm text-red-300">
                {error}
              </p>
            )}

            <button
              type="button"
              onClick={runAnalysis}
              disabled={loading}
              className="mt-8 w-full rounded-xl bg-reef-teal py-4 font-semibold text-ocean-950 transition hover:bg-teal-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Analyzing..." : "Analyze Reef"}
            </button>

            {loading && (
              <div className="mt-6 flex flex-col items-center gap-3">
                <div className="h-10 w-10 animate-spin rounded-full border-2 border-reef-teal border-t-transparent" />
                <p className="text-sm text-reef-teal">
                  {LOADING_STEPS[loadingStep] ?? LOADING_STEPS[LOADING_STEPS.length - 1]}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
