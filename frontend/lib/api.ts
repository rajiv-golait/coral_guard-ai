import type {
  AlertRequest,
  AlertResult,
  ConservationReport,
  OceanParams,
  PredictionResult,
  ReportRequest,
} from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const err = (await res.json()) as { detail?: string };
      if (err.detail) detail = err.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<{ status: string; models_loaded: boolean }> {
  const res = await fetch(`${API_URL}/api/health`);
  return handleResponse(res);
}

export async function predictCoral(
  image: File,
  params: OceanParams
): Promise<PredictionResult> {
  const form = new FormData();
  form.append("image", image);
  form.append("params", JSON.stringify(params));

  const res = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    body: form,
  });
  return handleResponse<PredictionResult>(res);
}

export async function generateReport(
  body: ReportRequest
): Promise<ConservationReport> {
  const res = await fetch(`${API_URL}/api/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleResponse<ConservationReport>(res);
}

export async function triggerAlert(body: AlertRequest): Promise<AlertResult> {
  const res = await fetch(`${API_URL}/api/alert`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleResponse<AlertResult>(res);
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(",")[1] ?? "";
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
