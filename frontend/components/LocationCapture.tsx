"use client";

import { useCallback, useState } from "react";

type GeoStatus = "idle" | "loading" | "success" | "error";

interface LocationCaptureProps {
  latitude: number;
  longitude: number;
  onLocationChange: (latitude: number, longitude: number) => void;
  disabled?: boolean;
}

function roundCoord(value: number): number {
  return Math.round(value * 10) / 10;
}

export default function LocationCapture({
  latitude,
  longitude,
  onLocationChange,
  disabled = false,
}: LocationCaptureProps) {
  const [status, setStatus] = useState<GeoStatus>("idle");
  const [message, setMessage] = useState<string | null>(null);

  const useBrowserLocation = useCallback(() => {
    if (disabled) return;

    if (!navigator.geolocation) {
      setStatus("error");
      setMessage("Geolocation is not supported in this browser.");
      return;
    }

    setStatus("loading");
    setMessage(null);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const lat = roundCoord(position.coords.latitude);
        const lon = roundCoord(position.coords.longitude);
        onLocationChange(lat, lon);
        setStatus("success");
        setMessage(
          `Location set (${lat}°, ${lon}°). Adjust sliders below if needed.`
        );
      },
      (err) => {
        setStatus("error");
        const texts: Record<number, string> = {
          1: "Location permission denied. Allow location access in your browser.",
          2: "Location unavailable. Try again or set coordinates manually.",
          3: "Location request timed out. Try again.",
        };
        setMessage(texts[err.code] ?? "Could not get your location.");
      },
      {
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 60_000,
      }
    );
  }, [disabled, onLocationChange]);

  return (
    <div className="rounded-xl border border-teal-500/25 bg-ocean-800/60 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-reef-teal">Site coordinates</h3>
          <p className="mt-1 text-xs text-slate-400">
            Use browser GPS or fine-tune with the latitude/longitude sliders below.
          </p>
        </div>
        <button
          type="button"
          onClick={useBrowserLocation}
          disabled={disabled || status === "loading"}
          className="inline-flex items-center gap-2 rounded-lg border border-reef-teal/50 bg-reef-teal/10 px-4 py-2 text-sm font-medium text-reef-teal transition hover:bg-reef-teal/20 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {status === "loading" ? (
            <>
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-reef-teal border-t-transparent" />
              Locating...
            </>
          ) : (
            <>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="h-4 w-4"
                aria-hidden
              >
                <path
                  fillRule="evenodd"
                  d="M11.54 22.351l.07.04.028.016a.76.76 0 00.723 0l.028-.015.071-.041a16.975 16.975 0 001.144-.742 19.58 19.58 0 002.683-2.282c1.944-1.99 3.963-4.98 3.963-8.827a8.25 8.25 0 00-16.5 0c0 3.846 2.02 6.837 3.963 8.827a19.58 19.58 0 002.682 2.282 16.975 16.975 0 001.145.742zM12 13.5a3 3 0 100-6 3 3 0 000 6z"
                  clipRule="evenodd"
                />
              </svg>
              Use my location
            </>
          )}
        </button>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-lg bg-ocean-900/80 px-3 py-2">
          <span className="text-slate-400">Latitude</span>
          <p className="font-mono text-white">{latitude}°</p>
        </div>
        <div className="rounded-lg bg-ocean-900/80 px-3 py-2">
          <span className="text-slate-400">Longitude</span>
          <p className="font-mono text-white">{longitude}°</p>
        </div>
      </div>

      {message && (
        <p
          className={`mt-3 text-xs ${
            status === "error" ? "text-red-300" : "text-slate-400"
          }`}
          role={status === "error" ? "alert" : "status"}
        >
          {message}
        </p>
      )}
    </div>
  );
}
