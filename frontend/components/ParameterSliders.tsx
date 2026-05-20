"use client";

import LocationCapture from "@/components/LocationCapture";
import type { OceanParams, SliderConfig } from "@/types";

export const SLIDER_CONFIGS: SliderConfig[] = [
  { key: "Latitude_Degrees", label: "Latitude", min: -90, max: 90, step: 0.1, unit: "°" },
  { key: "Longitude_Degrees", label: "Longitude", min: -180, max: 180, step: 0.1, unit: "°" },
  { key: "Depth_m", label: "Depth", min: 0, max: 50, step: 0.5, unit: "m" },
  { key: "Turbidity", label: "Turbidity", min: 0, max: 20, step: 0.1, unit: "NTU" },
  { key: "Cyclone_Frequency", label: "Cyclone Frequency", min: 0, max: 10, step: 0.1, unit: "" },
  { key: "ClimSST", label: "ClimSST", min: 20, max: 35, step: 0.1, unit: "°C" },
  { key: "SSTA", label: "SSTA", min: -3, max: 6, step: 0.1, unit: "°C" },
  { key: "TSA", label: "TSA", min: -2, max: 5, step: 0.1, unit: "°C" },
  { key: "Percent_Cover", label: "Percent Cover", min: 0, max: 100, step: 1, unit: "%" },
  { key: "Date_Year", label: "Date Year", min: 1980, max: 2025, step: 1, unit: "" },
];

export const DEFAULT_PARAMS: OceanParams = {
  Latitude_Degrees: -16.5,
  Longitude_Degrees: 145.8,
  Depth_m: 8,
  Turbidity: 2.5,
  Cyclone_Frequency: 1.2,
  ClimSST: 27.5,
  SSTA: 0.8,
  TSA: 1.1,
  Percent_Cover: 65,
  Date_Year: 2024,
};

interface ParameterSlidersProps {
  params: OceanParams;
  onChange: (params: OceanParams) => void;
  disabled?: boolean;
}

export default function ParameterSliders({
  params,
  onChange,
  disabled = false,
}: ParameterSlidersProps) {
  const update = (key: keyof OceanParams, value: number) => {
    onChange({ ...params, [key]: value });
  };

  const handleBrowserLocation = (latitude: number, longitude: number) => {
    onChange({
      ...params,
      Latitude_Degrees: latitude,
      Longitude_Degrees: longitude,
    });
  };

  return (
    <div className="grid gap-4 sm:grid-cols-1">
      <LocationCapture
        latitude={params.Latitude_Degrees}
        longitude={params.Longitude_Degrees}
        onLocationChange={handleBrowserLocation}
        disabled={disabled}
      />
      {SLIDER_CONFIGS.map((cfg) => (
        <div key={cfg.key} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span className="text-slate-300">{cfg.label}</span>
            <span className="font-mono text-reef-teal">
              {params[cfg.key]}
              {cfg.unit ? ` ${cfg.unit}` : ""}
            </span>
          </div>
          <input
            type="range"
            min={cfg.min}
            max={cfg.max}
            step={cfg.step}
            value={params[cfg.key]}
            onChange={(e) => update(cfg.key, parseFloat(e.target.value))}
          />
        </div>
      ))}
    </div>
  );
}
