export type HealthClass = "Healthy" | "Bleached" | "Dead";
export type RiskLevel = "CRITICAL" | "HIGH" | "MODERATE" | "LOW";
export type AlertType = "CRITICAL" | "HIGH";

export interface OceanParams {
  Latitude_Degrees: number;
  Longitude_Degrees: number;
  Depth_m: number;
  Turbidity: number;
  Cyclone_Frequency: number;
  ClimSST: number;
  SSTA: number;
  TSA: number;
  Percent_Cover: number;
  Date_Year: number;
}

export interface SliderConfig {
  key: keyof OceanParams;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
}

export interface PredictionResult {
  health_class: HealthClass;
  confidence: number;
  probabilities: Record<HealthClass, number>;
  cluster_id: number;
  cluster_name: string;
  is_anomaly: boolean;
  risk_level: RiskLevel;
  thermal_stress: number;
  light_index: number;
  sst_total: number;
}

export interface ConservationReport {
  executive_summary: string;
  risk_level: RiskLevel;
  key_threats: string[];
  immediate_actions: string[];
  preventive_measures: string[];
  monitoring_schedule: string;
  recovery_prognosis: string;
  scientific_context: string;
}

export interface ReportRequest extends PredictionResult {
  latitude: number;
  longitude: number;
  depth_m: number;
  turbidity: number;
  ssta: number;
  tsa: number;
  site_name: string;
}

export interface AlertRequest {
  health_class: HealthClass;
  confidence: number;
  cluster_name: string;
  is_anomaly: boolean;
  risk_level: RiskLevel;
  site_name: string;
  latitude: number;
  longitude: number;
  depth_m: number;
  executive_summary: string;
  immediate_actions: string[];
  override_email?: string;
  image_base64?: string;
  image_filename: string;
}

export interface AlertResult {
  triggered: boolean;
  alert_type: AlertType | null;
  email_sent: boolean;
  sms_sent: boolean;
  message: string;
}

export interface AnalysisSession {
  prediction: PredictionResult;
  report: ConservationReport;
  alert: AlertResult;
  params: OceanParams;
  siteName: string;
  imagePreview: string;
  timestamp: string;
}
