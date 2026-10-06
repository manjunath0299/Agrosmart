import { useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Bell,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  CloudUpload,
  Cpu,
  Droplets,
  Gauge,
  Leaf,
  Menu,
  Package,
  Play,
  Settings,
  ShieldCheck,
  Sprout,
  Syringe,
  Upload,
  Wifi,
  X,
  Zap,
} from "lucide-react";
import { motion } from "framer-motion";

import {
  analyzeLeaf,
  approveSpray,
} from "./services/api";

// ============================================================
// CONFIGURATION
// ============================================================

const DEVICE_ID = "ESP32-001";

const navigation = [
  {
    name: "AI Analysis",
    icon: Activity,
  },
  {
    name: "Treatments",
    icon: Syringe,
  },
  {
    name: "Devices",
    icon: Cpu,
  },
  {
    name: "Analytics",
    icon: BarChart3,
  },
  {
    name: "Notifications",
    icon: Bell,
  },
  {
    name: "Settings",
    icon: Settings,
  },
  {
    name: "Help",
    icon: CircleHelp,
  },
];


// ============================================================
// HELPERS
// ============================================================

function formatConfidence(value) {
  if (value === undefined || value === null) {
    return "--";
  }

  return `${Number(value).toFixed(1)}%`;
}

function formatDiseaseName(name) {
  if (!name) {
    return "--";
  }

  return name
    .replace(/___/g, " - ")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}


function getSeverityClass(level) {
  switch (level) {
    case "LOW":
      return "text-emerald-400";

    case "MEDIUM":
      return "text-amber-400";

    case "HIGH":
      return "text-orange-400";

    case "SEVERE":
      return "text-red-400";

    default:
      return "text-slate-400";
  }
}


function getSeverityRing(level) {
  switch (level) {
    case "LOW":
      return "stroke-emerald-400";

    case "MEDIUM":
      return "stroke-amber-400";

    case "HIGH":
      return "stroke-orange-400";

    case "SEVERE":
      return "stroke-red-400";

    default:
      return "stroke-slate-500";
  }
}


// ============================================================
// SEVERITY GAUGE
// ============================================================

function SeverityGauge({ severity, level }) {
  const numericSeverity =
    typeof severity === "number"
      ? Math.min(Math.max(severity, 0), 100)
      : 0;

  const radius = 68;
  const circumference = 2 * Math.PI * radius;

  const progress =
    circumference -
    (numericSeverity / 100) * circumference;

  return (
    <div className="relative flex h-52 w-52 items-center justify-center">

      <svg
        width="180"
        height="180"
        viewBox="0 0 180 180"
        className="-rotate-90"
      >
        <circle
          cx="90"
          cy="90"
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth="12"
          className="text-slate-800"
        />

        <circle
          cx="90"
          cy="90"
          r={radius}
          fill="none"
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={progress}
          className={`${getSeverityRing(
            level
          )} transition-all duration-700`}
        />
      </svg>

      <div className="absolute text-center">

        <div className="text-4xl font-bold text-white">
          {numericSeverity.toFixed(1)}%
        </div>

        <div
          className={`mt-1 text-sm font-semibold ${getSeverityClass(
            level
          )}`}
        >
          {level || "UNKNOWN"}
        </div>

      </div>
    </div>
  );
}


// ============================================================
// MAIN APP
// ============================================================

export default function App() {

  const fileInputRef = useRef(null);

  // ----------------------------------------------------------
  // UI STATE
  // ----------------------------------------------------------

  const [activePage, setActivePage] =
    useState("AI Analysis");

  const [mobileMenuOpen, setMobileMenuOpen] =
    useState(false);

  // ----------------------------------------------------------
  // IMAGE STATE
  // ----------------------------------------------------------

  const [selectedFile, setSelectedFile] =
    useState(null);

  const [previewUrl, setPreviewUrl] =
    useState(null);

  const [isDragging, setIsDragging] =
    useState(false);

  // ----------------------------------------------------------
  // AI STATE
  // ----------------------------------------------------------

  const [analysisResult, setAnalysisResult] =
    useState(null);

  const [isAnalyzing, setIsAnalyzing] =
    useState(false);

  // ----------------------------------------------------------
  // SPRAY STATE
  // ----------------------------------------------------------

  const [sprayResult, setSprayResult] =
    useState(null);

  const [isSpraying, setIsSpraying] =
    useState(false);

  // ----------------------------------------------------------
  // ERROR STATE
  // ----------------------------------------------------------

  const [error, setError] =
    useState(null);


  // ==========================================================
  // IMAGE HANDLING
  // ==========================================================

  const handleFileSelect = (file) => {

    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {

      setError(
        "Please select a valid leaf image."
      );

      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const newPreviewUrl =
      URL.createObjectURL(file);

    setSelectedFile(file);
    setPreviewUrl(newPreviewUrl);

    setAnalysisResult(null);
    setSprayResult(null);
    setError(null);
  };


  const handleFileInput = (event) => {

    const file =
      event.target.files?.[0];

    handleFileSelect(file);
  };


  const handleDrop = (event) => {

    event.preventDefault();

    setIsDragging(false);

    const file =
      event.dataTransfer.files?.[0];

    handleFileSelect(file);
  };


  const removeImage = () => {

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(null);
    setPreviewUrl(null);

    setAnalysisResult(null);
    setSprayResult(null);
    setError(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };


  // ==========================================================
  // TRY ANOTHER IMAGE
  // ==========================================================

  const handleTryAnother = () => {

    removeImage();

    // Allow React to finish clearing the previous file
    // before opening the browser file picker.
    setTimeout(() => {
      fileInputRef.current?.click();
    }, 50);
  };


  // ==========================================================
  // AI ANALYSIS
  // ==========================================================

  const handleAnalyze = async () => {

    if (!selectedFile) {

      setError(
        "Please upload a leaf image first."
      );

      return;
    }

    setIsAnalyzing(true);

    setError(null);
    setAnalysisResult(null);
    setSprayResult(null);

    try {

      const result =
        await analyzeLeaf(selectedFile);

      if (!result.success) {

        setError(
          result.message ||
            "AI analysis failed."
        );

        return;
      }

      setAnalysisResult(result);

    } catch (err) {

      console.error(err);

      setError(
        err.response?.data?.message ||
          err.message ||
          "Unable to connect to the AgroSmart backend."
      );

    } finally {

      setIsAnalyzing(false);
    }
  };


  // ==========================================================
  // APPROVE & SPRAY
  // ==========================================================

  const handleApproveSpray = async () => {

    if (!analysisResult) {

      setError(
        "Run AI analysis before spraying."
      );

      return;
    }

    if (!analysisResult.spray?.allowed) {

      setError(
        "Spraying is not allowed for this analysis."
      );

      return;
    }

    const confirmed = window.confirm(
      "Approve spray operation for this treatment?"
    );

    if (!confirmed) {
      return;
    }

    setIsSpraying(true);
    setError(null);
    setSprayResult(null);

    try {

      const result =
        await approveSpray({

          crop:
            analysisResult.disease.crop,

          disease:
            analysisResult.disease.name,

          severityLevel:
            analysisResult.severity.level,

          deviceId: DEVICE_ID,
        });

      setSprayResult(result);

    } catch (err) {

      console.error(err);

      setError(
        err.response?.data?.message ||
          err.message ||
          "Unable to send spray command."
      );

    } finally {

      setIsSpraying(false);
    }
  };


  // ==========================================================
  // RESULT REFERENCES
  // ==========================================================

  const disease =
    analysisResult?.disease;

  const severity =
    analysisResult?.severity;

  const treatment =
    analysisResult?.treatment;

  const spray =
    analysisResult?.spray;


  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="min-h-screen bg-[#07100d] text-white">

      {/* ======================================================
          MOBILE HEADER
      ====================================================== */}

      <header className="sticky top-0 z-40 flex items-center justify-between border-b border-white/5 bg-[#07100d]/95 px-5 py-4 backdrop-blur-xl lg:hidden">

        <div className="flex items-center gap-3">

          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500/15">

            <Sprout
              size={20}
              className="text-emerald-400"
            />

          </div>

          <span className="font-semibold">
            AgroSmart
          </span>

        </div>


        <button
          onClick={() =>
            setMobileMenuOpen(
              !mobileMenuOpen
            )
          }
          className="rounded-xl p-2 hover:bg-white/5"
        >
          {mobileMenuOpen ? (
            <X size={22} />
          ) : (
            <Menu size={22} />
          )}
        </button>

      </header>


      {/* ======================================================
          SIDEBAR
      ====================================================== */}

      <aside
        className={`
          fixed inset-y-0 left-0 z-50 w-64
          border-r border-white/5
          bg-[#08130f]
          transition-transform duration-300
          lg:translate-x-0
          ${
            mobileMenuOpen
              ? "translate-x-0"
              : "-translate-x-full"
          }
        `}
      >

        <div className="flex h-full flex-col">

          {/* Logo */}

          <div className="flex h-20 items-center gap-3 border-b border-white/5 px-6">

            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/15">

              <Leaf
                size={21}
                className="text-emerald-400"
              />

            </div>

            <div>

              <div className="font-bold tracking-tight">
                AgroSmart
              </div>

              <div className="text-[11px] text-slate-500">
                AI AGRICULTURE
              </div>

            </div>

          </div>


          {/* Navigation */}

          <nav className="flex-1 space-y-1 px-3 py-5">

            {navigation.map((item) => {

              const Icon = item.icon;

              const active =
                activePage === item.name;

              return (
                <button
                  key={item.name}
                  onClick={() => {

                    setActivePage(
                      item.name
                    );

                    setMobileMenuOpen(false);
                  }}
                  className={`
                    flex w-full items-center gap-3
                    rounded-xl px-4 py-3
                    text-sm transition
                    ${
                      active
                        ? "bg-emerald-500/10 text-emerald-400"
                        : "text-slate-400 hover:bg-white/5 hover:text-white"
                    }
                  `}
                >

                  <Icon size={18} />

                  <span>
                    {item.name}
                  </span>

                  {active && (
                    <ChevronRight
                      size={15}
                      className="ml-auto"
                    />
                  )}

                </button>
              );
            })}

          </nav>


          {/* Device Card */}

          <div className="border-t border-white/5 p-4">

            <div className="rounded-2xl border border-emerald-500/10 bg-emerald-500/5 p-4">

              <div className="mb-3 flex items-center justify-between">

                <div className="flex items-center gap-2">

                  <Wifi
                    size={15}
                    className="text-emerald-400"
                  />

                  <span className="text-xs font-medium">
                    Device
                  </span>

                </div>

                <span className="flex items-center gap-1.5 text-[10px] text-emerald-400">

                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />

                  ONLINE

                </span>

              </div>

              <div className="text-sm font-semibold">
                {DEVICE_ID}
              </div>

              <div className="mt-1 text-[11px] text-slate-500">
                MQTT Connected
              </div>

            </div>

          </div>

        </div>
      </aside>


      {/* ======================================================
          MAIN CONTENT
      ====================================================== */}

      <main className="min-h-screen lg:ml-64">

        <div className="mx-auto max-w-[1600px] px-5 py-6 sm:px-8 lg:px-10 lg:py-8">

          {/* ==================================================
              HEADER
          ================================================== */}

          <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">

            <div>

              <div className="mb-1 flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-emerald-400">

                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />

                AI Crop Intelligence

              </div>

              <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
                Plant Health Center
              </h1>

              <p className="mt-1 text-sm text-slate-500">
                Diagnose crop disease and manage
                precision treatment.
              </p>

            </div>


            <div className="flex items-center gap-3">

              <div className="hidden rounded-xl border border-white/5 bg-white/[0.02] px-4 py-2.5 sm:block">

                <div className="flex items-center gap-2">

                  <Activity
                    size={15}
                    className="text-emerald-400"
                  />

                  <span className="text-xs text-slate-400">
                    System
                  </span>

                  <span className="text-xs font-semibold text-emerald-400">
                    Operational
                  </span>

                </div>

              </div>


              <button className="relative rounded-xl border border-white/5 bg-white/[0.02] p-3 text-slate-400 hover:text-white">

                <Bell size={18} />

                <span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-emerald-400" />

              </button>

            </div>

          </div>


          {/* ==================================================
              ERROR
          ================================================== */}

          {error && (

            <motion.div
              initial={{
                opacity: 0,
                y: -10,
              }}
              animate={{
                opacity: 1,
                y: 0,
              }}
              className="mb-6 flex items-start gap-3 rounded-2xl border border-red-500/20 bg-red-500/5 p-4"
            >

              <AlertCircle
                size={20}
                className="mt-0.5 shrink-0 text-red-400"
              />

              <div className="flex-1">

                <div className="font-medium text-red-300">
                  Analysis Error
                </div>

                <div className="mt-1 text-sm text-red-400/80">
                  {error}
                </div>

              </div>

              <button
                onClick={() =>
                  setError(null)
                }
                className="text-red-400/60 hover:text-red-300"
              >
                <X size={17} />
              </button>

            </motion.div>
          )}


          {/* ==================================================
              UPLOAD + SYSTEM
          ================================================== */}

          <section className="mb-6 grid gap-6 xl:grid-cols-[1.5fr_1fr]">

            {/* Upload Card */}

            <div className="overflow-hidden rounded-3xl border border-white/5 bg-gradient-to-br from-emerald-500/[0.07] via-transparent to-transparent">

              <div className="p-6 sm:p-8">

                <div className="mb-6 flex items-start justify-between">

                  <div>

                    <div className="mb-2 flex items-center gap-2 text-emerald-400">

                      <Leaf size={17} />

                      <span className="text-xs font-semibold uppercase tracking-widest">
                        New Analysis
                      </span>

                    </div>

                    <h2 className="text-xl font-bold">
                      Upload Leaf Image
                    </h2>

                    <p className="mt-1 max-w-lg text-sm text-slate-500">
                      Upload a clear crop leaf image
                      for AI disease and severity
                      analysis.
                    </p>

                  </div>


                  <div className="hidden rounded-xl bg-emerald-500/10 p-3 sm:block">

                    <ShieldCheck
                      size={22}
                      className="text-emerald-400"
                    />

                  </div>

                </div>


                {/* ==================================================
                    DROP ZONE / PREVIEW
                ================================================== */}

                {!previewUrl ? (

                  <div
                    onDragOver={(event) => {

                      event.preventDefault();

                      setIsDragging(true);
                    }}
                    onDragLeave={() =>
                      setIsDragging(false)
                    }
                    onDrop={handleDrop}
                    onClick={() =>
                      fileInputRef.current?.click()
                    }
                    className={`
                      group cursor-pointer rounded-2xl
                      border border-dashed p-8
                      text-center transition-all sm:p-12
                      ${
                        isDragging
                          ? "border-emerald-400 bg-emerald-500/10"
                          : "border-white/10 bg-black/10 hover:border-emerald-500/30 hover:bg-emerald-500/[0.03]"
                      }
                    `}
                  >

                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={
                        handleFileInput
                      }
                    />

                    <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-500/10 transition-transform group-hover:scale-105">

                      <CloudUpload
                        size={28}
                        className="text-emerald-400"
                      />

                    </div>

                    <div className="font-semibold">
                      Drop your leaf image here
                    </div>

                    <div className="mt-1 text-sm text-slate-500">
                      or click to browse
                    </div>

                    <div className="mt-4 text-[11px] text-slate-600">
                      JPG, JPEG, PNG • Clear leaf
                      images recommended
                    </div>

                  </div>

                ) : (

                  <div className="relative overflow-hidden rounded-2xl border border-white/5 bg-black">

                    <img
                      src={previewUrl}
                      alt="Selected leaf"
                      className="max-h-[390px] w-full object-contain"
                    />


                    <button
                      onClick={removeImage}
                      className="absolute right-3 top-3 rounded-xl bg-black/70 p-2 text-white backdrop-blur hover:bg-black"
                      title="Remove image"
                    >
                      <X size={17} />
                    </button>


                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4 pt-10">

                      <div className="min-w-0">

                        <div className="truncate text-sm font-medium">
                          {selectedFile?.name}
                        </div>

                        <div className="text-xs text-slate-400">
                          {(
                            selectedFile?.size /
                            1024 /
                            1024
                          ).toFixed(2)}{" "}
                          MB
                        </div>

                      </div>

                    </div>

                  </div>
                )}


                {/* ==================================================
                    ANALYZE + TRY ANOTHER
                ================================================== */}

                <div className="mt-5 flex flex-col gap-3 sm:flex-row">

                  {/* Analyze */}

                  <button
                    onClick={handleAnalyze}
                    disabled={
                      !selectedFile ||
                      isAnalyzing
                    }
                    className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-emerald-500 px-5 py-3.5 text-sm font-bold text-[#06100c] transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
                  >

                    {isAnalyzing ? (
                      <>
                        <span className="h-4 w-4 animate-spin rounded-full border-2 border-[#06100c]/30 border-t-[#06100c]" />

                        Analyzing image...
                      </>
                    ) : (
                      <>
                        <Zap size={17} />

                        Analyze with AgroSmart AI
                      </>
                    )}

                  </button>


                  {/* Try Another */}

                  <button
                    onClick={handleTryAnother}
                    className="flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-5 py-3.5 text-sm font-semibold text-slate-300 transition hover:border-emerald-500/30 hover:bg-emerald-500/5 hover:text-emerald-300 sm:w-auto"
                  >

                    <Upload size={17} />

                    Try Another

                  </button>

                </div>

              </div>

            </div>


            {/* ==================================================
                SYSTEM OVERVIEW
            ================================================== */}

            <div className="rounded-3xl border border-white/5 bg-white/[0.02] p-6 sm:p-8">

              <div className="mb-6">

                <div className="text-xs font-semibold uppercase tracking-widest text-slate-500">
                  Connected System
                </div>

                <h2 className="mt-2 text-xl font-bold">
                  Precision Spray Network
                </h2>

              </div>


              <div className="space-y-3">

                {[
                  {
                    icon: Cpu,
                    label: "ESP32 Controller",
                    value: DEVICE_ID,
                    status: "ONLINE",
                  },
                  {
                    icon: Wifi,
                    label: "MQTT Broker",
                    value: "localhost:1883",
                    status: "CONNECTED",
                  },
                  {
                    icon: Droplets,
                    label: "Flow Sensor",
                    value: "Ready",
                    status: "READY",
                  },
                  {
                    icon: Syringe,
                    label: "Spray System",
                    value: "Simulation",
                    status: "SAFE",
                  },
                ].map((item) => {

                  const Icon = item.icon;

                  return (
                    <div
                      key={item.label}
                      className="flex items-center justify-between rounded-2xl border border-white/5 bg-black/10 p-4"
                    >

                      <div className="flex items-center gap-3">

                        <div className="rounded-xl bg-white/5 p-2.5">

                          <Icon
                            size={17}
                            className="text-slate-400"
                          />

                        </div>

                        <div>

                          <div className="text-sm font-medium">
                            {item.label}
                          </div>

                          <div className="mt-0.5 text-xs text-slate-600">
                            {item.value}
                          </div>

                        </div>

                      </div>

                      <span className="text-[10px] font-semibold tracking-wide text-emerald-400">
                        {item.status}
                      </span>

                    </div>
                  );
                })}

              </div>


              <div className="mt-5 rounded-2xl border border-amber-500/10 bg-amber-500/5 p-4">

                <div className="flex gap-3">

                  <ShieldCheck
                    size={18}
                    className="mt-0.5 shrink-0 text-amber-400"
                  />

                  <div>

                    <div className="text-xs font-semibold text-amber-300">
                      Simulation Mode
                    </div>

                    <p className="mt-1 text-xs leading-relaxed text-slate-500">
                      Physical spray execution is
                      currently disabled. Commands are
                      intended for the simulated ESP32.
                    </p>

                  </div>

                </div>

              </div>

            </div>

          </section>


          {/* ==================================================
              ANALYSIS RESULTS
          ================================================== */}

          {analysisResult && (

            <motion.section
              initial={{
                opacity: 0,
                y: 15,
              }}
              animate={{
                opacity: 1,
                y: 0,
              }}
              className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]"
            >

              {/* ==================================================
                  DISEASE
              ================================================== */}

              <div className="rounded-3xl border border-white/5 bg-white/[0.02] p-6 sm:p-8">

                <div className="mb-6 flex items-center justify-between">

                  <div>

                    <div className="text-xs font-semibold uppercase tracking-widest text-slate-500">
                      AI Diagnosis
                    </div>

                    <h2 className="mt-2 text-2xl font-bold">
                      {formatDiseaseName(disease?.name)}
                    </h2>

                  </div>

                  <div className="rounded-xl bg-emerald-500/10 p-3">

                    <Leaf
                      size={22}
                      className="text-emerald-400"
                    />

                  </div>

                </div>


                <div className="grid gap-3 sm:grid-cols-3">

                  <div className="rounded-2xl border border-white/5 bg-black/10 p-4">

                    <div className="text-xs text-slate-500">
                      Crop
                    </div>

                    <div className="mt-2 font-semibold">
                      {disease?.crop || "--"}
                    </div>

                  </div>


                  <div className="rounded-2xl border border-white/5 bg-black/10 p-4">

                    <div className="text-xs text-slate-500">
                      Confidence
                    </div>

                    <div className="mt-2 font-semibold text-emerald-400">
                      {formatConfidence(
                        disease?.confidence
                      )}
                    </div>

                  </div>


                  <div className="rounded-2xl border border-white/5 bg-black/10 p-4">

                    <div className="text-xs text-slate-500">
                      Disease Type
                    </div>

                    <div className="mt-2 truncate font-semibold">
                      {disease?.type || "--"}
                    </div>

                  </div>

                </div>


                <div className="mt-4 flex items-center gap-2 rounded-xl border border-emerald-500/10 bg-emerald-500/5 px-4 py-3">

                  <CheckCircle2
                    size={17}
                    className="text-emerald-400"
                  />

                  <span className="text-xs text-slate-400">

                    {disease?.status ===
                    "PREDICTION_ACCEPTED"
                      ? "Disease prediction accepted above the 70% confidence threshold."
                      : disease?.status ||
                        "Prediction processed."}

                  </span>

                </div>

              </div>


              {/* ==================================================
                  SEVERITY
              ================================================== */}

              <div className="rounded-3xl border border-white/5 bg-white/[0.02] p-6 sm:p-8">

                <div className="mb-3">

                  <div className="text-xs font-semibold uppercase tracking-widest text-slate-500">
                    Disease Severity
                  </div>

                </div>


                <div className="flex items-center justify-center">

                  <SeverityGauge
                    severity={
                      severity?.percentage
                    }
                    level={
                      severity?.level
                    }
                  />

                </div>


                <div className="grid grid-cols-2 gap-3">

                  <div className="rounded-2xl border border-white/5 bg-black/10 p-4">

                    <div className="text-xs text-slate-500">
                      Leaf Area
                    </div>

                    <div className="mt-2 text-sm font-semibold">
                      {severity?.leaf_pixels?.toLocaleString() ||
                        "--"}
                    </div>

                  </div>


                  <div className="rounded-2xl border border-white/5 bg-black/10 p-4">

                    <div className="text-xs text-slate-500">
                      Diseased Area
                    </div>

                    <div className="mt-2 text-sm font-semibold">
                      {severity?.disease_pixels?.toLocaleString() ||
                        "--"}
                    </div>

                  </div>

                </div>

              </div>


              {/* ==================================================
                  TREATMENT
              ================================================== */}

              <div className="rounded-3xl border border-white/5 bg-white/[0.02] p-6 sm:p-8">

                <div className="mb-6 flex items-center justify-between">

                  <div>

                    <div className="text-xs font-semibold uppercase tracking-widest text-slate-500">
                      Treatment Recommendation
                    </div>

                    <h2 className="mt-2 text-xl font-bold">
                      Treatment Plan
                    </h2>

                  </div>

                  <Package
                    size={22}
                    className="text-emerald-400"
                  />

                </div>


                {treatment?.available ? (

                  <div className="space-y-3">

                    <div className="rounded-2xl border border-emerald-500/10 bg-emerald-500/5 p-5">

                      <div className="text-xs text-slate-500">
                        Product
                      </div>

                      <div className="mt-2 font-semibold text-emerald-300">
                        {treatment.product}
                      </div>

                    </div>


                    <div className="grid grid-cols-2 gap-3">

                      <div className="rounded-2xl border border-white/5 bg-black/10 p-4">

                        <div className="text-xs text-slate-500">
                          Bottle
                        </div>

                        <div className="mt-2 text-xl font-bold">
                          #{treatment.bottle}
                        </div>

                      </div>


                      <div className="rounded-2xl border border-white/5 bg-black/10 p-4">

                        <div className="text-xs text-slate-500">
                          Controller Volume
                        </div>

                        <div className="mt-2 text-xl font-bold">

                          {treatment.amount_ml}

                          <span className="text-sm text-slate-500">
                            {" "}mL
                          </span>

                        </div>

                      </div>

                    </div>


                    <div className="rounded-xl border border-amber-500/10 bg-amber-500/5 p-3 text-xs leading-relaxed text-slate-500">

                      Prototype controller volume only.
                      This is not an agronomic pesticide
                      label dosage.

                    </div>

                  </div>

                ) : (

                  <div className="rounded-2xl border border-red-500/10 bg-red-500/5 p-5">

                    <div className="flex gap-3">

                      <AlertCircle
                        size={19}
                        className="shrink-0 text-red-400"
                      />

                      <div>

                        <div className="font-semibold text-red-300">
                          Treatment not available
                        </div>

                        <div className="mt-1 text-xs text-slate-500">
                          {treatment?.reason ||
                            "No validated treatment is configured for this condition."}
                        </div>

                      </div>

                    </div>

                  </div>

                )}

              </div>


              {/* ==================================================
                  SPRAY CONTROL
              ================================================== */}

              <div className="rounded-3xl border border-white/5 bg-white/[0.02] p-6 sm:p-8">

                <div className="mb-6">

                  <div className="text-xs font-semibold uppercase tracking-widest text-slate-500">
                    Precision Application
                  </div>

                  <h2 className="mt-2 text-xl font-bold">
                    Spray Control
                  </h2>

                </div>


                <div className="space-y-4">

                  <div className="flex items-center justify-between rounded-2xl border border-white/5 bg-black/10 p-4">

                    <div className="flex items-center gap-3">

                      <Gauge
                        size={18}
                        className="text-slate-400"
                      />

                      <span className="text-sm text-slate-400">
                        Severity
                      </span>

                    </div>

                    <span
                      className={`text-sm font-bold ${getSeverityClass(
                        severity?.level
                      )}`}
                    >
                      {severity?.level || "--"}
                    </span>

                  </div>


                  <div className="flex items-center justify-between rounded-2xl border border-white/5 bg-black/10 p-4">

                    <div className="flex items-center gap-3">

                      <Cpu
                        size={18}
                        className="text-slate-400"
                      />

                      <span className="text-sm text-slate-400">
                        Target Device
                      </span>

                    </div>

                    <span className="text-sm font-semibold">
                      {DEVICE_ID}
                    </span>

                  </div>


                  <div
                    className={`
                      rounded-2xl border p-4
                      ${
                        spray?.allowed
                          ? "border-emerald-500/10 bg-emerald-500/5"
                          : "border-red-500/10 bg-red-500/5"
                      }
                    `}
                  >

                    <div className="flex gap-3">

                      {spray?.allowed ? (

                        <CheckCircle2
                          size={19}
                          className="shrink-0 text-emerald-400"
                        />

                      ) : (

                        <AlertCircle
                          size={19}
                          className="shrink-0 text-red-400"
                        />

                      )}


                      <div>

                        <div
                          className={`text-sm font-semibold ${
                            spray?.allowed
                              ? "text-emerald-300"
                              : "text-red-300"
                          }`}
                        >

                          {spray?.allowed
                            ? "Spray authorized for approval"
                            : "Spray unavailable"}

                        </div>


                        <div className="mt-1 text-xs leading-relaxed text-slate-500">

                          {spray?.reason ||
                            "Treatment validation required."}

                        </div>

                      </div>

                    </div>

                  </div>


                  <button
                    onClick={
                      handleApproveSpray
                    }
                    disabled={
                      !spray?.allowed ||
                      isSpraying
                    }
                    className="flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-500 px-5 py-3.5 text-sm font-bold text-[#06100c] transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-30"
                  >

                    {isSpraying ? (

                      <>
                        <span className="h-4 w-4 animate-spin rounded-full border-2 border-[#06100c]/30 border-t-[#06100c]" />

                        Sending Command...
                      </>

                    ) : (

                      <>
                        <Play
                          size={17}
                          fill="currentColor"
                        />

                        Approve & Spray
                      </>

                    )}

                  </button>

                </div>

              </div>

            </motion.section>
          )}


          {/* ==================================================
              SPRAY RESULT
          ================================================== */}

          {sprayResult && (

            <motion.section
              initial={{
                opacity: 0,
                y: 10,
              }}
              animate={{
                opacity: 1,
                y: 0,
              }}
              className="mt-6 rounded-3xl border border-emerald-500/10 bg-emerald-500/5 p-6"
            >

              <div className="flex items-start gap-4">

                <div className="rounded-xl bg-emerald-500/10 p-3">

                  <CheckCircle2
                    size={23}
                    className="text-emerald-400"
                  />

                </div>


                <div className="flex-1">

                  <div className="text-lg font-bold text-emerald-300">
                    {sprayResult.status ||
                      "Spray command processed"}
                  </div>

                  <p className="mt-1 text-sm text-slate-400">
                    {sprayResult.message ||
                      "The spray request was processed by AgroSmart."}
                  </p>


                  {sprayResult.command && (

                    <div className="mt-4 rounded-2xl border border-white/5 bg-black/20 p-4">

                      <div className="grid gap-3 sm:grid-cols-3">

                        <div>

                          <div className="text-[11px] text-slate-600">
                            Command ID
                          </div>

                          <div className="mt-1 break-all text-xs font-medium">
                            {
                              sprayResult
                                .command
                                .command_id
                            }
                          </div>

                        </div>


                        <div>

                          <div className="text-[11px] text-slate-600">
                            Device
                          </div>

                          <div className="mt-1 text-xs font-medium">
                            {DEVICE_ID}
                          </div>

                        </div>


                        <div>

                          <div className="text-[11px] text-slate-600">
                            Status
                          </div>

                          <div className="mt-1 text-xs font-medium text-emerald-400">
                            COMMAND SENT
                          </div>

                        </div>

                      </div>

                    </div>
                  )}

                </div>

              </div>

            </motion.section>
          )}


          {/* ==================================================
              EMPTY STATE
          ================================================== */}

          {!analysisResult &&
            !selectedFile && (

              <section className="mt-6 grid gap-4 md:grid-cols-3">

                {[
                  {
                    icon: Activity,
                    title: "AI Disease Detection",
                    text: "Identify crop diseases using the trained disease classifier.",
                  },
                  {
                    icon: BarChart3,
                    title: "Severity Estimation",
                    text: "Estimate diseased leaf area using the human-GT-trained segmentation model.",
                  },
                  {
                    icon: Droplets,
                    title: "Precision Treatment",
                    text: "Map configured treatment rules to the appropriate spray controller.",
                  },
                ].map((item) => {

                  const Icon = item.icon;

                  return (
                    <div
                      key={item.title}
                      className="rounded-2xl border border-white/5 bg-white/[0.02] p-5"
                    >

                      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10">

                        <Icon
                          size={19}
                          className="text-emerald-400"
                        />

                      </div>

                      <h3 className="font-semibold">
                        {item.title}
                      </h3>

                      <p className="mt-2 text-xs leading-relaxed text-slate-500">
                        {item.text}
                      </p>

                    </div>
                  );
                })}

              </section>
            )}


          {/* ==================================================
              FOOTER
          ================================================== */}

          <footer className="mt-10 flex flex-col gap-2 border-t border-white/5 py-6 text-[11px] text-slate-600 sm:flex-row sm:items-center sm:justify-between">

            <div>
              AgroSmart AI • Precision Agriculture
            </div>

            <div className="flex items-center gap-2">

              <ShieldCheck size={13} />

              AI-assisted decision support

            </div>

          </footer>

        </div>

      </main>

    </div>
  );
}