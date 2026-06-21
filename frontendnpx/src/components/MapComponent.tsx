"use client"; // Tells Next.js that this runs only in the browser

import { MapContainer, TileLayer, Marker, Popup, useMapEvents, CircleMarker, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useState, useEffect, useMemo } from "react";

// УМНАЯ ПЕРЕМЕННАЯ: Берет адрес из настроек Докера или использует сервер по умолчанию
const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://windturbine.ink";

// --- INTERFACES & TYPES ---
interface Station {
  STN: number;
  station_name: string;
  lat: number;
  lon: number;
  avg_wind_speed: number;
}

interface GridPoint {
  cell_lat: number;
  cell_lon: number;
  ml_suitability_score?: number;
  suitability_color?: string;
  is_natura2000?: number;
}

interface WindExplorerPoint {
  cell_lat: number;
  cell_lon: number;
  wind_speed: number;
}

interface EvaluationResult {
  score: number;
  ml_label?: string;
  wind_speed_ms?: number;
  environment?: {
    population_density?: number;
    is_natura2000?: number;
  };
  warnings?: string[];
  error?: string;
}

interface WindPointDetails {
  annual_wind_speed?: number;
  monthly_wind_speed?: Record<string, number>;
  country?: string;
  error?: string;
}

interface CandidatePoint {
  rank: number;
  cell_lat: number;
  cell_lon: number;
  ml_suitability_score: number;
  kmeans_label: string;
  kmeans_rank: number;
  rf_empirical_probability: number;
  rf_empirical_label: string;
  wind_speed: number;
  population_density: number;
  dist_to_nearest_turbine_m: number;
  is_natura2000: number;
  suitability_color: string;
  very_suitable: boolean;
  short_reason: string;
}

// --- LEAFLET ICON FIX FOR NEXT.JS ---
const DefaultIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

// Smaller version for weather stations
const SmallIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [13, 20], 
  iconAnchor: [9, 30], 
});

L.Marker.prototype.options.icon = DefaultIcon;
// ----------------------------------------

interface MapClickProps {
  mode: "suitability" | "windExplorer";
  explorerSubMode: "netherlands" | "country";
  selectedCountry: string;
  setClickPos: (pos: { lat: number; lng: number } | null) => void;
  setEvaluation: (evalResult: EvaluationResult | null) => void;
  setWindPointDetails: (details: WindPointDetails | null) => void;
  setMapBbox: (bbox: { min_lat: number; max_lat: number; min_lon: number; max_lon: number } | null) => void;
}

// Map click handler component
function MapClickHandler({
  mode,
  explorerSubMode,
  selectedCountry,
  setClickPos,
  setEvaluation,
  setWindPointDetails,
  setMapBbox
}: MapClickProps) {
  const map = useMapEvents({
    click: async (e) => {
      const { lat, lng } = e.latlng;
      setClickPos({ lat, lng });
      
      // ОЧИЩАЕМ СТАРЫЕ ДАННЫЕ ПРИ КЛИКЕ, ЧТОБЫ ПОКАЗАТЬ ЗАГРУЗКУ И ИЗБЕЖАТЬ МИГАНИЯ
      setEvaluation(null);
      setWindPointDetails(null);

      if (mode === "suitability") {
        try {
          const res = await fetch(`${API_URL}/api/v1/turbines/evaluate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lat: lat, lon: lng, turbine_model: "Vestas_V164" }),
          });
          const data = await res.json();
          setEvaluation(data);
        } catch (err) {
          console.error("Evaluation error:", err);
          setEvaluation({ score: 0, error: "Failed to fetch" }); // Fallback error state
        }
      } else {
        try {
          const queryUrl = `${API_URL}/api/v1/wind-explorer/point?lat=${lat}&lon=${lng}&mode=${explorerSubMode}${
            explorerSubMode === "country" ? `&country=${selectedCountry}` : ""
          }`;
          const res = await fetch(queryUrl);
          const data = await res.json();
          setWindPointDetails(data);
        } catch (err) {
          console.error("Wind point lookup error:", err);
          setWindPointDetails({ error: "Failed to fetch" }); // Fallback error state
        }
      }
    },
    moveend: () => {
      const bounds = map.getBounds();
      setMapBbox({
        min_lat: bounds.getSouth(),
        max_lat: bounds.getNorth(),
        min_lon: bounds.getWest(),
        max_lon: bounds.getEast()
      });
    },
    zoomend: () => {
      const bounds = map.getBounds();
      setMapBbox({
        min_lat: bounds.getSouth(),
        max_lat: bounds.getNorth(),
        min_lon: bounds.getWest(),
        max_lon: bounds.getEast()
      });
    }
  });

  useEffect(() => {
    if (map) {
      const bounds = map.getBounds();
      setMapBbox({
        min_lat: bounds.getSouth(),
        max_lat: bounds.getNorth(),
        min_lon: bounds.getWest(),
        max_lon: bounds.getEast()
      });
    }
  }, [map, setMapBbox]);

  return null;
}

interface MapControllerProps {
  selectedCandidate: { lat: number; lon: number } | null;
}

function MapController({ selectedCandidate }: MapControllerProps) {
  const map = useMap();
  
  useEffect(() => {
    if (selectedCandidate) {
      map.flyTo([selectedCandidate.lat, selectedCandidate.lon], 12);
    }
  }, [selectedCandidate, map]);
  
  return null;
}

export default function MapComponent() {
  const [stations, setStations] = useState<Station[]>([]);
  const [gridData, setGridData] = useState<GridPoint[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);
  const [clickPos, setClickPos] = useState<{ lat: number; lng: number } | null>(null);
  const [isLoadingGrid, setIsLoadingGrid] = useState(true);

  // --- WIND EXPLORER STATES ---
  const [mode, setMode] = useState<"suitability" | "windExplorer">("suitability");
  const [explorerSubMode, setExplorerSubMode] = useState<"netherlands" | "country">("netherlands");
  const [selectedMonth, setSelectedMonth] = useState<string>("annual");
  const [selectedCountry, setSelectedCountry] = useState<string>("Netherlands");
  const [windExplorerGrid, setWindExplorerGrid] = useState<WindExplorerPoint[]>([]);
  const [isLoadingWindGrid, setIsLoadingWindGrid] = useState(false);
  const [windPointDetails, setWindPointDetails] = useState<WindPointDetails | null>(null);

  // --- TOP CANDIDATES / TIERS STATES ---
  const [tierCounts, setTierCounts] = useState<{ total_cells: number; tiers: { tier: string; count: number }[] } | null>(null);
  const [topCandidates, setTopCandidates] = useState<CandidatePoint[]>([]);
  const [candidateScope, setCandidateScope] = useState<"global" | "mapView">("global");
  const [candidateLimit, setCandidateLimit] = useState<number>(20);
  const [candidateMinScore, setCandidateMinScore] = useState<number>(60);
  const [candidateDiverse, setCandidateDiverse] = useState<boolean>(true);
  const [mapBbox, setMapBbox] = useState<{ min_lat: number; max_lat: number; min_lon: number; max_lon: number } | null>(null);
  const [isLoadingCandidates, setIsLoadingCandidates] = useState(false);
  const [candidatesError, setCandidatesError] = useState<string | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<{ lat: number; lon: number } | null>(null);
  const [isLegendOpen, setIsLegendOpen] = useState(false);

  // 1. Load the list of weather stations
  useEffect(() => {
    fetch(`${API_URL}/api/v1/stations`)
      .then((res) => res.json())
      .then((data) => setStations(data))
      .catch((err) => console.error("Error loading stations:", err));
  }, []);

  // 2. Load the ENTIRE ML grid
  useEffect(() => {
    fetch(`${API_URL}/api/v1/wind/all`)
      .then((res) => res.json())
      .then((data) => {
        setGridData(data);
        setIsLoadingGrid(false);
      })
      .catch((err) => {
        console.error("Error loading grid:", err);
        setIsLoadingGrid(false);
      });
  }, []);

  // 3. Load the Wind Explorer data dynamically
  useEffect(() => {
    if (mode !== "windExplorer") return;
    
    setIsLoadingWindGrid(true);
    
    let url = "";
    if (explorerSubMode === "netherlands") {
      url = `${API_URL}/api/v1/wind-explorer/netherlands?month=${selectedMonth}`;
    } else {
      url = `${API_URL}/api/v1/wind-explorer/country?country=${selectedCountry}`;
    }

    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        setWindExplorerGrid(data);
        setIsLoadingWindGrid(false);
      })
      .catch((err) => {
        console.error("Error loading wind explorer grid:", err);
        setIsLoadingWindGrid(false);
      });

  }, [mode, explorerSubMode, selectedMonth, selectedCountry]);

  // 3.5 Fetch Tiers Counts
  useEffect(() => {
    if (mode !== "suitability") return;
    
    fetch(`${API_URL}/api/v1/suitability/tiers`)
      .then((res) => res.json())
      .then((data) => setTierCounts(data))
      .catch((err) => console.error("Error loading tiers counts:", err));
  }, [mode]);

  // 3.6 Fetch Top Candidate Locations (Global Scope)
  useEffect(() => {
    if (mode !== "suitability" || candidateScope !== "global") return;
    
    setIsLoadingCandidates(true);
    setCandidatesError(null);
    
    const url = `${API_URL}/api/v1/suitability/top?limit=${candidateLimit}&min_score=${candidateMinScore}&diverse=${candidateDiverse}`;
    
    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch top candidates");
        return res.json();
      })
      .then((data) => {
        setTopCandidates(data.candidates || []);
        setIsLoadingCandidates(false);
      })
      .catch((err) => {
        console.error("Error loading top candidates:", err);
        setCandidatesError("Candidates lookup unavailable");
        setIsLoadingCandidates(false);
      });
  }, [mode, candidateScope, candidateLimit, candidateMinScore, candidateDiverse]);

  // 3.7 Fetch Top Candidate Locations (Map View Scope)
  useEffect(() => {
    if (mode !== "suitability" || candidateScope !== "mapView" || !mapBbox) return;
    
    setIsLoadingCandidates(true);
    setCandidatesError(null);
    
    const url = `${API_URL}/api/v1/suitability/top?limit=${candidateLimit}&min_score=${candidateMinScore}&diverse=${candidateDiverse}&min_lat=${mapBbox.min_lat}&max_lat=${mapBbox.max_lat}&min_lon=${mapBbox.min_lon}&max_lon=${mapBbox.max_lon}`;
    
    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch top candidates");
        return res.json();
      })
      .then((data) => {
        setTopCandidates(data.candidates || []);
        setIsLoadingCandidates(false);
      })
      .catch((err) => {
        console.error("Error loading top candidates:", err);
        setCandidatesError("Candidates lookup unavailable");
        setIsLoadingCandidates(false);
      });
  }, [mode, candidateScope, candidateLimit, candidateMinScore, candidateDiverse, mapBbox]);

  const getScoreColor = (score: number, isNatura: number) => {
    if (isNatura === 1) return "#64748b"; 
    if (score >= 80) return "#15803d"; 
    if (score >= 70) return "#22c55e"; 
    if (score >= 60) return "#84cc16"; 
    if (score >= 50) return "#14b8a6"; 
    if (score >= 40) return "#0ea5e9"; 
    if (score >= 25) return "#2563eb"; 
    return "#1e3a8a";                 
  };

  const getWindSpeedColor = (speed: number) => {
    if (speed >= 9.0) return "#15803d"; 
    if (speed >= 8.0) return "#22c55e"; 
    if (speed >= 7.0) return "#84cc16"; 
    if (speed >= 6.0) return "#14b8a6"; 
    if (speed >= 5.0) return "#0ea5e9"; 
    if (speed >= 4.0) return "#2563eb"; 
    return "#1e3a8a";                 
  };

  // --- КЭШИРОВАНИЕ ДАННЫХ ДЛЯ КАРТЫ (useMemo) ---
  // Избавляет от тормозов при кликах
  const suitabilityMarkers = useMemo(() => {
    return gridData.map((point, idx) => {
      const score = point.ml_suitability_score || 0;
      const isHighlySuitable = score >= 70;
      return (
        <CircleMarker
          key={`grid-${idx}`}
          center={[point.cell_lat, point.cell_lon]}
          radius={isHighlySuitable ? 3.5 : 2}
          pathOptions={{
            fillColor: point.suitability_color || getScoreColor(score, point.is_natura2000 || 0),
            stroke: isHighlySuitable,
            color: "#ffffff",
            weight: 1.5,
            fillOpacity: 0.85
          }}
        />
      );
    });
  }, [gridData]);

  const windExplorerMarkers = useMemo(() => {
    return windExplorerGrid.map((point, idx) => {
      const speed = point.wind_speed;
      const isHighWind = speed >= 8.0;
      return (
        <CircleMarker
          key={`wind-grid-${idx}`}
          center={[point.cell_lat, point.cell_lon]}
          radius={isHighWind ? 3 : 1.8}
          pathOptions={{
            fillColor: getWindSpeedColor(speed),
            stroke: isHighWind,
            color: "#ffffff",
            weight: 1,
            fillOpacity: 0.8
          }}
        />
      );
    });
  }, [windExplorerGrid]);
  // ----------------------------------------------

  return (
    <div className="flex flex-col md:flex-row h-screen w-screen overflow-hidden bg-gray-50">
      
      {/* LEFT PANEL (SIDEBAR) */}
      <div className="w-full h-[50vh] md:h-full md:w-1/3 lg:w-1/4 p-6 bg-white shadow-xl z-10 flex flex-col overflow-y-auto border-r border-gray-100 select-none order-last md:order-first">
        
        {/* App Title & Subtitle */}
        <div className="mb-6">
          <h1 className="text-2xl font-black text-blue-900 tracking-tight mb-1">
            Wind Turbine Analyzer
          </h1>
          <p className="text-gray-500 text-xs font-semibold leading-relaxed">
            Interactive map for exploring wind-turbine suitability and wind patterns.
          </p>
        </div>

        {/* Section: View Mode */}
        <div className="mb-6 bg-gray-50 border border-gray-200/60 rounded-xl p-4">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2.5">View Mode</h3>
          <div className="flex flex-col gap-2">
            <button
              onClick={() => {
                setMode("suitability");
                setClickPos(null);
              }}
              className={`w-full px-4 py-3 rounded-lg text-xs font-bold transition-all duration-200 text-left flex items-center justify-between ${
                mode === "suitability"
                  ? "bg-blue-600 text-white shadow-md shadow-blue-100"
                  : "bg-white border border-gray-200 text-gray-700 hover:bg-gray-100"
              }`}
            >
              <span>Suitability View</span>
              {mode === "suitability" && <span className="w-1.5 h-1.5 rounded-full bg-white" />}
            </button>
            <button
              onClick={() => {
                setMode("windExplorer");
                setClickPos(null);
              }}
              className={`w-full px-4 py-3 rounded-lg text-xs font-bold transition-all duration-200 text-left flex items-center justify-between ${
                mode === "windExplorer"
                  ? "bg-blue-600 text-white shadow-md shadow-blue-100"
                  : "bg-white border border-gray-200 text-gray-700 hover:bg-gray-100"
              }`}
            >
              <span>Wind Explorer</span>
              {mode === "windExplorer" && <span className="w-1.5 h-1.5 rounded-full bg-white" />}
            </button>
          </div>
        </div>

        {/* Section: Top Candidate Locations (Suitability View only) */}
        {mode === "suitability" && (
          <div className="mb-6 bg-blue-50/20 border border-blue-100 rounded-xl p-4 flex flex-col space-y-4 text-gray-800">
            <div>
              <h3 className="text-xs font-black text-blue-900 uppercase tracking-wider mb-2">Top Candidate Locations</h3>
              
              {/* Tiers Summary UI */}
              {tierCounts && (
                <div className="bg-white p-3 rounded-lg border border-gray-100/60 shadow-sm text-[11px] mb-3 space-y-1.5 font-bold">
                  <div className="text-xs text-blue-900 border-b pb-1 font-black mb-1">Netherlands Suitability Tiers</div>
                  <div className="flex justify-between items-center text-green-700">
                    <span>🟢 Very Suitable (≥70):</span>
                    <span>{tierCounts.tiers.find(t => t.tier === "Very Suitable")?.count || 0}</span>
                  </div>
                  <div className="flex justify-between items-center text-lime-700">
                    <span>🟢 Suitable (60-69):</span>
                    <span>{tierCounts.tiers.find(t => t.tier === "Suitable")?.count || 0}</span>
                  </div>
                  <div className="flex justify-between items-center text-blue-700">
                    <span>🔵 Medium (40-59):</span>
                    <span>{tierCounts.tiers.find(t => t.tier === "Medium")?.count || 0}</span>
                  </div>
                  <div className="flex justify-between items-center text-indigo-700">
                    <span>🔵 Low (&lt;40):</span>
                    <span>{tierCounts.tiers.find(t => t.tier === "Low")?.count || 0}</span>
                  </div>
                  <div className="flex justify-between items-center text-slate-500">
                    <span>⚫ Excluded (Natura 2000):</span>
                    <span>{tierCounts.tiers.find(t => t.tier === "Excluded")?.count || 0}</span>
                  </div>
                </div>
              )}

              {/* Sidebar controls for Candidate Filtering */}
              <div className="grid grid-cols-2 gap-2 mb-3">
                <div>
                  <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Scope</label>
                  <select
                    value={candidateScope}
                    onChange={(e) => setCandidateScope(e.target.value as "global" | "mapView")}
                    className="w-full text-xs border border-gray-200 rounded-md p-1.5 bg-white text-gray-800 font-bold focus:outline-none focus:ring-1 focus:ring-blue-500"
                  >
                    <option value="global">Global (NL)</option>
                    <option value="mapView">Current Map View</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Limit</label>
                  <select
                    value={candidateLimit}
                    onChange={(e) => setCandidateLimit(Number(e.target.value))}
                    className="w-full text-xs border border-gray-200 rounded-md p-1.5 bg-white text-gray-800 font-bold focus:outline-none focus:ring-1 focus:ring-blue-500"
                  >
                    <option value={10}>Top 10</option>
                    <option value={20}>Top 20</option>
                    <option value={50}>Top 50</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 mb-3">
                <div>
                  <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Min Score</label>
                  <select
                    value={candidateMinScore}
                    onChange={(e) => setCandidateMinScore(Number(e.target.value))}
                    className="w-full text-xs border border-gray-200 rounded-md p-1.5 bg-white text-gray-800 font-bold focus:outline-none focus:ring-1 focus:ring-blue-500"
                  >
                    <option value={60}>60+ (Suitable)</option>
                    <option value={70}>70+ (Very Suitable)</option>
                    <option value={80}>80+ (Excellent)</option>
                  </select>
                </div>
                <div className="flex flex-col justify-end">
                  <button
                    onClick={() => setCandidateDiverse(prev => !prev)}
                    className={`w-full py-1.5 rounded-md text-[11px] font-bold border transition ${
                      candidateDiverse
                        ? "bg-blue-50 border-blue-200 text-blue-700 shadow-sm"
                        : "bg-white border-gray-200 text-gray-600"
                    }`}
                  >
                    {candidateDiverse ? "✓ Diverse Mode" : "Cluster Mode"}
                  </button>
                </div>
              </div>

              <p className="text-[10px] text-gray-400 leading-tight italic bg-yellow-50/50 border border-yellow-100 p-2 rounded-md mb-3 font-semibold text-center">
                “These are screening candidates, not legally approved turbine sites.”
              </p>

              {/* Render Candidates List */}
              {isLoadingCandidates ? (
                <div className="text-center text-xs text-blue-600 font-bold py-4 animate-pulse">⏳ Refreshing candidates list...</div>
              ) : candidatesError ? (
                <div className="text-center text-xs text-red-500 font-bold py-2 bg-red-50 border border-red-100 rounded">{candidatesError}</div>
              ) : topCandidates.length === 0 ? (
                <div className="text-center text-xs text-gray-400 font-bold py-4">No candidates meet the score criteria in this area.</div>
              ) : (
                <div className="space-y-2 max-h-[220px] overflow-y-auto pr-1">
                  {topCandidates.map((cand) => (
                    <div
                      key={`cand-${cand.rank}`}
                      onClick={() => {
                        setSelectedCandidate({ lat: cand.cell_lat, lon: cand.cell_lon });
                        setClickPos({ lat: cand.cell_lat, lng: cand.cell_lon });
                        setEvaluation(null); // Очищаем данные перед запросом нового кандидата
                        fetch(`${API_URL}/api/v1/turbines/evaluate`, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ lat: cand.cell_lat, lon: cand.cell_lon, turbine_model: "Vestas_V164" }),
                        })
                          .then((res) => res.json())
                          .then((data) => setEvaluation(data))
                          .catch((err) => console.error("Candidate evaluate failed:", err));
                      }}
                      className="bg-white hover:bg-blue-50/50 border border-gray-150 p-2.5 rounded-lg shadow-sm cursor-pointer transition flex items-start gap-2 select-none"
                    >
                      <div className="bg-blue-600 text-white rounded-md text-[10px] font-black w-5 h-5 flex items-center justify-center shrink-0">
                        {cand.rank}
                      </div>
                      <div className="flex-1 space-y-1 text-[11px] font-bold text-gray-700">
                        <div className="flex justify-between items-center text-xs text-blue-900 font-black">
                          <span>Point {cand.cell_lat.toFixed(3)}, {cand.cell_lon.toFixed(3)}</span>
                          <span className="bg-green-100 text-green-800 text-[10px] font-extrabold px-1.5 py-0.5 rounded">
                            {cand.ml_suitability_score.toFixed(1)}%
                          </span>
                        </div>
                        <div className="text-[10px] text-gray-500 font-semibold leading-relaxed">
                          <b>K-Means:</b> {cand.kmeans_label} <br />
                          <b>Wind Speed:</b> {cand.wind_speed.toFixed(2)} m/s <br />
                          {cand.rf_empirical_probability > 0 && (
                            <><b>RF Siting Probability:</b> {(cand.rf_empirical_probability * 105).toFixed(0)}% <br /></>
                          )}
                          <span className="text-blue-700 italic block mt-1">“{cand.short_reason}”</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

            </div>
          </div>
        )}

        {/* Section: Controls */}
        {mode === "windExplorer" && (
          <div className="mb-6 bg-indigo-50/50 border border-indigo-100/50 rounded-xl p-4 space-y-4">
            <div>
              <label className="block text-[11px] font-bold text-gray-500 uppercase tracking-wider mb-2">Explorer Selector</label>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    setExplorerSubMode("netherlands");
                    setClickPos(null);
                  }}
                  className={`flex-1 py-2 rounded-lg text-xs font-bold border transition-all ${
                    explorerSubMode === "netherlands"
                      ? "bg-indigo-600 border-indigo-600 text-white shadow-sm font-extrabold"
                      : "bg-white border-gray-200 text-gray-600 hover:bg-gray-50 font-semibold"
                  }`}
                >
                  NL Monthly
                </button>
                <button
                  onClick={() => {
                    setExplorerSubMode("country");
                    setClickPos(null);
                  }}
                  className={`flex-1 py-2 rounded-lg text-xs font-bold border transition-all ${
                    explorerSubMode === "country"
                      ? "bg-indigo-600 border-indigo-600 text-white shadow-sm font-extrabold"
                      : "bg-white border-gray-200 text-gray-600 hover:bg-gray-50 font-semibold"
                  }`}
                >
                  Country Compare
                </button>
              </div>
            </div>

            {explorerSubMode === "netherlands" ? (
              <div>
                <label className="block text-[11px] font-bold text-gray-500 uppercase tracking-wider mb-1.5">Netherlands Month</label>
                <select
                  value={selectedMonth}
                  onChange={(e) => {
                    setSelectedMonth(e.target.value);
                    setClickPos(null);
                  }}
                  className="w-full text-xs border border-gray-200 rounded-lg p-2.5 bg-white text-gray-800 font-bold focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm"
                >
                  <option value="annual">Annual Average</option>
                  <option value="1">January</option>
                  <option value="2">February</option>
                  <option value="3">March</option>
                  <option value="4">April</option>
                  <option value="5">May</option>
                  <option value="6">June</option>
                  <option value="7">July</option>
                  <option value="8">August</option>
                  <option value="9">September</option>
                  <option value="10">October</option>
                  <option value="11">November</option>
                  <option value="12">December</option>
                </select>
              </div>
            ) : (
              <div>
                <label className="block text-[11px] font-bold text-gray-500 uppercase tracking-wider mb-1.5">Select Country</label>
                <select
                  value={selectedCountry}
                  onChange={(e) => {
                    setSelectedCountry(e.target.value);
                    setClickPos(null);
                  }}
                  className="w-full text-xs border border-gray-200 rounded-lg p-2.5 bg-white text-gray-800 font-bold focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm"
                >
                  <option value="Netherlands">Netherlands (Annual)</option>
                  <option value="Denmark">Denmark</option>
                  <option value="Ireland">Ireland</option>
                  <option value="Scotland">Scotland</option>
                  <option value="France">France</option>
                </select>
              </div>
            )}
          </div>
        )}

        {/* Section: How to read this view (Dynamic Help Card) */}
        <div className="mt-auto bg-blue-50/40 border border-blue-100/50 rounded-xl p-4">
          {mode === "suitability" ? (
            <div>
              <h3 className="text-xs font-bold text-blue-800 uppercase tracking-wider mb-2">Suitability View</h3>
              <p className="text-xs text-blue-900/80 leading-relaxed mb-3 font-medium">
                Use this view to explore candidate locations for wind-turbine development. The map combines wind, population density, Natura 2000 restrictions, turbine proximity, K-Means clusters, and Random Forest comparison outputs. Click a grid point to see the detailed assessment.
              </p>
              <ul className="list-disc pl-4 text-[11px] text-blue-900/90 space-y-1.5 font-semibold">
                <li><span className="font-extrabold text-green-700">Green</span> = stronger suitability</li>
                <li><span className="font-extrabold text-blue-700">Blue</span> = weaker suitability</li>
                <li><span className="font-extrabold text-slate-500">Grey</span> = Natura 2000 / restricted</li>
                <li>Highlighted points = very suitable candidates</li>
                <li>Click a point for score, warnings, and model details</li>
              </ul>
            </div>
          ) : explorerSubMode === "netherlands" ? (
            <div>
              <h3 className="text-xs font-bold text-indigo-800 uppercase tracking-wider mb-2">Netherlands Monthly Wind</h3>
              <p className="text-xs text-indigo-900/80 leading-relaxed mb-3 font-medium">
                Use this view to inspect wind speed only. The grid is coloured by interpolated KNMI wind speed for the selected month or annual average. Suitability, population, Natura, and machine-learning labels are hidden in this view.
              </p>
              <ul className="list-disc pl-4 text-[11px] text-indigo-900/95 space-y-1.5 font-semibold">
                <li><span className="font-extrabold text-blue-700">Blue</span> = lower wind speed</li>
                <li><span className="font-extrabold text-green-700">Green</span> = higher wind speed</li>
                <li>Select a month to compare seasonal patterns</li>
                <li>Click a point for wind-only details</li>
              </ul>
            </div>
          ) : (
            <div>
              <h3 className="text-xs font-bold text-indigo-800 uppercase tracking-wider mb-2">Country Wind Compare</h3>
              <p className="text-xs text-indigo-900/80 leading-relaxed mb-3 font-medium">
                Use this view to compare wind potential between countries. This is a wind-only comparison and does not include population density, Natura 2000, turbine distance, or suitability modelling.
              </p>
              <ul className="list-disc pl-4 text-[11px] text-indigo-900/95 space-y-1.5 font-semibold">
                <li>Select a country to view its wind grid</li>
                <li><span className="font-extrabold text-blue-700">Blue</span> = lower wind speed</li>
                <li><span className="font-extrabold text-green-700">Green</span> = higher wind speed</li>
                <li>Click a point for wind-only details</li>
                <li className="text-red-700">This is not a full turbine suitability comparison</li>
              </ul>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="pt-4 mt-6 border-t border-gray-100 text-[10px] text-gray-400 text-center font-bold uppercase tracking-wider">
          Sprint 2: ML-based Suitability Assessment
        </div>

      </div>

      {/* RIGHT PANEL (MAP) */}
      <div className="w-full h-[50vh] md:h-full md:flex-1 relative z-0 order-first md:order-last">
        
        {/* Grid loading indicator */}
        {(isLoadingGrid || (mode === "windExplorer" && isLoadingWindGrid)) && (
          <div className="absolute top-4 right-4 z-[1000] bg-white px-3 py-2 rounded-lg shadow-md text-xs font-bold animate-pulse flex items-center border border-gray-100 text-gray-700">
            <span className="mr-1.5">⏳</span> Loading Grid...
          </div>
        )}

        {/* Legend overlay */}
        <div className="absolute bottom-4 right-2 md:right-4 z-[1000] flex flex-col items-end">
          
          <button 
            onClick={() => setIsLegendOpen(!isLegendOpen)}
            className="md:hidden mb-2 bg-white px-3 py-2 rounded-lg shadow-md border border-gray-200 text-xs font-bold text-blue-900 flex items-center gap-1.5 active:bg-gray-50"
          >
            {isLegendOpen ? "▼ Hide Legend" : "▲ Show Legend"}
          </button>

          <div className={`${isLegendOpen ? "block" : "hidden"} md:block bg-white p-3.5 rounded-xl shadow-lg border border-gray-100 max-w-xs text-xs text-gray-800 font-medium select-none`}>
            {mode === "suitability" ? (
              <div>
                <h4 className="font-bold text-blue-900 mb-2 border-b border-gray-100 pb-1 text-sm">Suitability Rating</h4>
                <div className="space-y-1.5">
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#15803d]" />Excellent (≥80%)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#22c55e]" />Very Good (70-79%)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#84cc16]" />Good (60-69%)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#14b8a6]" />Average (50-59%)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#0ea5e9]" />Moderate (40-49%)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#2563eb]" />Poor (25-39%)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#1e3a8a]" />Very Poor (&lt;25%)</div>
                  <div className="flex items-center border-t border-gray-100 pt-1.5 mt-1.5"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#64748b]" />Natura 2000 (Protected)</div>
                  <div className="flex items-center mt-1"><span className="w-3.5 h-3.5 border border-white outline outline-1 outline-blue-400 rounded-full mr-2 bg-green-500" />Border = Highly Suitable</div>
                </div>
              </div>
            ) : (
              <div>
                <h4 className="font-bold text-indigo-900 mb-2 border-b border-gray-100 pb-1 text-sm">Wind Speed (m/s)</h4>
                <div className="space-y-1.5">
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#15803d]" />Excellent (≥9.0 m/s)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#22c55e]" />Very Good (8.0 - 8.9 m/s)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#84cc16]" />Good (7.0 - 7.9 m/s)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#14b8a6]" />Moderate (6.0 - 6.9 m/s)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#0ea5e9]" />Moderate-Low (5.0 - 5.9 m/s)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#2563eb]" />Low (4.0 - 4.9 m/s)</div>
                  <div className="flex items-center"><span className="w-3.5 h-3.5 rounded-full mr-2 bg-[#1e3a8a]" />Very Low (&lt;4.0 m/s)</div>
                </div>
                <p className="text-[10px] text-gray-400 mt-2 border-t border-gray-100 pt-1.5 font-semibold">
                  Netherlands & International Wind Explorer
                </p>
              </div>
            )}
          </div>
        </div>

        <MapContainer 
          center={[52.3, 4.9]} 
          zoom={7} 
          className="w-full h-full z-0"
          preferCanvas={true} 
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <MapClickHandler 
            mode={mode}
            explorerSubMode={explorerSubMode}
            selectedCountry={selectedCountry}
            setClickPos={setClickPos}
            setEvaluation={setEvaluation}
            setWindPointDetails={setWindPointDetails}
            setMapBbox={setMapBbox}
          />
          <MapController selectedCandidate={selectedCandidate} />

          {/* 1. ИСПОЛЬЗУЕМ КЭШИРОВАННЫЕ ТОЧКИ ВМЕСТО ВЫЗОВА .map() */}
          {mode === "suitability" && suitabilityMarkers}

          {/* 2. ИСПОЛЬЗУЕМ КЭШИРОВАННЫЕ ТОЧКИ */}
          {mode === "windExplorer" && windExplorerMarkers}

          {/* Draw weather stations */}
          {mode === "suitability" && stations.map((st) => (
            <Marker 
              key={st.STN} 
              position={[st.lat, st.lon]}
              icon={SmallIcon}
            >
              <Popup autoPan={false}>
                <b>{st.station_name}</b> <br />
                ID: {st.STN} <br />
                Wind: {st.avg_wind_speed} m/s
              </Popup>
            </Marker>
          ))}

          {/* Draw Suitability popup at user's click location */}
          {clickPos && mode === "suitability" && (
            <Popup position={[clickPos.lat, clickPos.lng]} autoPan={false}>
              <div className="p-1 min-w-[200px] text-gray-800">
                {!evaluation ? (
                  <div className="flex justify-center items-center h-24 text-sm font-bold text-blue-600 animate-pulse">
                    ⏳ Loading assessment...
                  </div>
                ) : evaluation.error ? (
                  <div className="text-red-500 font-bold">Error loading data</div>
                ) : (
                  <>
                    <h3 className="font-bold text-lg mb-1 border-b pb-1">Location Assessment</h3>
                    <div className="my-2">
                      {evaluation.score >= 60 ? (
                        <span className="bg-green-100 text-green-800 text-xs font-bold px-2.5 py-0.5 rounded">✅ SUITABLE</span>
                      ) : (
                        <span className="bg-red-100 text-red-800 text-xs font-bold px-2.5 py-0.5 rounded">❌ UNSUITABLE</span>
                      )}
                    </div>
                    <div className="space-y-1 text-sm font-semibold">
                      <p><b>Rating (ML):</b> <span className="text-blue-700 font-extrabold">{evaluation.score}%</span></p>
                      <p><b>Cluster:</b> {evaluation.ml_label}</p>
                      <p><b>Wind:</b> {evaluation.wind_speed_ms} m/s</p>
                      <p><b>Population:</b> {evaluation.environment?.population_density} p/km²</p>
                      <p><b>Natura 2000:</b> {evaluation.environment?.is_natura2000 ? "Yes" : "No"}</p>
                    </div>
                    {evaluation.warnings && evaluation.warnings.length > 0 && (
                      <div className="mt-2 text-[11px] text-yellow-800 bg-yellow-50 p-2 rounded border border-yellow-100 font-semibold">
                        <b>Warnings:</b>
                        <ul className="list-disc pl-3 mt-1">
                          {evaluation.warnings.map((w: string, i: number) => (
                            <li key={i}>{w}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </>
                )}
              </div>
            </Popup>
          )}

          {/* Draw Wind Explorer popup at user's click location */}
          {clickPos && mode === "windExplorer" && (
            <Popup position={[clickPos.lat, clickPos.lng]} autoPan={false}>
              <div className="p-1 min-w-[200px] text-gray-800">
                {!windPointDetails ? (
                  <div className="flex justify-center items-center h-24 text-sm font-bold text-indigo-600 animate-pulse">
                    ⏳ Fetching wind data...
                  </div>
                ) : windPointDetails.error ? (
                  <div className="text-red-500 font-bold">Error loading data</div>
                ) : (
                  <>
                    <h3 className="font-bold text-lg mb-1 border-b pb-1 text-indigo-900">
                      Wind Observation
                    </h3>
                    <div className="my-2">
                      <span className="bg-blue-100 text-blue-800 text-xs font-bold px-2.5 py-0.5 rounded">
                        💨 {explorerSubMode === "netherlands" 
                          ? `NL - ${selectedMonth === "annual" ? "Annual" : `Month ${selectedMonth}`}` 
                          : `${selectedCountry}`}
                      </span>
                    </div>
                    <div className="space-y-1.5 text-sm font-semibold">
                      <p><b>Coordinates:</b> {clickPos.lat.toFixed(4)}, {clickPos.lng.toFixed(4)}</p>
                      {explorerSubMode === "netherlands" ? (
                        <>
                          <p><b>Current Wind:</b> <span className="text-green-700 font-extrabold">
                            {selectedMonth === "annual" 
                              ? `${windPointDetails.annual_wind_speed?.toFixed(2)} m/s` 
                              : `${windPointDetails.monthly_wind_speed?.[selectedMonth]?.toFixed(2)} m/s`}
                          </span></p>
                          <p><b>Annual Mean:</b> {windPointDetails.annual_wind_speed?.toFixed(2)} m/s</p>
                          {windPointDetails.monthly_wind_speed && (
                            <div className="mt-2 text-[11px] bg-indigo-50 border border-indigo-100 rounded p-1.5 text-indigo-950 font-semibold">
                              <b className="block mb-1 border-b border-indigo-100 text-indigo-900">Monthly breakdown:</b>
                              <div className="grid grid-cols-3 gap-1 text-center font-mono">
                                <div>Jan: {windPointDetails.monthly_wind_speed["1"]?.toFixed(1)}</div>
                                <div>Feb: {windPointDetails.monthly_wind_speed["2"]?.toFixed(1)}</div>
                                <div>Mar: {windPointDetails.monthly_wind_speed["3"]?.toFixed(1)}</div>
                                <div>Apr: {windPointDetails.monthly_wind_speed["4"]?.toFixed(1)}</div>
                                <div>May: {windPointDetails.monthly_wind_speed["5"]?.toFixed(1)}</div>
                                <div>Jun: {windPointDetails.monthly_wind_speed["6"]?.toFixed(1)}</div>
                                <div>Jul: {windPointDetails.monthly_wind_speed["7"]?.toFixed(1)}</div>
                                <div>Aug: {windPointDetails.monthly_wind_speed["8"]?.toFixed(1)}</div>
                                <div>Sep: {windPointDetails.monthly_wind_speed["9"]?.toFixed(1)}</div>
                                <div>Oct: {windPointDetails.monthly_wind_speed["10"]?.toFixed(1)}</div>
                                <div>Nov: {windPointDetails.monthly_wind_speed["11"]?.toFixed(1)}</div>
                                <div>Dec: {windPointDetails.monthly_wind_speed["12"]?.toFixed(1)}</div>
                              </div>
                            </div>
                          )}
                        </>
                      ) : (
                        <>
                          <p><b>Country:</b> {windPointDetails.country || selectedCountry}</p>
                          <p><b>Annual Wind:</b> <span className="text-green-700 font-extrabold">{windPointDetails.annual_wind_speed?.toFixed(2)} m/s</span></p>
                        </>
                      )}
                    </div>
                  </>
                )}
              </div>
            </Popup>
          )}
        </MapContainer>
      </div>

    </div>
  );
}