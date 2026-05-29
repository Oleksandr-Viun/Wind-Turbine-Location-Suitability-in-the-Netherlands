"use client"; // Tells Next.js that this runs only in the browser

import { MapContainer, TileLayer, Marker, Popup, useMapEvents, CircleMarker, GeoJSON } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useState, useEffect } from "react";

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
  iconSize: [13, 20], // Reduced size
  iconAnchor: [9, 30], // Centered at bottom
});

L.Marker.prototype.options.icon = DefaultIcon;
// ----------------------------------------

export default function MapComponent() {
  const [stations, setStations] = useState<any[]>([]);
  const [gridData, setGridData] = useState<any[]>([]);
  const [suitableAreas, setSuitableAreas] = useState<any | null>(null);
  const [evaluation, setEvaluation] = useState<any | null>(null);
  const [clickPos, setClickPos] = useState<{ lat: number; lng: number } | null>(null);
  const [isLoadingGrid, setIsLoadingGrid] = useState(true);

  // 1. Load the list of weather stations
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/v1/stations")
      .then((res) => res.json())
      .then((data) => setStations(data))
      .catch((err) => console.error("Error loading stations:", err));
  }, []);

  // 2. Load the ENTIRE ML grid (17k+ points)
  useEffect(() => {
    setIsLoadingGrid(true);
    fetch("http://127.0.0.1:8000/api/v1/wind/all")
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

  // 3. Load the suitability patches (polygons)
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/v1/wind/suitable-areas")
      .then((res) => res.json())
      .then((data) => setSuitableAreas(data))
      .catch((err) => console.error("Error loading suitable areas:", err));
  }, []);

  // Function to select point color based on its rating (Blue -> Green style)
  const getScoreColor = (score: number, isNatura: number) => {
    if (isNatura === 1) return "#64748b"; // Slate gray for Natura 2000 (protected)
    
    // Blue -> Teal -> Green scale (Higher = More Green/Lighter)
    if (score >= 80) return "#15803d"; // Dark Green (Excellent)
    if (score >= 70) return "#22c55e"; // Green (Very Good)
    if (score >= 60) return "#84cc16"; // Lime (Good)
    if (score >= 50) return "#14b8a6"; // Teal (Average)
    if (score >= 40) return "#0ea5e9"; // Sky Blue (Moderate)
    if (score >= 25) return "#2563eb"; // Blue (Poor)
    return "#1e3a8a";                 // Dark Blue (Very Poor)
  };

  // Map click handler component
  function MapClickHandler() {
    useMapEvents({
      click: async (e) => {
        const { lat, lng } = e.latlng;
        setClickPos({ lat, lng });

        try {
          const res = await fetch("http://127.0.0.1:8000/api/v1/turbines/evaluate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lat: lat, lon: lng, turbine_model: "Vestas_V164" }),
          });
          const data = await res.json();
          setEvaluation(data);
        } catch (err) {
          console.error("Evaluation error:", err);
        }
      },
    });
    return null;
  }

  return (
    <div className="relative w-full h-full">
      {/* Grid loading indicator */}
      {isLoadingGrid && (
        <div className="absolute top-4 right-4 z-[1000] bg-white p-2 rounded shadow-md text-xs font-bold animate-pulse">
          ⏳ Loading ML Grid...
        </div>
      )}

      <MapContainer 
        center={[52.3, 4.9]} 
        zoom={7} 
        className="w-full h-full z-0"
        preferCanvas={true} // OPTIMIZATION: Render via Canvas for 17k points
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <MapClickHandler />

        {/* Draw the ENTIRE grid (Small dots) */}
        {gridData.map((point, idx) => {
          const score = point.ml_suitability_score;
          const isHighlySuitable = score >= 70;
          
          return (
            <CircleMarker
              key={`grid-${idx}`}
              center={[point.cell_lat, point.cell_lon]}
              radius={isHighlySuitable ? 3.5 : 2} // Larger for highly suitable
              pathOptions={{
                fillColor: point.suitability_color || getScoreColor(score, point.is_natura2000),
                stroke: isHighlySuitable, // Individual border for top locations
                color: "#ffffff",         // Bright white border
                weight: 1.5,              // Slightly thicker for better visibility
                fillOpacity: 0.85
              }}
            />
          );
        })}

        {/* Draw weather stations (Smaller pin markers) */}
        {stations.map((st) => (
          <Marker 
            key={st.STN} 
            position={[st.lat, st.lon]}
            icon={SmallIcon}
          >
            <Popup>
              <b>{st.station_name}</b> <br />
              ID: {st.STN} <br />
              Wind: {st.avg_wind_speed} m/s
            </Popup>
          </Marker>
        ))}

        {/* Draw popup at user's click location */}
        {clickPos && evaluation && !evaluation.error && (
          <Popup position={[clickPos.lat, clickPos.lng]}>
            <div className="p-1 min-w-[200px]">
              <h3 className="font-bold text-lg mb-1 border-b pb-1">Location Assessment</h3>
              
              <div className="my-2">
                {evaluation.score >= 60 ? (
                  <span className="bg-green-100 text-green-800 text-xs font-bold px-2.5 py-0.5 rounded">✅ SUITABLE</span>
                ) : (
                  <span className="bg-red-100 text-red-800 text-xs font-bold px-2.5 py-0.5 rounded">❌ UNSUITABLE</span>
                )}
              </div>

              <div className="space-y-1 text-sm">
                <p><b>Rating (ML):</b> <span className="text-blue-700 font-bold">{evaluation.score}%</span></p>
                <p><b>Cluster:</b> {evaluation.ml_label}</p>
                <p><b>Wind:</b> {evaluation.wind_speed_ms} m/s</p>
                <p><b>Population:</b> {evaluation.environment?.population_density} p/km²</p>
                <p><b>Natura 2000:</b> {evaluation.environment?.is_natura2000 ? "Yes" : "No"}</p>
              </div>
              
              {evaluation.warnings?.length > 0 && (
                <div className="mt-2 text-[11px] text-yellow-800 bg-yellow-50 p-2 rounded border border-yellow-100">
                  <b>Warnings:</b>
                  <ul className="list-disc pl-3 mt-1">
                    {evaluation.warnings.map((w: string, i: number) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </Popup>
        )}
      </MapContainer>
    </div>
  );
}
