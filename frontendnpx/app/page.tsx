"use client";

import dynamic from "next/dynamic";

// Dynamic import of the map (to disable server-side rendering for Leaflet)
const MapWithNoSSR = dynamic(() => import("@/src/components/MapComponent"), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-full">Loading map...</div>,
});

export default function Home() {
  return (
    <main className="flex h-screen w-screen bg-gray-50 flex-col md:flex-row">
      
      {/* LEFT PANEL (SIDEBAR) */}
      <div className="w-full md:w-1/3 lg:w-1/4 p-6 bg-white shadow-xl z-10 flex flex-col">
        <h1 className="text-2xl font-bold text-blue-900 mb-2">
          Wind Turbine Analyzer
        </h1>
        <p className="text-gray-600 text-sm mb-6">
          Interactive system for assessing Dutch territories for wind farm construction.
        </p>

        <div className="bg-blue-50 border border-blue-100 p-4 rounded-lg mb-4">
          <h2 className="font-semibold text-blue-800 mb-1">How to use:</h2>
          <ul className="list-decimal pl-4 text-sm text-blue-900 space-y-1">
            <li>Click any point on the map (land or EEZ).</li>
            <li>The frontend sends coordinates to the Python API.</li>
            <li>The API calculates wind speed and returns a business assessment.</li>
          </ul>
        </div>

        <div className="mt-auto pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-400 text-center">
            Sprint 2: ML-based Suitability Assessment
          </p>
        </div>
      </div>

      {/* RIGHT PANEL (MAP) */}
      <div className="flex-1 relative z-0">
        <MapWithNoSSR />
      </div>

    </main>
  );
}