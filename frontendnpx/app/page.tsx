"use client";

import dynamic from "next/dynamic";

// Dynamic import of the map (to disable server-side rendering for Leaflet)
const MapWithNoSSR = dynamic(() => import("@/src/components/MapComponent"), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-full bg-gray-50 text-blue-900 font-bold">Loading Wind Turbine Analyzer...</div>,
});

export default function Home() {
  return (
    <main className="h-screen w-screen overflow-hidden bg-gray-50">
      <MapWithNoSSR />
    </main>
  );
}
