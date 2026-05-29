"use client"; // <-- ВОТ ТА САМАЯ ВОЛШЕБНАЯ СТРОЧКА

import dynamic from "next/dynamic";

// Динамический импорт карты (чтобы отключить серверный рендеринг для Leaflet)
const MapWithNoSSR = dynamic(() => import("@/src/components/MapComponent"), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-full">Загрузка карты...</div>,
});

export default function Home() {
  return (
    <main className="flex h-screen w-screen bg-gray-50 flex-col md:flex-row">
      
      {/* ЛЕВАЯ ПАНЕЛЬ (САЙДБАР) */}
      <div className="w-full md:w-1/3 lg:w-1/4 p-6 bg-white shadow-xl z-10 flex flex-col">
        <h1 className="text-2xl font-bold text-blue-900 mb-2">
          Wind Turbine Analyzer
        </h1>
        <p className="text-gray-600 text-sm mb-6">
          Интерактивная система оценки территорий Нидерландов для строительства ветропарков.
        </p>

        <div className="bg-blue-50 border border-blue-100 p-4 rounded-lg mb-4">
          <h2 className="font-semibold text-blue-800 mb-1">Как использовать:</h2>
          <ul className="list-decimal pl-4 text-sm text-blue-900 space-y-1">
            <li>Кликните в любую точку на карте (на суше или в EEZ).</li>
            <li>Фронтенд отправит координаты в Python API.</li>
            <li>API рассчитает скорость ветра и вернет бизнес-оценку.</li>
          </ul>
        </div>

        <div className="mt-auto pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-400 text-center">
            Спринт 1: Математическая модель (Без Natura 2000)
          </p>
        </div>
      </div>

      {/* ПРАВАЯ ПАНЕЛЬ (КАРТА) */}
      <div className="flex-1 relative z-0">
        <MapWithNoSSR />
      </div>

    </main>
  );
}