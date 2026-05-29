"use client"; // Говорит Next.js, что это выполняется только в браузере

import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useState, useEffect } from "react";

// --- ФИКС ИКОНОК ЛИФЛЕТА ДЛЯ NEXT.JS ---
const DefaultIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = DefaultIcon;
// ----------------------------------------

export default function MapComponent() {
  const [stations, setStations] = useState<any[]>([]);
  const [evaluation, setEvaluation] = useState<any | null>(null);
  const [clickPos, setClickPos] = useState<{ lat: number; lng: number } | null>(null);

  // При загрузке компонента запрашиваем список станций у нашего Python API
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/v1/stations")
      .then((res) => res.json())
      .then((data) => setStations(data))
      .catch((err) => console.error("Ошибка загрузки станций:", err));
  }, []);

  // Компонент-обработчик кликов по карте
  function MapClickHandler() {
    useMapEvents({
      click: async (e) => {
        const { lat, lng } = e.latlng;
        setClickPos({ lat, lng });

        // Отправляем POST запрос на наш эндпоинт Evaluate
        try {
          const res = await fetch("http://127.0.0.1:8000/api/v1/turbines/evaluate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lat: lat, lon: lng, turbine_model: "Vestas_V164" }),
          });
          const data = await res.json();
          setEvaluation(data);
        } catch (err) {
          console.error("Ошибка оценки:", err);
        }
      },
    });
    return null;
  }

  return (
    <MapContainer center={[52.3, 4.9]} zoom={7} className="w-full h-full z-0">
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <MapClickHandler />

      {/* Рисуем метеостанции (черные точки) */}
      {stations.map((st) => (
        <Marker key={st.STN} position={[st.lat, st.lon]}>
          <Popup>
            <b>{st.station_name}</b> <br />
            ID: {st.STN} <br />
            Ветер: {st.avg_wind_speed} м/с
          </Popup>
        </Marker>
      ))}

      {/* Рисуем попап в месте клика пользователя */}
      {clickPos && evaluation && (
        <Popup position={[clickPos.lat, clickPos.lng]}>
          <div className="p-1">
            <h3 className="font-bold text-lg mb-1">Оценка локации</h3>
            {evaluation.suitable ? (
              <p className="text-green-600 font-bold mb-2">✅ Пригодно для стройки</p>
            ) : (
              <p className="text-red-600 font-bold mb-2">❌ Непригодно</p>
            )}
            <p><b>Скорость ветра:</b> {evaluation.wind_speed_ms} м/с</p>
            <p><b>Рейтинг:</b> {evaluation.score} / 100</p>
            
            {evaluation.warnings?.length > 0 && (
              <div className="mt-2 text-sm text-yellow-700 bg-yellow-100 p-2 rounded">
                <b>Предупреждения:</b>
                <ul className="list-disc pl-4">
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
  );
}