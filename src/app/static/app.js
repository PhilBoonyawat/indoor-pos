// ============================================================
// CONFIG — Update these to match your floor plan
// ============================================================
const CONFIG = {
    // Floor plan image
    floorPlanUrl: "/static/Level7Floor.png",
    floorPlanWidth: 858,
    floorPlanHeight: 782,

    pollInterval: 3000,
};


// ============================================================
// MAP SETUP — Floor plan as the entire map (no base tiles)
// ============================================================

// Use simple CRS so pixel coordinates map directly
const bounds = [[0, 0], [CONFIG.floorPlanHeight, CONFIG.floorPlanWidth]];

const map = L.map("map", {
    crs: L.CRS.Simple,        // pixel-based coordinates, no lat/lng
    minZoom: -2,
    maxZoom: 3,
    zoomControl: true,
    attributionControl: false,
});

// Add floor plan as the map itself
const floorPlanLayer = L.imageOverlay(
    CONFIG.floorPlanUrl,
    bounds
).addTo(map);

// Fit the view to the floor plan
map.fitBounds(bounds);
map.setMaxBounds(bounds.map(b => [b[0] - 100, b[1] - 100]));  // slight padding


// ============================================================
// POSITION MARKER
// ============================================================
const positionIcon = L.divIcon({
    className: "position-marker",
    html: `<div class="marker-pulse" style="
        width: 20px; height: 20px;
        background: #3b82f6;
        border: 3px solid #fff;
        border-radius: 50%;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.6);
    "></div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
});

// Start marker at center of floor plan
const startPos = [CONFIG.floorPlanHeight / 2, CONFIG.floorPlanWidth / 2];
const positionMarker = L.marker(startPos, { icon: positionIcon }).addTo(map);
positionMarker.bindPopup("Waiting for position...");


// ============================================================
// ROOM POSITIONS — Maps room names to pixel coords on floor plan
// ============================================================
let roomPositions = {};

async function loadRoomPositions() {
    try {
        const res = await fetch("/api/room-positions");
        roomPositions = await res.json();
        console.log("Room positions loaded:", roomPositions);

        // Add room labels to the map
        for (const [room, pos] of Object.entries(roomPositions)) {
            // Leaflet Simple CRS uses [y, x] but y is inverted (0 = top)
            const latlng = [CONFIG.floorPlanHeight - pos.y, pos.x];

            L.circleMarker(latlng, {
                radius: 6,
                fillColor: '#4ecca3',
                fillOpacity: 0.3,
                color: '#4ecca3',
                weight: 1,
            }).addTo(map);

            L.tooltip({ permanent: true, direction: 'top', className: 'room-tooltip' })
                .setLatLng(latlng)
                .setContent(room)
                .addTo(map);
        }
    } catch (e) {
        console.log("No room positions available");
    }
}

function getRoomCoords(room, posX, posY) {
    // Convert pixel position to Leaflet Simple CRS coordinates
    // In Simple CRS: [y, x] where y=0 is bottom, but our pixels y=0 is top
    // So we invert: leaflet_y = imageHeight - pixel_y

    if (posX > 0 || posY > 0) {
        return [CONFIG.floorPlanHeight - posY, posX];
    }

    // Fallback: check room_positions loaded from API
    const pos = roomPositions[room];
    if (pos) {
        return [CONFIG.floorPlanHeight - pos.y, pos.x];
    }

    // Last resort: center of floor plan
    return [CONFIG.floorPlanHeight / 2, CONFIG.floorPlanWidth / 2];
}


// ============================================================
// TRAIL
// ============================================================
let showTrail = true;
let trailPoints = [];
const trailLine = L.polyline([], {
    color: "#3b82f6",
    weight: 2,
    opacity: 0.5,
    dashArray: "5, 8",
}).addTo(map);

const trailDots = L.layerGroup().addTo(map);

function addTrailPoint(lat, lng) {
    if (!showTrail) return;
    trailPoints.push([lat, lng]);
    if (trailPoints.length > 50) trailPoints.shift();
    trailLine.setLatLngs(trailPoints);

    const dot = L.circleMarker([lat, lng], {
        radius: 3, fillColor: "#3b82f6", fillOpacity: 0.3, stroke: false,
    });
    trailDots.addLayer(dot);
    if (trailDots.getLayers().length > 50) {
        trailDots.removeLayer(trailDots.getLayers()[0]);
    }
}


// ============================================================
// MODEL SELECTION
// ============================================================
const modelSelect = document.getElementById("model-select");

async function loadModels() {
    try {
        const res = await fetch("/api/models");
        const data = await res.json();

        modelSelect.innerHTML = "";
        data.available.forEach(name => {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            if (name === data.active) opt.selected = true;
            modelSelect.appendChild(opt);
        });

        if (data.available.length === 0) {
            modelSelect.innerHTML = '<option value="">Demo Mode</option>';
        }
    } catch (e) {
        modelSelect.innerHTML = '<option value="">Offline</option>';
    }
}

modelSelect.addEventListener("change", async () => {
    const name = modelSelect.value;
    if (!name) return;

    try {
        await fetch("/api/model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_name: name }),
        });
        console.log("Switched to:", name);
    } catch (e) {
        console.error("Failed to switch model:", e);
    }
});


// ============================================================
// POLLING
// ============================================================
const modeBadge = document.getElementById("mode-badge");
const roomLabel = document.getElementById("room-label");
const confidenceLabel = document.getElementById("confidence-label");
const apsCount = document.getElementById("aps-count");
const modelName = document.getElementById("model-name");
const lastUpdate = document.getElementById("last-update");

async function fetchPosition() {
    try {
        const res = await fetch("/api/position");
        if (!res.ok) throw new Error("API error");
        const pos = await res.json();

        // Update marker position
        const coords = getRoomCoords(pos.room, pos.position_x, pos.position_y);
        positionMarker.setLatLng(coords);

        // Update popup
        positionMarker.setPopupContent(`
            <div class="position-popup">
                <strong>Room:</strong> ${pos.room}<br>
                <strong>Confidence:</strong> ${(pos.confidence * 100).toFixed(1)}%<br>
                <strong>Model:</strong> ${pos.model_used}<br>
                <strong>APs:</strong> ${pos.aps_detected}<br>
                <strong>Mode:</strong> ${pos.mode}
            </div>
        `);

        // Trail
        addTrailPoint(coords[0], coords[1]);

        // Update header
        roomLabel.textContent = `Room: ${pos.room}`;
        confidenceLabel.textContent = `${(pos.confidence * 100).toFixed(1)}%`;

        // Mode badge
        if (pos.mode === "live") {
            modeBadge.textContent = "LIVE";
            modeBadge.className = "mode-badge live";
        } else {
            modeBadge.textContent = "DEMO";
            modeBadge.className = "mode-badge demo";
        }

        // Info panel
        apsCount.textContent = pos.aps_detected;
        modelName.textContent = pos.model_used;
        lastUpdate.textContent = pos.timestamp
            ? new Date(pos.timestamp).toLocaleTimeString()
            : "—";

    } catch (e) {
        console.error("Fetch error:", e);
        modeBadge.textContent = "OFFLINE";
        modeBadge.className = "mode-badge demo";
    }
}


// ============================================================
// CONTROLS
// ============================================================
document.getElementById("btn-center").addEventListener("click", () => {
    map.fitBounds(bounds);
});

document.getElementById("btn-trail").addEventListener("click", (e) => {
    showTrail = !showTrail;
    e.target.textContent = showTrail ? "🔵" : "⚪";
    if (showTrail) {
        trailLine.addTo(map);
        trailDots.addTo(map);
    } else {
        map.removeLayer(trailLine);
        map.removeLayer(trailDots);
    }
});

document.getElementById("btn-clear").addEventListener("click", () => {
    trailPoints = [];
    trailLine.setLatLngs([]);
    trailDots.clearLayers();
});


// ============================================================
// INIT
// ============================================================
async function init() {
    await loadModels();
    await loadRoomPositions();
    fetchPosition();
    setInterval(fetchPosition, CONFIG.pollInterval);
}

init();