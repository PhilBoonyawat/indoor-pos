// ============================================================
// CONFIG — Update these to match your floor plan
// ============================================================
const CONFIG = {
    center: [51.5131, -0.1170],
    defaultZoom: 20,

    // Enable the floor plan overlay
    floorPlanUrl: "/static/floorplan.png",   // ← change from null

    // These need to match the real lat/lng corners of your building
    // You'll need to fine-tune these by checking Google Maps
    floorPlanBounds: [
        [51.51265041169237, -0.11728181865167508],  // Southwest corner
        [51.512482671238374, -0.11679127299528806]   // Northeast corner
    ],

    // Match your actual floor plan image dimensions
    floorPlanWidth: 858,   // ← update to your image's pixel width
    floorPlanHeight: 782,   // ← update to your image's pixel height
  }

// ============================================================
// MAP SETUP
// ============================================================
const map = L.map("map", {
    center: CONFIG.center,
    zoom: CONFIG.defaultZoom,
    zoomControl: true,
});

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 22,
    attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

// Floor plan overlay
let floorPlanLayer = null;
if (CONFIG.floorPlanUrl) {
    floorPlanLayer = L.imageOverlay(
        CONFIG.floorPlanUrl,
        CONFIG.floorPlanBounds,
        { opacity: 0.75, interactive: false }
    ).addTo(map);
}


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

const positionMarker = L.marker(CONFIG.center, { icon: positionIcon }).addTo(map);
positionMarker.bindPopup("Waiting for position...");


// ============================================================
// ROOM POSITIONS — Maps room names to lat/lng on the floor plan
// ============================================================
let roomPositions = {};

// Convert pixel position on floor plan to lat/lng
function pixelToLatLng(x, y) {
    const sw = CONFIG.floorPlanBounds[0];
    const ne = CONFIG.floorPlanBounds[1];
    const lat = sw[0] + (1 - y / CONFIG.floorPlanHeight) * (ne[0] - sw[0]);
    const lng = sw[1] + (x / CONFIG.floorPlanWidth) * (ne[1] - sw[1]);
    return [lat, lng];
}

// Load room positions from backend
async function loadRoomPositions() {
    try {
        const res = await fetch("/api/room-positions");
        roomPositions = await res.json();
        console.log("Room positions loaded:", roomPositions);
    } catch (e) {
        console.log("No room positions available — using default center");
    }
}

function getRoomLatLng(room, posX, posY) {
    // If we have pixel positions from the labeller tool, convert them
    if (posX > 0 || posY > 0) {
        return pixelToLatLng(posX, posY);
    }

    // Fallback: spread rooms in a grid pattern around center
    const roomOffsets = {
        "(S)7.01": [-0.00015, -0.00030],
        "(S)7.02": [-0.00015,  0.00000],
        "(S)7.03": [-0.00015,  0.00030],
        "(S)7.04": [ 0.00015, -0.00030],
        "(S)7.05": [ 0.00015,  0.00000],
        "(S)7.06": [ 0.00015,  0.00030],
    };

    const offset = roomOffsets[room] || [0, 0];
    return [CONFIG.center[0] + offset[0], CONFIG.center[1] + offset[1]];
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
        const latlng = getRoomLatLng(pos.room, pos.position_x, pos.position_y);
        positionMarker.setLatLng(latlng);

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
        addTrailPoint(latlng[0], latlng[1]);

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
    map.setView(positionMarker.getLatLng(), CONFIG.defaultZoom);
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