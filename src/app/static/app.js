// ============================================================
// CONFIG
// ============================================================
const CONFIG = {
    floorPlanUrl: "/static/Level7Floor.svg",
    pollInterval: 3000,
};

// ============================================================
// GLOBALS
// ============================================================
let map;
let positionMarker;
let roomPositions = {};
let bounds;

// SVG natural dimensions — must match the actual SVG viewBox
const SVG_W = 1626.667;
const SVG_H = 708;

// ============================================================
// COORDINATE HELPER
// SVG has y=0 at top-left. Leaflet CRS.Simple has y=0 at
// bottom-left. So: leaflet_lat = SVG_H - svg_y
//                  leaflet_lng = svg_x
// ============================================================
function svgToLatLng(svgX, svgY) {
    return [SVG_H - svgY, svgX];
}

// ============================================================
// INIT MAP
// ============================================================
function initMap() {
    bounds = [[0, 0], [SVG_H, SVG_W]];

    // Start with a very low minZoom; we will tighten it after fitBounds
    map = L.map("map", {
        crs: L.CRS.Simple,
        minZoom: -5,
        maxZoom: 3,
        zoomControl: true,
        attributionControl: false,
    });

    L.imageOverlay(CONFIG.floorPlanUrl, bounds).addTo(map);

    // Fit with NO padding so the SVG fills the whole container
    map.fitBounds(bounds, { padding: [0, 0] });

    // Lock the current zoom as the minimum so the user can never zoom out
    const fitZoom = map.getBoundsZoom(bounds, false);
    map.setMinZoom(fitZoom);

    // Restrict panning so the floor plan always covers the viewport
    map.setMaxBounds(bounds);

    initMarker();
    loadRoomPositions();
    loadModels();

    // Wire up control buttons
    document.getElementById("btn-center").addEventListener("click", () => {
        if (positionMarker) map.panTo(positionMarker.getLatLng());
    });
    document.getElementById("btn-trail").addEventListener("click", toggleTrail);
    document.getElementById("btn-clear").addEventListener("click", clearTrail);

    setInterval(fetchPosition, CONFIG.pollInterval);
    fetchPosition(); // immediate first fetch
}

// Start immediately — SVG dimensions are already known
initMap();

// ============================================================
// MARKER (BLUE DOT)
// ============================================================
function initMarker() {
    const positionIcon = L.divIcon({
        className: "",
        html: `<div class="marker-pulse" style="
            width: 20px;
            height: 20px;
            background: #3b82f6;
            border: 3px solid #fff;
            border-radius: 50%;
            box-shadow: 0 0 12px rgba(59,130,246,0.6);
        "></div>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10],
    });

    // Start in the centre of the floor plan
    positionMarker = L.marker(svgToLatLng(SVG_W / 2, SVG_H / 2), { icon: positionIcon })
        .addTo(map);
    positionMarker.bindPopup("Waiting for position...");
}

// ============================================================
// ROOM LABELS — placed directly on the floor plan
// ============================================================
async function loadRoomPositions() {
    try {
        const res = await fetch("/api/room-positions");
        roomPositions = await res.json();
        console.log(roomPositions);
        for (const [room, pos] of Object.entries(roomPositions)) {
            // pos.x / pos.y are SVG pixel coordinates
            const latlng = svgToLatLng(pos.x, pos.y);

            // Small teal circle
            L.circleMarker(latlng, {
                radius: 6,
                fillColor: "#3fffdc",
                fillOpacity: 0.35,
                color: "#3fffdc",
                weight: 1.5,
            }).addTo(map);

            // Permanent label that sticks to the circle at all zoom levels
            console.log(room, latlng);
            L.marker(latlng, {
                icon: L.divIcon({
                    className: "room-label-icon",
                    html: `<span class="room-tooltip">${room}</span>`,
                    iconAnchor: [0, 22],   // place label just above the dot
                }),
                interactive: false,
                zIndexOffset: 0,
            }).addTo(map);
        }

    } catch (e) {
        console.error("Failed to load room positions", e);
    }
}

// ============================================================
// TRAIL
// ============================================================
let trailPoints = [];
let trailVisible = true;
const trailLine = L.polyline([], {
    color: "#3b82f6",
    weight: 2,
    opacity: 0.5,
    dashArray: "5, 8",
}).addTo(map);

function addTrailPoint(latlng) {
    trailPoints.push(latlng);
    if (trailPoints.length > 50) trailPoints.shift();
    if (trailVisible) trailLine.setLatLngs(trailPoints);
}

function toggleTrail() {
    trailVisible = !trailVisible;
    trailLine.setLatLngs(trailVisible ? trailPoints : []);
}

function clearTrail() {
    trailPoints = [];
    trailLine.setLatLngs([]);
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
    } catch (e) {
        modelSelect.innerHTML = "<option>Offline</option>";
    }
}

modelSelect.addEventListener("change", async () => {
    await fetch("/api/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelSelect.value }),
    });
});

// ============================================================
// FETCH POSITION — move marker to predicted room
// ============================================================
let lastRoom = null;

async function fetchPosition() {
    try {
        const res = await fetch("/api/position");
        const pos = await res.json();

        // Determine coordinates: prefer explicit x/y from API,
        // fall back to roomPositions lookup
        let latlng;
        if ((pos.position_x > 0 || pos.position_y > 0)) {
            latlng = svgToLatLng(pos.position_x, pos.position_y);
        } else {
            const rp = roomPositions[pos.room];
            latlng = rp ? svgToLatLng(rp.x, rp.y) : svgToLatLng(SVG_W / 2, SVG_H / 2);
        }

        // Animate marker jump to new room
        positionMarker.setLatLng(latlng);

        // Add trail point
        addTrailPoint(latlng);

        // Update popup
        positionMarker.setPopupContent(`
            <strong>${pos.room}</strong><br>
            Confidence: ${(pos.confidence * 100).toFixed(1)}%<br>
            Model: ${pos.model_used}
        `);

        // ── UI updates ──
        document.getElementById("room-label").textContent = `Room: ${pos.room}`;
        document.getElementById("confidence-label").textContent =
            `${(pos.confidence * 100).toFixed(1)}%`;
        document.getElementById("aps-count").textContent = pos.aps_detected ?? "—";
        document.getElementById("model-name").textContent = pos.model_used ?? "—";

        const ts = pos.timestamp ? new Date(pos.timestamp).toLocaleTimeString() : "—";
        document.getElementById("last-update").textContent = ts;

        const badge = document.getElementById("mode-badge");
        badge.textContent = (pos.mode === "live") ? "LIVE" : "DEMO";
        badge.className = "mode-badge " + (pos.mode === "live" ? "live" : "demo");

        // Flash marker on room change
        if (pos.room !== lastRoom) {
            lastRoom = pos.room;
            flashMarker();
        }

    } catch (e) {
        console.error("Fetch failed", e);
    }
}

// ============================================================
// FLASH EFFECT on room change
// ============================================================
function flashMarker() {
    const el = positionMarker.getElement();
    if (!el) return;
    const dot = el.querySelector(".marker-pulse");
    if (!dot) return;

    dot.style.transform = "scale(1.8)";
    dot.style.transition = "transform 0.15s ease-out";
    setTimeout(() => {
        dot.style.transform = "scale(1)";
    }, 200);
}

// ============================================================
// MANUAL TESTING (browser console)
// ============================================================
window.jumpToRoom = function(roomName) {
    const pos = roomPositions[roomName];
    if (!pos) { console.warn("Room not found:", roomName); return; }
    const latlng = svgToLatLng(pos.x, pos.y);
    positionMarker.setLatLng(latlng);
    map.panTo(latlng);
};

// ============================================================
// RE-FIT ON RESIZE — keeps SVG filling the viewport
// ============================================================
window.addEventListener("resize", () => {
    if (!map) return;
    map.fitBounds(bounds, { padding: [0, 0] });
    const fitZoom = map.getBoundsZoom(bounds, false);
    map.setMinZoom(fitZoom);
});