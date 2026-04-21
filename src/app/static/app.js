/*
app.js - Frontend logic for indoor positioning system
*/

const CONFIG = {
    floorPlanUrl: "/static/Level7Floor.svg",
    pollInterval: 3000,
};

let map;
let positionMarker;
let roomPositions = {};
let bounds;

// SVG natural dimensions matching the actual image file
const SVG_W = 1626.667;
const SVG_H = 708;

function svgToLatLng(svgX, svgY) {
    return [SVG_H - svgY, svgX];
}

/*
Initialize Leaflet map with the floor plan as a static image overlay.
The SVG's coordinate system has (0,0) at the top-left, so we flip the Y-axis
when converting to Leaflet's lat/lng format.
*/
function initMap() {
    bounds = [[0, 0], [SVG_H, SVG_W]];

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

    document.getElementById("btn-center").addEventListener("click", () => {
        if (positionMarker) map.panTo(positionMarker.getLatLng());
    });
    document.getElementById("btn-trail").addEventListener("click", toggleTrail);
    document.getElementById("btn-clear").addEventListener("click", clearTrail);

    setInterval(fetchPosition, CONFIG.pollInterval);
    fetchPosition(); 
}

initMap();

/*
Position marker with a pulsing effect, created using a custom divIcon.
The marker starts at the center of the floor plan and updates based on API data.
*/
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

/*
Load room positions from the backend and place markers with labels on the map.
The API returns an object mapping room names to their SVG coordinates.
For each room, we place a small teal circle and a permanent label that stays
visible at all zoom levels.
*/
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
                    iconAnchor: [0, 22],  
                }),
                interactive: false,
                zIndexOffset: 0,
            }).addTo(map);
        }

    } catch (e) {
        console.error("Failed to load room positions", e);
    }
}


let trailPoints = [];
let trailVisible = true;
const trailLine = L.polyline([], {
    color: "#4635c5",
    weight: 2,
    opacity: 0.5,
    dashArray: "5, 8",
}).addTo(map);

/*
Trail management: keeps a history of recent positions and displays them as a dashed line.
*/
function addTrailPoint(latlng) {
    trailPoints.push(latlng);
    if (trailPoints.length > 50) trailPoints.shift();
    if (trailVisible) trailLine.setLatLngs(trailPoints);
}

/*
Toggle the visibility of the trail line without losing the history of points.
*/
function toggleTrail() {
    trailVisible = !trailVisible;
    trailLine.setLatLngs(trailVisible ? trailPoints : []);
}

/*
Clear the trail history and remove the line from the map.
*/
function clearTrail() {
    trailPoints = [];
    trailLine.setLatLngs([]);
}


const modelSelect = document.getElementById("model-select");

/*
Load available models from the backend and populate the dropdown menu.
*/
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


let lastRoom = null;

/*
Fetch the current position from the backend API and update the marker, popup, and UI labels.
The API response includes room name, confidence, model used, timestamp, and mode (live/demo).
If the room has changed since the last update, we trigger a flash effect on the marker.
*/
async function fetchPosition() {
    try {
        const res = await fetch("/api/position");
        const pos = await res.json();

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

/*
Flash the position marker with a quick scale animation to draw attention to room changes.
*/
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

window.addEventListener("resize", () => {
    if (!map) return;
    map.fitBounds(bounds, { padding: [0, 0] });
    const fitZoom = map.getBoundsZoom(bounds, false);
    map.setMinZoom(fitZoom);
});