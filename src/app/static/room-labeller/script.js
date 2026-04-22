const COLORS = [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
    '#8b5cf6', '#ec4899', '#06b6d4', '#f97316',
    '#14b8a6', '#6366f1', '#84cc16', '#e11d48'
];

let rooms = [];
let placingMode = false;
let placingRoomName = '';
let floorPlanLoaded = false;
let imageWidth = 0;
let imageHeight = 0;


const fileInput = document.getElementById('file-input');
const importInput = document.getElementById('import-input');
const canvasContainer = document.getElementById('canvas-container');
const uploadPrompt = document.getElementById('upload-prompt');
const roomList = document.getElementById('room-list');
const roomNameInput = document.getElementById('room-name-input');
const modeBadge = document.getElementById('mode-badge');
const instructionText = document.getElementById('instruction-text');
const cursorCoords = document.getElementById('cursor-coords');
const countEl = document.getElementById('count');

document.getElementById('upload-btn').addEventListener('click', () => fileInput.click());
document.getElementById('choose-image-btn').addEventListener('click', () => fileInput.click());
document.getElementById('add-room-btn').addEventListener('click', startPlacing);
document.getElementById('export-btn').addEventListener('click', exportJSON);
document.getElementById('import-btn').addEventListener('click', importJSON);

// Floor plan upload and rendering
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
        uploadPrompt.style.display = 'none';

        // Remove old wrapper if exists
        const old = document.querySelector('.floor-plan-wrapper');
        if (old) old.remove();

        const wrapper = document.createElement('div');
        wrapper.className = 'floor-plan-wrapper';

        const img = document.createElement('img');
        img.src = ev.target.result;
        img.id = 'floor-plan-img';
        img.onload = () => {
            imageWidth = img.naturalWidth;
            imageHeight = img.naturalHeight;
            floorPlanLoaded = true;
            instructionText.textContent = 'Add a room name -> click on the floor plan to place it';
            showToast('Floor plan loaded — ' + imageWidth + '×' + imageHeight + 'px');
            reRenderMarkers();
        };

        wrapper.appendChild(img);
        canvasContainer.appendChild(wrapper);

        // Click to place
        wrapper.addEventListener('click', handleCanvasClick);

        // Mouse move for coordinates
        wrapper.addEventListener('mousemove', (e) => {
            const rect = img.getBoundingClientRect();
            const x = Math.round((e.clientX - rect.left) / rect.width * imageWidth);
            const y = Math.round((e.clientY - rect.top) / rect.height * imageHeight);
            cursorCoords.textContent = `x: ${x}  y: ${y}`;
        });
    };
    reader.readAsDataURL(file);
});

/*
Places location marker on the rendered floor plan based on user clicks, allowing users to label rooms with custom names. 
Validations ensure that a floor plan is uploaded and a room name is provided before placement. 
The UI updates dynamically to reflect the current state and provides feedback through toast notifications.
*/
function startPlacing() {
    const name = roomNameInput.value.trim();
    if (!name) {
        showToast('Enter a room name first');
        roomNameInput.focus();
        return;
    }
    if (!floorPlanLoaded) {
        showToast('Upload a floor plan first');
        return;
    }
    if (rooms.find(r => r.name === name)) {
        showToast('Room "' + name + '" already exists');
        return;
    }

    placingMode = true;
    placingRoomName = name;
    modeBadge.textContent = 'PLACING: ' + name;
    modeBadge.className = 'mode-badge placing';
    instructionText.textContent = 'Click on the floor plan to place "' + name + '"';
}

/*
Handles click events on the floor plan canvas to place room markers. 
*/
function handleCanvasClick(e) {
    if (!placingMode) return;

    const img = document.getElementById('floor-plan-img');
    const rect = img.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) / rect.width * imageWidth);
    const y = Math.round((e.clientY - rect.top) / rect.height * imageHeight);

    const color = COLORS[rooms.length % COLORS.length];

    rooms.push({
        name: placingRoomName,
        x: x,
        y: y,
        color: color
    });

    placingMode = false;
    placingRoomName = '';
    roomNameInput.value = '';
    modeBadge.textContent = 'IDLE';
    modeBadge.className = 'mode-badge idle';
    instructionText.textContent = 'Add another room or export your positions';

    updateUI();
    showToast('Placed "' + rooms[rooms.length - 1].name + '" at (' + x + ', ' + y + ')');
}

// Allows pressing Enter in the room name input to start placing mode, improving UX for keyboard users.
roomNameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') startPlacing();
});

/*
Updates the UI to reflect the current list of rooms, including the count and rendering markers on the floor plan.
*/
function updateUI() {
    countEl.textContent = rooms.length;
    renderRoomList();
    reRenderMarkers();
}

/*
Renders the list of rooms in the sidebar with their names, coordinates, and a delete button.
*/
function renderRoomList() {
    if (rooms.length === 0) {
        roomList.innerHTML = `
            <div class="empty-rooms">
                <div class="icon">🏠</div>
                <p>No rooms yet.<br>Add a room name and click on the floor plan.</p>
            </div>
        `;
        return;
    }

    roomList.innerHTML = rooms.map((room, i) => `
        <div class="room-item" data-index="${i}">
            <div class="room-dot" style="background:${room.color}; box-shadow: 0 0 8px ${room.color}55"></div>
            <div class="room-info">
                <div class="room-name">${room.name}</div>
                <div class="room-coords">x: ${room.x}  y: ${room.y}</div>
            </div>
            <button class="room-delete" data-index="${i}">✕</button>
        </div>
    `).join('');

    // Delegate clicks on dynamically generated items
    roomList.querySelectorAll('.room-item').forEach(el => {
        el.addEventListener('click', () => highlightRoom(parseInt(el.dataset.index)));
    });
    roomList.querySelectorAll('.room-delete').forEach(el => {
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteRoom(parseInt(el.dataset.index));
        });
    });
}

/*
Fetches the latest position data from the server and updates the map marker and UI accordingly.
*/
function reRenderMarkers() {
    // Remove old markers
    document.querySelectorAll('.room-marker').forEach(m => m.remove());

    const img = document.getElementById('floor-plan-img');
    if (!img) return;

    const wrapper = img.parentElement;

    rooms.forEach(room => {
        const pxX = (room.x / imageWidth) * 100;
        const pxY = (room.y / imageHeight) * 100;

        const marker = document.createElement('div');
        marker.className = 'room-marker';
        marker.style.left = pxX + '%';
        marker.style.top = pxY + '%';

        marker.innerHTML = `
            <div class="marker-dot" style="background:${room.color}; border-color:${room.color}"></div>
            <div class="marker-label">${room.name}</div>
        `;

        wrapper.appendChild(marker);
    });
}

/*
Makes the clicked room item in the sidebar visually distinct to indicate it is active or selected.
*/
function highlightRoom(index) {
    document.querySelectorAll('.room-item').forEach((el, i) => {
        el.classList.toggle('active', i === index);
    });
}

/*
Deletes a room from the list based on its index and updates the UI accordingly.
*/
function deleteRoom(index) {
    const name = rooms[index].name;
    rooms.splice(index, 1);
    updateUI();
    showToast('Removed "' + name + '"');
}

/*
Exports coordinate data of all mapped rooms to a JSON file named room_positions.json.
*/
function exportJSON() {
    if (rooms.length === 0) {
        showToast('No rooms to export');
        return;
    }

    const output = {};
    rooms.forEach(room => {
        output[room.name] = {
            x: room.x,
            y: room.y
        };
    });

    const data = JSON.stringify(output, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'room_positions.json';
    a.click();
    URL.revokeObjectURL(url);

    showToast('Exported ' + rooms.length + ' rooms to room_positions.json');
}

/*
Imports room coordinate data from a user-selected JSON file and updates the UI with the new room information.
*/
function importJSON() {
    importInput.click();
}

importInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
        try {
            const data = JSON.parse(ev.target.result);
            rooms = [];
            let i = 0;
            for (const [name, pos] of Object.entries(data)) {
                rooms.push({
                    name: name,
                    x: pos.x,
                    y: pos.y,
                    color: COLORS[i % COLORS.length]
                });
                i++;
            }
            updateUI();
            showToast('Imported ' + rooms.length + ' rooms');
        } catch (err) {
            showToast('Invalid JSON file');
        }
    };
    reader.readAsText(file);
});

/*
Displays a temporary toast notification with the provided message.
*/
function showToast(message) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2500);
}