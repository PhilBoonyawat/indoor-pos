// ---------------- MAP SETUP ----------------

const map = L.map('map', {
  crs: L.CRS.Simple,
  minZoom: -2
})

const width = 2000
const height = 1400

const bounds = [[0,0],[height,width]]

L.imageOverlay("floorplan.png", bounds).addTo(map)

map.fitBounds(bounds)


// ---------------- ROOM STORAGE ----------------

const rooms = {}

function createRoom(name, coords){

  const room = L.polygon(coords,{
    color:"#444",
    weight:2,
    fillColor:"#888",
    fillOpacity:0.2
  }).addTo(map)

  room.bindTooltip(name,{
    permanent:true,
    direction:"center",
    className:"room-label"
  })

  rooms[name] = room
}


// ---------------- ROOM POLYGONS ----------------

createRoom("7.01",[
[608,1436],
[618,1722],
[404,1740],
[397,1421]
])

createRoom("7.02",[
[856,1368],
[858,1694],
[628,1718],
[622,1428]
])

createRoom("7.03",[
[874,1690],
[1230,1676],
[1232,1286],
[882,1296]
])

createRoom("7.04",[
[876,850],
[1212,844],
[1220,1276],
[862,1286]
])

createRoom("7.05",[
[872,340],
[1228,320],
[1240,808],
[892,832]
])

createRoom("7.06",[
[764,328],
[396,300],
[392,856],
[496,872],
[492,1036],
[736,1056],
[768,368]
])


// ---------------- USER LOCATION DOT ----------------

const userMarker = L.circleMarker([800,900],{
  radius:10,
  color:"#0066ff",
  fillColor:"#0099ff",
  fillOpacity:1
}).addTo(map)


// ---------------- HEATMAP LAYER ----------------

const heat = L.heatLayer([], {

  radius:70,
  blur:50,

  gradient:{
    0.1:"blue",
    0.3:"cyan",
    0.5:"lime",
    0.7:"yellow",
    1.0:"red"
  }

}).addTo(map)


// ---------------- RSSI NORMALIZATION ----------------

function normalizeRSSI(rssi){

  return Math.max(0, Math.min(1, (100 + rssi) / 70))

}


// ---------------- ADD HEAT INSIDE ROOM ----------------

function addSignalHeat(roomName, rssi){

  const room = rooms[roomName]

  if(!room) return

  const center = room.getBounds().getCenter()

  const intensity = normalizeRSSI(rssi)

  const points = []

  // generate multiple points inside room
  for(let i=0;i<12;i++){

    const offsetY = (Math.random()-0.5)*120
    const offsetX = (Math.random()-0.5)*120

    points.push([
      center.lat + offsetY,
      center.lng + offsetX,
      intensity
    ])

  }

  heat.addLatLng(points)

}


// ---------------- ROOM HIGHLIGHT ----------------

function highlightRoom(roomName){

  for(let r in rooms){

    rooms[r].setStyle({
      fillColor:"#888",
      fillOpacity:0.2
    })

  }

  if(rooms[roomName]){

    rooms[roomName].setStyle({
      fillColor:"#00ff88",
      fillOpacity:0.5,
      color:"#00cc66",
      weight:3
    })

    document.getElementById("roomDisplay").innerText = roomName

    const center = rooms[roomName].getBounds().getCenter()

    userMarker.setLatLng(center)

  }

}


// ---------------- RANDOM DEMO PREDICTION ----------------

const roomList = ["7.01","7.02","7.03","7.04","7.05","7.06"]

function randomPrediction(){

  const room = roomList[Math.floor(Math.random()*roomList.length)]

  highlightRoom(room)

  // simulate RSSI measurement
  const rssi = -40 - Math.random()*40

  addSignalHeat(room, rssi)

}


// update every 3 seconds
setInterval(randomPrediction,3000)