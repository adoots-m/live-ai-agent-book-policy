// Phone client: mic -> 16 kHz PCM, camera -> JPEG frames (+ accelerometer), speaker <- 24 kHz PCM.
"use strict";
const KEY = document.querySelector('meta[name="lca-key"]').content;
const $ = (id) => document.getElementById(id);
const FRAME_MS = 333;          // ~3 fps of high-res frames while connected
const FRAME_W = 1600;          // long side sent to the server
let ws = null, audioCtx = null, micNode = null, micStream = null, camStream = null, frameTimer = null;
let playHead = 0, playing = [], sweeping = false;
const imu = { alpha: null, beta: null, gamma: null, gx: null, gy: null, gz: null, screen_angle: 0 };

// ------------------------------------------------------------------ sensors
function onMotion(e) {
  const g = e.accelerationIncludingGravity || {};
  imu.gx = g.x; imu.gy = g.y; imu.gz = g.z;
  imu.screen_angle = (screen.orientation && screen.orientation.angle) || window.orientation || 0;
}
function onOrient(e) { imu.alpha = e.alpha; imu.beta = e.beta; imu.gamma = e.gamma; }
async function enableSensors() {
  try {
    if (typeof DeviceMotionEvent !== "undefined" && DeviceMotionEvent.requestPermission) {
      await DeviceMotionEvent.requestPermission();   // iOS: needs a user gesture
    }
    if (typeof DeviceOrientationEvent !== "undefined" && DeviceOrientationEvent.requestPermission) {
      await DeviceOrientationEvent.requestPermission();
    }
  } catch (e) { note("motion sensors unavailable: " + e); }
  window.addEventListener("devicemotion", onMotion);
  window.addEventListener("deviceorientation", onOrient);
}

// ------------------------------------------------------------------ audio
function b64(buf) {
  let s = ""; const b = new Uint8Array(buf);
  for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
  return btoa(s);
}
function resampleTo16k(input, rate) {
  const ratio = rate / 16000, n = Math.floor(input.length / ratio), out = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}
async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
  const src = audioCtx.createMediaStreamSource(micStream);
  micNode = audioCtx.createScriptProcessor(4096, 1, 1);
  micNode.onaudioprocess = (ev) => {
    if (!ws || ws.readyState !== 1) return;
    const pcm = resampleTo16k(ev.inputBuffer.getChannelData(0), audioCtx.sampleRate);
    ws.send(JSON.stringify({ type: "audio", data: b64(pcm.buffer) }));
  };
  src.connect(micNode); micNode.connect(audioCtx.destination);
}
function playPcm(b64data) {
  const bin = atob(b64data), n = bin.length / 2, buf = audioCtx.createBuffer(1, n, 24000), ch = buf.getChannelData(0);
  for (let i = 0; i < n; i++) { let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); if (v >= 0x8000) v -= 0x10000; ch[i] = v / 0x8000; }
  const node = audioCtx.createBufferSource(); node.buffer = buf; node.connect(audioCtx.destination);
  playHead = Math.max(playHead, audioCtx.currentTime + 0.02); node.start(playHead); playHead += buf.duration;
  playing.push(node); node.onended = () => { playing = playing.filter((x) => x !== node); };
}
function stopPlayback() { playing.forEach((n) => { try { n.stop(); } catch (_) {} }); playing = []; playHead = 0; }

// ------------------------------------------------------------------ camera
async function startCamera() {
  camStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1440 } }, audio: false });
  $("video").srcObject = camStream;
  await $("video").play().catch(() => {});
}
function startFrames() {
  const v = $("video"), c = $("grab"), ctx = c.getContext("2d");
  clearInterval(frameTimer);
  frameTimer = setInterval(() => {
    if (!ws || ws.readyState !== 1 || !v.videoWidth || ws.bufferedAmount > 3e6) return;
    const s = Math.min(1, FRAME_W / Math.max(v.videoWidth, v.videoHeight));
    c.width = Math.round(v.videoWidth * s); c.height = Math.round(v.videoHeight * s);
    ctx.drawImage(v, 0, 0, c.width, c.height);
    c.toBlob(async (blob) => {
      if (!blob || !ws || ws.readyState !== 1) return;
      ws.send(JSON.stringify({ type: "frame", data: b64(await blob.arrayBuffer()), imu: { ...imu } }));
    }, "image/jpeg", 0.85);
  }, FRAME_MS);
}

// ------------------------------------------------------------------ UI
function note(t) { const li = document.createElement("li"); li.textContent = t; $("notices").prepend(li); }
const lines = {};
function transcript(who, text, final) {
  const key = who + (lines[who + "_open"] || 0);
  let el = lines[key];
  if (!el) { el = document.createElement("div"); el.className = who; $("transcript").appendChild(el); lines[key] = el; }
  el.textContent = (who === "user" ? "You: " : "Agent: ") + text;
  if (final) lines[who + "_open"] = (lines[who + "_open"] || 0) + 1;
  $("transcript").scrollTop = 1e9;
}
function renderInventory(inv) {
  const c = inv.counts || {};
  $("counts").textContent = `${c.spines || 0} books (${c.legible || 0} readable) · ${c.items || 0} items · ${c.shelf_units || 0} bookcases`;
  const ol = $("books"); ol.innerHTML = "";
  (inv.books || []).slice().reverse().slice(0, 120).forEach((b) => {
    const li = document.createElement("li");
    if (b.title) li.textContent = `${b.title}${b.author ? " — " + b.author : ""}`;
    else if (b.legible) li.textContent = `“${b.spine_text}” (identifying…)`;
    else { li.textContent = b.spine_text ? `partial: “${b.spine_text}”` : "unreadable spine (logged with size)"; li.className = "unread"; }
    ol.appendChild(li);
  });
  const ul = $("items"); ul.innerHTML = "";
  (inv.items || []).forEach((i) => { const li = document.createElement("li"); li.textContent = `${i.category}: ${i.description}`; ul.appendChild(li); });
}
function renderStatus(s) {
  const chips = $("chips"); chips.innerHTML = "";
  const labels = { blur: "blurry", fast: "too fast", glare: "glare", dark: "too dark" };
  (s.problems || []).forEach((p) => { const d = document.createElement("span"); d.className = "chip"; d.textContent = labels[p] || p; chips.appendChild(d); });
  if (s.too_far) { const d = document.createElement("span"); d.className = "chip"; d.textContent = "step closer"; chips.appendChild(d); }
  const m = document.createElement("span"); m.className = "chip" + (s.segment_has_marker ? " ok" : ""); m.textContent = s.segment_has_marker ? "marker ✓" : "marker not seen"; chips.appendChild(m);
}
function setSweeping(on) {
  sweeping = on;
  $("start").disabled = on; $("end").disabled = !on;
  $("conn").textContent = on ? "● recording sweep" : "connected"; $("conn").className = "pill " + (on ? "rec" : "on");
}

// ------------------------------------------------------------------ socket
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws?k=${encodeURIComponent(KEY)}`);
  ws.onopen = () => { $("conn").textContent = "connected"; $("conn").className = "pill on"; startFrames(); };
  ws.onclose = () => { $("conn").textContent = "offline"; $("conn").className = "pill off"; clearInterval(frameTimer); $("start").disabled = true; $("end").disabled = true; };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case "ready": $("start").disabled = false; $("locale").textContent = `${m.country} · ${m.currency}`; break;
      case "audio": playPcm(m.data); break;
      case "interrupted": stopPlayback(); break;
      case "transcript": transcript(m.who, m.text, m.final); break;
      case "inventory": renderInventory(m); break;
      case "status": renderStatus(m); break;
      case "notice": note(m.text); break;
      case "locale": $("locale").textContent = `${m.country} · ${m.currency}`; $("country").value = m.country; break;
      case "sweep": setSweeping(m.active); if (m.processing) { $("packet").hidden = false; $("packet").textContent = "Building the claim packet…"; } break;
      case "packet_ready": {
        const p = $("packet"); p.hidden = false; const s = m.summary || {};
        p.innerHTML = `<b>Packet ready</b> — ${s.books} books (${s.identified} identified), ${s.currency} ${s.books_replacement} replacement,
          ${s.review_items} review lines, built in ${s.time_to_packet_s}s<br>
          <a href="${m.report}" target="_blank">Report</a><a href="${m.json}" target="_blank">claim_packet.json</a><a href="${m.zip}">Download zip</a>`;
        $("start").disabled = true; $("end").disabled = true; break;
      }
      case "tool": if (m.name !== "get_sweep_status") note(`tool ${m.name}: ${JSON.stringify(m.result).slice(0, 120)}`); break;
      case "error": note("ERROR: " + m.message); break;
    }
  };
}

$("connect").onclick = async () => {
  $("connect").disabled = true;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    await enableSensors();
    if (!$("video").src) await startCamera();
    await startMic();
    connect();
  } catch (e) { note("Could not start: " + e); $("connect").disabled = false; }
};
$("start").onclick = () => ws && ws.send(JSON.stringify({ type: "control", action: "start_sweep" }));
$("end").onclick = () => ws && ws.send(JSON.stringify({ type: "control", action: "end_sweep" }));
$("country").onchange = (e) => ws && ws.send(JSON.stringify({ type: "control", action: "set_country", country: e.target.value }));
$("file").onchange = (e) => {   // desktop testing: use a recorded video as the camera
  const f = e.target.files[0]; if (!f) return;
  const v = $("video"); v.srcObject = null; v.src = URL.createObjectURL(f); v.loop = false; v.play();
};
