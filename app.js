import {
  DEFAULT_MODEL_PATH,
  CLASS_LABELS,
  DEFAULT_ROLES,
  SPECIFIC_GESTURE_IDS,
  loadYoloModel,
  runYoloInference
} from './yolo.js';

import {
  initMediaPipe,
  sendFrameToMediaPipe,
  getLatestMpResult
} from './mediapipe.js';

let classRoles = [...DEFAULT_ROLES];
try {
  const saved = JSON.parse(localStorage.getItem('safedrive_roles_v2'));
  if (Array.isArray(saved) && saved.length === DEFAULT_ROLES.length) {
    classRoles = saved;
  } else {
    localStorage.setItem('safedrive_roles_v2', JSON.stringify(DEFAULT_ROLES));
  }
} catch (e) {
  classRoles = [...DEFAULT_ROLES];
}

let alertLogs = [];
try {
  alertLogs = JSON.parse(localStorage.getItem('yolo_its_violation_photos') || '[]');
} catch (e) {
  alertLogs = [];
}

let soundMode = localStorage.getItem('safedrive_sound_mode') || 'mp3';
let confThreshold = parseFloat(localStorage.getItem('safedrive_conf_thresh')) || 0.25;
let alarmDelaySec = parseFloat(localStorage.getItem('safedrive_alarm_delay')) || 1.0;
let earThreshold = parseFloat(localStorage.getItem('safedrive_ear_thresh')) || 0.20;
let marThreshold = 0.55;

// Cập nhật thông số
window.addEventListener('storage', (e) => {
  if (e.key === 'safedrive_roles_v2') {
    try { classRoles = JSON.parse(e.newValue) || [...DEFAULT_ROLES]; } catch (_) {}
  } else if (e.key === 'safedrive_sound_mode') {
    soundMode = e.newValue || 'mp3';
  } else if (e.key === 'safedrive_conf_thresh') {
    confThreshold = parseFloat(e.newValue) || 0.25;
  } else if (e.key === 'safedrive_alarm_delay') {
    alarmDelaySec = parseFloat(e.newValue) || 1.0;
  } else if (e.key === 'safedrive_ear_thresh') {
    earThreshold = parseFloat(e.newValue) || 0.20;
  }
});

// DOM Elements
const video = document.getElementById('webcam');
const overlayCanvas = document.getElementById('overlay-canvas') || document.createElement('canvas');
const overlayCtx = overlayCanvas.getContext('2d');

let prepCanvas = document.getElementById('prep-canvas');
if (!prepCanvas) {
  prepCanvas = document.createElement('canvas');
  prepCanvas.width = 640;
  prepCanvas.height = 640;
}
const prepCtx = prepCanvas.getContext('2d', { willReadFrequently: true });
const projectSound = document.getElementById('project-sound');

const viewport = document.getElementById('viewport');
const flashOverlay = document.getElementById('flash-overlay');
const fpsPill = document.getElementById('fps-pill');
const timerPill = document.getElementById('timer-pill');
const violationBar = document.getElementById('violation-status-bar');

let isMonitoring = false;
let isInferring = false;
let rafId = null;

let isMirrored = true;
let useBGR = false;
let useMediaPipe = true;
let autoSnapshotEnabled = true;

let distractionStartTime = null;
let lastAlertLoggedAt = 0;
let totalFrames = 0;
let safeFrames = 0;
let sessionStartTime = Date.now();
let toastTimeout = null;

// Khởi tạo
loadYoloModel(DEFAULT_MODEL_PATH, 'best.onnx');
initMediaPipe();

// Brain
async function inferenceLoop() {
  if (!isMonitoring) return;

  if (isInferring) {
    rafId = requestAnimationFrame(inferenceLoop);
    return;
  }

  isInferring = true;
  const t0 = performance.now();
  let detections = [];
  let classScores = new Array(CLASS_LABELS.length).fill(0);

  try {
    if (video && video.readyState >= 2) {
      if (useMediaPipe) {
        await sendFrameToMediaPipe(video);
      }

      const yoloOut = await runYoloInference(
        video,
        prepCtx,
        overlayCanvas,
        useBGR,
        classRoles,
        confThreshold
      );
      detections = yoloOut.detections;
      classScores = yoloOut.classScores;
    }
  } catch (err) {
    console.error("Inference error:", err);
  } finally {
    isInferring = false;
  }

  const mpResult = getLatestMpResult();
  if (useMediaPipe && mpResult.hasFace) {
    const { ear, mar, pitchRatio, yawRatio } = mpResult;

    const isEyesClosed = ear < earThreshold;
    const isYawning = mar > marThreshold;
    const isHeadDown = pitchRatio > 1.42;
    const isHeadTurned = yawRatio < 0.42 || yawRatio > 2.35;

    if (isEyesClosed) {
      classScores[1] = Math.max(classScores[1], 0.95);
      detections.push({ classId: 1, score: 0.95, priority: 1, source: 'MediaPipe (EAR)', box: null });
    }
    if (isHeadDown) {
      classScores[3] = Math.max(classScores[3], 0.85);
      detections.push({ classId: 3, score: 0.85, priority: 3, source: 'MediaPipe (Pose)', box: null });
    }
    if (isHeadTurned) {
      classScores[9] = Math.max(classScores[9], 0.82);
      detections.push({ classId: 9, score: 0.82, priority: 3, source: 'MediaPipe (Pose)', box: null });
    }
    if (isYawning) {
      classScores[10] = Math.max(classScores[10], 0.80);
      detections.push({ classId: 10, score: 0.80, priority: 3, source: 'MediaPipe (MAR)', box: null });
    }
  }

  // FPS
  const inferMs = Math.round(performance.now() - t0);
  const fps = Math.min(60, Math.round(1000 / Math.max(16, inferMs)));
  if (fpsPill) fpsPill.textContent = `⚡ ${fps} FPS | ${inferMs} ms`;

  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  // Ưu tiên
  let activeDistraction = null;

  detections.forEach(det => {
    const role = classRoles[det.classId];
    const isFromMediaPipe = det.source && det.source.startsWith('MediaPipe');

    if (!det.priority) det.priority = 2;

    if ((role === 'danger' || isFromMediaPipe) && SPECIFIC_GESTURE_IDS.has(det.classId)) {
      if (!activeDistraction) {
        activeDistraction = det;
      } else if (det.priority < activeDistraction.priority) {
        activeDistraction = det;
      } else if (det.priority === activeDistraction.priority && det.score > activeDistraction.score) {
        activeDistraction = det;
      }
    }
  });

  // Thanh trạng thái
  if (violationBar) {
    if (activeDistraction) {
      const alertMessages = {
        1: "CẢNH BÁO: NGỦ GẬT!",
        3: "CẢNH BÁO: CÚI GỤC ĐẦU!",
        4: "CẢNH BÁO: ĐANG ĂN UỐNG!",
        6: "CẢNH BÁO: SỬ DỤNG ĐIỆN THOẠI!",
        7: "CẢNH BÁO: CHỈNH RADIO!",
        9: "CẢNH BÁO: LÀM VIỆC RIÊNG!",
        10: "CẢNH BÁO: MỆT MỎI!"
      };
      violationBar.innerHTML = alertMessages[activeDistraction.classId] || `🚨 CẢNH BÁO: ${CLASS_LABELS[activeDistraction.classId].toUpperCase()}`;
      violationBar.style.borderLeftColor = "var(--color-red)";
      violationBar.style.color = "var(--color-red)";
    } else if (useMediaPipe && !mpResult.hasFace) {
      violationBar.innerHTML = "Chưa phát hiện khuôn mặt";
      violationBar.style.borderLeftColor = "var(--color-grey-500)";
      violationBar.style.color = "var(--color-grey-500)";
    } else {
      violationBar.innerHTML = "AN TOÀN (Đang tập trung)";
      violationBar.style.borderLeftColor = "var(--color-black)";
      violationBar.style.color = "var(--color-black)";
    }
  }

  try {
    localStorage.setItem('safedrive_live_scores', JSON.stringify(classScores));
  } catch (_) {}

  updateSafetyState(activeDistraction);
  updateSessionClock();

  if (isMonitoring) {
    rafId = requestAnimationFrame(inferenceLoop);
  }
}

function updateSafetyState(distraction) {
  totalFrames++;

  if (distraction) {
    if (!distractionStartTime) distractionStartTime = performance.now();
    const elapsedSec = (performance.now() - distractionStartTime) / 1000;
    if (timerPill) timerPill.textContent = `Mất tập trung: ${elapsedSec.toFixed(1)}s`;

    const labelName = CLASS_LABELS[distraction.classId];
    const confPct = Math.round(distraction.score * 100);

    if (elapsedSec >= alarmDelaySec) {
      if (viewport) viewport.className = 'viewport-wrapper state-danger';
      const kpiAlarm = document.getElementById('kpi-alarm-state');
      if (kpiAlarm) {
        kpiAlarm.textContent = 'BÁO ĐỘNG!';
        kpiAlarm.style.color = 'var(--color-red)';
      }

      triggerAlarmSound(labelName);
      logViolationWithAutoPhoto(labelName, confPct, elapsedSec, distraction.source);
    } else {
      if (viewport) viewport.className = 'viewport-wrapper';
      const kpiAlarm = document.getElementById('kpi-alarm-state');
      if (kpiAlarm) {
        kpiAlarm.textContent = 'Đang kiểm tra';
        kpiAlarm.style.color = 'var(--color-black)';
      }
    }
  } else {
    safeFrames++;
    distractionStartTime = null;
    if (timerPill) timerPill.textContent = `Mất tập trung: 0.0s`;
    if (viewport) viewport.className = 'viewport-wrapper state-safe';
    const kpiAlarm = document.getElementById('kpi-alarm-state');
    if (kpiAlarm) {
      kpiAlarm.textContent = 'An toàn';
      kpiAlarm.style.color = 'var(--color-black)';
    }
  }

  const safetyScore = Math.round((safeFrames / totalFrames) * 100);
  const kpiScore = document.getElementById('kpi-score');
  if (kpiScore) kpiScore.textContent = `${safetyScore}%`;
}

// Chụp ảnh
function captureViolationPhoto(behaviorName, confPct, durationSec, sourceName) {
  const snapCanvas = document.createElement('canvas');
  const w = (video && video.videoWidth) || 640;
  const h = (video && video.videoHeight) || 480;
  snapCanvas.width = w;
  snapCanvas.height = h;
  const sCtx = snapCanvas.getContext('2d');

  sCtx.fillStyle = '#0B0C0E';
  sCtx.fillRect(0, 0, w, h);

  if (video && video.readyState >= 2) {
    sCtx.save();
    if (isMirrored) {
      sCtx.translate(w, 0);
      sCtx.scale(-1, 1);
    }
    sCtx.drawImage(video, 0, 0, w, h);
    sCtx.restore();
  }

  const timeStr = new Date().toLocaleString('vi-VN');
  sCtx.fillStyle = '#D91B24';
  sCtx.fillRect(0, h - 38, w, 38);
  sCtx.fillStyle = '#FFFFFF';
  sCtx.font = 'bold 13px Arial, sans-serif';
  sCtx.fillText(`VI PHẠM: ${behaviorName.toUpperCase()} (${confPct}% - ${sourceName}) | ${timeStr}`, 12, h - 14);

  if (flashOverlay) {
    flashOverlay.classList.add('active');
    setTimeout(() => flashOverlay.classList.remove('active'), 180);
  }

  return snapCanvas.toDataURL('image/jpeg', 0.82);
}

function saveLogsToStorage() {
  if (alertLogs.length > 25) alertLogs = alertLogs.slice(0, 25);
  try {
    localStorage.setItem('yolo_its_violation_photos', JSON.stringify(alertLogs));
  } catch (e) {
    alertLogs = alertLogs.slice(0, 10);
    try { localStorage.setItem('yolo_its_violation_photos', JSON.stringify(alertLogs)); } catch (_) {}
  }
}

function showSnapshotToast(photoDataUrl, behaviorName, timeStr) {
  const toast = document.getElementById('snapshot-toast');
  if (!toast) return;
  const imgEl = document.getElementById('toast-img');
  const descEl = document.getElementById('toast-desc');
  if (imgEl) imgEl.src = photoDataUrl;
  if (descEl) descEl.textContent = `${behaviorName} • ${timeStr}`;
  toast.classList.add('show');
  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toast.classList.remove('show'), 2800);
}

function logViolationWithAutoPhoto(behavior, confPct, durationSec, sourceName = 'AI') {
  const now = Date.now();
  if (now - lastAlertLoggedAt < 3500) return;
  lastAlertLoggedAt = now;

  const timeStr = new Date().toLocaleTimeString('vi-VN');
  const photoDataUrl = autoSnapshotEnabled
    ? captureViolationPhoto(behavior, confPct, durationSec, sourceName)
    : null;

  alertLogs.unshift({
    timeStr,
    behavior,
    confPct,
    sourceName,
    duration: `${durationSec.toFixed(1)}s`,
    photo: photoDataUrl
  });

  saveLogsToStorage();
  renderHistoryTableAndGallery();

  if (photoDataUrl) {
    showSnapshotToast(photoDataUrl, behavior, timeStr);
  }
}

function renderHistoryTableAndGallery() {
  const alertsEl = document.getElementById('kpi-alerts');
  if (alertsEl) alertsEl.textContent = alertLogs.length;

  const tbody = document.getElementById('log-tbody');
  if (tbody) {
    if (alertLogs.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="5" style="text-align:center; color: var(--color-grey-500); padding: 18px;">
            Chưa có vi phạm nào. Ảnh bằng chứng sẽ tự động chụp khi phát hiện mất tập trung.
          </td>
        </tr>
      `;
    } else {
      tbody.innerHTML = alertLogs.map((item, idx) => `
        <tr>
          <td>
            ${item.photo
              ? `<img src="${item.photo}" class="log-thumb" onclick="openLightbox(${idx})">`
              : `<span>Không chụp</span>`}
          </td>
          <td style="font-weight:700;">${item.timeStr}</td>
          <td style="color: var(--color-red); font-weight: 700;">${item.behavior}</td>
          <td>${item.sourceName || 'AI'} (${item.confPct}%)</td>
          <td>
            ${item.photo
              ? `<button onclick="openLightbox(${idx})" style="padding:4px 8px;font-size:0.74rem;">🔍 Xem</button>`
              : `-`}
          </td>
        </tr>
      `).join('');
    }
  }

  const galleryGrid = document.getElementById('gallery-grid');
  if (galleryGrid) {
    if (alertLogs.length === 0) {
      galleryGrid.innerHTML = `<p style="color:var(--color-grey-500); grid-column: 1/-1; text-align:center; padding: 32px;">Chưa có ảnh vi phạm nào được lưu.</p>`;
    } else {
      galleryGrid.innerHTML = alertLogs.map((item, idx) => `
        <div class="gallery-card">
          <img src="${item.photo}" alt="${item.behavior}" onclick="openLightbox(${idx})">
          <div class="gallery-meta">
            <div style="font-weight:700; color:var(--color-red);">${item.behavior}</div>
            <div style="color:var(--color-grey-700);">${item.timeStr} • ${item.sourceName || 'AI'}</div>
            <div class="gallery-actions">
              <button onclick="openLightbox(${idx})">Xem</button>
              <button class="primary" onclick="downloadPhoto(${idx})">⬇Tải</button>
              <button class="danger-btn" onclick="deletePhoto(${idx})"></button>
            </div>
          </div>
        </div>
      `).join('');
    }
  }
}

window.openLightbox = function(index) {
  const item = alertLogs[index];
  if (!item || !item.photo) return;
  document.getElementById('lightbox-title').textContent = `BẰNG CHỨNG VI PHẠM: ${item.behavior}`;
  document.getElementById('lightbox-img').src = item.photo;
  document.getElementById('lightbox-meta').textContent = `Thời gian: ${item.timeStr} | Nguồn: ${item.sourceName || 'AI'} (${item.confPct}%) | Kéo dài: ${item.duration}`;
  document.getElementById('download-single-btn').onclick = () => downloadPhoto(index);
  document.getElementById('lightbox-modal').classList.add('active');
};

window.downloadPhoto = function(index) {
  const item = alertLogs[index];
  if (!item || !item.photo) return;
  const a = document.createElement('a');
  a.href = item.photo;
  a.download = `ViPham_${item.timeStr.replace(/[:\s\/]/g, '_')}.jpg`;
  a.click();
};

window.deletePhoto = function(index) {
  alertLogs.splice(index, 1);
  saveLogsToStorage();
  renderHistoryTableAndGallery();
};

renderHistoryTableAndGallery();

// Báo động
let audioCtx = null;
let lastBeepTime = 0;
let lastVoiceTime = 0;

function ensureAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass();
  }
  if (audioCtx.state === 'suspended') audioCtx.resume();
}

function playFallbackSiren(mode) {
  ensureAudioContext();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  const currentTime = audioCtx.currentTime;

  if (mode === 'siren') {
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(750, currentTime);
    osc.frequency.linearRampToValueAtTime(1350, currentTime + 0.16);
    osc.frequency.linearRampToValueAtTime(750, currentTime + 0.32);
  } else {
    osc.type = 'square';
    osc.frequency.setValueAtTime(1100, currentTime);
  }

  gain.gain.setValueAtTime(0.3, currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, currentTime + 0.32);

  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start(currentTime);
  osc.stop(currentTime + 0.32);
}

function triggerAlarmSound(behaviorName) {
  const mode = localStorage.getItem('safedrive_sound_mode') || soundMode || 'mp3';
  if (mode === 'mute') return;

  const now = performance.now();

  if (mode === 'mp3' && projectSound) {
    if (now - lastBeepTime < 700) return;
    lastBeepTime = now;
    projectSound.currentTime = 0;
    projectSound.play().catch(() => playFallbackSiren('siren'));
    return;
  }

  if (mode === 'voice') {
    if (now - lastVoiceTime > 3200 && 'speechSynthesis' in window) {
      lastVoiceTime = now;
      window.speechSynthesis.cancel();
      const cleanName = behaviorName.replace(/\s*\([^)]*\)/g, '');
      const utter = new SpeechSynthesisUtterance(`Cảnh báo! Phát hiện ${cleanName}. Vui lòng tập trung lái xe!`);
      utter.lang = 'vi-VN';
      utter.rate = 1.15;
      window.speechSynthesis.speak(utter);
    }
    return;
  }

  if (now - lastBeepTime < 350) return;
  lastBeepTime = now;
  playFallbackSiren(mode);
}

// Camera
async function startCamera() {
  try {
    ensureAudioContext();
    if (rafId) cancelAnimationFrame(rafId);
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(t => t.stop());
    }
    video.src = '';

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
      });
    } catch (_) {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }

    video.srcObject = stream;
    await video.play();
    isMonitoring = true;

    const btn = document.getElementById('toggle-cam-btn');
    if (btn) {
      btn.textContent = 'TẠM DỪNG';
      btn.className = 'primary';
    }
    rafId = requestAnimationFrame(inferenceLoop);
  } catch (err) {
    if (violationBar) {
      violationBar.innerHTML = "Không thể mở camera: " + err.message;
      violationBar.style.color = "var(--color-red)";
    }
  }
}

document.getElementById('toggle-cam-btn')?.addEventListener('click', () => {
  ensureAudioContext();
  const btn = document.getElementById('toggle-cam-btn');
  if (isMonitoring) {
    isMonitoring = false;
    if (rafId) cancelAnimationFrame(rafId);
    if (btn) {
      btn.textContent = 'TIẾP TỤC';
      btn.className = 'danger-btn';
    }
    if (violationBar) {
      violationBar.innerHTML = "Đang tạm dừng";
      violationBar.style.color = "var(--color-grey-500)";
    }
  } else {
    startCamera();
  }
});

document.getElementById('view-all-photos-btn')?.addEventListener('click', () => {
  document.getElementById('gallery-modal')?.classList.add('active');
});
document.getElementById('close-gallery-btn')?.addEventListener('click', () => {
  document.getElementById('gallery-modal')?.classList.remove('active');
});
document.getElementById('close-lightbox-btn')?.addEventListener('click', () => {
  document.getElementById('lightbox-modal')?.classList.remove('active');
});
document.getElementById('clear-gallery-btn')?.addEventListener('click', () => {
  alertLogs = [];
  saveLogsToStorage();
  renderHistoryTableAndGallery();
});

function updateSessionClock() {
  const sec = Math.floor((Date.now() - sessionStartTime) / 1000);
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = String(sec % 60).padStart(2, '0');
  const durEl = document.getElementById('kpi-duration');
  if (durEl) durEl.textContent = `${m}:${s}`;
}

document.getElementById('export-csv-btn')?.addEventListener('click', () => {
  if (alertLogs.length === 0) return;
  const rows = [
    ['Thoi gian', 'Hanh vi vi pham', 'Nguon AI', 'Do tin cay (%)', 'Thoi luong'],
    ...alertLogs.map(r => [r.timeStr, r.behavior, r.sourceName || 'AI', r.confPct, r.duration])
  ];
  const csvContent = 'data:text/csv;charset=utf-8,\uFEFF' + rows.map(e => e.join(',')).join('\n');
  const link = document.createElement('a');
  link.setAttribute('href', encodeURI(csvContent));
  link.setAttribute('download', 'Nhat ky vi pham.csv');
  link.click();
});

document.getElementById('clear-log-btn')?.addEventListener('click', () => {
  alertLogs = [];
  saveLogsToStorage();
  renderHistoryTableAndGallery();
});