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
  getLatestMpResult,
  drawMediaPipeLandmarks
} from './mediapipe.js';

// Trạng thái lưu trữ cục bộ
let classRoles = JSON.parse(localStorage.getItem('yolov12_its_roles')) || [...DEFAULT_ROLES];
let alertLogs = JSON.parse(localStorage.getItem('yolo_its_violation_photos') || '[]');

// DOM Elements
const video = document.getElementById('webcam');
const overlayCanvas = document.getElementById('overlay-canvas');
const overlayCtx = overlayCanvas.getContext('2d');
const prepCanvas = document.getElementById('prep-canvas');
const prepCtx = prepCanvas.getContext('2d', { willReadFrequently: true });
const projectSound = document.getElementById('project-sound');

const viewport = document.getElementById('viewport');
const flashOverlay = document.getElementById('flash-overlay');
const modelDot = document.getElementById('model-dot');
const modelStatusText = document.getElementById('model-status-text');
const mpDot = document.getElementById('mp-dot');
const mpStatusText = document.getElementById('mp-status-text');

const alertBanner = document.getElementById('alert-banner');
const alertText = document.getElementById('alert-text');
const alertConf = document.getElementById('alert-conf');
const fpsPill = document.getElementById('fps-pill');
const timerPill = document.getElementById('timer-pill');

const violationBar = document.getElementById('violation-status-bar');
const earDisplay = document.getElementById('ear-display');
const marDisplay = document.getElementById('mar-display');
const poseDisplay = document.getElementById('pose-display');
// Biến trạng thái vận hành
let isMonitoring = false;
let isInferring = false;
let rafId = null;

let isMirrored = true;
let useBGR = false;
let useMediaPipe = true;
let autoSnapshotEnabled = true;

let confThreshold = 0.25;
let alarmDelaySec = 1.0;
let earThreshold = 0.20;
let marThreshold = 0.55;

let distractionStartTime = null;
let lastAlertLoggedAt = 0;
let totalFrames = 0;
let safeFrames = 0;
let sessionStartTime = Date.now();
let toastTimeout = null;

// =========================================================================
// 1. KHỞI TẠO 2 MODULE AI (YOLOv12n & MEDIAPIPE)
// =========================================================================
function handleYoloStatus(state, message) {
  const statusText = document.getElementById('model-status-text');
  const modelDot = document.getElementById('model-dot');
  
  if (statusText) statusText.textContent = message; // Chỉ chạy khi thẻ tồn tại
  if (modelDot) {
    modelDot.classList.remove('online', 'danger-dot');
    if (state === 'online') modelDot.classList.add('online');
    if (state === 'error') modelDot.classList.add('danger-dot');
  }
}

function handleMpStatus(state, message) {
  const mpStatusText = document.getElementById('mp-status-text');
  const mpDot = document.getElementById('mp-dot');
  
  if (mpStatusText) mpStatusText.textContent = message;
  if (mpDot) {
    mpDot.classList.remove('online', 'danger-dot');
    if (state === 'online') mpDot.classList.add('online');
    if (state === 'error') mpDot.classList.add('danger-dot');
  }
}
loadYoloModel(DEFAULT_MODEL_PATH, 'best.onnx', handleYoloStatus);
initMediaPipe(handleMpStatus);

// =========================================================================
// 2. HIỂN THỊ 11 THANH NHÃN HÀNH VI
// =========================================================================
function getRoleLabel(role) {
  if (role === 'safe') return 'An toàn';
  if (role === 'danger') return 'Cảnh báo';
  return 'Bỏ qua';
}

function getRoleColor(role) {
  if (role === 'safe') return 'var(--color-black)';
  if (role === 'danger') return 'var(--color-red)';
  return 'var(--color-grey-500)';
}

window.toggleClassRole = function(idx) {
  const order = ['danger', 'safe', 'ignore'];
  const nextIdx = (order.indexOf(classRoles[idx]) + 1) % order.length;
  classRoles[idx] = order[nextIdx];
  localStorage.setItem('yolov12_its_roles', JSON.stringify(classRoles));
  document.getElementById('class-bars').innerHTML = '';
  renderClassBars();
};

document.getElementById('reset-roles-btn').addEventListener('click', () => {
  classRoles = [...DEFAULT_ROLES];
  localStorage.setItem('yolov12_its_roles', JSON.stringify(classRoles));
  document.getElementById('class-bars').innerHTML = '';
  renderClassBars();
});

function renderClassBars(scoresArray = new Array(CLASS_LABELS.length).fill(0)) {
  const container = document.getElementById('class-bars');
  if (container.children.length !== CLASS_LABELS.length) {
    container.innerHTML = CLASS_LABELS.map((name, idx) => {
      const role = classRoles[idx] || 'danger';
      return `
        <div class="class-item">
          <div class="class-item-head">
            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${idx}. ${name}">${idx}. ${name}</span>
            <div style="display:flex;align-items:center;gap:6px;">
              <span id="prob-text-${idx}" style="font-weight:700;">0%</span>
              <span class="role-pill role-${role}" onclick="toggleClassRole(${idx})">
                ${getRoleLabel(role)}
              </span>
            </div>
          </div>
          <div class="progress-track">
            <div id="prob-bar-${idx}" class="progress-fill" style="background: ${getRoleColor(role)}"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  CLASS_LABELS.forEach((_, idx) => {
    const pct = Math.min(100, Math.round((scoresArray[idx] || 0) * 100));
    const txtEl = document.getElementById(`prob-text-${idx}`);
    const barEl = document.getElementById(`prob-bar-${idx}`);
    if (txtEl) txtEl.textContent = `${pct}%`;
    if (barEl) barEl.style.width = `${pct}%`;
  });
}

renderClassBars();

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
    if (video.readyState >= 2) {
      // 1. Gọi Module MediaPipe
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

  // Lấy các phần tử giao diện
  const violationBar = document.getElementById('violation-status-bar');
  const earDisplay = document.getElementById('ear-display');
  const marDisplay = document.getElementById('mar-display');
  const poseDisplay = document.getElementById('pose-display');

  // 3. Hợp nhất số liệu từ MediaPipe vào bảng 11 nhãn
  const mpResult = getLatestMpResult();
  if (useMediaPipe && mpResult.hasFace) {
    const { ear, mar, pitchRatio, yawRatio } = mpResult;
    if (earDisplay) earDisplay.textContent = ear.toFixed(2);
    if (marDisplay) marDisplay.textContent = mar.toFixed(2);

    const isEyesClosed = ear < earThreshold;
    const isYawning = mar > marThreshold;
    const isHeadDown = pitchRatio > 1.42;
    const isHeadTurned = yawRatio < 0.42 || yawRatio > 2.35;

    if (earDisplay) earDisplay.className = isEyesClosed ? 'val alert' : 'val';
    if (marDisplay) marDisplay.className = isYawning ? 'val alert' : 'val';

    // Đẩy lỗi của MediaPipe vào mảng (Kèm mức độ ưu tiên: Nhắm mắt = 1, Tư thế = 3)
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
  } else {
    if (earDisplay) earDisplay.textContent = "--";
    if (marDisplay) marDisplay.textContent = "--";
    if (poseDisplay) poseDisplay.textContent = useMediaPipe ? "Chưa phát hiện mặt" : "Đã tắt MediaPipe";
  }

  // 4. Vẽ kết quả & Xét quyền ưu tiên cảnh báo (YOLO Đồ vật > MediaPipe Tư thế)
  const inferMs = Math.round(performance.now() - t0);
  const fps = Math.min(60, Math.round(1000 / Math.max(16, inferMs)));
  const fpsPill = document.getElementById('fps-pill');
  if (fpsPill) fpsPill.textContent = `⚡ ${fps} FPS | ${inferMs} ms`;

  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (useMediaPipe && mpResult.hasFace && mpResult.landmarks) {
    drawMediaPipeLandmarks(overlayCtx, overlayCanvas, mpResult.landmarks, isMirrored);
  }

  let activeDistraction = null;

  detections.forEach(det => {
    const role = classRoles[det.classId];
    const isFromMediaPipe = det.source && det.source.startsWith('MediaPipe');

    // Gán cấp độ ưu tiên cho YOLO (Ưu tiên 2: Cao hơn Tư thế đầu/Ngáp của MediaPipe)
    if (!det.priority) {
      det.priority = 2; 
    }

    if ((role === 'danger' || isFromMediaPipe) && SPECIFIC_GESTURE_IDS.has(det.classId)) {
      if (!activeDistraction) {
        activeDistraction = det;
      } else {
        // LUẬT ƯU TIÊN: Số priority nhỏ hơn sẽ được ưu tiên chọn trước!
        // (1: Nhắm mắt -> 2: Điện thoại/Uống nước/Radio -> 3: Cúi đầu/Quay mặt/Ngáp)
        if (det.priority < activeDistraction.priority) {
          activeDistraction = det;
        } else if (det.priority === activeDistraction.priority && det.score > activeDistraction.score) {
          activeDistraction = det;
        }
      }
    }

    // Vẽ Bounding Box của YOLO
    if (det.box) {
      let [x, y, w, h] = det.box;
      if (isMirrored) {
        x = overlayCanvas.width - x - w;
      }

      const bgColor = role === 'safe' ? '#0B0C0E' : '#D91B24';
      overlayCtx.strokeStyle = bgColor;
      overlayCtx.lineWidth = 3;
      overlayCtx.strokeRect(x, y, w, h);

      const tag = `${CLASS_LABELS[det.classId]} (${Math.round(det.score * 100)}%)`;
      overlayCtx.font = 'bold 13px Arial, sans-serif';
      const textW = overlayCtx.measureText(tag).width + 14;
      overlayCtx.fillStyle = bgColor;
      overlayCtx.fillRect(x, Math.max(0, y - 24), textW, 24);
      overlayCtx.fillStyle = '#FFFFFF';
      overlayCtx.fillText(tag, x + 7, Math.max(16, y - 7));
    }
  });

  // 5. Cập nhật Thanh Trạng Thái Hợp Nhất (Hiển thị cả lỗi YOLO lẫn MediaPipe)
  if (violationBar) {
    if (activeDistraction) {
      const alertMessages = {
        1: "🚨 CẢNH BÁO: NHẮM MẮT / NGỦ GẬT!",
        3: "🚨 CẢNH BÁO: CÚI GỤC ĐẦU!",
        4: "🚨 CẢNH BÁO: ĐANG ĂN / UỐNG NƯỚC!",
        6: "🚨 CẢNH BÁO: SỬ DỤNG ĐIỆN THOẠI!",
        7: "🚨 CẢNH BÁO: CHỈNH RADIO / MÀN HÌNH!",
        9: "🚨 CẢNH BÁO: QUAY MẶT / MẤT TẬP TRUNG!",
        10: "⚠️ CẢNH BÁO: MỆT MỎI / NGÁP NGỦ!"
      };
      violationBar.innerHTML = alertMessages[activeDistraction.classId] || `🚨 CẢNH BÁO: ${CLASS_LABELS[activeDistraction.classId].toUpperCase()}`;
      violationBar.style.borderLeftColor = "var(--color-red)";
      violationBar.style.color = "var(--color-red)";
    } else if (useMediaPipe && !mpResult.hasFace) {
      violationBar.innerHTML = "⚠️ Chưa phát hiện khuôn mặt";
      violationBar.style.borderLeftColor = "var(--color-grey-500)";
      violationBar.style.color = "var(--color-grey-500)";
    } else {
      violationBar.innerHTML = "✅ AN TOÀN (Đang tập trung)";
      violationBar.style.borderLeftColor = "var(--color-black)";
      violationBar.style.color = "var(--color-black)";
    }
  }

  renderClassBars(classScores);
  updateSafetyState(activeDistraction, detections[0]);
  updateSessionClock();

  if (isMonitoring) {
    rafId = requestAnimationFrame(inferenceLoop);
  }
}

function updateSafetyState(distraction, topDetection) {
  totalFrames++;

  if (distraction) {
    if (!distractionStartTime) distractionStartTime = performance.now();
    const elapsedSec = (performance.now() - distractionStartTime) / 1000;
    timerPill.textContent = `⏱️ Mất tập trung: ${elapsedSec.toFixed(1)}s`;

    const labelName = CLASS_LABELS[distraction.classId];
    const confPct = Math.round(distraction.score * 100);
    alertConf.textContent = `${confPct}% • ${distraction.source}`;

    if (elapsedSec >= alarmDelaySec) {
      viewport.className = 'viewport-wrapper state-danger';
      alertBanner.className = 'alert-banner danger';
      alertText.textContent = `CẢNH BÁO VI PHẠM: ${labelName.toUpperCase()} (${elapsedSec.toFixed(1)}s)`;
      document.getElementById('kpi-alarm-state').textContent = 'BÁO ĐỘNG!';
      document.getElementById('kpi-alarm-state').style.color = 'var(--color-red)';

      triggerAlarmSound(labelName);
      logViolationWithAutoPhoto(labelName, confPct, elapsedSec, distraction.source);
    } else {
      viewport.className = 'viewport-wrapper';
      alertBanner.className = 'alert-banner warning';
      alertText.textContent = `⚠️ Phát hiện: ${labelName} (${elapsedSec.toFixed(1)}s / ${alarmDelaySec}s)`;
      document.getElementById('kpi-alarm-state').textContent = 'Đang kiểm tra';
      document.getElementById('kpi-alarm-state').style.color = 'var(--color-black)';
    }
  } else {
    safeFrames++;
    distractionStartTime = null;
    timerPill.textContent = `⏱️ Mất tập trung: 0.0s`;
    viewport.className = 'viewport-wrapper state-safe';
    alertBanner.className = 'alert-banner safe';
    alertText.textContent = `Trạng thái: Tài xế đang tập trung lái xe an toàn`;
    alertConf.textContent = topDetection ? `${Math.round(topDetection.score * 100)}%` : 'An toàn';
    document.getElementById('kpi-alarm-state').textContent = 'An toàn';
    document.getElementById('kpi-alarm-state').style.color = 'var(--color-black)';
  }

  const safetyScore = Math.round((safeFrames / totalFrames) * 100);
  document.getElementById('kpi-score').textContent = `${safetyScore}%`;
}

// =========================================================================
// 4. CHỤP ẢNH BẰNG CHỨNG TỰ ĐỘNG & QUẢN LÝ LỊCH SỬ
// =========================================================================
function captureViolationPhoto(behaviorName, confPct, durationSec, sourceName) {
  const snapCanvas = document.createElement('canvas');
  const w = overlayCanvas.width || 640;
  const h = overlayCanvas.height || 480;
  snapCanvas.width = w;
  snapCanvas.height = h;
  const sCtx = snapCanvas.getContext('2d');

  sCtx.fillStyle = '#0B0C0E';
  sCtx.fillRect(0, 0, w, h);

  if (video.readyState >= 2) {
    sCtx.save();
    if (isMirrored) {
      sCtx.translate(w, 0);
      sCtx.scale(-1, 1);
    }
    sCtx.drawImage(video, 0, 0, w, h);
    sCtx.restore();
  }

  sCtx.drawImage(overlayCanvas, 0, 0, w, h);

  const timeStr = new Date().toLocaleString('vi-VN');
  sCtx.fillStyle = '#D91B24';
  sCtx.fillRect(0, h - 38, w, 38);
  sCtx.fillStyle = '#FFFFFF';
  sCtx.font = 'bold 13px Arial, sans-serif';
  sCtx.fillText(`🚨 VI PHẠM: ${behaviorName.toUpperCase()} (${confPct}% - ${sourceName}) | ${timeStr}`, 12, h - 14);

  flashOverlay.classList.add('active');
  setTimeout(() => flashOverlay.classList.remove('active'), 180);

  return snapCanvas.toDataURL('image/jpeg', 0.82);
}

function saveLogsToStorage() {
  if (alertLogs.length > 25) alertLogs = alertLogs.slice(0, 25);
  try {
    localStorage.setItem('yolo_its_violation_photos', JSON.stringify(alertLogs));
  } catch (e) {
    alertLogs = alertLogs.slice(0, 10);
    localStorage.setItem('yolo_its_violation_photos', JSON.stringify(alertLogs));
  }
}

function showSnapshotToast(photoDataUrl, behaviorName, timeStr) {
  const toast = document.getElementById('snapshot-toast');
  document.getElementById('toast-img').src = photoDataUrl;
  document.getElementById('toast-desc').textContent = `${behaviorName} • ${timeStr}`;
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
  // Cập nhật số liệu an toàn (kiểm tra phần tử tồn tại trước khi gán)
  const alertsEl = document.getElementById('kpi-alerts');
  const galleryCountEl = document.getElementById('gallery-count');
  if (alertsEl) alertsEl.textContent = alertLogs.length;
  if (galleryCountEl) galleryCountEl.textContent = alertLogs.length;

  // Render bảng lịch sử vi phạm
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

  // Render kho ảnh gallery (Đã gộp đúng dòng chuỗi thời gian và nguồn AI để tránh lỗi ngắt dòng)
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
            <div style="color:var(--color-grey-700);">🕒 ${item.timeStr} • ${item.sourceName || 'AI'}</div>
            <div class="gallery-actions">
              <button onclick="openLightbox(${idx})">🔍 Xem</button>
              <button class="primary" onclick="downloadPhoto(${idx})">⬇️ Tải</button>
              <button class="danger-btn" onclick="deletePhoto(${idx})">🗑️</button>
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
  document.getElementById('lightbox-title').textContent = `🚨 BẰNG CHỨNG VI PHẠM: ${item.behavior}`;
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

// =========================================================================
// 5. HỆ THỐNG ÂM THANH BÁO ĐỘNG
// =========================================================================
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
  const mode = document.getElementById('sound-type').value;
  if (mode === 'mute') return;

  const now = performance.now();

  if (mode === 'mp3') {
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

document.getElementById('test-sound-btn').addEventListener('click', () => {
  triggerAlarmSound("Mất tập trung");
});

// =========================================================================
// 6. SỰ KIỆN NÚT BẤM & ĐIỀU KHIỂN CAMERA
// =========================================================================
async function startCamera() {
  try {
    ensureAudioContext();
    if (rafId) cancelAnimationFrame(rafId);
    const facingMode = 'user';
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(t => t.stop());
    }
    video.src = '';
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    isMonitoring = true;
    const btn = document.getElementById('toggle-cam-btn');
    btn.textContent = 'TẠM DỪNG';
    btn.className = 'primary';
    rafId = requestAnimationFrame(inferenceLoop);
  } catch (err) {
    alertText.textContent = "Không thể mở Camera: " + err.message;
  }
}

document.getElementById('toggle-cam-btn').addEventListener('click', () => {
  ensureAudioContext();
  const btn = document.getElementById('toggle-cam-btn');
  if (isMonitoring) {
    isMonitoring = false;
    if (rafId) cancelAnimationFrame(rafId);
    btn.textContent = 'TIẾP TỤC';
    btn.className = 'danger-btn';
    alertText.textContent = '⏸️ Đã tạm dừng hệ thống giám sát AI';
  } else {
    startCamera();
  }
});

document.getElementById('view-all-photos-btn').addEventListener('click', () => {
  document.getElementById('gallery-modal').classList.add('active');
});
document.getElementById('close-gallery-btn').addEventListener('click', () => {
  document.getElementById('gallery-modal').classList.remove('active');
});
document.getElementById('close-lightbox-btn').addEventListener('click', () => {
  document.getElementById('lightbox-modal').classList.remove('active');
});
document.getElementById('clear-gallery-btn').addEventListener('click', () => {
  alertLogs = [];
  saveLogsToStorage();
  renderHistoryTableAndGallery();
});

document.getElementById('alarm-delay-slider').addEventListener('input', (e) => {
  alarmDelaySec = parseFloat(e.target.value);
  document.getElementById('delay-val').textContent = `${alarmDelaySec.toFixed(1)} giây`;
});
document.getElementById('conf-slider').addEventListener('input', (e) => {
  confThreshold = parseInt(e.target.value, 10) / 100;
  document.getElementById('conf-val').textContent = `${e.target.value}%`;
});
document.getElementById('ear-slider').addEventListener('input', (e) => {
  earThreshold = parseFloat(e.target.value);
  document.getElementById('ear-thresh-val').textContent = earThreshold.toFixed(2);
});

function updateSessionClock() {
  const sec = Math.floor((Date.now() - sessionStartTime) / 1000);
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = String(sec % 60).padStart(2, '0');
  document.getElementById('kpi-duration').textContent = `${m}:${s}`;
}

document.getElementById('export-csv-btn').addEventListener('click', () => {
  if (alertLogs.length === 0) return;
  const rows = [
    ['Thoi gian', 'Hanh vi vi pham', 'Nguon AI', 'Do tin cay (%)', 'Thoi luong'],
    ...alertLogs.map(r => [r.timeStr, r.behavior, r.sourceName || 'AI', r.confPct, r.duration])
  ];
  const csvContent = 'data:text/csv;charset=utf-8,\uFEFF' + rows.map(e => e.join(',')).join('\n');
  const link = document.createElement('a');
  link.setAttribute('href', encodeURI(csvContent));
  link.setAttribute('download', 'nhat_ky_canh_bao_safedrive.csv');
  link.click();
});

document.getElementById('clear-log-btn').addEventListener('click', () => {
  alertLogs = [];
  saveLogsToStorage();
  renderHistoryTableAndGallery();
});