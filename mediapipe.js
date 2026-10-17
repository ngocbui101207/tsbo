const FaceMesh = window.FaceMesh;

let faceMesh = null;
let latestMpResult = {
  hasFace: false,
  ear: 0.30,
  mar: 0.10,
  pitchRatio: 1.0,
  yawRatio: 1.0,
  landmarks: null
};

//Khoảng cách Euclid 2D giữa 2 điểm mốc
function dist2D(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

//Tỷ lệ mở mắt [góc ngoài, mí trên 1, mí trên 2, góc trong, mí dưới 2, mí dưới 1]
function calculateEAR(lm, indices) {
  const v1 = dist2D(lm[indices[1]], lm[indices[5]]);
  const v2 = dist2D(lm[indices[2]], lm[indices[4]]);
  const h  = dist2D(lm[indices[0]], lm[indices[3]]);
  return (v1 + v2) / (2.0 * h + 1e-6);
}

// Tỷ lệ mở miệng
function calculateMAR(lm) {
  const v1 = dist2D(lm[81], lm[178]);
  const v2 = dist2D(lm[13], lm[14]);
  const v3 = dist2D(lm[311], lm[402]);
  const h  = dist2D(lm[78], lm[308]);
  return (v1 + v2 + v3) / (3.0 * h + 1e-6);
}

// Khởi tạo Google MediaPipe Face Mesh (478 điểm 3D)
export async function initMediaPipe(onStatusChange) {
  try {
    faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    faceMesh.onResults((results) => {
      if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
        latestMpResult = {
          hasFace: false,
          ear: 0,
          mar: 0,
          pitchRatio: 1,
          yawRatio: 1,
          landmarks: null
        };
        return;
      }

      const lm = results.multiFaceLandmarks[0];
      const leftEAR = calculateEAR(lm, [362, 385, 387, 263, 373, 380]);
      const rightEAR = calculateEAR(lm, [33, 160, 158, 133, 153, 144]);
      const avgEAR = (leftEAR + rightEAR) / 2.0;
      const mar = calculateMAR(lm);

      // Tính tỷ lệ quay ngang (Yaw) và cúi đầu (Pitch)
      const nose = lm[1];
      const forehead = lm[10];
      const chin = lm[152];
      const leftCheek = lm[234];
      const rightCheek = lm[454];

      const distLeft = Math.abs(nose.x - leftCheek.x);
      const distRight = Math.abs(rightCheek.x - nose.x);
      const yawRatio = distLeft / (distRight + 1e-5);

      const distUp = Math.abs(nose.y - forehead.y);
      const distDown = Math.abs(chin.y - nose.y);
      const pitchRatio = distUp / (distDown + 1e-5);

      latestMpResult = {
        hasFace: true,
        ear: avgEAR,
        mar: mar,
        pitchRatio,
        yawRatio,
        landmarks: lm
      };
    });

    if (onStatusChange) onStatusChange('online', 'MediaPipe: Hoạt động');
  } catch (e) {
    console.warn("MediaPipe init error:", e);
    if (onStatusChange) onStatusChange('error', 'MediaPipe: Lỗi nạp');
  }
}

export async function sendFrameToMediaPipe(video) {
  if (faceMesh && video.readyState >= 2) {
    await faceMesh.send({ image: video });
  }
  return latestMpResult;
}

export function getLatestMpResult() {
  return latestMpResult;
}

// Vẽ các điểm mốc mắt, mũi, miệng, cằm lên màn hình giám sát
export function drawMediaPipeLandmarks(overlayCtx, overlayCanvas, lm, isMirrored) {
  if (!lm) return;
  const w = overlayCanvas.width;
  const h = overlayCanvas.height;

  overlayCtx.save();
  const keyPoints = [33, 133, 362, 263, 13, 14, 78, 308, 1, 152];
  overlayCtx.fillStyle = '#FFFFFF';
  overlayCtx.strokeStyle = '#D91B24';
  overlayCtx.lineWidth = 1.5;

  keyPoints.forEach(idx => {
    const pt = lm[idx];
    const px = (isMirrored ? (1 - pt.x) : pt.x) * w;
    const py = pt.y * h;
    overlayCtx.beginPath();
    overlayCtx.arc(px, py, 2.5, 0, Math.PI * 2);
    overlayCtx.fill();
    overlayCtx.stroke();
  });
  overlayCtx.restore();
}