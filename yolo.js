const ort = window.ort;

export const DEFAULT_MODEL_PATH = './best.onnx';
export const INPUT_SIZE = 640;

export const CLASS_LABELS = [
  "Nhãn 0 (Bỏ qua)",                 // 0: '0'
  "Nhắm mắt (close_eye)",            // 1: 'close_eye'
  "Lái xe an toàn (correct)",        // 2: 'correct'
  "Ngủ gật / Gục đầu (dozing)",      // 3: 'dozing'
  "Uống nước / Ăn uống (drinking)",  // 4: 'drinking'
  "Khuôn mặt bình thường (face)",    // 5: 'face'
  "Dùng điện thoại (mobile)",        // 6: 'mobile'
  "Chỉnh Radio / Màn hình (radio)",  // 7: 'radio'
  "Nói chuyện (talking)",            // 8: 'talking'
  "Làm việc riêng / Quay đầu (work)",// 9: 'work'
  "Ngáp / Mệt mỏi (yawn)"            // 10: 'yawn'
];

export const DEFAULT_ROLES = [
  'ignore', // 0: '0' (Bỏ qua)
  'ignore', // 1: 'close_eye' -> YOLO bỏ qua, nhường MediaPipe
  'safe',   // 2: 'correct'
  'ignore', // 3: 'dozing'    -> YOLO bỏ qua, nhường MediaPipe
  'danger', // 4: 'drinking'  -> YOLO bắt
  'safe',   // 5: 'face'
  'danger', // 6: 'mobile'    -> YOLO bắt
  'danger', // 7: 'radio'     -> YOLO bắt
  'ignore', // 8: 'talking'   -> Bỏ qua hoàn toàn
  'ignore', // 9: 'work'      -> YOLO bỏ qua, nhường MediaPipe
  'ignore'  // 10: 'yawn'     -> YOLO bỏ qua, nhường MediaPipe
];

export const SPECIFIC_GESTURE_IDS = new Set([1, 3, 4, 6, 7, 9, 10]);

const PIXEL_COUNT = INPUT_SIZE * INPUT_SIZE;
const reusableFloat32Buffer = new Float32Array(3 * PIXEL_COUNT);
const reusableInputTensor = new ort.Tensor('float32', reusableFloat32Buffer, [1, 3, INPUT_SIZE, INPUT_SIZE]);

let session = null;
let letterboxMeta = { scale: 1, padX: 0, padY: 0, srcW: 640, srcH: 480 };

export async function loadYoloModel(source, labelName = 'best.onnx', onStatusChange) {
  try {
    if (onStatusChange) onStatusChange('loading', `YOLO: Đang nạp ${labelName}...`);
    session = await ort.InferenceSession.create(source, {
      executionProviders: ['wasm']
    });
    if (onStatusChange) onStatusChange('online', `YOLO: Đã nạp ${labelName}`);
    return true;
  } catch (err) {
    console.warn("Chưa nạp được best.onnx:", err);
    if (onStatusChange) onStatusChange('error', "YOLO: Chờ nạp best.onnx");
    return false;
  }
}

export function isYoloReady() {
  return session !== null;
}

function preprocessFrameLetterbox(video, prepCtx, overlayCanvas, useBGR) {
  const srcW = video.videoWidth || 640;
  const srcH = video.videoHeight || 480;

  if (overlayCanvas.width !== srcW || overlayCanvas.height !== srcH) {
    overlayCanvas.width = srcW;
    overlayCanvas.height = srcH;
  }

  const scale = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = Math.round((INPUT_SIZE - newW) / 2 - 0.1);
  const padY = Math.round((INPUT_SIZE - newH) / 2 - 0.1);

  letterboxMeta = { scale, padX, padY, srcW, srcH };

  prepCtx.fillStyle = 'rgb(114, 114, 114)';
  prepCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  prepCtx.drawImage(video, 0, 0, srcW, srcH, padX, padY, newW, newH);

  const imgData = prepCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

  if (!useBGR) {
    for (let i = 0; i < PIXEL_COUNT; i++) {
      reusableFloat32Buffer[i] = imgData[i * 4] / 255.0;                       // R
      reusableFloat32Buffer[i + PIXEL_COUNT] = imgData[i * 4 + 1] / 255.0;     // G
      reusableFloat32Buffer[i + 2 * PIXEL_COUNT] = imgData[i * 4 + 2] / 255.0; // B
    }
  } else {
    for (let i = 0; i < PIXEL_COUNT; i++) {
      reusableFloat32Buffer[i] = imgData[i * 4 + 2] / 255.0;                   // B
      reusableFloat32Buffer[i + PIXEL_COUNT] = imgData[i * 4 + 1] / 255.0;     // G
      reusableFloat32Buffer[i + 2 * PIXEL_COUNT] = imgData[i * 4] / 255.0;     // R
    }
  }

  return reusableInputTensor;
}

/**
 * Tính chỉ số chồng lấp IoU (Intersection over Union) cho thuật toán NMS
 */
function iou(boxA, boxB) {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
  const yB = Math.min(boxA[1] + boxA[3], boxB[1] + boxB[3]);
  const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  return inter / (boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter + 1e-6);
}

/**
 * Giải mã Tensor đầu ra của YOLOv12n [1, 15, 8400]
 */
function decodeYolo(outputTensor, classRoles, confThreshold) {
  const data = outputTensor.data;
  const dims = outputTensor.dims;
  const numClasses = Math.min(CLASS_LABELS.length, dims[1] - 4);
  const numAnchors = dims[2];
  const classScores = new Array(CLASS_LABELS.length).fill(0);
  const candidates = [];

  for (let c = 0; c < numClasses; c++) {
    const offset = (4 + c) * numAnchors;
    let maxC = 0;
    for (let i = 0; i < numAnchors; i++) {
      if (data[offset + i] > maxC) maxC = data[offset + i];
    }
    classScores[c] = maxC;
  }

  for (let i = 0; i < numAnchors; i++) {
    for (let c = 1; c < numClasses; c++) {
      if (classRoles[c] === 'ignore') continue;
      const score = data[(4 + c) * numAnchors + i];
      if (score >= confThreshold) {
        const cx = data[0 * numAnchors + i];
        const cy = data[1 * numAnchors + i];
        const w  = data[2 * numAnchors + i];
        const h  = data[3 * numAnchors + i];

        const x1 = (cx - w / 2 - letterboxMeta.padX) / letterboxMeta.scale;
        const y1 = (cy - h / 2 - letterboxMeta.padY) / letterboxMeta.scale;
        candidates.push({
          classId: c,
          score,
          source: 'YOLOv12n',
          box: [x1, y1, w / letterboxMeta.scale, h / letterboxMeta.scale]
        });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  const detections = [];
  for (const cand of candidates) {
    let keep = true;
    for (const kept of detections) {
      if (cand.classId === kept.classId && iou(cand.box, kept.box) > 0.45) {
        keep = false;
        break;
      }
    }
    if (keep) detections.push(cand);
    if (detections.length >= 10) break;
  }

  return { detections, classScores };
}

/**
 * Hàm chạy suy luận YOLOv12n trên 1 khung hình
 */
export async function runYoloInference(video, prepCtx, overlayCanvas, useBGR, classRoles, confThreshold) {
  if (!session) {
    overlayCanvas.width = video.videoWidth || 640;
    overlayCanvas.height = video.videoHeight || 480;
    return {
      detections: [],
      classScores: new Array(CLASS_LABELS.length).fill(0)
    };
  }

  const inputTensor = preprocessFrameLetterbox(video, prepCtx, overlayCanvas, useBGR);
  const results = await session.run({ [session.inputNames[0]]: inputTensor });
  const outTensor = results[session.outputNames[0]];
  const decoded = decodeYolo(outTensor, classRoles, confThreshold);

  // Giải phóng bộ nhớ Tensor đầu ra ngay sau khi đọc xong
  if (typeof outTensor.dispose === 'function') {
    outTensor.dispose();
  }

  return decoded;
}