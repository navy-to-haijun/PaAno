// 使用命令
// python -m http.server 8080

// c1 显示滚动历史，最多保留 1M 点；c2 显示最近一帧。
const HISTORY_MAX_POINTS = 20000;
const VOLTAGE_COLOR = new WebglPlotBundle.ColorRGBA(0.1, 0.9, 0.45, 1);
const SCORE_COLOR = new WebglPlotBundle.ColorRGBA(1, 0.25, 0.2, 1);

// 同一个 canvas 内：电压在上半区，分数在下半区，中间留空避免重叠。
const VOLTAGE_CENTER_Y = 0.55;
const VOLTAGE_HALF_HEIGHT = 0.4;
const SCORE_MIN_Y = -0.9;
const SCORE_HEIGHT = 0.8;

function resizeCanvas(canvas) {
  // 按设备像素比设置真实画布尺寸，避免高分屏下曲线模糊。
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
}

function resizeView(view) {
  resizeCanvas(view.canvas);
  view.plot.viewport(0, 0, view.canvas.width, view.canvas.height);
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

function mapVoltage(value) {
  // 后端传来的电压已经在 [-1, 1]，这里压到上半区 [0.15, 0.95]。
  return VOLTAGE_CENTER_Y + clamp(value, -1, 1) * VOLTAGE_HALF_HEIGHT;
}

function mapScore(value) {
  // 后端传来的分数已经在 [0, 1]，这里压到下半区 [-0.9, -0.1]。
  return SCORE_MIN_Y + clamp(value, 0, 1) * SCORE_HEIGHT;
}

function createLine(color, points, emptyY) {
  const line = new WebglPlotBundle.WebglLine(color, points);
  line.arrangeX();
  fillLineY(line, emptyY);
  return line;
}

function createView(canvasId, points) {
  const canvas = document.getElementById(canvasId);
  resizeCanvas(canvas);

  const plot = new WebglPlotBundle.WebglPlot(canvas);
  const voltageLine = createLine(VOLTAGE_COLOR, points, VOLTAGE_CENTER_Y);
  const scoreLine = createLine(SCORE_COLOR, points, SCORE_MIN_Y);
  plot.addLine(voltageLine);
  plot.addLine(scoreLine);

  return { canvas, plot, voltageLine, scoreLine, capacity: points };
}

function fillLineY(line, value) {
  for (let i = 0; i < line.numPoints; i++) {
    line.setY(i, value);
  }
}

function setLineXForPoints(line, points) {
  const denominator = Math.max(1, points - 1);
  for (let i = 0; i < line.numPoints; i++) {
    line.setX(i, i < points ? -1 + (2 * i) / denominator : 2);
  }
}

function writeFrameToLines(view, targetStart, voltage, scores, sourceStart, count) {
  const voltageXY = view.voltageLine.xy;
  const scoreXY = view.scoreLine.xy;

  for (let i = 0; i < count; i++) {
    const targetYIndex = (targetStart + i) * 2 + 1;
    const sourceIndex = sourceStart + i;
    voltageXY[targetYIndex] = mapVoltage(voltage[sourceIndex]);
    scoreXY[targetYIndex] = mapScore(scores[sourceIndex]);
  }
}

function shiftLineYLeft(line, removeCount, currentLength, emptyY) {
  const xy = line.xy;
  const keepCount = Math.max(0, currentLength - removeCount);

  for (let i = 0; i < keepCount; i++) {
    xy[i * 2 + 1] = xy[(i + removeCount) * 2 + 1];
  }
  for (let i = keepCount; i < currentLength; i++) {
    xy[i * 2 + 1] = emptyY;
  }
}

function parseFrame(buffer) {
  // 后端二进制帧格式：
  // 前 8 字节为两个 uint32：voltageLength 和 scoreLength。
  // 后面依次是 voltage float32[] 和 scores float32[]。
  if (buffer.byteLength < 8) {
    return null;
  }

  const view = new DataView(buffer);
  // true 表示小端序，对应 Python struct.Struct("<II")。
  const voltageLength = view.getUint32(0, true);
  const scoreLength = view.getUint32(4, true);
  const voltageBytes = voltageLength * Float32Array.BYTES_PER_ELEMENT;
  const scoreBytes = scoreLength * Float32Array.BYTES_PER_ELEMENT;
  const expectedBytes = 8 + voltageBytes + scoreBytes;

  


  if (buffer.byteLength < expectedBytes) {
    return null;
  }

  return {
    // 这里不复制数据，直接用 ArrayBuffer 视图读取，减少前端开销。
    voltage: new Float32Array(buffer, 8, voltageLength),
    scores: new Float32Array(buffer, 8 + voltageBytes, scoreLength),
  };
}

const historyView = createView("c1", HISTORY_MAX_POINTS);
let latestView = createView("c2", 1);
const historyChunks = [];
let historyLength = 0;
let drawPending = false;

function removeOldestHistoryChunk() {
  const removeCount = historyChunks.shift() || 0;
  if (removeCount <= 0) {
    return;
  }

  shiftLineYLeft(historyView.voltageLine, removeCount, historyLength, VOLTAGE_CENTER_Y);
  shiftLineYLeft(historyView.scoreLine, removeCount, historyLength, SCORE_MIN_Y);
  historyLength -= removeCount;
}

function appendHistory(frame) {
  const frameLength = Math.min(frame.voltage.length, frame.scores.length);
  if (frameLength <= 0) {
    return;
  }

  const count = Math.min(frameLength, HISTORY_MAX_POINTS);
  const sourceStart = frameLength - count;

  if (count === HISTORY_MAX_POINTS) {
    historyChunks.length = 0;
    historyChunks.push(count);
    historyLength = 0;
    writeFrameToLines(historyView, 0, frame.voltage, frame.scores, sourceStart, count);
    historyLength = count;
    return;
  }

  while (historyLength + count > HISTORY_MAX_POINTS && historyChunks.length > 0) {
    removeOldestHistoryChunk();
  }

  if (historyLength + count > HISTORY_MAX_POINTS) {
    const removeCount = historyLength + count - HISTORY_MAX_POINTS;
    shiftLineYLeft(historyView.voltageLine, removeCount, historyLength, VOLTAGE_CENTER_Y);
    shiftLineYLeft(historyView.scoreLine, removeCount, historyLength, SCORE_MIN_Y);
    historyLength -= removeCount;
  }

  writeFrameToLines(historyView, historyLength, frame.voltage, frame.scores, sourceStart, count);
  historyLength += count;
  historyChunks.push(count);
}

function ensureLatestViewCapacity(points) {
  const capacity = Math.max(1, points);
  if (latestView.capacity === capacity) {
    return;
  }

  latestView.plot.removeAllLines();
  latestView.voltageLine = createLine(VOLTAGE_COLOR, capacity, VOLTAGE_CENTER_Y);
  latestView.scoreLine = createLine(SCORE_COLOR, capacity, SCORE_MIN_Y);
  latestView.plot.addLine(latestView.voltageLine);
  latestView.plot.addLine(latestView.scoreLine);
  latestView.capacity = capacity;
}

function renderLatestFrame(frame) {
  const frameLength = Math.min(frame.voltage.length, frame.scores.length);
  const count = Math.min(frameLength, HISTORY_MAX_POINTS);
  const sourceStart = frameLength - count;


  // 显示最大最小值
  const voltageMax = Math.max(...frame.voltage);
  const voltageMin = Math.min(...frame.voltage);
  const scoreMax = Math.max(...frame.scores);
  const scoreMin = Math.min(...frame.scores);
  console.log(`voltage: [${voltageMin.toFixed(2)}, ${voltageMax.toFixed(2)}], scores: [${scoreMin.toFixed(2)}, ${scoreMax.toFixed(2)}]`);

  ensureLatestViewCapacity(count);
  setLineXForPoints(latestView.voltageLine, Math.max(1, count));
  setLineXForPoints(latestView.scoreLine, Math.max(1, count));

  if (count > 0) {
    writeFrameToLines(latestView, 0, frame.voltage, frame.scores, sourceStart, count);
  } else {
    fillLineY(latestView.voltageLine, VOLTAGE_CENTER_Y);
    fillLineY(latestView.scoreLine, SCORE_MIN_Y);
  }
}

function requestDraw() {
  if (drawPending) {
    return;
  }

  drawPending = true;
  requestAnimationFrame(() => {
    historyView.plot.update();
    latestView.plot.update();
    drawPending = false;
  });
}

// 默认连接当前页面所在主机的 8765 端口，便于换设备 IP 后不用改代码。
const wsHost = "172.101.1.2";
const ws = new WebSocket(`ws://${wsHost}:8765`);
ws.binaryType = "arraybuffer";

ws.onmessage = (event) => {
  // 显示数据只来自推理结果队列经 WebSocket 发来的帧。
  const frame = parseFrame(event.data);
  if (!frame) {
    return;
  }



  appendHistory(frame);
  renderLatestFrame(frame);
  requestDraw();
};

window.addEventListener("resize", () => {
  // 浏览器窗口变化时同步调整两个画布的真实尺寸。
  resizeView(historyView);
  resizeView(latestView);
  requestDraw();
});

requestDraw();
