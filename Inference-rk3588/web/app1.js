// 每个画布最多显示的点数，需要和示波器当前采样点数保持同一量级。
const MAX_POINTS = 10000;

function resize(canvas) {
  // 按设备像素比设置真实画布尺寸，避免高分屏下曲线模糊。
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
}

function createLine(canvasId, color) {
  // 创建一个 WebGL 曲线视图：一个 canvas、一个 plot、一个 line。
  const canvas = document.getElementById(canvasId);
  resize(canvas);

  const plot = new WebglPlotBundle.WebglPlot(canvas);
  const line = new WebglPlotBundle.WebglLine(color, MAX_POINTS);
  line.arrangeX();
  line.constY(0);
  plot.addLine(line);

  return { canvas, plot, line };
}

// c1 显示电压波形。
const voltageView = createLine(
  "c1",
  new WebglPlotBundle.ColorRGBA(0.1, 0.9, 0.45, 1)
);

// c2 显示每个采样点对应的异常分数。
const scoreView = createLine(
  "c2",
  new WebglPlotBundle.ColorRGBA(1, 0.25, 0.2, 1)
);

function clearTail(line, start) {
  // 当前帧点数少于 MAX_POINTS 时，把尾部清零，避免残留上一帧旧曲线。
  for (let i = start; i < line.numPoints; i++) {
    line.setY(i, 0);
  }
}

function drawArray(line, data, transform) {
  // 将一段 Float32Array 写入 WebGL 曲线，transform 用于做显示范围映射。
  const points = Math.min(line.numPoints, data.length);
  for (let i = 0; i < points; i++) {
    line.setY(i, transform(data[i]));
  }
  clearTail(line, points);
}

function clampUnit(value) {
  // 电压在后端已经缩放，这里再兜底限制到 WebGL 可视范围。
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(-1, Math.min(1, value));
}

function scaleScore(value) {
  // 异常分数通常是 [0, 1]，映射到 WebGL 的 [-1, 1] 方便占满第二个画布。
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(-1, Math.min(1, value * 2 - 1));
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

// 默认连接当前页面所在主机的 8765 端口，便于换设备 IP 后不用改代码。
const wsHost = window.location.hostname || "127.0.0.1";
const ws = new WebSocket(`ws://${wsHost}:8765`);
ws.binaryType = "arraybuffer";

ws.onmessage = (event) => {
  // 显示数据只来自推理结果队列经 WebSocket 发来的帧。
  const frame = parseFrame(event.data);
  if (!frame) {
    return;
  }

  drawArray(voltageView.line, frame.voltage, clampUnit);
  drawArray(scoreView.line, frame.scores, scaleScore);
};

window.addEventListener("resize", () => {
  // 浏览器窗口变化时同步调整两个画布的真实尺寸。
  resize(voltageView.canvas);
  resize(scoreView.canvas);
});

function loop() {
  // requestAnimationFrame 持续刷新 WebGL 画布，数据由 ws.onmessage 更新。
  voltageView.plot.update();
  scoreView.plot.update();
  requestAnimationFrame(loop);
}

loop();
