// 获取创建的画布
const canvas = document.getElementById("my_canvas");
const devicePixelRatio = window.devicePixelRatio || 1;
// 获取真实的像素
canvas.width = canvas.clientWidth * devicePixelRatio;
canvas.height = canvas.clientHeight * devicePixelRatio;

// 创建 WebGL plot
const numX = canvas.width;

const color = new WebglPlotBundle.ColorRGBA(
  Math.random(),
  Math.random(),
  Math.random(),
  1
);
// 创建一条线
const line = new WebglPlotBundle.WebglLine(color, numX);
// 创建绘图器
const wglp = new WebglPlotBundle.WebglPlot(canvas);

// 初始化 X
line.arrangeX();

wglp.addLine(line);

// 循环
function loop() {
  update();
  wglp.update();
  requestAnimationFrame(loop);
}

function update() {
  const freq = 0.001;
  const amp = 0.5;
  const noise = 0.1;

  for (let i = 0; i < line.numPoints; i++) {
    const ySin = Math.sin(Math.PI * i * freq * Math.PI * 2);
    const yNoise = Math.random() - 0.5;

    line.setY(i, ySin * amp + yNoise * noise);
  }
}

requestAnimationFrame(loop);