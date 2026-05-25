const N = 1000000;  // 缓冲区大小
function resize(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
}

// ===== canvas 1： 绘制电压值 =====
const canvas1 = document.getElementById("c1");
resize(canvas1);
const wglp1 = new WebglPlotBundle.WebglPlot(canvas1);

// 线颜色绿色
const line1 = new WebglPlotBundle.WebglLine(
  new WebglPlotBundle.ColorRGBA(0, 1, 0, 1),
  N
);
// 自动等分X轴
line1.arrangeX();
wglp1.addLine(line1);


let buffer = new Float32Array(N);
let writeIndex = 0;     // 写入位置
let totalPoints = 0;    // 已接收总点数（用于判断是否满）

// ===== canvas 2 =====
const canvas2 = document.getElementById("c2");
resize(canvas2);
const wglp2 = new WebglPlotBundle.WebglPlot(canvas2);

const line2 = new WebglPlotBundle.WebglLine(
  new WebglPlotBundle.ColorRGBA(1, 0, 0, 1),
  10000
);

line2.arrangeX();
wglp2.addLine(line2);




// ===== WebSocket =====
const ws = new WebSocket("ws://192.168.62.56:8765");
ws.binaryType = "arraybuffer"; // 开启二进制模式

// 添加连接状态显示
ws.onopen = () => {
  console.log("WebSocket连接成功");
};

ws.onclose = () => {
  console.log("WebSocket连接断开");
};

ws.onerror = (error) => {
  console.error("WebSocket错误:", error);
};

ws.onmessage = (event) => {

  const voltageArray = new Float32Array(event.data);
  const points = voltageArray.length;

  console.log(`接收到数据: 点数=${points}, 电压范围=[${Math.min(...voltageArray).toFixed(2)}V, ${Math.max(...voltageArray).toFixed(2)}V]`);
  
  // 循环写入缓冲区
  for (let i = 0; i < points; i++) {
    buffer[writeIndex] = voltageArray[i];
    writeIndex = (writeIndex + 1) % N;  // 到达末尾后回到开头

    // 将波形画到第二个canvas上
    line2.setY(i, voltageArray[i]);
  }
  
  // 更新已接收总数（最多N）
  totalPoints = Math.min(N, totalPoints + points);
  
  // 显示缓冲区中的所有有效数据
  // 注意：需要从 writeIndex 开始显示到末尾，再到开头
  for (let i = 0; i < totalPoints; i++) {
    const idx = (writeIndex - totalPoints + i + N) % N;
    line1.setY(i, buffer[idx]);
  }


};

// ===== render loop =====
function loop() {
  wglp1.update();
  wglp2.update();
  requestAnimationFrame(loop);
}

loop();


// python -m http.server 8080
