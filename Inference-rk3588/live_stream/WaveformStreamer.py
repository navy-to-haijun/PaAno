import asyncio
import struct
import numpy as np
import websockets
from utils.LoggerConfig import init_logger
from live_stream.RigolOscilloscope import Oscilloscope
import os
from datetime import datetime

logger = init_logger()

class WaveformStreamer:
    """
    示波器数据流式传输：通过 WebSocket 将波形数据以二进制格式推送web上
    """
    def __init__(self, oscilloscope, host: str = "0.0.0.0", port: int = 8765):
        self.osc = oscilloscope
        self.host = host
        self.port = port
        self.server = None
        self._is_running = False

    async def _handler(self, websocket):
        """
        处理单个客户端连接的内部协程
        """
        logger.info(f"WebSocket 客户端已连接: {websocket.remote_address}")
        
        # 获取波形参数
        try:
            waveform_para = self.osc.get_waveform_para()
            logger.info(f"已获取波形参数: {waveform_para}")
        except Exception as e:
            logger.error(f"初始化获取波形参数失败: {e}")
            return

        try:
            while self._is_running:
                # 采集数据
                data = self.osc.get_waveform_data(waveform_para)
                time_list, voltage_list = data
                if len(voltage_list) == 0:
                    await asyncio.sleep(0.1)
                    continue
                # 保存数据到本地
                self._save_data_to_file(voltage_list)
                
                # 缩放数据：将voltage_list缩放到[-1, 1]范围内(规定电压范围为±10V)
                voltage_arr = np.array(voltage_list, dtype=np.float32)
                voltage_arr = np.clip(voltage_arr, -10.0, 10.0) / 15.0
                logger.info(f"采集到数据: 点数={len(voltage_list)}, 电压范围=[{min(voltage_arr):.2f}V, {max(voltage_arr):.2f}V]")
                # 发送二进制数据(只发送电压值)
                await websocket.send(voltage_arr.tobytes())
                logger.info(f"已向 {websocket.remote_address} 发送二进制帧，点数: {voltage_arr.size}")

                # 4. 适当释放 CPU 权限
                await asyncio.sleep(0.05)

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"客户端断开连接: {websocket.remote_address}, 状态码: {e.code}")
        except Exception as e:
            logger.error(f"传输时发生异常: {e}")
    
    def _save_data_to_file(self, voltage_arr):
        """
        将电压数据保存到本地txt文件
        
        Args:
            voltage_arr: 电压数据数组
        """
        try:
            # 创建保存目录（如果不存在）
            save_dir = getattr(self, 'save_dir', './waveform_data')
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名（使用时间戳，前面加data）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
            filename = f"data_{timestamp}.txt"
            filepath = os.path.join(save_dir, filename)
            
            # 只写入电压值，每个值一行
            with open(filepath, 'w', encoding='utf-8') as f:
                for voltage in voltage_arr:
                    f.write(f"{voltage:.6f}\n")
            
            logger.info(f"数据已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存数据文件失败: {e}")

    async def _serve(self):
        """
        启动并保持服务器运行
        """
        self._is_running = True
        async with websockets.serve(self._handler, self.host, self.port):
            logger.info(f"WebSocket 二进制传输服务已在 ws://{self.host}:{self.port} 启动")
            await asyncio.Future()  # 永久保持运行

    def start(self):
        """
        启动websocket服务
        """
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            self._is_running = False
            logger.info("WebSocket 服务已被用户手动停止")



   

if __name__ == '__main__':
    logger.info("初始化示波器控制...")
    
    # 实例化示波器
    osc = Oscilloscope("192.168.0.4")

    try:
        # 配置基本参数与波形参数
        if osc.set_basic_para() and osc.set_waveform_para():
            logger.info("示波器硬件参数配置成功。")
            
            # 实例化传输器
            streamer = WaveformStreamer(oscilloscope=osc, host="0.0.0.0", port=8765)
            
            # 4. 启动流式传输
            streamer.start()
            
    finally:
        #确保在退出时安全关闭示波器连接
        logger.info("正在关闭与示波器的连接...")
        osc.close()