import asyncio
import queue
import struct

import numpy as np
import websockets

from utils.LoggerConfig import init_logger


logger = init_logger()


class WaveformStreamer:
    """
    WebSocket 推理结果发送器。

    该类只负责显示链路：从 result_queue 获取推理完成后的电压和异常分数，
    打包成前端可解析的二进制帧，并广播给所有已连接的 WebSocket 客户端。

    二进制帧格式：
    - uint32 voltage_len
    - uint32 score_len
    - float32[voltage_len] scaled voltage values
    - float32[score_len] anomaly scores
    """

    HEADER = struct.Struct("<II")

    def __init__(self, result_queue: queue.Queue, host: str = "0.0.0.0", port: int = 8765):
        """初始化发送器，result_queue 是唯一的数据来源。"""
        if result_queue is None:
            raise ValueError("必须提供 result_queue。")

        self.result_queue = result_queue
        self.host = host
        self.port = port
        self._is_running = False
        self._clients = set()
        self._latest_payload = None

    async def _handler(self, websocket):
        """维护单个 WebSocket 客户端连接。"""
        logger.info(f"WebSocket 客户端已连接：{websocket.remote_address}")

        self._clients.add(websocket)
        try:
            # 新客户端连接时先推送最近一帧，避免页面长时间空白。
            if self._latest_payload is not None:
                await websocket.send(self._latest_payload)
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            logger.info(f"WebSocket 客户端已断开：{websocket.remote_address}")

    async def _broadcast_loop(self, stop_event=None):
        """持续从推理结果队列取帧，并广播给在线客户端。"""
        while self._is_running:
            if stop_event is not None and stop_event.is_set():
                break

            # 发送端只关心最新画面，队列中旧帧会在 _drain_latest_result 中丢弃。
            frame = self._drain_latest_result()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            payload = self._pack_frame(frame)
            self._latest_payload = payload

            if not self._clients:
                continue

            # 单个客户端发送失败不会影响其他客户端。
            await asyncio.gather(
                *(self._send_to_client(client, payload) for client in tuple(self._clients)),
                return_exceptions=True,
            )

    async def _send_to_client(self, websocket, payload: bytes):
        """向单个客户端发送一帧数据，异常连接会被移除。"""
        try:
            await websocket.send(payload)
        except websockets.exceptions.ConnectionClosed:
            self._clients.discard(websocket)
        except Exception:
            self._clients.discard(websocket)
            logger.exception("发送 WebSocket 数据帧失败。")

    def _drain_latest_result(self):
        """取出 result_queue 中最新的一帧，并丢弃队列里已经过期的帧。"""
        latest = None
        while True:
            try:
                latest = self.result_queue.get_nowait()
                self.result_queue.task_done()
            except queue.Empty:
                return latest

    def _pack_frame(self, frame) -> bytes:
        """将推理结果帧转换为前端协议需要的二进制数据。"""
        voltage = getattr(frame, "voltage", None)
        scores = getattr(frame, "scores", None)

        if voltage is None:
            raise ValueError("推理结果帧缺少 voltage 数据。")
        if scores is None:
            raise ValueError("推理结果帧缺少 scores 数据。")

        voltage_arr = self._prepare_voltage(voltage)
        # score_arr = self._prepare_scores(scores)
        # 开头和结尾各去掉10个点，避免 WebGL 曲线渲染时边界异常
        score_arr = np.ascontiguousarray(np.asarray(scores, dtype=np.float32).reshape(-1))
        score_arr = score_arr[10:-10] if score_arr.size > 20 else score_arr

        # 每个电压采样点都应有一个对应的异常分数。
        if voltage_arr.size != score_arr.size:
            raise ValueError(
                f"电压长度 ({voltage_arr.size}) 与异常分数长度 ({score_arr.size}) 不一致。"
            )

        header = self.HEADER.pack(voltage_arr.size, score_arr.size)
        return header + voltage_arr.tobytes() + score_arr.tobytes()

    @staticmethod
    def _prepare_voltage(voltage) -> np.ndarray:
        """将原始电压转换为前端 WebGL 曲线使用的 float32 连续数组。"""
        voltage_arr = np.asarray(voltage, dtype=np.float32).reshape(-1)
        voltage_arr = np.clip(voltage_arr, -10.0, 10.0) / 15.0
        # 开头和结尾各去掉10个点，避免 WebGL 曲线渲染时边界异常。
        if voltage_arr.size > 20:
            voltage_arr = voltage_arr[10:-10]
        return np.ascontiguousarray(voltage_arr)
    @staticmethod
    def _prepare_scores(scores) -> np.ndarray:
        """将原始分数转换为前端 WebGL 曲线使用的 float32 连续数组。"""
        score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        # 由[0,1] 缩放到 [-1.1]
        score_arr = np.clip(score_arr, 0, 1) * 2 - 1
        return np.ascontiguousarray(score_arr)

    async def serve(self, stop_event=None):
        """启动 WebSocket 服务，直到外部 stop_event 触发或服务退出。"""
        self._is_running = True
        try:
            async with websockets.serve(self._handler, self.host, self.port):
                logger.info(f"WebSocket 服务已启动：ws://{self.host}:{self.port}")
                await self._broadcast_loop(stop_event=stop_event)
        finally:
            self._is_running = False
