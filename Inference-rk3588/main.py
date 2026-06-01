import argparse
import asyncio
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from inference.inference import RKNNPaAnoInference
from live_stream.RigolOscilloscope import Oscilloscope
from live_stream.WaveformStreamer import WaveformStreamer
from utils.LoggerConfig import init_logger


"""
实时流水线入口：
1. 采集线程从示波器读取电压波形，并写入 raw_queue。
2. 推理线程从 raw_queue 取波形，调用 RKNN 推理得到每个点的异常分数。
3. WebSocket 协程只从 result_queue 取推理后的结果并发送给前端。

注意：这里不修改 RigolOscilloscope.py 和 inference.py 的内部逻辑，只调用它们已有的接口。
"""

logger = init_logger()

# 两个队列都只保留 2 帧，避免无限堆积导致实时性变差。
QUEUE_SIZE = 2

# 默认连接参数可以通过环境变量覆盖，便于部署到不同设备。
DEFAULT_OSC_IP = os.getenv("OSCILLOSCOPE_IP", "172.101.1.1")
DEFAULT_HOST = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("WEBSOCKET_PORT", "8765"))


@dataclass
class WaveformFrame:
    """采集线程产出的原始波形帧。"""

    seq: int  # 帧序号，用来追踪采集、推理和发送的对应关系。
    timestamp: float  # 采集完成时间戳。
    time_ms: np.ndarray  # 示波器返回的时间轴，单位为毫秒。
    voltage: np.ndarray  # 示波器解析后的电压数组。


@dataclass
class InferenceFrame:
    """推理线程产出的显示帧，包含电压和每个点的异常分数。"""

    seq: int  # 对应 WaveformFrame.seq。
    timestamp: float  # 原始采集时间戳。
    infer_timestamp: float  # 推理完成时间戳。
    voltage: np.ndarray  # 原始电压曲线，发送前由 WaveformStreamer 做显示缩放。
    scores: np.ndarray  # 点异常分数，长度应与 voltage 对齐。


def load_config(config_path: Path) -> dict:
    """读取 RKNN 推理配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def put_latest(result_queue: queue.Queue, item: InferenceFrame) -> None:
    """
    非阻塞写入推理结果队列。

    result_queue 只用于前端显示，队列满时丢弃旧结果，保留最新结果，避免发送端拖慢推理线程。
    """
    while True:
        try:
            result_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                # 发送端只关心最新结果，旧结果被丢弃后要调用 task_done 与 get 配对。
                result_queue.get_nowait()
                result_queue.task_done()
            except queue.Empty:
                continue


def align_scores(scores: np.ndarray, target_size: int) -> np.ndarray:
    """
    对齐点异常分数长度，防止数据长度不一致
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == target_size:
        return scores
    if scores.size == 0:
        logger.warning(f"推理没有返回异常分数，使用 0 数组发送。")
        return np.zeros(target_size, dtype=np.float32)
    if scores.size > target_size:
        logger.warning(
            f"异常分数长度 {scores.size} 大于波形长度 {target_size}，将截断多余分数。"
        )
        return scores[:target_size]

    logger.warning(
        f"异常分数长度 {scores.size} 小于波形长度 {target_size}，将使用边缘值补齐。"
    )
    return np.pad(scores, (0, target_size - scores.size), mode="edge").astype(np.float32)


def acquire_worker(
    osc: Oscilloscope,
    waveform_para: dict,
    raw_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """
    采集线程。

    采集线程只负责调用示波器接口取电压值，并通过 raw_queue 交给推理线程。
    raw_queue 是阻塞队列：推理来不及时采集线程会等待，保证采到的数据会进入推理流程。
    """
    seq = 0
    logger.info(f"采集线程已启动。")

    while not stop_event.is_set():
        try:
            # 直接调用 RigolOscilloscope.py 已有解析逻辑，拿到时间轴和电压值。
            waveform_data = osc.get_waveform_data(waveform_para)
            if waveform_data is None:
                continue

            time_list, voltage_list = waveform_data
            if not voltage_list:
                continue

            # 将一次示波器采样封装成帧，便于后续日志和结果对应。
            frame = WaveformFrame(
                seq=seq,
                timestamp=time.time(),
                time_ms=np.asarray(time_list, dtype=np.float32),
                voltage=np.asarray(voltage_list, dtype=np.float32),
            )

            # raw_queue 满时阻塞等待，形成对采集速度的反压。
            while not stop_event.is_set():
                try:
                    raw_queue.put(frame, timeout=0.1)
                    logger.info(f"采集完成：帧序号={seq}，点数={frame.voltage.size}")
                    break
                except queue.Full:
                    continue

            seq += 1
        except Exception:
            logger.exception(f"采集线程发生异常。")
            stop_event.set()

    logger.info(f"采集线程已停止。")


def infer_worker(
    inferencer: RKNNPaAnoInference,
    raw_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """
    推理线程。

    推理线程从 raw_queue 阻塞取采集帧，调用 inference.py 的实时推理入口。
    推理后的 voltage + scores 再写入 result_queue，供 WebSocket 协程发送。
    """
    logger.info(f"推理线程已启动。")

    while not stop_event.is_set():
        try:
            # 没有采集数据时短暂等待，避免线程空转占用 CPU。
            frame = raw_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            infer_start = time.time()
            # 调用推理
            scores = inferencer.evaluate(use_livestream=True, data=frame.voltage)
            scores = align_scores(scores, frame.voltage.size)

            # 推理结果队列是发送端唯一数据来源，发送端不直接读取采集队列。
            result = InferenceFrame(
                seq=frame.seq,
                timestamp=frame.timestamp,
                infer_timestamp=time.time(),
                voltage=frame.voltage,
                scores=scores,
            )
            put_latest(result_queue, result)
            logger.info(f"推理完成：帧序号={frame.seq}，点数={frame.voltage.size}，耗时={time.time() - infer_start:.3f}s")
        except Exception:
            logger.exception(f"推理线程处理帧 {frame.seq} 时发生异常。")
            stop_event.set()
        finally:
            # 与 raw_queue.get 配对，方便后续如需 join 队列时能正确工作。
            raw_queue.task_done()

    logger.info(f"推理线程已停止。")


async def run_pipeline(args: argparse.Namespace) -> None:
    """
    启动完整实时流水线。

    线程负责阻塞型任务：示波器采集、RKNN 推理。
    asyncio 协程负责 WebSocket 网络发送，避免发送阻塞影响采集和推理。
    """
    # 采集数据队列: 推理线程负责阻塞取数。
    raw_queue = queue.Queue(maxsize=QUEUE_SIZE)

    # 推理数据队列: 发送队列负责取数。
    result_queue = queue.Queue(maxsize=QUEUE_SIZE)

    # 所有线程和协程共用的停止信号。
    stop_event = threading.Event()

    logger.info(f"正在加载推理配置：{args.config}")
    inferencer = RKNNPaAnoInference(load_config(args.config))

    logger.info(f"正在连接示波器：{args.osc_ip}")
    osc = Oscilloscope(args.osc_ip)

    threads = []
    try:
        # 示波器参数初始化
        if not osc.set_basic_para():
            raise RuntimeError("设置示波器基础参数失败。")
        if not osc.set_waveform_para():
            raise RuntimeError("设置示波器波形参数失败。")

        waveform_para = osc.get_waveform_para()
        logger.info(f"示波器波形参数：{waveform_para}")

        # 两个线程启动后会并行工作：采集 N+1 帧时，推理线程可处理 N 帧。
        threads = [
            threading.Thread(
                target=acquire_worker,
                args=(osc, waveform_para, raw_queue, stop_event),
                daemon=True,
                name="AcquireThread",
            ),
            threading.Thread(
                target=infer_worker,
                args=(inferencer, raw_queue, result_queue, stop_event),
                daemon=True,
                name="InferThread",
            ),
        ]

        for worker in threads:
            worker.start()

        # WebSocket 只消费 result_queue，即推理完成后的数据。
        streamer = WaveformStreamer(result_queue=result_queue, host=args.host, port=args.port)
        await streamer.serve(stop_event=stop_event)
    except KeyboardInterrupt:
        logger.info(f"用户中断程序。")
    finally:
        # 退出时通知线程停止，并关闭示波器连接。
        stop_event.set()
        for worker in threads:
            worker.join(timeout=2.0)
        logger.info(f"正在关闭示波器连接。")
        osc.close()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行示波器采集 -> RKNN 推理 -> WebSocket 发送流水线。")
    parser.add_argument("--osc-ip", default=DEFAULT_OSC_IP, help="Rigol 示波器 IP 地址。")
    parser.add_argument("--host", default=DEFAULT_HOST, help="WebSocket 监听地址。")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="WebSocket 监听端口。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "inference" / "evlconfig.yaml",
        help="推理配置文件路径。",
    )
    return parser.parse_args()


def main() -> None:
    """程序入口。"""
    asyncio.run(run_pipeline(parse_args()))


if __name__ == "__main__":
    main()
