import asyncio
import websockets
import json
import math
import random
import time
N = 1000
async def send_data(ws):
    t = 0
    while True:
        t += 0.05

        ch1 = []
        ch2 = []

        for i in range(N):
            # 通道1：正弦 + 噪声
            y1 = math.sin(i * 0.02 + t) * 0.5 + random.uniform(-0.0, 0.0)

            # 通道2：频率不同 + 噪声
            y2 = math.sin(i * 0.03 + t * 1.2) * 0.4 + random.uniform(-0.0, 0.0)

            ch1.append(y1)
            ch2.append(y2)

        await ws.send(json.dumps({
            "ch1": ch1,
            "ch2": ch2
        }))

        await asyncio.sleep(0.03)


async def main():
    async with websockets.serve(send_data, "0.0.0.0", 8765):
        print("WebSocket running at ws://0.0.0.0:8765")
        await asyncio.Future()

asyncio.run(main())