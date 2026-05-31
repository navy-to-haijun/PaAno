# 485信号质量判断

## 参考论文

论文：[PaAno: Patch-Based Representation Learning for Time-Series Anomaly Detection](https://arxiv.org/abs/2602.01359)

github：[PaAno](https://github.com/jinnnju/PaAno)

## 架构

```
               AcquireThread
      │
      ▼

 queue.Queue(maxsize=2)

      │
      ▼

 InferThread

      │                   MainThread
                        │
                    asyncio
                        │
                        ▼

                 WebSocketSender

                        ▲
                        │

              infer_to_web
             SimpleQueue()

                        ▲
                        │

                  InferThread

                        ▲
                        │

              acq_to_infer
            Queue(maxsize=2)

                        ▲
                        │

                 AcquireThread
      ▼

 queue.SimpleQueue()

      │
      ▼

 asyncio WebSocket
```



## PaAno算法

![img](attachments/train.png)
