from live_stream.RigolOscilloscope import Oscilloscope
from live_stream.WaveformStreamer import WaveformStreamer
from utils.LoggerConfig import init_logger
logger = init_logger()

def plot_waveform_offline():
    logger.info("单次绘制波形...")
    # 创建示波器实例
    osc = Oscilloscope("192.168.0.104")
    # 设置基本参数
    res = osc.set_basic_para()
    if res:
        logger.info("基本参数设置完成")
    # 设置波形参数
    res = osc.set_waveform_para()
    if res:
        # 获取波形参数
        para = osc.get_waveform_para()
        logger.info(f"波形参数: {para}")
    # 获取波形数据
    time_list, voltage_list = osc.get_waveform_data(para)
    logger.info(voltage_list[0:1000])

    # 保存波形数据
    # osc.save_waveform_to_npz(time_list, voltage_list, filename="date/waveform5.npz")
    # 绘制波形
    osc.plot_waveform(time_list, voltage_list)

def live_streaming():
    logger.info("实时传输数据...")
     # 实例化示波器
    osc = Oscilloscope("172.101.1.1")
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




if __name__ == '__main__':
    # plot_waveform_offline()
    live_streaming()
    
   