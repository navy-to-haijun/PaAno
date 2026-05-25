import pyvisa
import time
import struct
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.LoggerConfig import init_logger


logger = init_logger()

class Oscilloscope:
    '''
    创建示波器类，实现对示波器的操作
    '''
    def __init__(self, ip_address):
        self.ip_address = ip_address
        self.rm = pyvisa.ResourceManager()
        try:
            self.inst = self.rm.open_resource(f'TCPIP0::{self.ip_address}::INSTR')
        except:
            logger.error(f'无法连接到示波器, IP地址: {self.ip_address}')
            exit()
        
        logger.info(f'已连接到示波器, IP地址: {self.ip_address}')

        # 通过发送*IDN?命令查询设备信息
        resp = self.inst.query("*IDN?").strip()
        if resp:
            logger.info(f'示波器信息: {resp}')
        else:
            logger.error('无法获取示波器的信息')
            self.close()
            exit()
    
    def write_and_check(self, command, expected_response):
        """
        写命令并检查响应
        
        :param command: 写命令
        :param expected_response: 期望响应
        """
        # 发送命令
        self.inst.write(command)
        # # 构造查询命令
        query_cmd = command.split()[0] + "?"  
        # 读取响应
        response = self.inst.query(query_cmd).strip()
        if response == expected_response:
            logger.info(f'命令 "{command}" 执行成功，响应符合预期: {response}')
            return True
        else:
            logger.info(f'命令 "{command}" 执行失败，预期响应: {expected_response}，实际响应: {response}')
            return False
    def set_basic_para(self, channel=1):
        """
        set_basic_para 的 Docstring
        
        :param self: 说明
        :param channel: 通道编号, 默认为1
        """
        commands = [
                # (命令, 期望响应)
                (f":CHANnel{channel}:DISPlay ON", "1"),               # 打开通道
                (f":CHANnel{channel}:SCALe 2", "2.000000E+00"),       # 设置通道电压刻度为2V/div
                (f":CHANnel{channel}:OFFSet 0", "0.000000E+00"),      # 设置通道的垂直偏移为0V
                (f":TIMebase:MODE MAIN", "MAIN"),                     # 设置时间基准为MAIN YT模式
                (f":TIMebase:SCALe 0.00001", "1.000000E-5"),            # 设置时间刻度为500us/div
                (f":TRIGger:MODE EDGE", "EDGE"),                     # 设置触发模式为边沿触发
                (f":TRIGger:EDGE:SOURce CHAN{channel}", f"CHAN{channel}"),  # 设置触发源为通道1
                (f":TRIGger:EDGE:LEVel 1", "1.000000E0"),        # 设置触发水平为1V
                (f":CHANnel1:PROBe 10", "10"),                     # 设置探头衰减比为10X
                # (f":TRIGger:SWEep SINGle", "SING"),               # 设置触发模式为单次触发
            ]
        
        self.inst.write(":RUN")  # 运行示波器
        
        # 顺序执行命令
        for command, exp_response in commands:
            status = self.write_and_check(command, exp_response)
            if not status:
                return False

        return True
        
    def set_waveform_para(self, channel=1):
        """
        设置波形参数
        
        :param channel: 通道编号, 默认为1
        :return: 波形数据
        """

        commands = [
                # (命令, 期望响应)
                (f":WAV:SOUR CHAN{channel}", f"CHAN{channel}"),    # 设置通道
                ("WAV:MODE RAW", "RAW"),                           # 设置读取数据模式: 读取内存中的波形数据(读取时需要停止示波器)      
                ("WAV:FORM WORD", "WORD"),                         # 设置数据的返回格式：一个波形点占用2个字节
                (f":ACQuire:MDEPth 10k", "1.0000E+04"),           # 设置波形深度(要求采样速率控制在20M) 
                ("WAVeform:POINts 10000", "10000"),              # 设置波形点数
                ("WAVeform:STARt 1", "1"),                         # 设置起始点
                ("WAVeform:STOP 10000", "10000"),                # 设置结束点
            ]
        
        # 顺序执行命令
        for command, exp_response in commands:
            status = self.write_and_check(command, exp_response)
            if not status:
                return False
            
        logger.info("波形参数设置完成")
        return True
    def __wait_acquire_done(self, timeout=10.0):
        start = time.time()
        while time.time() - start < timeout:
            status = self.inst.query(":TRIGger:STATus?").strip()
            if status in ("STOP"):
                return True
            time.sleep(1)
        logger.error("等待示波器采集完成超时")
        return False
    
    def __parse_waveform_data(self, date: bytes, preamble: dict):
        """
        解析示波器的数据
        
        :param date: 示波器的原始数据
        :type date: bytes
        :param preamble: 示波器波形参数
        :type preamble: dict
        """
        # 解析TMC 头: #NXXXXXX #: TMC 规定的头标志符,  N 表述数据长度字段的位数, XXXXXX 表述数据长度
        if date[0:1] != b'#':
            logger.error("波形数据格式错误，缺少 '#' 头")
            exit()
        # 获取长度字段的位数
        length_size = int(date[1:2].decode('ascii'))
        # 获取数据长度
        data_length = int(date[2:2 + length_size].decode('ascii'))
        # 获取数据开始，结束的索引
        date_start_index = 2 + length_size
        date_end_index = date_start_index + data_length
        # 获取原始数据部分
        raw_data = date[date_start_index:date_end_index]
        # 计算波形点数
        count = data_length // 2
        logger.info(f"解析到的波形点数: {count} 字节")
        # 解析获取原始的ADC值 ，小端模式，每个点占用2字节
        adc_values  = struct.unpack(f'<{count}H', raw_data)
        # 获取有用的preamble的参数
        yorigin = preamble['yorigin']
        yreference = preamble['yreference']
        yincrement = preamble['yincrement']

        xorigin = preamble['xorigin']
        xincrement = preamble['xincrement']

        # 将ADC值转换为电压值
        time_ms_list = []
        voltage_list = []
        for i, adc in enumerate(adc_values):
            voltage = (adc - yorigin - yreference) * yincrement
            time = (i * xincrement + xorigin) * 1000  # 转换为毫秒
            time_ms_list.append(time)
            voltage_list.append(voltage)

        return time_ms_list, voltage_list

    def get_waveform_data(self, waveform_para: dict):
        """
        获取波形数据(只解析RAW模式, 数据类型为WORD)

        :return: 波形数据
        """
        # 获取波形数据(二进制模式)
        # 单次触发
        self.write_and_check(":TRIGger:SWEep SINGle", "SING")
        # 等待采集完成
        if not self.__wait_acquire_done(timeout=5.0):
            return None
        
        # 读取波形数据
        self.inst.write(":WAVeform:DATA?")
        date = self.inst.read_raw()
        # 解析波形数据
        time_list, voltage_list = self.__parse_waveform_data(date, waveform_para)
        return time_list, voltage_list
    
    def plot_waveform(self, time_list, voltage_list):
        """
        绘制波形

        :param time_list: 时间列表
        :param voltage_list: 电压列表
        """
        if time_list is None or voltage_list is None: 
            logger.error("波形数据为空")
            return
        logger.info("绘制波形")
        plt.figure(figsize=(12, 6))
        plt.plot(time_list, voltage_list)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (V)')
        plt.title('Waveform')
        plt.show()

    def save_waveform_to_npz(self, time_list, voltage_list, filename="waveform.npz"):
        """
        保存波形数据到NPZ文件

        :param time_list: 时间列表
        :param voltage_list: 电压列表
        :param filename: 文件名
        """
        if time_list is None or voltage_list is None: 
            logger.error("波形数据为空")
            return
        logger.info(f"保存波形数据到文件: {filename}")
        
        # 目录不存在则创建
        save_dir = os.path.dirname(filename)
        if save_dir and not os.path.exists(save_dir):
            logger.info(f"创建目录: {save_dir}")
            os.makedirs(save_dir)
        try:
            logger.info("保存波形数据")
            # 转换为numpy数组
            time_arr = np.asarray(time_list, dtype=np.float64)
            voltage_arr = np.asarray(voltage_list, dtype=np.float64)

            np.savez(
                filename,
                time=time_arr,
                voltage=voltage_arr,
                points=time_arr.size
            )
            logger.info(f"波形已保存: {filename} (points={time_arr.size})")
        except Exception as e:
            logger.error(f"保存波形数据失败: {e}")
            


    def get_waveform_para(self):
        """
        获取波形参数

        :return: 波形参数
        """
        # 获取波形参数
        response = self.inst.query(":WAVeform:PREamble?")
        values = response.strip().split(',')   # 分割响应字符串

        if len(values) != 10:
            logger.error(f"PREamble 参数数量错误: {response}")
            exit()
        # 解析参数

        result = {
            "format": int(values[0]),               # 数据格式 (0: BYTE, 1: WORD, 2: ASCII)
            "type": int(values[1]),                 # 数据类型 (0: NORMal, 1: MAXimum, 2: RAW)
            "points": int(values[2]),               # 波形点数
            "count": int(values[3]),                # 采集点数，在平均采样方式下为平均次数，其它方式下为 1
            "xincrement": float(values[4]),         # X 方向上的相邻两点之间的时间差
            "xorigin": float(values[5]),            # X 方向上的起始时间
            "xreference": float(values[6]),         # X 方向上数据点的参考时间基准
            "yincrement": float(values[7]),         # Y 方向上波形的步进值
            "yorigin": int(values[8]),            # Y 方向上相对于“垂直参考位置” 的垂直偏移
            "yreference": int(values[9])          # Y 方向的垂直参考位置
        }
        # 派生参数
        result['sample_rate_MSa'] = 1 / result["xincrement"] / 1000000                  # 采样率 (采样点数(M)/秒)
        result['time_span_ms'] = result["xincrement"] * result['points'] * 1000          # 采样时长 (毫秒)
        return result
    

    def close(self):
        self.inst.close()
        self.rm.close()
    
    def webscoket_stream(self):
        pass
        

if __name__ == '__main__' :
    logger.info("示波器流式读数测试...")

    # 创建示波器实例
    osc = Oscilloscope("192.168.0.4")
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

    # 保存波形数据
    # osc.save_waveform_to_npz(time_list, voltage_list, filename="date/waveform5.npz")
    # 绘制波形
    osc.plot_waveform(time_list, voltage_list)



    # 关闭连接
    osc.close()
