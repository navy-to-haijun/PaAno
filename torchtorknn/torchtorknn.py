
from rknn.api import RKNN
import numpy as np
import os
if __name__ == '__main__':

    model = '../PaAno-model/output/trained_model.pt'


    # 创建RKNN对象
    rknn = RKNN(verbose=True, verbose_file='./rknn.log')

    # 配置参数
    print('--> Config model')
    rknn.config(target_platform='rk3588')
    print('done')

    # 加载模型
    print('--> Loading model')
    input_size_list = [[-1, 1, 256]]
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

     # 编译模型
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # 转化模型
    print('--> Export rknn model')
    ret = rknn.export_rknn('../PaAno-model/output/rk3588-PaAno.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # 初始化推理环境
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)

    # 推理
    # 产生随机输入数据: 格式[[1, 1, 64]]
    
    data = np.random.randn(10, 1, 256).astype(np.float32)

    outputs = rknn.inference(inputs=[data])
    print(outputs)
    print(outputs[0].shape)




    rknn.release()

    

