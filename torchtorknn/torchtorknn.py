
from rknn.api import RKNN
import numpy as np
import os
if __name__ == '__main__':

    model = '../trained_encoder.pt'

    input_size_list = [[1, 1, 64]]


    rknn = RKNN(verbose=True, verbose_file='./rknn.log')

    print('--> Config model')
    rknn.config(target_platform='rk3566')
    print('done')

    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

     # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn('./test.rknn')
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
    
    data = np.random.randn(1, 1, 64).astype(np.float32)

    outputs = rknn.inference(inputs=[data])
    print(outputs)
    print(outputs[0].shape)




    rknn.release()

    

