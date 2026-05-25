from rknn.api import RKNN

rknn = RKNN(verbose=True)

# 加载模型
ret = rknn.load_rknn(path='./test.rknn')
if ret != 0:
    print('Load rknn model failed')
    exit(ret)
outputs = rknn.inference(inputs=[img], data_format=['nhwc'])
