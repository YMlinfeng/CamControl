import numpy as np

# 加载文件
data = np.load('/m2v_intern/mengzijie/depthanythingv3/npz/000_运镜参考_068_video.npz')
data = np.load('/m2v_intern/mengzijie/depthanythingv3/synthetic_npz/fisheye_04_forward_medium.npz')
data = np.load('/m2v_intern/mengzijie/depthanythingv3/hitchcock/hitchcock_05_dolly_out_d8.npz')
print(data)

# 1. 查看这个“压缩包”里有哪些变量（变量名/键值）
print("内部包含的变量名:", data.files)

# 2. 查看具体某个变量的内容和维度
for key in data.files:
    array = data[key]
    print(f"\n变量名: {key}")
    print(f"数据形状 (Shape): {array.shape}")
    print(f"数据类型 (Dtype): {array.dtype}")
    # 打印前几行看看数据长什么样
    print("数据样例:")
    print(array[:1]) # 只看前两个



'''

内部包含的变量名: ['extrinsics', 'intrinsics']

变量名: extrinsics
数据形状 (Shape): (125, 3, 4)
数据类型 (Dtype): float64
数据样例:
[[[ 9.67223763e-01 -6.08905870e-03 -2.53852457e-01  9.87506962e+00]
  [ 7.83670880e-03  9.99952018e-01  5.87382168e-03  3.05962861e-01]
  [ 2.53804535e-01 -7.67066842e-03  9.67225134e-01 -8.37901783e+00]]

 [[ 9.71428156e-01 -3.44118779e-03 -2.37308741e-01  8.74182892e+00]
  [ 4.84297099e-03  9.99974072e-01  5.32428641e-03  2.26103738e-01]
  [ 2.37284273e-01 -6.32144138e-03  9.71419692e-01 -7.48032713e+00]]]

变量名: intrinsics
数据形状 (Shape): (125, 3, 3)
数据类型 (Dtype): float32
数据样例:
[[[454.56903   0.      252.     ]
  [  0.      444.0466  140.     ]
  [  0.        0.        1.     ]]

 [[455.5099    0.      252.     ]
  [  0.      444.3739  140.     ]
  [  0.        0.        1.     ]]]


'''