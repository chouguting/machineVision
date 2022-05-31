import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


def _rule1_final_check(array):
    n_135_is_zero = False
    n_357_is_zero = False
    if array[1] == 0 or array[3] == 0 or array[5] == 0:
        n_135_is_zero = True
    if array[3] == 0 or array[5] == 0 or array[7] == 0:
        n_357_is_zero = True
    return n_135_is_zero and n_357_is_zero

def _rule2_final_check(array):
    n_137_is_zero = False
    n_157_is_zero = False
    if array[1] == 0 or array[3] == 0 or array[7] == 0:
        n_137_is_zero = True
    if array[1] == 0 or array[5] == 0 or array[7] == 0:
        n_157_is_zero = True
    return n_137_is_zero and n_157_is_zero


rotation_index = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]

# generate the filters for rules 1 and 2
filters1_list = []
filters2_list = []

for neighbor_one_count in range(2, 7, 1):
    neighbor_array = np.ones(neighbor_one_count)
    neighbor_array = np.append(neighbor_array, [0.0] * (8 - neighbor_one_count))
    #前兩個規則代表了鄰居中的1繞一圈看的話是要相連再一起的
    #所以以鄰居有兩個1的為例
    #[1 1 0]
    #[0 p 0]
    #[0 0 0]
    #我只需要把這個kernel一格格的旋轉一圈
    #只要符合最後一個條件(final_check)就可以作為kernel
    #print("neighbor_one_count:", neighbor_one_count)
    for rotation in range(8):
        #檢查最後一個條件 如果可以 才能作為kernel用
        if _rule1_final_check(neighbor_array):
            full_kernel = np.zeros((3, 3, 1))
            for i in range(8):
                full_kernel[rotation_index[i]] = neighbor_array[i]
            #print(full_kernel)
            full_kernel[1, 1] = 1
            full_kernel = full_kernel * 2.0 - 1  #將kernel轉成-1~1
            filters1_list.append(full_kernel)
        neighbor_array = np.roll(neighbor_array, 1)

for neighbor_one_count in range(2, 7, 1):
    neighbor_array = np.ones(neighbor_one_count)
    neighbor_array = np.append(neighbor_array, [0.0] * (8 - neighbor_one_count))
    #前兩個規則代表了鄰居中的1繞一圈看的話是要相連再一起的
    #所以以鄰居有兩個1的為例
    #[1 1 0]
    #[0 p 0]
    #[0 0 0]
    #我只需要把這個kernel一格格的旋轉一圈
    #只要符合最後一個條件(final_check)就可以作為kernel
    # print("neighbor_one_count:", neighbor_one_count)
    for rotation in range(8):
        if _rule2_final_check(neighbor_array):
            full_kernel = np.zeros((3, 3, 1))
            for i in range(8):
                full_kernel[rotation_index[i]] = neighbor_array[i]
            #print(full_kernel)
            full_kernel[1, 1] = 1
            full_kernel = full_kernel * 2.0 - 1  #將kernel轉成-1~1
            filters2_list.append(full_kernel)
        neighbor_array = np.roll(neighbor_array, 1)


filters1 = np.asarray(filters1_list)
filters1 = np.moveaxis(filters1, 0, -1)  #把(34,3,3,1)轉成(3,3,1,34)
filters1 = tf.constant(filters1, dtype=tf.float32)

filters2 = np.asarray(filters2_list)
for i in range(filters2.shape[0]):
    for j in range(filters2.shape[1]):
        for k in range(filters2.shape[2]):
            print(filters2[i, j, k], end='')
        print("\n")
    print("\n")
filters2 = np.moveaxis(filters2, 0, -1)  #把(34,3,3,1)轉成(3,3,1,34)
filters2 = tf.constant(filters2, dtype=tf.float32)

