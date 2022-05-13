import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

filters1_list = []

rotation_index = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]


class AdaptiveThreshold:
    # init是constructor
    def __init__(self, blockSize, C):
        # 1個blockSize*blockSize＊1的KERNEL
        self.filters = tf.ones((blockSize, blockSize, 1, 1), dtype=tf.float32) / blockSize ** 2
        self.C = tf.constant(C, dtype=tf.float32)

    def __call__(self, inputs):
        # hint: tf.nn.conv2d, tf.where
        mean = tf.nn.conv2d(inputs, self.filters, strides=[1, 1], padding='SAME')
        x = tf.where(inputs > (mean - self.C), 1.0, 0.0)

        return x  # return the resultant image, where 1 represents above the threshold and 0 represents below the threshold


# 檢查rule 1的最後一個規則
def rule1_final_check(array):
    n_135_is_zero = False
    n_357_is_zero = False
    if array[1] == 0 or array[3] == 0 or array[5] == 0:
        n_135_is_zero = True
    if array[3] == 0 or array[5] == 0 or array[7] == 0:
        n_357_is_zero = True
    return n_135_is_zero and n_357_is_zero


# 檢查rule 2的最後一個規則
def rule2_final_check(array):
    n_137_is_zero = False
    n_157_is_zero = False
    if array[1] == 0 or array[3] == 0 or array[7] == 0:
        n_137_is_zero = True
    if array[1] == 0 or array[5] == 0 or array[7] == 0:
        n_157_is_zero = True
    return n_137_is_zero and n_157_is_zero


filters1_list = []
filters2_list = []

for neighbor_one_count in range(2, 7, 1):
    neighbor_array = np.ones(neighbor_one_count)
    neighbor_array = np.append(neighbor_array, [0.0] * (8 - neighbor_one_count))
    # 前兩個規則代表了鄰居中的1繞一圈看的話是要相連再一起的
    # 所以以鄰居有兩個1的為例
    # [1 1 0]
    # [0 p 0]
    # [0 0 0]
    # 我只需要把這個kernel一格格的旋轉一圈
    # 只要符合最後一個條件(final_check)就可以作為kernel
    print("neighbor_one_count:", neighbor_one_count)
    for rotation in range(8):
        if rule1_final_check(neighbor_array):
            full_kernel = np.zeros((3, 3, 1))
            for i in range(8):
                full_kernel[rotation_index[i]] = neighbor_array[i]
            full_kernel[1, 1] = 1
            print(full_kernel)
            full_kernel = full_kernel * 2.0 - 1  # 將kernel轉成-1~1
            filters1_list.append(full_kernel)
        neighbor_array = np.roll(neighbor_array, 1)

for neighbor_one_count in range(2, 7, 1):
    neighbor_array = np.ones(neighbor_one_count)
    neighbor_array = np.append(neighbor_array, [0.0] * (8 - neighbor_one_count))
    # 前兩個規則代表了鄰居中的1繞一圈看的話是要相連再一起的
    # 所以以鄰居有兩個1的為例
    # [1 1 0]
    # [0 p 0]
    # [0 0 0]
    # 我只需要把這個kernel一格格的旋轉一圈
    # 只要符合最後一個條件(final_check)就可以作為kernel
    print("neighbor_one_count:", neighbor_one_count)
    for rotation in range(8):
        if rule2_final_check(neighbor_array):
            full_kernel = np.zeros((3, 3, 1))
            for i in range(8):
                full_kernel[rotation_index[i]] = neighbor_array[i]
            print(full_kernel)
            full_kernel[1, 1] = 1
            full_kernel = full_kernel * 2.0 - 1  # 將kernel轉成-1~1
            filters2_list.append(full_kernel)
        neighbor_array = np.roll(neighbor_array, 1)

filters1 = np.asarray(filters1_list)
filters1 = np.moveaxis(filters1, 0, -1)  # 把(34,3,3,1)轉成(3,3,1,34)

filters1 = np.asarray(filters1_list)
filters1 = np.moveaxis(filters1, 0, -1)  # 把(34,3,3,1)轉成(3,3,1,34)
filters1 = tf.constant(filters1, dtype=tf.float32)
print(filters1[:,:,:,2])

# 下載測試影像
url = 'https://evatronix.com/images/en/offer/printed-circuits-board/Evatronix_Printed_Circuits_Board_01_1920x1080.jpg'
testimage = tf.keras.utils.get_file('pcb.jpg', url)  # 記得改網址的話 檔名也要改掉

# 讀入測試影像
inputs = cv2.imread(testimage)

# 轉成灰階影像
inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2GRAY)  # 轉成灰階
inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
inputs = inputs[tf.newaxis, :, :, tf.newaxis]  # 擴增兩個維度

binary = AdaptiveThreshold(61, -8)(inputs)
plt.imshow(tf.squeeze(binary).numpy() * 255, cmap='gray')  # 把大於零的部分會被設為255，小於零的部分會被設為0
image = binary * 2 - 1  # 把前景的部分設為1，背景的部分設為-1

# 存下AdaptiveThresholding結果
# binary這張圖現去掉大小為1的維度，再把大於零的部分會被設為255，小於零的部分會被設為0
plt.imshow(tf.squeeze(binary).numpy() * 255, cmap='gray')  # 把大於零的部分會被設為255，小於零的部分會被設為

x = tf.pad(image, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), constant_values=-1.0)
# tf.nn.conv2d, tf.math.reduce_max, tf.where
pass_one_has_modified = False
pass_two_has_modified = False
# add your code for rule 1
pass1_result = tf.nn.conv2d(x, filters1, strides=[1, 1], padding='VALID')
pass1_result_max = tf.math.reduce_max(pass1_result, axis=-1, keepdims=True)
nine_count_1 = (tf.math.reduce_max(pass1_result, axis=-1, keepdims=True) == 9.0)
nine_location = tf.where(pass1_result_max==tf.constant(9.0,dtype=tf.float32))
print("number of nine:", tf.where(pass1_result_max==tf.constant(9.0,dtype=tf.float32)).shape[0])
pass1_result_max = tf.where(pass1_result_max == 9.0, 1.0, -1.0)
pass_one_has_modified = False
if (tf.where(pass1_result_max==tf.constant(9.0,dtype=tf.float32)).shape[0]) > 0:
    pass_one_has_modified = True

inputs2 = tf.where(pass1_result_max==tf.constant(9.0,dtype=tf.float32), tf.constant(-1.0,dtype=tf.float32), inputs)
difference = tf.where(inputs!=inputs2)
print("HHHH")