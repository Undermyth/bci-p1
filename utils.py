import numpy as np
from numba import njit

def not_moving_filter(cursor_pos, accumulate_step = 10, threshold = 5):
    mask = np.zeros(cursor_pos.shape[0])
    for i in range(len(cursor_pos)):
        if i - accumulate_step < 0:
            mask[i] = 0
        else:
            if np.linalg.norm(cursor_pos[i] - cursor_pos[i - accumulate_step]) >= threshold:
                mask[i] = 1
            else:
                mask[i] = 0
    return mask

def get_bins(time_stamps, time_bin, mask):
    record_interval = time_stamps[1] - time_stamps[0]
    index_interval = int(np.round(time_bin / record_interval))

    start_index = 0
    bins = []
    while start_index + index_interval < len(time_stamps):
        if np.all(mask[start_index: start_index + index_interval]):
            bins.append([start_index, start_index + index_interval]) # [, )
            start_index += index_interval
        else:
            start_index += 1
        
    return np.stack(bins, axis=0)

# [important] numba is needed for speed
@njit
def collect_spikes(spike_stamps, bin_indexs, time_stamps):
    num_bins = bin_indexs.shape[0]
    bins = np.zeros(num_bins)
    for spike_time in spike_stamps:
        for index in range(num_bins):  # 使用range替代enumerate，因为numba对range支持更好
            start_index, end_index = bin_indexs[index]
            if time_stamps[start_index] <= spike_time <= time_stamps[end_index]:
                bins[index] += 1
                break
    return bins

def get_angles(cursor_pos, bin_indexs):
    angle_vecs = np.zeros((bin_indexs.shape[0], 2))
    for index, (start_index, end_index) in enumerate(bin_indexs):
        angle_vec = cursor_pos[end_index] - cursor_pos[start_index]
        angle_vec = angle_vec / np.linalg.norm(angle_vec)
        angle_vecs[index] = angle_vec
    return angle_vecs

def aggregate_angle_bins(angle_vecs, angle_bins):
    angles = np.degrees(np.arctan2(angle_vecs[:, 1], angle_vecs[:, 0]))
    bin_edges = np.linspace(-180, 180, angle_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(angles, bin_edges) - 1
    aggregated_angle_vecs = np.stack([np.cos(np.radians(bin_centers)), np.sin(np.radians(bin_centers))], axis=1)
    return aggregated_angle_vecs, bin_indices

def aggregate_spike_bins(spike_counts, bin_indices, angle_bins):
    aggregated_values = np.bincount(bin_indices, weights=spike_counts, minlength=angle_bins)
    return aggregated_values

def get_velocity(cursor_pos, time_stamps, bin_indexs):
    velocity = np.zeros((bin_indexs.shape[0], 2))
    for index, (start_index, end_index) in enumerate(bin_indexs):
        velocity[index] = (cursor_pos[end_index] - cursor_pos[start_index]) / (time_stamps[end_index] - time_stamps[start_index])
    return velocity

def get_position(cursor_pos, time_stamps, bin_indexs):
    position = np.zeros((bin_indexs.shape[0], 2))
    for index, (start_index, end_index) in enumerate(bin_indexs):
        position[index] = cursor_pos[start_index]
    return position

import numpy as np

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x0):
        """
        初始化卡尔曼滤波器。
        
        参数:
        A: 状态转移矩阵 (n x n)
        B: 控制矩阵 (n x m)，如果没有控制输入，可以设置为零矩阵
        H: 测量矩阵 (k x n)
        Q: 过程噪声协方差矩阵 (n x n)
        R: 测量噪声协方差矩阵 (k x k)
        P: 初始误差协方差矩阵 (n x n)
        x0: 初始状态 (n x 1)
        """
        self.A = A  # 状态转移矩阵
        self.B = B  # 控制矩阵
        self.H = H  # 测量矩阵
        self.Q = Q  # 过程噪声协方差
        self.R = R  # 测量噪声协方差
        self.P = P  # 初始误差协方差
        self.x = x0  # 初始状态估计

    def predict(self, u=None):
        """
        时间更新（预测步骤）
        
        参数:
        u: 控制输入向量 (m x 1)，如果没有控制输入，可以传入 None。
        """
        if u is None:
            u = np.zeros((self.B.shape[1], 1))  # 若无控制输入，设为零向量

        # 状态预测
        self.x = np.dot(self.A, self.x)
        
        # 协方差预测
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x

    def update(self, z):
        """
        测量更新（校正步骤）
        
        参数:
        z: 当前测量值 (k x 1)
        """
        # 计算 Kalman 增益
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # 测量预测协方差
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman 增益

        # 状态更新
        y = z - np.dot(self.H, self.x)  # 计算测量残差（创新）
        self.x = self.x + np.dot(K, y)

        # 协方差更新
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - np.dot(K, self.H)) @ self.P

    def get_state(self):
        """
        获取当前的状态估计。
        """
        return self.x

    def get_covariance(self):
        """
        获取当前的误差协方差。
        """
        return self.P

def linear_fit(x, y):
    '''
    fitting linear transformation Y = X * W, 
    X.shape = [N, D], W.shape = [D, K]
    y.shape = [N, K]
    '''
    pseudo_inverse = np.linalg.inv(x.T @ x)
    return pseudo_inverse @ x.T @ y
