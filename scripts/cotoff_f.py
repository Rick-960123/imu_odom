import numpy as np
from scipy.fft import fft

def calculate_cutoff_frequency(data, sample_rate, threshold):
    # 执行傅里叶变换
    spectrum = np.abs(fft(data))
    
    # 计算频率轴
    freq_axis = np.fft.fftfreq(len(data), 1/sample_rate)
    
    # 计算总能量
    total_energy = np.sum(spectrum)
    
    # 计算能量下降的阈值
    threshold_energy = total_energy * threshold
    
    # 找到能量下降到阈值以下的频率点
    cutoff_freq = freq_axis[np.where(np.cumsum(spectrum) <= threshold_energy)][-1]
    
    return cutoff_freq

# 示例数据
data = np.random.randn(1000)  # 替换为你的实际数据
sample_rate = 1000  # 替换为采样率

# 计算截至频率，这里选择能量下降到总能量的0.95以下作为阈值
threshold = 0.7
cutoff_frequency = calculate_cutoff_frequency(data, sample_rate, threshold)

print("截至频率: {:.2f} Hz".format(cutoff_frequency))
