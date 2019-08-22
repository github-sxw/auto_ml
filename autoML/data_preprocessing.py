# -*- coding: utf-8 -*-
# @Time : 2019/8/7 9:26
# @Author : sxw
import os
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt, lfilter, firwin, get_window
import matplotlib
import matplotlib.pyplot as plt
import math


def get_array(array):
    """统一转换为np.array格式的数据"""
    if isinstance(array, np.ndarray):
        if len(array.shape) == 1:
            return array
        elif len(array.shape) == 2 and (array.shape[0] == 1 or array.shape[1] == 1):
            return array.reshape(-1)
        else:
            raise TypeError("The dimension of the numpy.array must be 1 or 2")
    elif isinstance(array, (list, pd.Series)):
        array = np.array(array)
        return get_array(array)
    else:
        raise TypeError("Input must be a numpy.array, list or pandas.Series")


def moving_smooth(series, box_pts=5, n_inter=1):
    """
    调用numpy.convolve函数，并优化边缘处理
    :param series: 要平滑的序列
    :param box_pts: 滑动窗口的大小
    :param n_inter: 平滑次数
    :return: 与原序列相同长度的平滑之后的序列
    """
    array = get_array(series)
    if box_pts % 2 == 1 and box_pts > 0:
        for i in range(n_inter):
            box = np.ones(box_pts) / box_pts
            temp_array = array.copy()
            middle_slice = np.convolve(temp_array, box, mode='valid')
            r = np.arange(1, box_pts - 1, 2)
            start_slice = np.cumsum(array[:box_pts - 1])[::2] / r
            stop_slice = (np.cumsum(array[:-box_pts:-1])[::2] / r)[::-1]
            smooth_array = np.concatenate((start_slice, middle_slice, stop_slice))
            array = smooth_array.copy()
        return array
    else:
        raise ValueError("param 'box_pts' must be odd number")


def lms_filtering(xn, dn, order, mu, itr):
    """
    自适应最小均方误差(LMS)算法
    :param xn:输入信号
    :param dn:参考信号
    :param order:滤波器阶数
    :param mu:收敛因子
    :param itr:采样的点数
    :return:yn为输出信号
            en为误差信号
    """
    en = np.zeros(itr)
    W = [np.zeros(order - 1) for _ in range(itr)]
    for k in range(itr)[order-1:itr]:
        x = xn[k - order + 1:k].copy()[::-1]    # 取定给定的反向的个数
        d = x.mean()                            # 用原始信号n时刻前order-1个点代替参考信号的n时刻的值
        y = np.sum(W[k - 1] * x)
        en[k] = d - y
        W[k] = np.add(W[k - 1], 2 * mu * en[k] * x)  # 更新权重

    yn = np.inf * np.ones(len(xn))              # 求最优时滤波器的输出序列
    for k in range(len(xn))[order - 1:len(xn)]:
        x = xn.copy()[k - order + 1:k].copy()[::-1]
        yn[k] = np.sum(W[len(W) - 1] * x)

    return yn, en


class IirFiltering:
    """
    调用Scipy.signal.butter中的Butterworth滤波器,属于无限冲激响应（IIR）滤波器
    """
    def __init__(self, order=20, sampling_frequency=25600):
        self.order = order                              # 滤波器的阶数
        self.sampling_frequency = sampling_frequency    # 采样频率

    def low_filtering(self, cutoff_frequency, series):
        """
        低通滤波
        :param cutoff_frequency: 信号的截止频率
        :param series: 要过滤的目标信号
        :return: 低通滤波过滤之后的信号
        """
        wn = 2 * cutoff_frequency / self.sampling_frequency
        array = get_array(series)
        b, a = butter(self.order, wn, btype="lowpass", output="ba")
        filtered_array = filtfilt(b, a, array)
        # filtered_array = lfilter(b, a, array)
        return filtered_array

    def high_filtering(self, cutoff_frequency, series):
        """
        高通滤波
        :param cutoff_frequency: 信号的截止频率
        :param series: 要过滤的目标信号
        :return: 高通滤波过滤之后的信号
        """
        wn = 2 * cutoff_frequency / self.sampling_frequency
        array = get_array(series)
        b, a = butter(self.order, wn, btype="highpass", output="ba")
        filtered_array = filtfilt(b, a, array)
        # filtered_array = lfilter(b, a, array)
        return filtered_array

    def band_filtering(self, lowcut, highcut, series):
        """
        带通滤波
        :param lowcut:信号截止频率下限
        :param highcut:信号截止频率上限
        :param series:要过滤的目标信号
        :return:带通滤波过滤之后的信号
        """
        low = 2 * lowcut / self.sampling_frequency
        high = 2 * highcut / self.sampling_frequency
        array = get_array(series)
        b, a = butter(self.order, [low, high], btype="bandpass", output="ba")
        filtered_array = filtfilt(b, a, array)
        return filtered_array


class FirFiltering:
    """
    Use firwin with a window to create a lowpass FIR filter
    """
    def __init__(self, window_name='hanning', order=20, sampling_frequency=25600):
        """
        :param window_name:窗函数的名称,参照scipy.signal.get_window函数中的窗函数名称
        :param order:The number of taps in the FIR filter
        :param sampling_frequency:采样频率
        """
        self.len_filter = order + 1
        self.window = window_name
        self.sampling_frequency = sampling_frequency

    def low_filtering(self, cutoff_frequency, series):
        """
        低通滤波
        :param cutoff_frequency: 信号的截止频率
        :param series: 要过滤的目标信号
        :return: 低通滤波过滤之后的信号
        """
        array = get_array(series)
        taps = firwin(self.len_filter, 2 * cutoff_frequency / self.sampling_frequency,
                      window=self.window, pass_zero='lowpass')
        filtered_array = filtfilt(taps, 1.0, array)
        return filtered_array

    def high_filtering(self, cutoff_frequency, series):
        """
        高通滤波
        :param cutoff_frequency: 信号的截止频率
        :param series: 要过滤的目标信号
        :return: 高通滤波过滤之后的信号
        """
        array = get_array(series)
        taps = firwin(self.len_filter, 2 * cutoff_frequency / self.sampling_frequency,
                      window=self.window,  pass_zero='highpass')
        filtered_array = filtfilt(taps, 1.0, array)
        return filtered_array

    def band_filtering(self, lowcut, highcut, series):
        """
        带通滤波
        :param lowcut:信号截止频率下限
        :param highcut:信号截止频率上限
        :param series:要过滤的目标信号
        :return:带通滤波过滤之后的信号
        """
        array = get_array(series)
        low = 2 * lowcut / self.sampling_frequency
        high = 2 * highcut / self.sampling_frequency
        taps = firwin(self.len_filter, [low, high], window=self.window, pass_zero='bandpass')
        filtered_array = filtfilt(taps, 1.0, array)
        return filtered_array


class Windows:
    def __init__(self, nx=20):
        self.N = nx       # 窗函数的大小

    def rectangle(self):
        rectangle_window = np.ones(self.N)
        return rectangle_window

    def hanning(self):
        hanning_window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (self.N - 1)) for n in range(self.N)])
        return hanning_window

    def hamming(self):
        hamming_window = np.array([0.53836 - 0.46164 * np.cos(2 * np.pi * n / (self.N - 1)) for n in range(self.N)])
        return hamming_window

    def fejer(self):
        """三角窗（费杰窗）"""
        fejer_window = np.array([2 / self.N * (self.N / 2 - np.abs(n - (self.N - 1) / 2)) for n in range(self.N)])
        return fejer_window

    def flap_top(self):
        """平顶窗"""
        flap_top_window = np.array([1 - 1.93 * np.cos(2 * np.pi * n / (self.N - 1))
                                    + 1.29 * np.cos(4 * np.pi * n / (self.N - 1))
                                    - 0.388 * np.cos(6 * np.pi * n / (self.N - 1))
                                    + 0.032 * np.cos(8 * np.pi * n / (self.N - 1)) for n in range(self.N)])
        return flap_top_window
        # window_name = 'flattop'
        # return get_window(window_name, self.N)

    def gaussian(self, xigema=0.5):
        """高斯窗"""
        if xigema > 0.5 or xigema <= 0:
            raise ValueError("0<σ<=0.5")
        gaussian_windows = [np.power(np.e, -0.5 * np.power((n - (self.N - 1) / 2) /
                                                           (2 * xigema * (self.N - 1)), 2)) for n in range(self.N)]
        return gaussian_windows


if __name__ == '__main__':
    # N = 500
    # fs = 5
    # n = [2 * math.pi * fs * t / N for t in range(N)]
    # axis_x = np.linspace(0, 1, num=N)
    # myfont = matplotlib.font_manager.FontProperties(fname='C:\\Windows\\Fonts\\simsun.ttc')
    # x = [math.sin(i) for i in n]       # 5Hz的正弦信号
    # x1 = [math.sin(i*5) for i in n]    # 25Hz的正弦信号
    # x2 = [math.sin(i*10) for i in n]   # 50Hz的正弦信号
    # xx = [x[i] + x1[i] + x2[i] for i in range(len(x))]
    #
    # plt.subplot(241)
    # plt.plot(axis_x, x)
    # plt.title('5Hz的正弦信号', fontproperties=myfont)
    # plt.axis('tight')
    # plt.subplot(242)
    # plt.plot(axis_x, x1)
    # plt.title('25HZ的正弦信号', fontproperties=myfont)
    # plt.axis('tight')
    # plt.subplot(243)
    # plt.plot(axis_x, x2)
    # plt.title('50Hz的正弦信号', fontproperties=myfont)
    # plt.axis('tight')
    # plt.subplot(244)
    # plt.plot(axis_x, xx)
    # plt.title('5HZ、25HZ、50Hz的正弦信号', fontproperties=myfont)
    # plt.axis('tight')
    #
    # # filtering = FirFiltering('hamming', 60, 500)   # FIR
    # filtering = IirFiltering(9, 500)             # IIR滤波
    #
    # sf1 = filtering.low_filtering(25, xx)
    #
    # plt.subplot(245)
    # plt.plot(axis_x, sf1)
    # plt.title('低通滤波后', fontproperties=myfont)
    # plt.axis('tight')
    #
    # sf2 = filtering.high_filtering(40, xx)
    #
    # plt.subplot(246)
    # plt.plot(axis_x, sf2)
    # plt.title(u'高通滤波后', fontproperties=myfont)
    # plt.axis('tight')
    #
    # sf3 = filtering.band_filtering(15, 40, xx)
    # plt.subplot(247)
    # plt.plot(axis_x, sf3)
    # plt.title('带通滤波后', fontproperties=myfont)
    # plt.axis('tight')
    #
    # sf4 = moving_smooth(xx, 11, 10)
    # plt.subplot(248)
    # plt.plot(axis_x, sf4)
    # plt.title('移动平滑之后', fontproperties=myfont)
    # plt.axis('tight')
    # n = 200
    # win = Windows(n)
    # # result = win.rectangle()
    # # result = win.hanning()
    # # result = win.hamming()
    # # result = win.fejer()
    # result = win.flap_top()
    # # result = win.gaussian()
    # plt.scatter(np.array(range(n)), result)
    # plt.show()

    # 参数初始化
    itr = 10000  # 采样的点数
    noise_size = itr
    X = np.linspace(0, 4 * np.pi, itr, endpoint=True)
    Y = np.sin(X)
    signal_array = Y  # [0.0]*noise_size
    noise_array = np.random.normal(0, 0.3, noise_size)
    """noise_array = []
    for x in range(itr):
        noise_array.append(random.gauss(mu,sigma))"""
    signal_noise_array = signal_array + noise_array
    order = 64  # 滤波器的阶数
    mu = 0.0001  # 步长因子
    xs = signal_noise_array
    xn = xs  # 原始输入端的信号为被噪声污染的正弦信号
    dn = signal_array  # 对于自适应对消器，用dn作为期望
    # 调用LMS算法

    (yn, en) = lms_filtering(xn, dn, order, mu, itr)

    # 画出图形
    plt.figure(1)
    plt.plot(xn, label="$xn$")
    plt.plot(dn, label="$dn$")
    plt.xlabel("Time(s)")
    plt.ylabel("Volt")
    plt.title("original signal xn and desired signal dn")
    plt.legend()

    plt.figure(2)
    # plt.plot(xn,label="$xn$")
    # plt.plot(xn,label="$xn$")
    plt.plot(dn, label="$dn$")
    plt.plot(yn, label="$yn$")
    plt.xlabel("Time(s)")
    plt.ylabel("Volt")
    plt.title("original signal xn and processing signal yn")
    plt.legend()

    plt.figure(3)
    plt.plot(en, label="$en$")
    plt.xlabel("Time(s)")
    plt.ylabel("Volt")
    plt.title("error between processing signal yn and desired voltage dn")
    plt.legend()
    plt.show()



