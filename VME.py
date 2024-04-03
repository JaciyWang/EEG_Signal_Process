# 变分模式提取 Variational Mode Extraction
def vme(signal, Alpha=3500, omega_int=0.0028, tau=0, tol=1e-6):
        # 对输入信号进行镜像扩展，使其长度变为原来的两倍
        T = len(signal)
        f_mir = np.zeros(T * 2)
        f_mir[:T // 2] = signal[T // 2 - 1::-1]  # 前一半的逆序
        f_mir[T // 2:T * 3 // 2] = signal
        f_mir[T * 3 // 2:2 * T] = signal[T - 1:T // 2 - 1:-1]  # 后一半逆序
        f = f_mir

        # 计算时间域和频域的离散化坐标
        T = len(f)
        t = np.arange(1, T+1)/T   # t: 0->T
        udiff = tol + np.finfo(float).eps  # ------ update step
        omega_axis = t - 0.5 - 1 / T

        # 计算输入信号的傅里叶变换(Hilbert transform)，并将其变为单边谱
        f_hat = np.fft.fftshift(np.fft.fft(f))
        f_hat_onesided = f_hat.copy()
        f_hat_onesided[:T // 2] = 0

        N = 300  # 最大迭代次数设置为300
        omega_d = np.zeros(N)
        omega_d[0] = omega_int  # 初始化中心频率为初始猜测值

        dual_vector = np.zeros((N, len(omega_axis)), dtype=complex)  # 初始化对偶变量向量为零向量
        u_hat_d = np.zeros((N, len(omega_axis)), dtype=complex)  # 初始化模式谱的变化为零矩阵

        # 循环来迭代更新模式谱、模式中心频率和对偶变量，直到收敛或达到最大迭代次数
        n = 0
        while udiff > tol and n < N-1:
            # 基于变分原理和拉格朗日乘子法，更新模式谱 ud
            u_hat_d[n+1, :] = (f_hat_onesided +
                               (u_hat_d[n, :] * (Alpha**2) * (omega_axis - omega_d[n])**4) +
                               dual_vector[n, :]/2) / \
                              ((1 + (Alpha**2) * (omega_axis - omega_d[n])**4) *
                               (1 + 2*Alpha**2 * (omega_axis - omega_d[n])**4))  # **2
            # 基于模式谱的加权平均，更新模式中心频率 omega_d
            tmp = np.abs(u_hat_d[n + 1, T//2:T]) ** 2
            omega_d[n + 1] = (omega_axis[T//2:T] @ tmp.T.conjugate()) / np.sum(tmp)
            # 基于对偶上升法，更新对偶变量
            dual_vector[n+1, :] = dual_vector[n, :] + (tau * (f_hat_onesided - (u_hat_d[n, :])) / (1 + (Alpha ** 2) * (omega_axis - omega_d[n]) ** 4))
            n = n+1

            if n == 1:
                udiff = np.finfo(float).eps
            ttmp = u_hat_d[n, :] - u_hat_d[n - 1, :]
            udiff = udiff + 1 / T * ttmp @ np.conj(ttmp).T
            udiff = abs(udiff)
            # print(udiff)

        # 信号重构
        N = min(N, n)  # 选择最小的迭代次数和主循环计数器作为最终的迭代次数
        omega_d = omega_d.reshape((len(omega_d), 1))
        # print(N, omega_d.shape, u_hat_d.shape)
        omega = omega_d[:N, :]
        # 从最终的模式谱中提取单边谱，并用其共轭补全双边谱
        u_hatd = np.zeros((T, 1), dtype=complex)
        # 将u_hat_d的第N层，从T/2到T的部分赋给u_hatd的从T/2到T的部分
        l = len(u_hat_d[N, T//2:T])
        u_hatd[T//2:T, :] = u_hat_d[N, T//2:T].reshape((l, 1))
        # 将u_hat_d的第N层，从T/2到T的部分的共轭赋给u_hatd的从T / 2 + 1 到2的部分，按照逆序
        u_hatd[T//2:0:-1, :] = np.conj(u_hat_d[N, T//2:T].reshape((l, 1)))
        u_hatd[0, :] = np.conj(u_hatd[-1, :])

        # 对双边谱进行逆傅里叶变换，得到时间域的模式信号
        u_d = np.zeros((1, len(t)))
        # 对u_hatd的第一列进行逆傅里叶变换，并取实部，赋给u_d的第一行
        u_d[0, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hatd[:, 0])))

        # 将u_d从T/4+1到3*T/4的部分赋给自己，去除镜像扩展的部分，得到原始长度的模式信号
        u_d = u_d[:, T//4:3*T//4]
        u_hatd = np.fft.fftshift(np.fft.fft(u_d[0, :])).T

        # u_d：所提取的模式，u_hatd：所提取模式的频谱，omega：估计的模式中心频率
        return u_d, u_hatd, omega


# 检测窗口信号sig中的峰值，并返回峰值的索引k1，用了vme函数来计算信号的能量包络eeb，然后根据eeb的局部极大值和中位数绝对偏差来筛选峰值。
# k1是眨眼尖峰的信号帧id
def detect(sig, fs):
    # 正态化
    sig = (sig - sig.mean()) / sig.std()

    sig = np.concatenate([np.zeros(fs//8), sig, np.zeros(3*fs//8)])
    eeb, u_hatd, omega = vme(sig, 3500, 0.0028, 0, 1e-6)  # 3000
    eeb = np.squeeze(eeb)  # (1, 700) => (700, )

    # 找到eeb中的谷峰
    k, _ = signal.find_peaks(eeb, distance=50)

    r = eeb[k]
    k1 = []
    m = np.median(np.abs(eeb)) / 0.6745 * np.sqrt(2 * np.log(len(eeb)))
    for i in range(len(r)):
        if r[i] > m:
            k1.append(k[i])
        else:
            pass

    # plt.title('eog')
    # plt.plot(np.arange(len(eeb)) / fs, eeb)
    # for each in k1:
    #     plt.axvline(each/fs, color='red')
    # plt.show()

    k1 = [i - fs//8 for i in k1]
    return k1
