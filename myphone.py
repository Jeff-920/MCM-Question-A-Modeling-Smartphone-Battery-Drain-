import numpy as np
from scipy.signal import savgol_filter
import cv2
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class softModules():
    def __init__(self, time_resolution: int | float, default_charge: float, if_discharge: bool = True, all_time: float = 3.):
        self.time_resolution = time_resolution
        self.if_discharge = if_discharge
        self.total_charge = default_charge
        self.discharge_time = all_time  # hours
        self.time_squ = np.linspace(0, all_time, int(time_resolution)) 


    def screen_use(self, use_time: float, start_time: float, cpu_cost: float, gpu_cost: float):
        """
        use_time 表示使用比例，比如百分之20的时间使用屏幕就输入0.2
        一次性使用完屏幕时间（因为没有考虑屏幕开关导致的损失，假设可忽略）
        start_time 表示在某时刻点亮屏幕
        screen_cost 表示对cpu增加的使用率
        """
        if start_time + use_time > self.discharge_time:
            print(f"屏幕使用时间过长，仅使用{self.discharge_time - start_time}小时")
        mask = (self.time_squ < (start_time + use_time)) & (self.time_squ > start_time)
        cpu_use_rate = (np.zeros_like(self.time_squ) + cpu_cost) * mask
        gpu_use_rate = (np.zeros_like(self.time_squ) + gpu_cost) * mask
        return cpu_use_rate, gpu_use_rate, start_time, use_time          

    def game_use(self, use_time: float, start_time: float, cpu_cost: float, gpu_cost: float):
        if start_time + use_time > self.discharge_time:
            print(f"游戏使用时间过长，仅使用{self.discharge_time - start_time}小时")
            use_time = self.discharge_time - start_time
        mask = (self.time_squ < (start_time + use_time)) & (self.time_squ > start_time)
        cpu_use_rate = (np.zeros_like(self.time_squ) + cpu_cost) * mask
        gpu_use_rate = (np.zeros_like(self.time_squ) + gpu_cost) * mask
        return cpu_use_rate, gpu_use_rate   
        
    def wifi_use(self, use_time: float, start_time: float, cpu_cost: float, mod: str):
        if start_time + use_time > self.discharge_time:
            print(f"wifi使用时间过长，仅使用{self.discharge_time - start_time}小时")
            use_time = self.discharge_time - start_time
        mask = (self.time_squ < (start_time + use_time)) & (self.time_squ > start_time)
        use_rate = (np.zeros_like(self.time_squ) + cpu_cost) * mask
        return use_rate         
    
    def gps_use(self, use_time: float, start_time: float, cpu_cost: float):
        if start_time + use_time > self.discharge_time:
            print(f"gps使用时间过长，仅使用{self.discharge_time - start_time}小时")
            use_time = self.discharge_time - start_time
        mask = (self.time_squ < (start_time + use_time)) & (self.time_squ > start_time)
        use_rate = (np.zeros_like(self.time_squ) + cpu_cost) * mask
        return use_rate
    
    def screen_rgb(self, img: str, target_res: tuple):
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise ValueError("无法加载图片，检查路径是否正确")

        resized_img = cv2.resize(img, target_res, interpolation=cv2.INTER_AREA)
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        kr = 0.006
        kg = 0.002
        kb = 0.006
        R = rgb_img[:, :, 0].astype(np.float32)
        G = rgb_img[:, :, 1].astype(np.float32)
        B = rgb_img[:, :, 2].astype(np.float32)
        pixel_weight = kr * R + kg * G + kb * B
        total_sum = np.sum(pixel_weight)
        
        return total_sum
    
    def wifi_power(self, use_time: float, start_time: float, cpu_cost: float, mod: str):
        # 分段函数
        # 关断、接受、发送、监听
        # 没有数据qAq，假设使用WiFi期间内，休眠、接受、发送、监听，循环
        wifi_temp = np.sin(self.time_squ) * 250 + 250 # mW
        mask = (self.time_squ < (start_time + use_time)) & (self.time_squ > start_time)
        if mod == 'wifi':
            return wifi_temp * mask
        elif mod == '5g':
            return wifi_temp * mask * 4
        elif mod == '4g':
            return wifi_temp * mask * 2
    
    def gps_power(self, use_time: float, start_time: float, cpu_cost: float):
        # 同理，循环 休眠、捕获、追踪
        gps_temp = np.sin(self.time_squ) * 15 + 15 # mW
        mask = (self.time_squ < (start_time + use_time)) & (self.time_squ > start_time)
        return gps_temp * mask


class myPhone():
    def __init__(self, time_resolution: int | float, default_charge: float = 4800, if_discharge: bool = True,
                 all_time: float = 3., cpu_fre: float = 3.75, cpu_fre_rest: float = 0.4, cpu_cores: int = 6,
                 cpu_volt: float = 0.7, cpu_volt_max: float = 1.2, screen_nit: int = 850, screen_freq: int = 120,
                 screen_size: tuple = (2622, 1206), screen_area: float = 115, ppi: int = 460):
        self.cpu_cores = cpu_cores
        self.cpu_fre = cpu_fre
        self.cpu_fre_rest = cpu_fre_rest
        self.cpu_volt = cpu_volt
        self.cpu_volt_max = cpu_volt_max
        self.screen_ppi = ppi
        self.screen_nit = screen_nit
        self.screen_freq = screen_freq
        self.screen_size = screen_size
        self.screen_area = screen_area * 1e-4  # m2
        self.time_resolution = int(time_resolution)  
        self.if_discharge = if_discharge
        self.total_charge = default_charge  # mAh
        self.discharge_time = all_time  # hours
        self.time_squ = np.linspace(0, all_time, self.time_resolution)
        self.software = softModules(time_resolution, default_charge, if_discharge, all_time)

    def cpu_use(self, game_use: np.ndarray, screen_use: np.ndarray, 
                      wifi_use: np.ndarray, gps_use: np.ndarray, default_use: float = 0.05):
        use_rate = np.zeros_like(self.time_squ) + screen_use + game_use + wifi_use + gps_use + default_use
        use_freq = np.zeros_like(self.time_squ) + self.cpu_fre_rest + (use_rate - default_use) * self.cpu_fre
        use_volt = np.zeros_like(self.time_squ) + self.cpu_volt + (use_rate - default_use) * self.cpu_volt_max

        use_freq = np.clip(use_freq, 0, self.cpu_fre)
        use_volt = np.clip(use_volt, 0, self.cpu_volt_max)
        return use_rate, use_freq, use_volt
    
    def gpu_use(self, game_use: np.ndarray, screen_use: np.ndarray):
        use_rate = np.zeros_like(self.time_squ) + screen_use + game_use
        return use_rate

    def ram_use(self, game_use: np.ndarray, screen_use: np.ndarray, 
                      wifi_use: np.ndarray, gps_use: np.ndarray, default_use: float = 0.01):
        use_rate = np.zeros_like(self.time_squ) + screen_use + game_use + wifi_use + gps_use + default_use
        return use_rate

    def Pt(self, software_set: dict, input_software: list, k: list, return_components: bool = False):
        screen_cpu_rate = np.zeros_like(self.time_squ)
        screen_gpu_rate = np.zeros_like(self.time_squ)
        game_gpu_rate = np.zeros_like(self.time_squ)
        game_cpu_rate = np.zeros_like(self.time_squ)
        wifi_cpu_rate = np.zeros_like(self.time_squ)
        gps_cpu_rate = np.zeros_like(self.time_squ)
        P_screen = np.zeros_like(self.time_squ)
        P_wifi = np.zeros_like(self.time_squ)
        P_gps = np.zeros_like(self.time_squ)
        for name in input_software:
            setting = software_set[name]
            if name == 'screen':
                screen_cpu_rate, screen_gpu_rate, _, _ = self.software.screen_use(** setting)
                screen_use = (self.time_squ < setting['start_time'] + setting['use_time']) & (self.time_squ > setting['start_time'])
                p_pix = 1e-5 * self.screen_nit * self.screen_area * 33.124022 / 0.8  # W
                p_dri = 1.5 * 15  # mW
                P_screen = (p_pix * 1e3 + p_dri) * screen_use # mW
            elif name == 'game':
                game_cpu_rate, game_gpu_rate = self.software.game_use(**setting)
            elif name == 'wifi':
                wifi_cpu_rate = self.software.wifi_use(**setting)
                P_wifi = self.software.wifi_power(**setting) * 0.05
            elif name == 'gps':
                gps_cpu_rate = self.software.gps_use(**setting)
                P_gps = self.software.gps_power(**setting)
            else:
                print(f"警告：无{name}模块，跳过")
        cpu_rate, cpu_fre, cpu_volt = self.cpu_use(game_cpu_rate, screen_cpu_rate, wifi_cpu_rate, gps_cpu_rate)
        gpu_rate = self.gpu_use(game_gpu_rate, screen_gpu_rate)
        ram_rate = self.ram_use(game_cpu_rate, screen_cpu_rate, wifi_cpu_rate, gps_cpu_rate)
        P_cpu = cpu_fre * (cpu_volt ** 2) * cpu_rate * 100
        P_gpu = 120 * self.screen_size[0] * self.screen_size[1] * self.screen_ppi * gpu_rate * 1e-9
        P_raw = ram_rate * 300 * 0.1
        P_cpu, P_gpu, P_raw = np.clip(P_cpu, 0, None), np.clip(P_gpu, 0, None), np.clip(P_raw, 0, None)
        if return_components:
            return P_cpu, P_gpu, P_raw, P_gps, P_screen, P_wifi
        return k[0] * P_cpu + k[1] * P_gpu + k[2] * P_raw + k[3] * P_gps + k[4] * P_screen + k[5] * P_wifi
    
    def postprocess(self, temperature: float, software_set: dict, input_software: list, k: list, alpha: float, window_length: int = 5, polyorder: int = 2):

        factor = 1 - alpha * abs(temperature - 23)
        factor = np.clip(factor, 0.1, None)
        y = self.Pt(software_set, input_software, k)
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= len(y):
            window_length = len(y) - 1 if len(y) > 1 else 1
        y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
        y_corrected = y_smooth / factor
        return y_corrected


class SOC():
    def __init__(self, revolution: int = 100):
        pass

    def read_ocv_soc(self, file_name):
        soc = []
        ocv = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    s = float(parts[0])
                    o = float(parts[1])
                    soc.append(s)
                    ocv.append(o)
        return np.array(soc), np.array(ocv)
    
    def ocv_soc_fun(self, soc, a, b, c, d, m, n):
        # VOC = a + b · (−lns)m + c · s + d · en(s−1)
        x_1 = np.log(1 / soc) ** np.exp(m)
        x_2 = np.exp(n * (soc - 1))
        return a + b * x_1 + c * soc + d * x_2
    
    # def ocv_soc_fun(self, soc, a, b):
    #     VOC = a + b · (−lns)m + c · s + d · en(s−1)
    #     x_1 = np.log(1 / soc) ** np.exp(m)
    #     x_2 = np.exp(n * (soc - 1))
    #     g = a * soc  + b
    #     return g # a + b * x_1 + c * soc + d * x_2
    
    def fit_fun(self, file_name):
        # init_params = [3.2, 0.1, 0.002, 0.1, 0.5, 5.0]
        init_params = [3.2, 0.1]
        soc, ocv = self.read_ocv_soc(file_name=file_name)
        params, _ = curve_fit(self.ocv_soc_fun, soc, ocv, p0=init_params,  maxfev=10000)
        plt.figure(figsize=(6, 4))
        plt.scatter(soc[::4], ocv[::4], label='Mean OCV-SOC Data', color="#1DCEDB", s=25)
        plt.plot(soc, self.ocv_soc_fun(soc, *params), label='Fitted Curve', color="#1DCEDB", linewidth=2.5)
        plt.xlabel('SOC')
        plt.ylabel('OCV (V)')
        plt.legend()
        plt.savefig(r"soc_data\soc_ocv_fit.png", dpi=300)
        print(f"R²: {r2_score(ocv, self.ocv_soc_fun(soc, *params))}")
        print(f"Fitted Parameters: {params}")
        return params
    
