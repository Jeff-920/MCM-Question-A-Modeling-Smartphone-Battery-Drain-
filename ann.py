import numpy as np
from scipy.signal import savgol_filter
import cv2
import pickle
from typing import List, Dict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class FitPlotter:
    def __init__(self, figsize=(6, 4), colors=None, title_fontsize=14, label_fontsize=14):
        self.figsize = figsize
        self.colors = colors or {
            'loss': "#359AEC", 'fit_scatter': "#F2A4E1", 'ideal_line': "#FA3041",
            'residuals': '#E76F51', 'components': '#4CC9F0'
        }
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize

    def plot_loss(self, history, savepath='train_loss.png'):
        plt.figure(figsize=self.figsize)
        n = len(history.get('mse_norm', []))
        if n == 0:
            print('No history to plot for loss')
            return
        x = np.arange(1, n+1)
        plt.plot(x, history['mse_norm'], color=self.colors['loss'], linewidth=2)
        plt.xscale('log')
        plt.xlabel('Epoch (log scale)', fontsize=self.label_fontsize)
        plt.ylabel('Normalized MSE', fontsize=self.label_fontsize)
        plt.title('Training Normalized Loss', fontsize=self.title_fontsize)
        plt.grid(alpha=0.3)
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"Saved loss plot: {savepath}")

    def plot_fit(self, preds, actual, savepath='fit_scatter.png'):
        preds = np.asarray(preds)
        actual = np.asarray(actual)
        plt.figure(figsize=self.figsize)
        plt.scatter(actual, preds, color=self.colors['fit_scatter'], alpha=0.6, s=10)
        mn = min(actual.min(), preds.min())
        mx = max(actual.max(), preds.max())
        plt.plot([mn, mx], [mn, mx], color=self.colors['ideal_line'], linestyle='--', linewidth=2)
        plt.xlabel('Actual (mW)', fontsize=self.label_fontsize)
        plt.ylabel('Predicted (mW)', fontsize=self.label_fontsize)
        plt.title('Model Fit: Predicted vs Actual', fontsize=self.title_fontsize)
        plt.grid(alpha=0.3)
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"Saved fit plot: {savepath}")

    def plot_residuals(self, preds, actual, savepath='residuals.png'):
        preds = np.asarray(preds)
        actual = np.asarray(actual)
        errors = preds - actual
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        # histogram limited to [-100, 100]
        bins = np.linspace(-100, 100, 41)
        ax1.hist(errors, bins=bins, color=self.colors['residuals'], alpha=0.7)
        ax1.set_xlim(-100, 100)
        ax1.set_title('Residuals Distribution')
        ax1.set_xlabel('Error (mW)')
        ax1.set_ylabel('Count')
        ax2.plot(errors, color=self.colors['residuals'], linewidth=1)
        ax2.set_title('Residuals over Samples')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Error (mW)')
        # clip y axis to [-100, 100]
        ax2.set_ylim(-100, 100)
        plt.tight_layout()
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"Saved residuals plot: {savepath}")

    def plot_components(self, k_vals, comp_means, labels=None, savepath='components.png'):
        labels = labels or ['P_cpu','P_gpu','P_raw','P_gps','P_screen','P_wifi']
        contrib = np.array(k_vals) * np.array(comp_means)
        total = contrib.sum()
        pct = contrib / (total + 1e-12) * 100
        fig, ax = plt.subplots(figsize=(10,6))
        bars = ax.bar(labels, contrib, color=self.colors['components'], alpha=0.8)
        ax.set_ylabel('Absolute Contribution (mW)', fontsize=self.label_fontsize)
        ax.set_title('Component Contributions to Total Power', fontsize=self.title_fontsize)
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{pct[i]:.1f}%", ha='center', va='bottom', fontsize=10)
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"Saved components plot: {savepath}")


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
        self.time_resolution = int(time_resolution)  # 修正为int
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
        # 物理约束：频率/电压非负且不超过最大值
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


def build_component_dataset(samples_list: List[Dict], phone: myPhone, input_software: list,
                            include_temperature: bool = False, use: bool = False):
    """
    Build dataset from samples.
    - include_temperature: whether to append sample['temperature'] as a feature
    Note: alpha is not used as an input feature because the model predicts alpha.
    """
    X_list = []
    y_list = []
    for sample in samples_list:
        comps = phone.Pt(sample['software_set'], input_software, k=[0,0,0,0,0,0], return_components=True)
        means = [float(np.mean(c)) if np.asarray(c).size > 0 else 0.0 for c in comps]
        if include_temperature:
            means.append(float(sample.get('temperature', 23.0)))
        X_list.append(means)
        y_list.append(float(sample['power_consumption_mw']))
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    if use:
        return X, y, comps
    else:
        return X, y


class TorchANN(nn.Module):
    def __init__(self, input_dim=7, hidden_sizes=(32, 64, 128, 64, 32), output_dim=7):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_sizes)
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
        # 权重初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def train_torch_ann(samples_list: List[Dict], phone: myPhone, input_software: list,
                    epochs: int = 1000, lr: float = 1e-3, batch_size: int = 64,
                    device: str = 'cpu', hidden_sizes=(32, 64, 128, 128, 128, 32, 16),
                    normalize_y: bool = True, include_temperature: bool = True,
                    val_ratio: float = 0.15, seed: int = 42):
    if torch is None:
        raise RuntimeError('PyTorch 未安装，请先 pip install torch')
    X_all, y_all = build_component_dataset(samples_list, phone, input_software, include_temperature=include_temperature)
    x_mean = X_all.mean(axis=0, keepdims=True)
    x_std = X_all.std(axis=0, keepdims=True) + 1e-9
    Xn_all = (X_all - x_mean) / x_std

    if normalize_y:
        y_mean = y_all.mean(axis=0, keepdims=True)
        y_std = y_all.std(axis=0, keepdims=True) + 1e-9
        yn_all = (y_all - y_mean) / y_std
    else:
        y_mean = np.array([[0.0]], dtype=np.float32)
        y_std = np.array([[1.0]], dtype=np.float32)
        yn_all = y_all

    X_t = torch.from_numpy(Xn_all).float()
    y_t = torch.from_numpy(yn_all).float()
    y_orig_t = torch.from_numpy(y_all).float()
    x_mean_t = torch.from_numpy(x_mean).float().to(device)
    x_std_t = torch.from_numpy(x_std).float().to(device)
    y_mean_t = torch.from_numpy(y_mean).float().to(device)
    y_std_t = torch.from_numpy(y_std).float().to(device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TorchANN(input_dim=X_t.shape[1], hidden_sizes=hidden_sizes, output_dim=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    history = {'mse_norm': [], 'mse_orig': [], 'mae_orig': [], 'r2_orig': []}
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb) 

            k_raw = out[:, :6]
            alpha_raw = out[:, 6]
 
            k = torch.nn.functional.softplus(k_raw)  
            alpha = torch.sigmoid(alpha_raw) * 0.1  
           
            comps_n = xb[:, :6]
            if xb.shape[1] > 6:
                temps_orig = xb[:, 6] * x_std_t[0, 6] + x_mean_t[0, 6]
            else:
                temps_orig = torch.zeros(xb.size(0), device=xb.device)
            factor = 1.0 - alpha * torch.abs(temps_orig - 23.0)
            factor = torch.clamp(factor, min=0.1)

            p_model = torch.sum(k * comps_n, dim=1) / factor
            p_model = p_model.unsqueeze(1)
            loss = criterion(p_model, yb)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
        epoch_loss = running_loss / len(dataset)
        scheduler.step(epoch_loss)

        model.eval()
        with torch.no_grad():
            out_full = model(X_t.to(device))  # (N,7)
            k_raw_f = out_full[:, :6]
            alpha_raw_f = out_full[:, 6]
            k_f = torch.nn.functional.softplus(k_raw_f)
            alpha_f = torch.sigmoid(alpha_raw_f) * 0.1
            comps_n_full = X_t[:, :6].to(device)
            if X_t.shape[1] > 6:
                temps_orig_full = X_t[:, 6].to(device) * x_std_t[0, 6] + x_mean_t[0, 6]
            else:
                temps_orig_full = torch.zeros(X_t.size(0), device=device)
            factor_full = 1.0 - alpha_f * torch.abs(temps_orig_full - 23.0)
            factor_full = torch.clamp(factor_full, min=0.1)
            preds_n_full = (torch.sum(k_f * comps_n_full, dim=1) / factor_full).unsqueeze(1)
            preds_orig_full = preds_n_full * y_std_t + y_mean_t
            y_orig_dev = y_orig_t.to(device)
            mse_orig = float(torch.mean((preds_orig_full - y_orig_dev) ** 2).cpu().numpy())
            mae_orig = float(torch.mean(torch.abs(preds_orig_full - y_orig_dev)).cpu().numpy())
            # compute R^2 on original scale for the full training set this epoch
            ss_res = float(torch.sum((preds_orig_full - y_orig_dev) ** 2).cpu().numpy())
            ss_tot = float(torch.sum((y_orig_dev - torch.mean(y_orig_dev)) ** 2).cpu().numpy())
            if ss_tot > 0:
                r2_orig = 1.0 - (ss_res / ss_tot)
            else:
                r2_orig = 1.0
        # record history
        history['mse_norm'].append(epoch_loss)
        history['mse_orig'].append(mse_orig)
        history['mae_orig'].append(mae_orig)
        history['r2_orig'].append(r2_orig)
        if epoch == 1 or epoch % max(1, epochs//10) == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} - MSE_norm: {epoch_loss:.6f} | MSE_orig: {mse_orig:.4f} | MAE_orig: {mae_orig:.4f}")
        model.train()

    X_all, y_all = build_component_dataset(samples_list, phone, input_software, include_temperature=include_temperature)
    Xn_all = (X_all - x_mean) / x_std
    X_t_all = torch.from_numpy(Xn_all).float().to(device)
    with torch.no_grad():
        out_all = model(X_t_all)
        k_raw_all = out_all[:, :6]
        alpha_raw_all = out_all[:, 6]
        k_all = torch.nn.functional.softplus(k_raw_all)
        alpha_all = torch.sigmoid(alpha_raw_all) * 0.1
        comps_n_all = X_t_all[:, :6]
        if X_t_all.shape[1] > 6:
            temps_orig_all = X_t_all[:, 6] * x_std_t[0, 6] + x_mean_t[0, 6]
        else:
            temps_orig_all = torch.zeros(X_t_all.size(0), device=device)
        factor_all = 1.0 - alpha_all * torch.abs(temps_orig_all - 23.0)
        factor_all = torch.clamp(factor_all, min=0.1)
        preds_n_all = (torch.sum(k_all * comps_n_all, dim=1) / factor_all).unsqueeze(1)
        preds_orig_all = preds_n_all.cpu().numpy().flatten() * float(y_std) + float(y_mean)
    y_all_flat = y_all.flatten()
    mse = float(np.mean((preds_orig_all - y_all_flat) ** 2))
    mae = float(np.mean(np.abs(preds_orig_all - y_all_flat)))
    # compute R^2 on full dataset
    ss_res_all = float(np.sum((preds_orig_all - y_all_flat) ** 2))
    ss_tot_all = float(np.sum((y_all_flat - np.mean(y_all_flat)) ** 2))
    r2 = 1.0 - (ss_res_all / ss_tot_all) if ss_tot_all > 0 else 1.0
    print(f"\n训练完成（全量评估）：MSE={mse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
    torch.save({'model_state': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}, 'ann_torch.pt')
    metrics = {'MSE': mse, 'MAE': mae, 'R2': r2}

    return model, x_mean, x_std, y_mean, y_std, metrics, history, preds_orig_all, y_all_flat


if __name__ == '__main__':

    ALL_TIME = 24  # hours
    with open("battery_samples_list_filtered_24.pkl", "rb") as f:
        samples_list = pickle.load(f)

    if len(samples_list) > 0:
        phone = myPhone(time_resolution=2401, default_charge=4800, all_time=ALL_TIME)
        INPUT_SW = ['screen', 'game', 'wifi', 'gps']
        device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
        model, x_mean, x_std, y_mean, y_std, metrics, history, preds, actual = train_torch_ann(samples_list, phone, INPUT_SW, epochs=5000, lr=1e-3, batch_size=256, device=device)

        try:
            plotter = FitPlotter()
            plotter.plot_loss(history, savepath='train_loss.png')
            plotter.plot_fit(preds, actual, savepath='fit_scatter.png')
            plotter.plot_residuals(preds, actual, savepath='residuals.png')

            with torch.no_grad():
                model_cpu = model.cpu()
                X_all, y_all = build_component_dataset(samples_list, phone, INPUT_SW, include_temperature=True)
                Xn_all = (X_all - x_mean) / x_std
                X_t_all = torch.from_numpy(Xn_all).float()
                out_all = model_cpu(X_t_all)
                k_raw_all = out_all[:, :6]
                k_all = torch.nn.functional.softplus(k_raw_all).numpy()
                mean_k = np.mean(k_all, axis=0)
                comp_means = np.mean(X_all[:, :6], axis=0)
            plotter.plot_components(mean_k, comp_means, savepath='components.png')
        except Exception as e:
            print(f"绘图失败: {e}")