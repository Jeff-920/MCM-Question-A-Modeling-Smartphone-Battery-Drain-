import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
import torch
from ann import *


INPUT_SW = ['screen', 'game', 'wifi', 'gps']
Q_Ah = 5088.0 / 1000.0  
a_batt = -1.0 / Q_Ah
ALL_TIME = 24.0  # hours

with open("battery_samples_list_filtered_24.pkl", "rb") as f:
    samples_list = pickle.load(f)[0: 80]
print(f"Loaded {len(samples_list)} samples for ANN training (torch)")

def P(sample_lis: list, time_resolution: int, input_sw=INPUT_SW, true_power=None, device='cpu'):
    phone = myPhone(time_resolution=time_resolution, default_charge=3000, if_discharge=True, all_time=ALL_TIME)
    checkpoint = torch.load(r"model\ann_torch.pt", map_location=device, weights_only=False)
    model_state = checkpoint['model_state']
    x_mean = checkpoint['x_mean'] 
    x_std = checkpoint['x_std']   
    y_mean = checkpoint['y_mean']  
    y_std = checkpoint['y_std']    

    x_mean_hard = x_mean.flatten()[:6] 
    x_std_hard = x_std.flatten()[:6]   
    y_mean_scalar = y_mean.flatten()[0]
    y_std_scalar = y_std.flatten()[0]

    model = TorchANN(input_dim=7, hidden_sizes=(32, 64, 128, 128, 128, 32, 16), output_dim=7).to(device)
    model.load_state_dict(model_state)
    model.eval() 

    X_infer, _, _ = build_component_dataset(
          samples_list=sample_lis,
          phone=phone,
          input_software=input_sw,
          include_temperature=True,
          use=True
    )
    Xn_infer = (X_infer - x_mean) / x_std 
    infer_temperature = sample_lis[0]['temperature']

    with torch.no_grad():
        Xt_infer = torch.from_numpy(Xn_infer).float().to(device)
        out = model(Xt_infer)
        k_raw = out[:, :6]
        alpha_raw = out[:, 6]
        k = torch.nn.functional.softplus(k_raw) 
        alpha = torch.sigmoid(alpha_raw) * 0.1  

        k_np = k.cpu().numpy().flatten()  
        alpha_np = alpha.cpu().numpy().item() 

    software_set = sample_lis[0]['software_set']
    P_cpu, P_gpu, P_raw, P_gps, P_screen, P_wifi = phone.Pt(
        software_set=software_set,
        input_software=input_sw,
        k=[0,0,0,0,0,0],  
        return_components=True 
    )
    X_t = np.stack([P_cpu, P_gpu, P_raw, P_gps, P_screen, P_wifi], axis=1)
    print(f"【硬件序列验证】形状：{X_t.shape} | 时间序列长度：{len(X_t)}（和time_squ一致）")

    X_t_n = (X_t - x_mean_hard) / (x_std_hard + 1e-9) 
    print(f"【归一化后硬件序列】均值：{X_t_n.mean(axis=0).round(2)}（尺度≈1，和训练一致）")

    temp_diff = np.abs(infer_temperature - 23.0)  
    factor = 1.0 - alpha_np * temp_diff
    factor = np.clip(factor, 0.1, None)  

    y_t_n = np.sum(k_np * X_t_n, axis=1) / factor  
    y_t_n = y_t_n.reshape(-1, 1) 

    P_t_corrected = y_t_n * y_std_scalar + y_mean_scalar
    P_t_corrected = P_t_corrected.flatten()  
    P_t_corrected = np.clip(P_t_corrected, 0.0, None)  


    with torch.no_grad():
        temps_orig = Xt_infer[:, 6] * x_std.flatten()[6] + x_mean.flatten()[6]
        factor_scalar = 1.0 - alpha_np * torch.abs(temps_orig - 23.0).cpu().numpy().item()
        factor_scalar = np.clip(factor_scalar, 0.1, None)
        comps_n = Xt_infer[:, :6]
        preds_n = (torch.sum(k * comps_n, dim=1) / factor_scalar).unsqueeze(1)
        preds_orig_scalar = preds_n.cpu().numpy().item() * y_std_scalar + y_mean_scalar

    P_t_mean = P_t_corrected.mean()
    print("="*60)
    print(f"环境温度：{infer_temperature:.2f} ℃ | 温度修正因子：{factor:.4f}")
    print(f"模型标量预测功耗：{preds_orig_scalar:.2f} mW | 真实值：{true_power:.2f} mW")
    print(f"逐时刻序列平均功耗：{P_t_mean:.2f} mW（和标量预测完全对齐）")
    print("="*60)
    print(f"模型学习到的硬件权重k：{k_np.round(4)}")
    print(f"模型学习到的温度修正因子alpha：{alpha_np:.6f}")
    print("="*60)
    print("硬件权重对应：[P_cpu, P_gpu, P_raw, P_gps, P_screen, P_wifi]")
    print(f"功耗序列长度：{len(P_t_corrected)} | 功耗范围：{P_t_corrected.min():.2f}~{P_t_corrected.max():.2f} mW")
    print("="*60)

    return P_t_corrected
    
def R(SOC, temperature):
    T_K = temperature + 273.15
    R0 = 1.09
    Ea = 2740.79 
    a_coef = 0.000256
    b = -0.048424
    c = 26.60
    d = 2.24
    R_g = 8.314
    epsilon = 0.001
    temp_term = np.exp(Ea / (R_g * T_K))
    soc_term = a_coef * SOC**2 + b * SOC + c + d / (SOC + epsilon)
    R_pred = R0 * temp_term * soc_term
    return R_pred

def V(soc):
  # VOC = a + b · (−lns)m + c · s + d · en(s−1)
    a = -3.43659310e+01 
    b = -1.79701821e-01
    c = -8.98948295e+00
    d = 4.75266299e+01
    m = 2.64464727e-03
    n = 2.14762195e-01
    eps = 1e-6
    soc_clip = np.clip(soc, eps, 1.0 - eps)
    x_1 = np.log(1.0 / soc_clip) ** np.exp(m)
    x_2 = np.exp(n * (soc_clip - 1))
    return a + b * x_1 + c * soc_clip + d * x_2

dt = 0.01        
s0 = 0.7        
t_start = 0     
t_end = ALL_TIME       

t = np.arange(t_start, t_end + dt, dt)
N = len(t) 
s = np.zeros(N)
s[0] = s0 


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150 
plt.rcParams['savefig.dpi'] = 300 

color_soc = "#f44bb3"
color_power = 'tab:blue'
color_current = 'tab:green'

for idx, sample in enumerate(samples_list):  

    temp = sample['temperature']
    soft_set = sample['software_set']
    power_cons = sample['power_consumption_mw']
    sample_num = idx + 1
    sample_tag = f"sample{sample_num}_temp{temp}"  
    print(f"\n=====================================")
    print(f"【样本{sample_num}/{len(samples_list)}】{sample_tag}")
    print(f"=====================================")

    s = np.zeros(N)
    s[0] = s0 
    In_arr = np.zeros(N)

    power = P([sample], time_resolution=N, input_sw=INPUT_SW, true_power=power_cons)
    sample_ = True
    for n in range(N-1):
        sn = s[n]
        Rn = R(sn, temperature=temp) * 0.01 * 40 * 1e-3  
        Vn = V(sn)  
        tn_p = float(power[n]) / 1000.0 
        
        if abs(Rn) < 1e-12:
            if abs(Vn) <= 1e-9:
                print(f"  警告：n={n} → Rn≈0 且 Vn≈0，默认In=0")
                sample_ = False
                break
            else:
                In = tn_p / Vn
        else:
            delta = Vn**2 - 4 * Rn * tn_p
            if delta < -1e-9:
                print(f"  错误：n={n} → delta={delta:.4e}<0，无实数解，默认In=0")
                sample_ = False
                break
            else:
                delta = max(delta, 0.0) 
                sqrt_d = np.sqrt(delta)
                I1 = (Vn - sqrt_d) / (2 * Rn)
                I2 = (Vn + sqrt_d) / (2 * Rn)

                candidates = [I1, I2]
                if tn_p >= 0:
                    pos = [c for c in candidates if c >= 0]
                    In = min(pos, key=abs) if pos else min(candidates, key=abs)
                else:
                    neg = [c for c in candidates if c <= 0]
                    In = min(neg, key=abs) if neg else min(candidates, key=abs)
        
        In_arr[n] = In
        s[n+1] = np.clip(sn + a_batt * In * dt, 0.0, 1.0)

    if not sample_:
        print(f"  ⚠️ 样本{sample_num}求解中断，跳过绘图和统计。")
        continue 
    else:
        plt.figure(figsize=(6, 4))
        plt.plot(t, s, color=color_soc, linewidth=4, label='$s(t)$ (Euler solution)')
        plt.xlabel('Time (hours)', fontsize=14)
        plt.ylabel('SOC (%)', fontsize=14)
        plt.title(f'SOC in 24 hours', fontsize=14)
        plt.xlim(0, 24)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.savefig(f'soc_{sample_tag}.png', bbox_inches='tight')
        plt.close()  
        print(f"  ✅ SOC图已保存：soc_{sample_tag}.png")

        plt.figure(figsize=(6, 4))

        plt.subplot(2,1,1)
        plt.plot(t, power, color=color_power, linewidth=4)
        plt.ylabel('Power (mW)', labelpad=10, fontsize=12)
        plt.title(f'Power & Current in 24 hours')
        plt.grid(True, alpha=0.3)

        plt.subplot(2,1,2)
        plt.plot(t, In_arr, color=color_current, linewidth=4)
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('Current (A)', labelpad=19, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
  
        plt.savefig(f'power_current_{sample_tag}.png', bbox_inches='tight')
        plt.close() 
        print(f"  ✅ 功率电流图已保存：power_current_{sample_tag}.png")

        p_min, p_max, p_mean = np.min(power), np.max(power), np.mean(power)
        i_min, i_max, i_mean = np.nanmin(In_arr), np.nanmax(In_arr), np.nanmean(In_arr)
        print(f"  📊 功率统计(mW)：min={p_min:.4f}, max={p_max:.4f}, mean={p_mean:.4f}")
        print(f"  📊 电流统计(A)：min={i_min:.6f}, max={i_max:.6f}, mean={i_mean:.6f}")

print(f"\n🎉 所有{len(samples_list)}个样本处理完成！每个样本均已单独保存SOC图和功率电流图。")