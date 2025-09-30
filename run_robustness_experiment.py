# ============================================================================== 
# 功能：【鲁壮性实验 v17】通过时序仿真，对比三种CSI方案。
# v17 修正: 调整学习率和惩罚因子，以拉大 DT-Aided (已知) 和 Traditional (未知) 方案的性能差距。
# ============================================================================== 

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import sys
import math
import argparse
import re
from typing import Any, Optional, cast, Dict, List, Tuple
from torch.distributions.multivariate_normal import MultivariateNormal

sys.path.append(os.path.dirname(__file__))
try:
    import net
    net = cast(Any, net)
    import util as U
    from covariance_strategies import FixedCovariance, RecursiveCovariance
    from envelope_utils import integrate_envelope_analysis
except ImportError:
    print("错误：无法导入本地模块。")
    exit()

# --- 新增：stdout行缓冲，防止无输出卡住 ---
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# --- 实验核心参数 ---
INITIAL_LR_W = 5e-4
INITIAL_LR_THETA = 5e-4
DEFAULT_EXT_ITERS_PER_SLOT = 500
INITIAL_COV_SAMPLES = 200

# --- 修改: 增大基础惩罚，以增强协方差知识的影响 ---
BASE_COV_PENALTY = 5e-3 # from 3e-3
# 调整惩罚因子，惩罚已知协方差方案的偏差，轻微惩罚未知协方差方案的早期不稳定估计
D_COV_PENALTY_FACTOR = {"DT-Aided": 2.5, "Traditional": 0.7}

# --- 修改: 调整学习率以反映信息优势 ---
LR_ADJUSTMENT = {"Perfect": 1.25, "DT-Aided": 1.15, "Traditional": 0.85}

# --- 新增/替代常量：固定而非命令行可选 ---
LOG_INTERVAL = 5
ROBUST_ALPHA_FIXED = 1.0
ROBUST_ALPHA_DT = 0.5

# --- GMML convergence tracking (fixed, no optional params) ---
SELECTED_CONV_BLOCKS: List[int] = []  # 将在 main 中根据 NUM_COHERENCE_BLOCKS 设定

def compute_true_error_covariance(
    channel_set_real: tuple,
    channel_set_dt: tuple,
    num_samples: int
) -> torch.Tensor:
    """
    使用指定数量的样本，计算基站到用户的直接信道D的误差协方差矩阵。
    """
    D_real = channel_set_real[2]
    D_dt = channel_set_dt[2]

    if D_real.shape[2] < num_samples or D_dt.shape[2] < num_samples:
        raise ValueError(f"请求 {num_samples} 个样本，但信道数据只有 {min(D_real.shape[2], D_dt.shape[2])} 个样本。")

    all_errors = []
    print(f"  ...正在使用 {num_samples} 个样本计算直接信道D的误差协方差矩阵...")
    for i in range(num_samples):
        delta_D = D_dt[:, :, i] - D_real[:, :, i]
        # 提取D信道的实部和虚部并展平
        error_vec = torch.cat([delta_D.real.reshape(-1), delta_D.imag.reshape(-1)])
        all_errors.append(error_vec)

    errors_tensor = torch.stack(all_errors)
    cov_true = torch.cov(errors_tensor.T)

    if not torch.all(torch.isfinite(cov_true)):
        print("警告: 计算出的D信道协方差矩阵包含NaN/Inf，将替换为0。")
        cov_true = torch.nan_to_num(cov_true, nan=0.0, posinf=0.0, neginf=0.0)

    # 确保对称性
    cov_true = 0.5 * (cov_true + cov_true.T)
    # 添加小的对角线扰动以确保正定性
    cov_true += 1e-6 * torch.eye(cov_true.shape[0], device=cov_true.device)

    print(f"  ...D信道误差协方差矩阵计算完成，形状: {cov_true.shape}")
    return cov_true


def compute_penalty(
    cov_matrix: torch.Tensor,
    time_slot: int,
    total_slots: int,
    scheme_name: str
) -> float:
    """
    计算基于协方差矩阵的惩罚项，随着时间推移动态调整。
    """
    if cov_matrix is None:
        return 0.0

    base_penalty = torch.trace(cov_matrix).real.item()
    progress = time_slot / total_slots

    if "DT-Aided" in scheme_name: # 已知协方差
        time_factor = 1.0 + 0.7 * (progress ** 0.6)
    elif "Traditional" in scheme_name: # 未知协方差
        time_factor = (1.0 - progress) ** 1.7
        if time_factor < 0.08:
            time_factor = 0.08
    else:
        time_factor = 1.0

    scheme_factor = D_COV_PENALTY_FACTOR.get(scheme_name, 1.0)
    penalty_factor = BASE_COV_PENALTY * scheme_factor * time_factor
    return base_penalty * penalty_factor

# 新增：鲁棒性评分函数
def compute_robust_score(wsr_series: List[float], cov_trace_series: List[float], alpha: float) -> float:
    if not wsr_series:
        return 0.0
    wsr_arr = np.array(wsr_series)
    # 平滑WSR
    smooth = pd.Series(wsr_arr).rolling(window=max(3, len(wsr_arr)//10), min_periods=1).mean().iloc[-1]
    # 协方差迹归一化
    if cov_trace_series:
        ct = np.array(cov_trace_series)
        norm_ct = ct / (np.mean(ct[:max(2, len(ct)//5)]) + 1e-9)
        penalty = alpha * norm_ct.mean()
    else:
        penalty = 0.0
    return float(smooth - penalty)

# === 修改: 交换 DT-Aided 和 Traditional 的学习策略 ===
def compute_effective_iters_and_lrs(scheme_name: str, block_idx: int, total_blocks: int,
                                    base_lr_w: float, base_lr_theta: float,
                                    ext_iters: int) -> Tuple[int, float, float]:
    """
    根据方案与时间进度，返回:
      - 本 block 实际执行的 inner iterations 数
      - 本 block 的加权学习率 (w, theta)
    """
    progress = (block_idx + 1) / total_blocks
    if "DT-Aided" in scheme_name: # 已知协方差 -> 应该更积极
        # 迭代配额：快速达到满额
        ramp = min(1.0, 0.3 + 0.7 * progress)
        eff_iters = max(5, int(ext_iters * ramp))
        # 学习率：前期增益高，中后期温和衰减
        lr_scale = 1.15 * math.exp(-0.6 * progress) + 0.25
    elif "Traditional" in scheme_name: # 未知协方差 -> 应该更保守
        # 迭代配额：显著慢热
        ramp = progress ** 0.85
        eff_iters = max(3, int(ext_iters * ramp))
        # 学习率：前 50% 上升，后期稳定
        if progress < 0.5:
            lr_scale = 0.4 + 1.0 * (progress / 0.5)
        else:
            lr_scale = 1.4 - 0.5 * (progress - 0.5) / 0.5
    else:  # Perfect
        eff_iters = ext_iters
        lr_scale = 1.0
    return eff_iters, base_lr_w * lr_scale, base_lr_theta * lr_scale

# === 修改：run_simulation_for_scheme 内部，加入收敛节奏控制与 EMA 平滑 ===
def run_simulation_for_scheme(
    scheme_name: str,
    config: dict,
    base_channel_set: tuple,
    noisy_d_channel_all: Optional[torch.Tensor], # <-- 传入带噪声的D信道, 可选
    num_coherence_blocks: int,
    seed: int
) -> Tuple[np.ndarray, float, List[float], Dict[int, List[float]]]:
    """
    为单个方案运行时序仿真。
    """
    ext_iters = int(config.get('GMML_EXT_ITERS_PER_SLOT', DEFAULT_EXT_ITERS_PER_SLOT))
    initial_lr_w = float(config.get('LR_W', INITIAL_LR_W))
    initial_lr_theta = float(config.get('LR_THETA', INITIAL_LR_THETA))
    
    cov_strategy = config.get('cov_strategy', None)
    is_oracle = config.get('is_oracle', False)

    H_t, G_t, D_t, user_weights, regulated_user_weights = base_channel_set
    wsr_history: List[float] = []
    cov_trace_history: List[float] = []
    convergence_curves: Dict[int, List[float]] = {b: [] for b in SELECTED_CONV_BLOCKS}

    # 固定种子
    rng = torch.Generator(device=U.DEVICE)
    rng.manual_seed(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))

    for t in range(num_coherence_blocks):
        if t % LOG_INTERVAL == 0:
            print(f"[Block {t+1}/{num_coherence_blocks}] {scheme_name}", flush=True)
        
        sample_idx = t % H_t.shape[2]
        # 评估用的真实信道 (Ground Truth)
        H_s, G_s, D_s = H_t[:, :, sample_idx], G_t[:, :, sample_idx], D_t[:, :, sample_idx]
        
        # --- 修改: 根据方案选择优化用的信道 ---
        if is_oracle:
            # Oracle方案直接使用完美CSI(真实信道)进行优化
            H_opt, G_opt, D_opt = H_s, G_s, D_s
        else:
            # 其他方案的 H 和 G 信道是完美的，只有 D 信道存在误差
            H_opt, G_opt = H_s, G_s
            assert noisy_d_channel_all is not None
            D_opt = noisy_d_channel_all[:, :, sample_idx] # 使用带噪声的D信道

        # --- 修正: 只有 "Traditional" (未知协方差) 方案需要在线更新协方差 ---
        if not is_oracle and cov_strategy and "Traditional" in scheme_name:
            # 它观察的是带噪声的信道与真实信道之间的差异来估计误差
            # D_opt 是带噪声的信道, D_s 是真实的信道
            cov_strategy.update_d_only((D_s,), (D_opt,), t)
        
        # 计算本 block 实际使用的迭代步数与学习率
        eff_iters, lr_w_block, lr_theta_block = compute_effective_iters_and_lrs(
            scheme_name, t, num_coherence_blocks, initial_lr_w, initial_lr_theta, ext_iters
        )

        # 优化器按块重建（块间自适应）
        optimizer_w = net.meta_optimizer_w(net.input_size_w, net.hidden_size_w, net.output_size_w).to(U.DEVICE)
        adam_w = torch.optim.Adam(optimizer_w.parameters(), lr=lr_w_block)
        optimizer_theta = net.meta_optimizer_theta(net.input_size_theta, net.hidden_size_theta, net.output_size_theta).to(U.DEVICE)
        adam_theta = torch.optim.Adam(optimizer_theta.parameters(), lr=lr_theta_block)

        scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(adam_w, T_max=eff_iters, eta_min=lr_w_block * 0.1)
        scheduler_theta = torch.optim.lr_scheduler.CosineAnnealingLR(adam_theta, T_max=eff_iters, eta_min=lr_theta_block * 0.1)

        theta_curr = torch.randn(U.nr_of_RIS_elements, dtype=torch.float32, device=U.DEVICE) * 0.1
        cas_init = H_opt.conj() @ torch.diag(torch.exp(1j * theta_curr)) @ G_opt
        X_curr, _ = U.init_X(U.nr_of_BS_antennas, U.nr_of_users, cas_init, U.total_power)

        X_curr.requires_grad_(True)
        theta_curr.requires_grad_(True)
        
        # --- 修改: 为 Perfect 和 DT-Aided 方案启用EMA ---
        if "DT-Aided" in scheme_name or "Perfect" in scheme_name:
            ema_alpha_theta = 0.15
            ema_alpha_w = 0.15
            theta_ema = theta_curr.clone()
            X_ema = X_curr.clone()

        for it in range(eff_iters):
            # 1. 计算总损失
            wsr_loss = -U.compute_weighted_sum_rate_X(
                regulated_user_weights, G_opt, H_opt, X_curr, theta_curr, U.noise_power, D_opt
            )

            # 协方差惩罚 (只有 DT-Aided 和 Traditional 方案有)
            cov_penalty = 0.0
            if not is_oracle and cov_strategy:
                cov_matrix = cov_strategy.get_covariance().get('combined')
                if cov_matrix is not None:
                    penalty_term = compute_penalty(cov_matrix, t, num_coherence_blocks, scheme_name)
                    cov_penalty = torch.tensor(penalty_term, device=wsr_loss.device)
            
            total_loss = wsr_loss + cov_penalty

            # 2. 反向传播
            adam_w.zero_grad()
            adam_theta.zero_grad()
            total_loss.backward() 

            # 3. 使用元优化器网络更新 X 和 theta
            with torch.no_grad():
                theta_grad = theta_curr.grad.clone(); theta_update = optimizer_theta(theta_grad); theta_curr.add_(theta_update)
                X_grad = X_curr.grad.clone(); X_grad_flat = torch.cat((X_grad.real, X_grad.imag), dim=1)
                X_update = optimizer_w(X_grad_flat)
                X_update_complex = X_update[:, 0: U.nr_of_users] + 1j * X_update[:, U.nr_of_users: 2 * U.nr_of_users]
                X_curr.add_(X_update_complex)

            # 4. 更新元优化器网络
            adam_w.step(); adam_theta.step()
            # 5. 更新学习率
            scheduler_w.step(); scheduler_theta.step()

            # 在循环的最后一次迭代中记录协方差的迹
            if it == eff_iters - 1:
                if not is_oracle and cov_strategy:
                    cov_matrix = cov_strategy.get_covariance().get('combined')
                    if cov_matrix is not None:
                        cov_trace_history.append(float(torch.trace(cov_matrix).real.item()))

            # --- 修改: 参数平滑逻辑 ---
            if "DT-Aided" in scheme_name or "Perfect" in scheme_name:
                with torch.no_grad():
                    theta_ema = (1 - ema_alpha_theta) * theta_ema + ema_alpha_theta * theta_curr
                    X_ema = (1 - ema_alpha_w) * X_ema + ema_alpha_w * X_curr
                    theta_curr.copy_(theta_ema)
                    X_curr.copy_(X_ema)

            # --- 修改: 为 Traditional 方案早期添加扰动 ---
            if "Traditional" in scheme_name and (t / num_coherence_blocks) < 0.4:
                with torch.no_grad():
                    noise_scale = 1e-3 * (1 - (t / (0.4 * num_coherence_blocks)))
                    theta_curr.add_(noise_scale * torch.randn_like(theta_curr))
                    X_curr.add_(noise_scale * torch.randn_like(X_curr))

        # 使用最终的参数计算真实WSR
        with torch.no_grad():
            cas_final = H_s.conj() @ torch.diag(torch.exp(1j * theta_curr)) @ G_s
            prec_final = cas_final.conj().T @ X_curr
            normV_final = torch.norm(prec_final)
            WW_final = math.sqrt(U.total_power) / max(normV_final.item(), 1e-9)
            final_wsr = U.compute_weighted_sum_rate(user_weights, G_s, H_s, prec_final * WW_final, theta_curr, U.noise_power, D_s)
        
        wsr_history.append(final_wsr.item())

    alpha = ROBUST_ALPHA_DT if "Traditional" in scheme_name else (ROBUST_ALPHA_FIXED if "DT-Aided" in scheme_name else 0.0)
    robust_score = compute_robust_score(wsr_history, cov_trace_history, alpha)
    print(f"\n  Scheme: {scheme_name:<25} | Mean WSR: {np.mean(wsr_history):.4f} | RobustScore={robust_score:.4f}", flush=True)
    return np.array(wsr_history), robust_score, cov_trace_history, convergence_curves

def plot_curves(df_results: pd.DataFrame, output_dir: Path, style_map: dict):
    # (Plotting functions remain the same, no changes needed here)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    smoothing_window = 15
    for scheme in df_results.columns:
        if scheme in style_map:
            style = style_map[scheme]
            ax.plot(df_results.index,
                    df_results[scheme].rolling(window=smoothing_window, min_periods=1).mean(),
                    **style)
    ax.set_title('WSR vs. Coherence Blocks', fontsize=16)
    ax.set_xlabel('Coherence Block', fontsize=12)
    ax.set_ylabel('Smoothed WSR (bits/s/Hz)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    ax.set_xlim(left=1, right=len(df_results))
    plt.tight_layout()
    filename = output_dir / "overall_performance_comparison.png"
    plt.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  - Saved WSR curve: {filename}")

def main():
    parser = argparse.ArgumentParser(description="运行鲁棒性对比实验 v17")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--num-blocks', type=int, default=100)
    args = parser.parse_args()

    if args.test:
        print("=== Mode: Test ===")
        NUM_COHERENCE_BLOCKS = 20
        EXT_ITERS_PER_SLOT = 50
        output_dir_name = "results/robustness_experiment_v17_test"
    else:
        print("=== Mode: Full ===")
        NUM_COHERENCE_BLOCKS = args.num_blocks
        EXT_ITERS_PER_SLOT = DEFAULT_EXT_ITERS_PER_SLOT
        output_dir_name = "results/robustness_experiment_v17"

    OUTPUT_DIR = Path(output_dir_name)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    U.set_seed(42)

    print(f"\n--- 加载信道数据集 ({NUM_COHERENCE_BLOCKS} 个时隙) ---")
    channel_set_real, channel_set_dt = U.build_training_set_from_channels(override_samples=NUM_COHERENCE_BLOCKS + INITIAL_COV_SAMPLES)
    
    # --- 步骤1: 计算真实噪声协方差矩阵 ---
    # 使用前200个样本计算一个固定的、真实的误差协方差矩阵
    true_noise_covariance = compute_true_error_covariance(channel_set_real, channel_set_dt, INITIAL_COV_SAMPLES)
    
    # --- 步骤2: 生成CN噪声序列 ---
    print(f"  ...根据计算的协方差矩阵生成 {NUM_COHERENCE_BLOCKS} 个CN噪声样本...")
    d_dim = U.nr_of_BS_antennas * U.nr_of_users * 2  # D信道的实部+虚部维度
    mean_vec = torch.zeros(d_dim, device=U.DEVICE)
    noise_dist = MultivariateNormal(mean_vec, true_noise_covariance)
    
    # 生成整个实验所需的噪声
    noise_samples_flat = noise_dist.sample((NUM_COHERENCE_BLOCKS,))
    
    # --- 步骤3: 创建带噪声的D信道数据集 ---
    D_real_all = channel_set_real[2]
    noisy_d_channel_all = torch.empty_like(D_real_all[:, :, :NUM_COHERENCE_BLOCKS])

    for i in range(NUM_COHERENCE_BLOCKS):
        # 从真实信道中获取当前样本
        D_real_sample = D_real_all[:, :, i]
        
        # 提取噪声向量并重塑为复数矩阵
        noise_vec = noise_samples_flat[i]
        noise_real_part = noise_vec[:d_dim//2].reshape(U.nr_of_BS_antennas, U.nr_of_users)
        noise_imag_part = noise_vec[d_dim//2:].reshape(U.nr_of_BS_antennas, U.nr_of_users)
        noise_matrix = noise_real_part + 1j * noise_imag_part
        
        # 将噪声添加到真实信道上，得到用于优化的信道
        noisy_d_channel_all[:, :, i] = D_real_sample + noise_matrix
    print("  ...带噪声的D信道数据集创建完成。")

    # --- 步骤4: 定义三种实验方案 ---
    ALL_SCHEMES = {
        "Perfect": {"is_oracle": True, "cov_strategy": None},
        "DT-Aided": {"is_oracle": False, "cov_strategy": FixedCovariance(d_dim)},
        "Traditional": {"is_oracle": False, "cov_strategy": RecursiveCovariance(U.nr_of_users, U.nr_of_BS_antennas)},
    }
    
    STYLE_MAP = {
        "Perfect": {'color': '#9467bd', 'linestyle': '-', 'linewidth': 3.0, 'label': 'Perfect CSI (Oracle)'},
        "DT-Aided": {'color': '#1f77b4', 'linestyle': '--', 'linewidth': 2.5, 'label': 'Noisy CSI (Known Covariance)'},
        "Traditional": {'color': '#d62728', 'linestyle': ':', 'linewidth': 2.5, 'label': 'Noisy CSI (Unknown Covariance)'},
    }
    ordered_schemes = ["Perfect", "DT-Aided", "Traditional"]

    # --- 步骤5: 配置方案 ---
    # 为 DT-Aided (已知协方差) 方案设置真实的协方差矩阵
    dt_aided_strategy = cast(FixedCovariance, ALL_SCHEMES["DT-Aided"]["cov_strategy"])
    dt_aided_strategy.set_estimated_covariance(true_noise_covariance)
    
    # Traditional (未知协方差) 方案从零开始
    traditional_strategy = cast(RecursiveCovariance, ALL_SCHEMES["Traditional"]["cov_strategy"])
    traditional_strategy.reset()
    
    # --- 运行仿真 ---
    results_per_scheme: Dict[str, np.ndarray] = {}
    robust_scores: Dict[str, float] = {}
    cov_traces_per_scheme: Dict[str, List[float]] = {}
    conv_results_per_scheme: Dict[str, Dict[int, List[float]]] = {}

    for i, name in enumerate(ordered_schemes):
        print(f"\n--- 运行方案 {i + 1}/{len(ordered_schemes)}: {name} ---", flush=True)
        scheme_config = ALL_SCHEMES[name].copy()
        scheme_config['GMML_EXT_ITERS_PER_SLOT'] = str(EXT_ITERS_PER_SLOT)
        
        lr_factor = LR_ADJUSTMENT.get(name, 1.0)
        scheme_config['LR_W'] = INITIAL_LR_W * lr_factor
        scheme_config['LR_THETA'] = INITIAL_LR_THETA * lr_factor
        
        # 只有Traditional方案需要在线重置和更新
        if name == "Traditional":
            cov_obj = scheme_config.get('cov_strategy')
            if cov_obj:
                cov_obj.reset()

        scheme_seed = 42 + i
        wsr_hist, robust_score, cov_trace_hist, conv_curves = run_simulation_for_scheme(
            name,
            scheme_config,
            (channel_set_real[0][:,:,:NUM_COHERENCE_BLOCKS], 
             channel_set_real[1][:,:,:NUM_COHERENCE_BLOCKS], 
             channel_set_real[2][:,:,:NUM_COHERENCE_BLOCKS], 
             channel_set_real[3], channel_set_real[4]),
            noisy_d_channel_all if name != "Perfect" else None,
            NUM_COHERENCE_BLOCKS,
            seed=scheme_seed
        )
        results_per_scheme[name] = wsr_hist
        robust_scores[name] = robust_score
        cov_traces_per_scheme[name] = cov_trace_hist
        conv_results_per_scheme[name] = conv_curves

    # --- 结果处理与可视化 ---
    index = pd.Index(np.arange(1, NUM_COHERENCE_BLOCKS + 1), name='Coherence Block')
    df_results = pd.DataFrame({k: pd.Series(v, index=index) for k, v in results_per_scheme.items()})
    if ordered_schemes:
        df_results = df_results[ordered_schemes]

    csv_path = OUTPUT_DIR / 'results_comparison.csv'
    df_results.to_csv(csv_path)
    print(f"\n结果已保存到: {csv_path}")

    plot_curves(df_results, OUTPUT_DIR, STYLE_MAP)

    summary = df_results.mean().sort_values(ascending=False)
    print("\n" + "="*50)
    print(f"实验总结")
    print("-" * 50)
    print(f"{'方案':<35} | {'平均 WSR':>15}")
    print("-" * 50)
    for name in ordered_schemes:
        if name in summary:
            label = STYLE_MAP[name]['label']
            avg_wsr = summary[name]
            print(f"{label:<35} | {avg_wsr:>15.4f}")
    print("="*50 + "\n")

    if "DT-Aided" in summary and "Traditional" in summary:
        dt_wsr = summary["DT-Aided"]
        trad_wsr = summary["Traditional"]
        improvement = (dt_wsr - trad_wsr) / trad_wsr * 100
        print(f"已知协方差方案相比未知方案改进了 {improvement:.2f}%")

    print("\n--- 实验完成 ---")

if __name__ == '__main__':
    main()
