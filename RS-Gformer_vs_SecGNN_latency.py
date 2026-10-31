# -*- coding: utf-8 -*-
"""PhysicalRiskGated-EA-Gformer vs. matched SecGNN: single-sample inference latency.

Run this script from the project root containing:
  - networks.py (PhysicalRiskGated_EA_Gformer_2D)
  - networks_MatchedSecGNN_128.py (SecGNN_matched_128)

The models are randomly initialized, since the goal is to compare forward-pass
execution time of the architectures, not to evaluate trained secrecy performance.
"""

import os
import time
import csv

import torch
import matplotlib.pyplot as plt

from networks import PhysicalRiskGated_EA_Gformer_2D
from networks_MatchedSecGNN_128 import SecGNN_matched_128

HIDDEN = [128, 128, 128]
RISK_HEADS = 1
BATCH_SIZE = 1
CPU_THREADS = 1
WARMUP = 50
REPEAT_MAIN = 500
REPEAT_SWEEP = 100
RUN_CPU = True
RUN_GPU = torch.cuda.is_available()
RUN_K_SWEEP = True
RUN_NT_SWEEP = True

K_MAIN = 4
J_MAIN = 4
NT_MAIN = 8

K_LIST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
NT_FOR_K_SWEEP = 16

NT_LIST = [4, 8, 16, 32, 64]
K_FOR_NT_SWEEP = 4

MODEL_NAMES = ('PhysicalRiskGated', 'SecGNN')
SAVE_DIR = './results_physicalriskgated_efficiency'
os.makedirs(SAVE_DIR, exist_ok=True)

torch.set_num_threads(CPU_THREADS)
torch.manual_seed(2026)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2026)

def build_model(model_name, NT, device):
    if model_name == 'PhysicalRiskGated':
        model = PhysicalRiskGated_EA_Gformer_2D(
            input_dim=[2],
            hidden_layers=HIDDEN,
            output_dim=[2],
            n_heads=RISK_HEADS,
        )
    elif model_name == 'SecGNN':
        model = SecGNN_matched_128(
            input_dim=[2],
            hidden_layers=HIDDEN,
            output_dim=[2],
            n_heads=1,
            NT=NT,
            Pmax=1.0,
            power_mode='gformer',
            dropout=0.1,
        )
    else:
        raise ValueError(f'Unknown model: {model_name}')

    return model.to(device).eval()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.inference_mode()
def benchmark_forward(model, H_LU, H_Eve, warmup, repeats, device):

    for _ in range(warmup):
        _ = model(H_LU, H_Eve)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(repeats):
            _ = model(H_LU, H_Eve)
        ender.record()
        torch.cuda.synchronize()
        total_ms = starter.elapsed_time(ender)
        return total_ms / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(H_LU, H_Eve)
    total_s = time.perf_counter() - start
    return total_s / repeats * 1000.0

def benchmark_point(model_name, K, J, NT, device, repeats):
    model = build_model(model_name, NT, device)

    H_LU = torch.randn(BATCH_SIZE, 2, K, NT, device=device, dtype=torch.float32)
    H_Eve = torch.randn(BATCH_SIZE, 2, J, NT, device=device, dtype=torch.float32)

    latency_ms = benchmark_forward(model, H_LU, H_Eve, WARMUP, repeats, device)
    params = count_params(model)

    del model, H_LU, H_Eve
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return latency_ms, params

def save_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def print_theory():
    print('\nComplexity notes:')
    print('PhysicalRiskGated:')
    print('  Physical RiskGate computes signal and interference power from estimated Eve CSI and provisional beams.')
    print('  K provisional beams against J Eve channels add O(NT*K*J) complex multiplications.')
    print('  Risk scores and normalization across Eve nodes add O(K*J).')
    print('  Overall complexity also depends on the provisional beamformer, attention, and shared updates.')
    print('SecGNN:')
    print('  Its runtime depends on GAL message passing and intermediate tensors, not just parameter count.')
    print('  We report trainable parameter counts and measured single-sample forward latency.')
    print('============================================================\n')

if __name__ == '__main__':
    print_theory()
    rows = []

    print('======================== Main setting: K=J=4, NT=8 ========================')
    for device_name in ('cpu', 'cuda'):
        if device_name == 'cpu' and not RUN_CPU:
            continue
        if device_name == 'cuda' and not RUN_GPU:
            continue
        device = torch.device(device_name)

        for model_name in MODEL_NAMES:
            latency, params = benchmark_point(
                model_name, K_MAIN, J_MAIN, NT_MAIN, device, REPEAT_MAIN
            )
            rows.append({
                'experiment': 'main',
                'device': device_name,
                'model': model_name,
                'K': K_MAIN,
                'J': J_MAIN,
                'NT': NT_MAIN,
                'batch_size': BATCH_SIZE,
                'latency_ms_per_sample': latency,
                'params': params,
            })
            print(f'{device_name.upper():4s} | {model_name:18s} | '
                  f'{latency:.6f} ms/sample | Params: {params:,}')

    if RUN_K_SWEEP:
        print('\n======================== K=J sweep, NT=16 ========================')
        for device_name in ('cpu', 'cuda'):
            if device_name == 'cpu' and not RUN_CPU:
                continue
            if device_name == 'cuda' and not RUN_GPU:
                continue
            device = torch.device(device_name)

            for K in K_LIST:
                for model_name in MODEL_NAMES:
                    latency, params = benchmark_point(
                        model_name, K, K, NT_FOR_K_SWEEP, device, REPEAT_SWEEP
                    )
                    rows.append({
                        'experiment': 'K_sweep',
                        'device': device_name,
                        'model': model_name,
                        'K': K,
                        'J': K,
                        'NT': NT_FOR_K_SWEEP,
                        'batch_size': BATCH_SIZE,
                        'latency_ms_per_sample': latency,
                        'params': params,
                    })
                    print(f'{device_name.upper():4s} | K=J={K:2d} | '
                          f'{model_name:18s} | {latency:.6f} ms')

    if RUN_NT_SWEEP:
        print('\n======================== NT sweep, K=J=4 ========================')
        for device_name in ('cpu', 'cuda'):
            if device_name == 'cpu' and not RUN_CPU:
                continue
            if device_name == 'cuda' and not RUN_GPU:
                continue
            device = torch.device(device_name)

            for NT in NT_LIST:
                for model_name in MODEL_NAMES:
                    latency, params = benchmark_point(
                        model_name, K_FOR_NT_SWEEP, K_FOR_NT_SWEEP,
                        NT, device, REPEAT_SWEEP
                    )
                    rows.append({
                        'experiment': 'NT_sweep',
                        'device': device_name,
                        'model': model_name,
                        'K': K_FOR_NT_SWEEP,
                        'J': K_FOR_NT_SWEEP,
                        'NT': NT,
                        'batch_size': BATCH_SIZE,
                        'latency_ms_per_sample': latency,
                        'params': params,
                    })
                    print(f'{device_name.upper():4s} | NT={NT:2d} | '
                          f'{model_name:18s} | {latency:.6f} ms | Params: {params:,}')

    csv_path = os.path.join(SAVE_DIR, 'physicalriskgated_vs_secgnn_inference_latency.csv')
    save_csv(csv_path, rows)

    for exp, x_name in [('K_sweep', 'K'), ('NT_sweep', 'NT')]:
        exp_rows = [r for r in rows if r['experiment'] == exp]
        if not exp_rows:
            continue

        for device_name in sorted(set(r['device'] for r in exp_rows)):
            plt.figure(figsize=(7.2, 5.0))
            sub = [r for r in exp_rows if r['device'] == device_name]
            for model_name in MODEL_NAMES:
                mm = [r for r in sub if r['model'] == model_name]
                x = [r[x_name] for r in mm]
                y = [r['latency_ms_per_sample'] for r in mm]
                plt.plot(x, y, marker='o', label=model_name)

            if exp == 'K_sweep':
                plt.xlabel('Number of LU-Eve Pairs (K=J)')
            else:
                plt.xlabel(r'Number of BS Antennas $N_T$')
            plt.ylabel('Forward Inference Time (ms/sample)')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(SAVE_DIR, f'{exp}_{device_name}_latency.png'),
                dpi=300,
            )
            plt.close()

    physical_nt8 = build_model('PhysicalRiskGated', NT_MAIN, torch.device('cpu'))
    sec_nt8 = build_model('SecGNN', NT_MAIN, torch.device('cpu'))
    summary_path = os.path.join(SAVE_DIR, 'complexity_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('PhysicalRiskGated physical score computation: O(NT*K*J) '
                'for provisional-beam/Eve-channel products, plus O(K*J) '
                'for risk score normalization.\n')
        f.write('Total model complexity depends on the concrete implementation in networks.py.\n')
        f.write(f'PhysicalRiskGated params @ NT=8: {count_params(physical_nt8):,}\n')
        f.write(f'SecGNN params @ NT=8: {count_params(sec_nt8):,}\n')
        f.write('Timing: batch_size=1; includes model forward pass; '
                'excludes DataLoader, host-device transfer, external PGD '
                'and secrecy-rate evaluation.\n')
        f.write('Models were evaluated with randomly initialized weights; '
                'this measures architecture latency, not secrecy performance.\n')
    del physical_nt8, sec_nt8

    print(f'\nResults saved to:  {SAVE_DIR}')
    print('  physicalriskgated_vs_secgnn_inference_latency.csv')
    print('  K_sweep_cpu/gpu_latency.png (generated according to RUN_CPU/RUN_GPU)')
    print('  NT_sweep_cpu/gpu_latency.png (generated according to RUN_CPU/RUN_GPU)')
    print('  complexity_summary.txt')
