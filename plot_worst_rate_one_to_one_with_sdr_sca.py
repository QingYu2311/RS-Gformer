# -*- coding: utf-8 -*-
import os
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from generate_geometry_rician_strong_eve import load_dataset
from networks import EA_Gformer_2D,PhysicalRiskGated_EA_Gformer_2D
from networks_MatchedSecGNN_128 import SecGNN_matched_128
from function_cal_security_rate import cal_dataset_security_rate_eavesdroppers_colluding_MRC,\
                                       cal_dataset_security_rate_eavesdroppers_noncolluding_max,\
                                       cal_dataset_security_rate_one_to_one
from loss_function_PGD_relative_epsilon import cal_robust_PGD_adversarial_loss_relative

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

def load_model_for_inference(model_path,model_name,train_mode,config):
    layers = config['model']['layers']
    if model_name == 'EA_Gformer_2D':
        model = EA_Gformer_2D(input_dim=[2],hidden_layers=layers,output_dim=[2],n_heads=1)
    elif model_name == 'PhysicalRiskGated_EA_Gformer_2D':
        model = PhysicalRiskGated_EA_Gformer_2D(input_dim=[2],hidden_layers=layers,output_dim=[2],n_heads=1)
    elif model_name == 'SecGNN_matched_128':
        model = SecGNN_matched_128(input_dim=[2],hidden_layers=layers,output_dim=[2],n_heads=1,NT=NT,dropout=0.0,Pmax=1.0)
    else:
        raise ValueError(f'Unregistered model: {model_name}')
    model.load_state_dict(torch.load(model_path,map_location=device),strict=False)
    model.to(device); model.eval()
    print(f'Loaded model: {model_name} | {train_mode} | {model_path}')
    return model

def get_model_path(model_name,train_mode):
    if train_mode == 'Robust':
        return (f'./models/robust_models/model_Robust_{model_name}_'
                f'K{K}_Eve{Eve}_NT{NT}_KR{rician_K_dB}dB_'
                f'SNRLU{model_SNR_LU}_SNREve{model_SNR_Eve}_mu{model_mu}_maxepoch{max_epoch}.pth')
    elif train_mode == 'NonRobust':
        return (f'./models/nonrobust_models/model_NonRobust_{model_name}_'
                f'K{K}_Eve{Eve}_NT{NT}_KR{rician_K_dB}dB_'
                f'SNRLU{model_SNR_LU}_SNREve{model_SNR_Eve}_mu{model_mu}_maxepoch{max_epoch}.pth')
    raise ValueError("train_mode must be 'Robust' or 'NonRobust'")

def model_forward(model,H_K,H_Eve_est):
    return model(channel_scale*H_K,channel_scale*H_Eve_est)

def load_test_data():
    root_path = './dataset_exp01_scenario'
    dataset_K_test = load_dataset(
        f'{root_path}/dataset_K_test/dataset_test_K={K}_NT={NT}_KR={rician_K_dB}dB_number={number_test}.pt')
    dataset_Eve_test_est = load_dataset(
        f'{root_path}/dataset_Eve_test_est/dataset_test_est_Eve={Eve}_NT={NT}_KR={rician_K_dB}dB_mu={model_mu}_number={number_test}.pt')
    return DataLoader(dataset_K_test,batch_size=batch_size,shuffle=False),\
           DataLoader(dataset_Eve_test_est,batch_size=batch_size,shuffle=False)

def cal_mu_to_worst_rate(model_name,train_mode):
    model = load_model_for_inference(get_model_path(model_name,train_mode),model_name,train_mode,config)
    dataloader_K_test,dataloader_Eve_test_est = load_test_data()
    rate_list = []
    print(f'Evaluating {model_name} + {train_mode} worst-case SSR versus mu==========')
    for mu_attack in mu_list:
        rate_sum,sample_number = 0.0,0
        for x_test_K,x_test_Eve_est in zip(dataloader_K_test,dataloader_Eve_test_est):
            x_test_K = x_test_K.to(device); x_test_Eve_est = x_test_Eve_est.to(device)
            real_batch_size = x_test_K.shape[0]
            with torch.no_grad():
                V_pred = model_forward(model,x_test_K,x_test_Eve_est)
            evaluation_worst = cal_robust_PGD_adversarial_loss_relative(
                H_K=x_test_K,H_Eve_est=x_test_Eve_est,V=V_pred,
                cal_security_rates_func=cal_security_rates_func,batch_size=real_batch_size,
                SNR_dB_LU=model_SNR_LU,SNR_dB_Eve=model_SNR_Eve,mu=mu_attack,
                pgd_steps=pgd_steps_eval,pgd_lr=pgd_lr,
                use_conjugate=False,init_mode='zero',objective_mode='rate')
            rate_sum += evaluation_worst.detach().item()*real_batch_size
            sample_number += real_batch_size
        rate = rate_sum/sample_number
        rate_list.append(rate)
        print(f'mu={mu_attack:.3f}: worst-case secrecy rate={rate:.6f} bps/Hz')
    return rate_list

def cal_SNR_to_worst_rate(model_name,train_mode):
    model = load_model_for_inference(get_model_path(model_name,train_mode),model_name,train_mode,config)
    dataloader_K_test,dataloader_Eve_test_est = load_test_data()
    rate_list = []
    print(f'Evaluating {model_name} + {train_mode} worst-case SSR versus LU SNR==========')
    for SNR_dB_LU in SNR_dB_LU_list:
        rate_sum,sample_number = 0.0,0
        for x_test_K,x_test_Eve_est in zip(dataloader_K_test,dataloader_Eve_test_est):
            x_test_K = x_test_K.to(device); x_test_Eve_est = x_test_Eve_est.to(device)
            real_batch_size = x_test_K.shape[0]
            with torch.no_grad():
                V_pred = model_forward(model,x_test_K,x_test_Eve_est)
            evaluation_worst = cal_robust_PGD_adversarial_loss_relative(
                H_K=x_test_K,H_Eve_est=x_test_Eve_est,V=V_pred,
                cal_security_rates_func=cal_security_rates_func,batch_size=real_batch_size,
                SNR_dB_LU=SNR_dB_LU,SNR_dB_Eve=model_SNR_Eve,mu=model_mu,
                pgd_steps=pgd_steps_eval,pgd_lr=pgd_lr,
                use_conjugate=False,init_mode='zero',objective_mode='rate')
            rate_sum += evaluation_worst.detach().item()*real_batch_size
            sample_number += real_batch_size
        rate = rate_sum/sample_number
        rate_list.append(rate)
        print(f'SNR_LU={SNR_dB_LU}dB: worst-case secrecy rate={rate:.6f} bps/Hz')
    return rate_list

def parse_sdr_sca_baseline(x_list,data):
    if data is None:
        return None
    if isinstance(data,dict):
        return [float(data[x]) for x in x_list]
    if isinstance(data,(list,tuple)):
        if len(data) != len(x_list):
            raise ValueError('SDR-SCA values must match x-axis length')
        return [float(x) for x in data]
    raise TypeError('SDR-SCA baseline must be dict, list, tuple, or None')

def plot_worst_rate(x_list, rate_list_list, x_mode, sdr_sca_baseline=None):
    print('Plotting curves==========')

    fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=150)

    color_dict = {
        'EA_Gformer_2D': '#0072BD',
        'PairPrior_EA_Gformer_2D': '#45BD00',
        'RiskGated_EA_Gformer_2D': '#D95319',
        'SINRPrior_EA_Gformer_2D': '#77AC30',
        'PairPrior_NoCrossAttention_EA_Gformer_2D': '#A2142F',
        'PairAware_Hard_EA_Gformer_2D':"#A28814",
        'PairAwareMask_EA_Gformer_2D': '#4DBEEE',
        'PairAwareMask_RiskGated_EA_Gformer_2D': '#7E2F8E',
        'PhysicalRiskGated_EA_Gformer_2D': '#D95319',
        'SecGNN_matched_128': '#EDB120',
    }
    marker_dict = {
        'EA_Gformer_2D': '^',
        'PhysicalRiskGated_EA_Gformer_2D': 'o',
        'PairPrior_EA_Gformer_2D': 'o',
        'RiskGated_EA_Gformer_2D': 's',
        'PairAware_Hard_EA_Gformer_2D': '^',
        'SecGNN_matched_128': 'v'
    }

    label_dict = {
        'PhysicalRiskGated_EA_Gformer_2D': 'PRG-Gformer',
        'RiskGated_EA_Gformer_2D': 'Risk-Gated EA-Gformer',
        'EA_Gformer_2D': 'EA-Gformer',
        'SecGNN_2D': 'SecGNN',
        'SecGNN_matched_128': 'SecGNN'
    }

    for (model_name, train_mode), rates in zip(model_setting_list, rate_list_list):

        color = color_dict.get(model_name, '#7F7F7F')
        marker = marker_dict.get(model_name, 'o')
        linestyle = '-'
        markerfacecolor = color

        label_name = label_dict.get(model_name, model_name)

        ax.plot(x_list,rates,linestyle=linestyle,marker=marker,markersize=6.5,linewidth=1.8,
                color=color,label=label_name,markerfacecolor=markerfacecolor,markeredgecolor=color,markeredgewidth=1.2)

    sdr_rates = parse_sdr_sca_baseline(x_list, sdr_sca_baseline)

    if sdr_rates is not None:
        color = '#4D4D4D'
        ax.plot(x_list,sdr_rates,linestyle='-.',marker='D',
            markersize=6.2,linewidth=1.7,color=color,label='SDR-SCA',
            markerfacecolor='white',markeredgecolor=color,markeredgewidth=1.1,zorder=2
        )

    if x_mode == 'mu':
        ax.set_ylim(top=10.0)
        ax.set_xlabel(r'$\mu$', fontsize=17)
        ax.legend(fontsize=18, loc='lower left',frameon=True,framealpha=1.0,edgecolor='#777777',fancybox=False)
        save_path = (
            f'./graphs/exp01c_snr/'
            f'WorstRate_vs_Mu_K{K}_NT{NT}_Eve{Eve}_'
            f'KR{rician_K_dB}dB_SNRLU{model_SNR_LU}_'
            f'SNREve{model_SNR_Eve}_model_mu{model_mu}.png'
        )

    else:
        ax.set_ylim(top=14.0)
        ax.set_xlabel('LU SNR (dB)', fontsize=18)
        ax.legend(fontsize=18, loc='best',frameon=True,framealpha=1.0,edgecolor='#777777',fancybox=False)
        save_path = (
            f'./graphs/exp01c_snr/'
            f'WorstRate_vs_SNR_K{K}_NT{NT}_Eve{Eve}_'
            f'KR{rician_K_dB}dB_SNREve{model_SNR_Eve}_'
            f'mu{model_mu}.png'
        )

    ax.set_xticks(x_list)

    ax.set_ylabel('Worst-case Sum Secrecy Rate (bps/Hz)',fontsize=16)
    ax.tick_params(axis='both',labelsize=16,direction='in',length=4,width=0.8)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('#555555')

    ax.grid(True,linestyle='-',linewidth=0.6,color='#BFBFBF',alpha=0.45)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    ax.margins(x=0.025, y=0.06)

    fig.tight_layout(pad=0.8)
    os.makedirs('./graphs/exp01c_snr', exist_ok=True)
    fig.savefig(save_path,dpi=400,bbox_inches='tight',facecolor='white')
    pdf_path = save_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Figure saved:  {save_path}')
    print(f'PDF saved:  {pdf_path}')

if __name__=='__main__':
    with open('./config.yaml','r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cal_security_rates_func_dict = {
        'one':cal_dataset_security_rate_one_to_one,
        'max':cal_dataset_security_rate_eavesdroppers_noncolluding_max,
        'mrc':cal_dataset_security_rate_eavesdroppers_colluding_MRC}
    cal_security_rates_func = cal_security_rates_func_dict[config.get('loss','one')]

    K = config['system']['K']; Eve = config['system']['Eve']; NT = config['system']['NT']
    model_mu = 0.1
    rician_K_dB = config['system'].get('rician_K_dB',10.0)
    model_SNR_LU = config['system']['SNR_dB_LU']; model_SNR_Eve = config['system']['SNR_dB_Eve']
    batch_size = config['training']['batch_size']; max_epoch = config['training']['max_epoch']
    number_test = config['training'].get('number_test',10000)
    pgd_steps_eval = config['training'].get('pgd_steps_eval',10); pgd_lr = config['training'].get('pgd_lr',0.2)
    gpu_id = config.get('device',{}).get('gpu_id',0)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    channel_scale = float(config['training'].get('channel_scale',1.0))
    print(f'Device: {device}, channel_scale={channel_scale:.4e}')

    mu_list = [0.01,0.05,0.10,0.15,0.20]
    SNR_dB_LU_list = [-5,0,5,10,15]

    model_setting_list = [

        ('PhysicalRiskGated_EA_Gformer_2D','Robust'),

        ('EA_Gformer_2D','NonRobust'),

        ('SecGNN_matched_128','NonRobust'),
    ]

    sdr_sca_mu_baseline_insideball = None

    sdr_sca_mu_baseline_insideball = {0.01:9.63,0.05:9.11,0.10:8.35,0.15:7.65,0.20:6.92}

    sdr_sca_snr_baseline_insideball = {-5:0.91,0:2.19,5:4.54,10:8.35,15:13.57,20:19.29}

    mu_rate_list_list = []
    for model_name,train_mode in model_setting_list:
        mu_rate_list_list.append(cal_mu_to_worst_rate(model_name,train_mode))
    plot_worst_rate(mu_list,mu_rate_list_list,'mu',sdr_sca_mu_baseline_insideball)

    snr_rate_list_list = []
    for model_name,train_mode in model_setting_list:
        snr_rate_list_list.append(cal_SNR_to_worst_rate(model_name,train_mode))
    plot_worst_rate(SNR_dB_LU_list,snr_rate_list_list,'snr',sdr_sca_snr_baseline_insideball)
