# -*- coding: utf-8 -*-
import os
import glob
import re
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from generate_geometry_rician_strong_eve import load_dataset
from networks import EA_Gformer_2D, PhysicalRiskGated_EA_Gformer_2D
from networks_MatchedSecGNN_128 import SecGNN_matched_128

from function_cal_security_rate import cal_dataset_security_rate_one_to_one
from loss_function_PGD_relative_epsilon import cal_robust_PGD_adversarial_loss_relative

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

mu_list = []
SNR_dB_LU_list = []
model_setting_list = []
sample_number_list = []

def load_model_for_inference(model_path,model_name,train_mode,config):
    layers = [128,128,128]
    if model_name == 'EA_Gformer_2D':
        model = EA_Gformer_2D(input_dim=[2],hidden_layers=layers,output_dim=[2],n_heads=1)
    elif model_name == 'PhysicalRiskGated_EA_Gformer_2D':
        model = PhysicalRiskGated_EA_Gformer_2D(input_dim=[2],hidden_layers=layers,output_dim=[2],n_heads=1)
    elif model_name == 'SecGNN_matched_128':
        model = SecGNN_matched_128(input_dim=[2],hidden_layers=layers,output_dim=[2],n_heads=1,NT=NT,dropout=0.0,Pmax=1.0)
    else:
        raise ValueError(f'Unregistered model: {model_name}')
    model.load_state_dict(torch.load(model_path,map_location=device,weights_only=True),strict=True)
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

sample_checkpoint_overrides = {}

def get_sample_number_model_path(model_name,sample_number,train_mode):
    if train_mode not in ('Robust','NonRobust'):
        raise ValueError(f'Unsupported training mode: {train_mode}')
    model_dir = './models/robust_models_for_sample_number'
    prefix = (f'model_{train_mode}_{model_name}_'
              f'K{K}_Eve{Eve}_NT{NT}_KR{rician_K_dB}dB_'
              f'SNRLU{model_SNR_LU}_SNREve{model_SNR_Eve}_mu{model_mu}_')
    paths = sorted(set(glob.glob(os.path.join(model_dir,prefix+'*.pth'))+
                       glob.glob(os.path.join(model_dir,prefix+'*.pt'))))
    matched = []
    for path in paths:
        m = re.search(r'(?:^|_)(?:sample_number|Ntrain|sample|number)=?(\d+)(?=_|\.pth?$)',os.path.basename(path))
        if m and int(m.group(1)) == int(sample_number):
            matched.append(path)
    key = (model_name,train_mode,int(sample_number))
    if key in sample_checkpoint_overrides:
        chosen = os.path.normpath(sample_checkpoint_overrides[key])
        if chosen not in [os.path.normpath(p) for p in matched]:
            raise ValueError(f'Selected checkpoint does not match model configuration or sample count: {chosen}\nCandidates: {matched}')
        return chosen
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        raise ValueError(f'{model_name} | {train_mode} | Ntrain={sample_number} has multiple checkpoints; '
                         f'select using independent validation and set sample_checkpoint_overrides:\n'+ '\n'.join(matched))
    raise FileNotFoundError(f'Checkpoint not found: {model_name} | {train_mode} | Ntrain={sample_number} checkpoint.'
                            f'\nSearch directory: {model_dir}\nRequired prefix: {prefix}\nFiles with matching configuration: {[os.path.basename(p) for p in paths]}')

def cal_sample_number_to_worst_rate(model_name,train_mode):
    dataloader_K_test,dataloader_Eve_test_est = load_test_data()
    rate_list = []
    print(f'Evaluating {model_name} + {train_mode} worst-case SSR versus training sample count==========')
    for sample_number in sample_number_list:
        model_path = get_sample_number_model_path(model_name,sample_number,train_mode)
        model = load_model_for_inference(model_path,model_name,train_mode,config)
        rate_sum,test_number = 0.0,0
        for x_test_K,x_test_Eve_est in zip(dataloader_K_test,dataloader_Eve_test_est):
            x_test_K = x_test_K.to(device); x_test_Eve_est = x_test_Eve_est.to(device)
            real_batch_size = x_test_K.shape[0]
            with torch.no_grad():
                V_pred = model_forward(model,x_test_K,x_test_Eve_est)
            evaluation_worst = cal_robust_PGD_adversarial_loss_relative(
                H_K=x_test_K,H_Eve_est=x_test_Eve_est,V=V_pred,
                cal_security_rates_func=cal_security_rates_func,batch_size=real_batch_size,
                SNR_dB_LU=model_SNR_LU,SNR_dB_Eve=model_SNR_Eve,mu=model_mu,
                pgd_steps=pgd_steps_eval,pgd_lr=pgd_lr,
                use_conjugate=False,init_mode='zero',objective_mode='rate')
            rate_sum += evaluation_worst.detach().item()*real_batch_size
            test_number += real_batch_size
        rate = rate_sum/test_number
        rate_list.append(rate)
        print(f'{model_name} | {train_mode} | Ntrain={sample_number} | '
              f'worst-case secrecy rate={rate:.6f} bps/Hz | checkpoint={model_path}')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rate_list

def parse_fixed_numerical_baseline(x_list,data):
    if data is None:
        return None
    if isinstance(data,(int,float)):
        return [float(data)]*len(x_list)
    if isinstance(data,dict):
        return [float(data[x]) for x in x_list]
    if isinstance(data,(list,tuple)):
        if len(data) != len(x_list):
            raise ValueError('Numerical baseline length must match x-axis length')
        return [float(x) for x in data]
    raise TypeError('Numerical baseline must be scalar, dict, list, tuple, or None')

def plot_epoch_to_worst_rate(history_dict,sdr_sca_baseline=None):
    print('Plotting worst-case SSR versus epoch==========')
    fig,ax = plt.subplots(figsize=(8.5,5.8),dpi=120)
    color_dict = {'RiskGated_EA_Gformer_2D':'#7220ED',
                  'PhysicalRiskGated_EA_Gformer_2D': '#7220ED',
                  'SecGNN_matched_128':'#E69F00'
                  }
    marker_dict = {'RiskGated_EA_Gformer_2D':'o',
                   'PhysicalRiskGated_EA_Gformer_2D': 'o',
                   'SecGNN_matched_128':'v'}

    label_dict = {'PhysicalRiskGated_EA_Gformer_2D': 'Physical Risk-Gated EA-Gformer',
                  'RiskGated_EA_Gformer_2D': 'RiskGated_EA_Gformer',
                  'SecGNN_matched_128': 'SecGNN'
    }
    valid_epoch = None

    for model_name,history in history_dict.items():
        if history is None:
            print(f'Warning: {model_name} has no epoch history; skipping')
            continue
        epoch = history['epoch'];
        train_rate = history['train_worst_rate'];
        test_rate = history['test_worst_rate']

        if not (len(epoch)==len(train_rate)==len(test_rate)):
            raise ValueError(f'{model_name}: epoch/train_worst_rate/test_worst_rate lengths must match')
        valid_epoch = epoch

        color = color_dict.get(model_name,'#7F7F7F'); marker = marker_dict.get(model_name,'o')
        label_name = label_dict.get(model_name, model_name)
        ax.plot(epoch,train_rate,'--',marker=marker,markersize=5.2,markevery=2,linewidth=1.8,color=color,
                label=f'{label_name} + Train Worst',markerfacecolor='white',markeredgecolor=color,markeredgewidth=1.1,alpha=0.85)
        ax.plot(epoch,test_rate,'-',marker=marker,markersize=5.8,markevery=2,linewidth=2.2,color=color,
                label=f'{label_name} + Test Worst',markerfacecolor=color,markeredgecolor=color,markeredgewidth=1.1)
        annotate_idx = list(range(0,len(epoch),10))
        if len(epoch)-1 not in annotate_idx:
            annotate_idx.append(len(epoch)-1)

    if valid_epoch is None:
        raise ValueError('No epoch history available for plotting')

    sdr_rates = parse_fixed_numerical_baseline(valid_epoch,sdr_sca_baseline)
    if sdr_rates is not None:
        ax.plot(valid_epoch,sdr_rates,'-.',linewidth=2.3,color='#4D4D4D',label='SDR-SCA')

    ax.set_xlabel('Epoch',fontsize=15)

    ax.set_ylabel('Worst-case Sum Secrecy Rate (bps/Hz)',fontsize=15)

    ax.set_xticks(list(range(0,max(valid_epoch)+1,25))+[199])
    ax.set_xlim(0, max(valid_epoch)+1)
    ax.set_ylim(0, 9)
    ax.tick_params(axis='both',labelsize=15,direction='in')

    ax.grid(True,linestyle='-',linewidth=0.6,color='#BFBFBF',alpha=0.45)
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    ax.legend(fontsize=15, loc='best',frameon=True,framealpha=1.0,edgecolor='#777777',fancybox=False)
    ax.margins(x=0.02,y=0.10)
    fig.tight_layout()
    os.makedirs('./graphs/exp01c_sample_number',exist_ok=True)
    save_path = (f'./graphs/exp01c_sample_number/WorstRate_vs_Epoch_RiskGated'
                 f'K{K}_NT{NT}_Eve{Eve}_KR{rician_K_dB}dB_SNRLU{model_SNR_LU}_'
                 f'SNREve{model_SNR_Eve}_mu{model_mu}.png')
    fig.savefig(save_path,dpi=400,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(f'Figure saved: {save_path}')

def plot_sample_number_to_worst_rate(sample_number_list,rate_dict,sdr_sca_baseline=None,train_mode_dict=None):
    print('Plotting worst-case SSR versus training sample count==========')
    fig,ax = plt.subplots(figsize=(8.5,5.8),dpi=120)
    color_dict = {'RiskGated_EA_Gformer_2D':'#7220ED',
                  'PhysicalRiskGated_EA_Gformer_2D': '#7220ED',
                  'SecGNN_matched_128':'#E69F00',
                  'LittleSecGNN':"#008EE6",
                  }
    marker_dict = {'RiskGated_EA_Gformer_2D':'o',
                   'PhysicalRiskGated_EA_Gformer_2D': 'o',
                   'SecGNN_matched_128':'v'}

    label_dict = {'PhysicalRiskGated_EA_Gformer_2D': 'PRG-Gformer',
                  'RiskGated_EA_Gformer_2D': 'RiskGated_EA_Gformer',
                  'SecGNN_matched_128': 'SecGNN'
                }
    for model_name,rate_list in rate_dict.items():
        if len(sample_number_list) != len(rate_list):
            raise ValueError(f'{model_name}: sample_number_list and rate_list lengths must match')
        color = color_dict.get(model_name,'#7F7F7F'); marker = marker_dict.get(model_name,'o')
        label_name = label_dict.get(model_name,model_name)
        train_mode = (train_mode_dict or {}).get(model_name,'Robust')
        train_label = 'Robust Train' if train_mode=='Robust' else 'Non-Robust Train'
        ax.plot(sample_number_list,rate_list,'-',marker=marker,markersize=7.5,linewidth=2.1,color=color,
                label=f'{label_name} + {train_label}',markerfacecolor=color,markeredgecolor=color,markeredgewidth=1.3)

    sdr_rates = parse_fixed_numerical_baseline(sample_number_list,sdr_sca_baseline)
    if sdr_rates is not None:
        color = '#4D4D4D'
        ax.plot(sample_number_list,sdr_rates,linestyle='-.',linewidth=2.3,color=color,label='SDR-SCA')

    ax.set_xscale('log')
    ax.set_xticks(sample_number_list)
    ax.set_xticklabels([str(x) for x in sample_number_list])
    ax.set_xlabel('Number of Training Samples',fontsize=15)

    ax.set_ylabel('Worst-case Sum Secrecy Rate (bps/Hz)',fontsize=15)
    ax.set_ylim(0, 9)

    ax.tick_params(axis='both',labelsize=15,direction='in')
    ax.grid(True,which='both',linestyle='--',linewidth=0.8,alpha=0.28)
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    ax.legend(fontsize=14, loc='best',frameon=True,framealpha=1.0,edgecolor='#777777',fancybox=False)
    ax.margins(x=0.03,y=0.10)
    fig.tight_layout()
    os.makedirs('./graphs/exp01c_sample_number',exist_ok=True)
    save_path = (f'./graphs/exp01c_sample_number/WorstRate_vs_SampleNumber_RiskGated_SecGNN_'
                 f'K{K}_NT{NT}_Eve{Eve}_KR{rician_K_dB}dB_SNRLU{model_SNR_LU}_'
                 f'SNREve{model_SNR_Eve}_mu{model_mu}.png')
    fig.savefig(save_path,dpi=400,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(f'Figure saved: {save_path}')

if __name__=='__main__':
    with open('./config.yaml','r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cal_security_rates_func_dict = {
        'one':cal_dataset_security_rate_one_to_one}
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

    mu_list = [0.0,0.05,0.10,0.15,0.20]
    SNR_dB_LU_list = [-5,0,5,10,15,20]
    
    history_epoch_to_rates_nonrobust_secgnn_matched_128 = {
    'epoch': [
        0,5,10,15,20,25,30,35,40,45,
        50,55,60,65,70,75,80,85,90,95,
        100,105,110,115,120,125,130,135,140,145,
        150,155,160,165,170,175,180,185,190,195,199
    ],

    'train_worst_rate': [
        1.222524,2.376403,3.394239,3.822661,4.101188,
        4.283960,4.443080,4.565491,4.640397,4.737240,
        4.806334,4.850161,4.893259,4.940190,4.963274,
        5.002675,5.094294,5.117514,5.130360,5.133242,
        5.150740,5.165799,5.165692,5.171391,5.180269,
        5.193836,5.221620,5.228406,5.227650,5.235210,
        5.233566,5.239697,5.242940,5.246141,5.252430,
        5.253817,5.256952,5.258163,5.258987,5.258041,
        5.259336
    ],
    'test_worst_rate': [
        1.602524,2.713144,3.692177,4.085793,4.333098,
        4.487864,4.621863,4.721740,4.776433,4.855145,
        4.907975,4.937214,4.967226,5.002419,5.014973,
        5.044930,5.128077,5.143697,5.149727,5.146494,
        5.158507,5.168646,5.164126,5.165867,5.171194,
        5.181576,5.206503,5.210726,5.207671,5.213169,
        5.209675,5.214147,5.215902,5.217768,5.222859,
        5.223172,5.225343,5.225690,5.225739,5.227097,
        5.226888
    ],
    'test_true_rate': [
        2.835400,4.226124,5.329695,5.737787,6.001658,
        6.169939,6.301363,6.400077,6.463037,6.528828,
        6.592041,6.618158,6.647359,6.684831,6.699256,
        6.728926,6.817473,6.828666,6.832370,6.838569,
        6.849068,6.849853,6.854908,6.856603,6.862354,
        6.866666,6.895446,6.898116,6.898545,6.901454,
        6.900860,6.901783,6.904867,6.904173,6.912230,
        6.913227,6.914807,6.915314,6.915440,6.916291,
        6.915839
    ],
    'lr': 2e-3
}
    history_epoch_to_rates_robust_physicalriskgated_128 = {
    'epoch': [
        0,5,10,15,20,25,30,35,40,45,
        50,55,60,65,70,75,80,85,90,95,
        100,105,110,115,120,125,130,135,140,145,
        150,155,160,165,170,175,180,185,190,195,199
    ],
    'train_worst_rate': [
        2.584791,7.279334,7.576324,7.646152,7.689484,
        7.717832,7.743026,7.764295,7.782977,7.796526,
        7.806127,7.819250,7.824700,7.835469,7.845816,
        7.852887,7.890060,7.894947,7.898048,7.899179,
        7.901538,7.903551,7.905462,7.907279,7.909605,
        7.910470,7.923683,7.924773,7.924736,7.925857,
        7.926053,7.926920,7.927519,7.928110,7.933036,
        7.933382,7.933513,7.933681,7.934030,7.934310,
        7.934418
    ],
    'test_worst_rate': [
        6.027259,7.359993,7.505613,7.677193,7.626721,
        7.680537,7.742639,7.731454,7.781302,7.767958,
        7.797020,7.820007,7.810610,7.808348,7.841025,
        7.839618,7.869796,7.864711,7.871989,7.874542,
        7.871437,7.877374,7.878242,7.878788,7.879624,
        7.880625,7.889492,7.888824,7.890630,7.889954,
        7.888839,7.890835,7.890982,7.889894,7.894178,
        7.894062,7.893681,7.893615,7.893556,7.893903,
        7.895362
    ],
    'test_true_rate': [
        7.565540,8.815911,8.989873,9.134421,9.088627,
        9.081454,9.162357,9.097397,9.236593,9.156246,
        9.197845,9.218504,9.182044,9.213069,9.225447,
        9.244544,9.266153,9.289642,9.273455,9.270019,
        9.260740,9.275936,9.273972,9.267143,9.246710,
        9.273943,9.259688,9.287855,9.264842,9.268638,
        9.256780,9.279917,9.281905,9.270683,9.269882,
        9.281758,9.284665,9.267097,9.285447,9.285191,
        9.276827
    ]
}

    history_epoch_dict = {
        'PhysicalRiskGated_EA_Gformer_2D':history_epoch_to_rates_robust_physicalriskgated_128,
        'SecGNN_matched_128':history_epoch_to_rates_nonrobust_secgnn_matched_128,
        }

    sdr_sca_fixed_rate = 8.35
    plot_epoch_to_worst_rate(history_epoch_dict,sdr_sca_fixed_rate)

    sample_number_list = [1000,5000,50000,100000,200000]
    sample_model_setting_list = [
        ('PhysicalRiskGated_EA_Gformer_2D','Robust'),
        ('SecGNN_matched_128','NonRobust')]
    sample_number_rate_dict = {}
    sample_train_mode_dict = {}
    for sample_model_name,train_mode in sample_model_setting_list:
        sample_number_rate_dict[sample_model_name] = cal_sample_number_to_worst_rate(sample_model_name,train_mode)
        sample_train_mode_dict[sample_model_name] = train_mode
    plot_sample_number_to_worst_rate(sample_number_list,sample_number_rate_dict,
                                     sdr_sca_fixed_rate,train_mode_dict=sample_train_mode_dict)
