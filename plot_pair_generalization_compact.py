# -*- coding: utf-8 -*-
import os,glob,re,torch,yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from generate_geometry_rician_strong_eve import load_dataset
from networks import EA_Gformer_2D,PhysicalRiskGated_EA_Gformer_2D
from networks_MatchedSecGNN_128 import SecGNN_matched_128
from function_cal_security_rate import cal_dataset_security_rate_one_to_one
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
    if train_mode not in ('Robust','NonRobust'):
        raise ValueError("train_mode must be 'Robust' or 'NonRobust'")
    prefix = 'robust' if train_mode=='Robust' else 'nonrobust'
    if pair_train_distribution == 'exponential':
        model_dir = f'./models/{prefix}_models_pair_generalization'
        pattern = (f'{model_dir}/model_{train_mode}_{model_name}_'
                   f'Kmean{Kmean}_Evemean{Evemean}_Kmax{max_pair_train}_NT{NT}_'
                   f'KR{rician_K_dB}dB_SNRLU{model_SNR_LU}_SNREve{model_SNR_Eve}_'
                   f'mu{model_mu}_maxepoch200.pth')
    elif pair_train_distribution == 'uniform':
        model_dir = f'./models/{prefix}_models_pair_generalization_uniform'
        pattern = (f'{model_dir}/model_{train_mode}_{model_name}_Uniform_'
                   f'Kmin{min_pair_train}_Kmax{max_pair_train}_NT{NT}_'
                   f'KR{rician_K_dB}dB_SNRLU{model_SNR_LU}_SNREve{model_SNR_Eve}_'
                   f'mu{model_mu}_maxepoch200.pth')
    else:
        raise ValueError("pair_train_distribution must be 'exponential' or 'uniform'")

    matched = sorted(glob.glob(pattern))
    if len(matched)==0:
        raise FileNotFoundError(f'Checkpoint not found: {model_name} + {train_mode} checkpoint，Search path: {pattern}')

    selection_K = max(pair_test_list)
    best_matched = [p for p in matched if re.search(rf'_bestK{selection_K}Worst\.pth$',os.path.basename(p))]
    if best_matched: matched = best_matched
    def get_maxepoch(path):
        m = re.search(r'_maxepoch(\d+)',os.path.basename(path))
        return int(m.group(1)) if m else -1
    matched = sorted(matched,key=lambda p:(get_maxepoch(p),os.path.basename(p)))
    if len(matched)>1:
        print(f'Warning: {model_name} + {train_mode} has multiple checkpoints; selecting: {matched[-1]}')
    return matched[-1]

def model_forward(model,H_K,H_Eve_est):
    return model(channel_scale*H_K,channel_scale*H_Eve_est)

def load_test_data(pair_number):
    root_path = './dataset_pair_generalization_rician_strong_eve'
    dataset_K_test = load_dataset(
        f'{root_path}/dataset_K_test/dataset_test_K={pair_number}_NT={NT}_KR={rician_K_dB}dB_number={number_test}.pt')
    dataset_Eve_test_true = load_dataset(
        f'{root_path}/dataset_Eve_test_true/dataset_test_true_Eve={pair_number}_K={pair_number}_NT={NT}_KR={rician_K_dB}dB_number={number_test}.pt')
    dataset_Eve_test_est = load_dataset(
        f'{root_path}/dataset_Eve_test_est/dataset_test_est_Eve={pair_number}_K={pair_number}_NT={NT}_KR={rician_K_dB}dB_mu={model_mu}_number={number_test}.pt')
    return DataLoader(dataset_K_test,batch_size=batch_size,shuffle=False),\
           DataLoader(dataset_Eve_test_true,batch_size=batch_size,shuffle=False),\
           DataLoader(dataset_Eve_test_est,batch_size=batch_size,shuffle=False)

def cal_pair_to_rate(model_name,train_mode):
    model = load_model_for_inference(get_model_path(model_name,train_mode),model_name,train_mode,config)
    worst_sum_list,true_sum_list,worst_user_list,true_user_list = [],[],[],[]
    print(f'Evaluating {model_name} + {train_mode} generalization versus LU-Eve pairs==========')
    for pair_number in pair_test_list:
        dataloader_K_test,dataloader_Eve_test_true,dataloader_Eve_test_est = load_test_data(pair_number)
        worst_sum,true_sum,sample_number = 0.0,0.0,0
        for x_test_K,x_test_Eve_true,x_test_Eve_est in zip(dataloader_K_test,dataloader_Eve_test_true,dataloader_Eve_test_est):
            x_test_K = x_test_K.to(device); x_test_Eve_true = x_test_Eve_true.to(device); x_test_Eve_est = x_test_Eve_est.to(device)
            real_batch_size = x_test_K.shape[0]
            with torch.no_grad():
                V_pred = model_forward(model,x_test_K,x_test_Eve_est)
            evaluation_worst = cal_robust_PGD_adversarial_loss_relative(
                H_K=x_test_K,H_Eve_est=x_test_Eve_est,V=V_pred,
                cal_security_rates_func=cal_security_rates_func,batch_size=real_batch_size,
                SNR_dB_LU=model_SNR_LU,SNR_dB_Eve=model_SNR_Eve,mu=model_mu,
                pgd_steps=pgd_steps_eval,pgd_lr=pgd_lr,use_conjugate=False,init_mode='zero',objective_mode='rate')
            with torch.no_grad():
                evaluation_true = cal_security_rates_func(
                    H_K=x_test_K,H_Eve=x_test_Eve_true,V=V_pred,batch_size=real_batch_size,
                    SNR_dB_LU=model_SNR_LU,SNR_dB_Eve=model_SNR_Eve)
            worst_sum += evaluation_worst.detach().item()*real_batch_size
            true_sum += evaluation_true.detach().item()*real_batch_size
            sample_number += real_batch_size
        worst_rate = worst_sum/sample_number; true_rate = true_sum/sample_number
        worst_user = worst_rate/pair_number; true_user = true_rate/pair_number
        worst_sum_list.append(worst_rate); true_sum_list.append(true_rate)
        worst_user_list.append(worst_user); true_user_list.append(true_user)
        mode = 'ID' if pair_number<=max_pair_train else 'OOD'
        print(f'{train_mode} | K=Eve={pair_number:2d} [{mode}] | Worst Sum Rate:{worst_rate:.6f} | Worst Per-user Rate:{worst_user:.6f} | '
              f'True Sum Rate:{true_rate:.6f} | True Per-user Rate:{true_user:.6f}')
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {'worst_sum':worst_sum_list,'true_sum':true_sum_list,'worst_per_user':worst_user_list,'true_per_user':true_user_list}

def parse_sdr_sca_baseline(x_list,data):
    if data is None: return None
    if isinstance(data,dict): return [float(data[x]) for x in x_list]
    if isinstance(data,(list,tuple)):
        if len(data)!=len(x_list): raise ValueError('SDR-SCA values must match x-axis length')
        return [float(x) for x in data]
    raise TypeError('SDR-SCA baseline must be dict, list, tuple, or None')

def plot_pair_rate(x_list,rate_dict,metric_mode,sdr_sca_baseline=None):
    print(f'Plotting {metric_mode} curves==========')
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    color_dict = {
        'EA_Gformer_2D': '#0072BD',
        'PairPrior_EA_Gformer_2D': '#45BD00',
        'RiskGated_EA_Gformer_2D': '#D95319',
        'PairAwareB_RiskGated_EA_Gformer_2D': '#A2142F',
        'PairAwareMask_EA_Gformer_2D': '#4DBEEE',
        'PairAwareMask_RiskGated_EA_Gformer_2D': '#7E2F8E',
        'PhysicalRiskGated_EA_Gformer_2D': '#D95319',
        'SecGNN_matched_128': '#EDB120',
    }
    marker_dict = {
        'EA_Gformer_2D':'^','RiskGated_EA_Gformer_2D':'s',
        'PhysicalRiskGated_EA_Gformer_2D':'o','SecGNN_matched_128':'v','LittleSecGNN':'D'}
    label_dict = {
        'EA_Gformer_2D':'EA-Gformer','RiskGated_EA_Gformer_2D':'Risk-Gated EA-Gformer',
        'PhysicalRiskGated_EA_Gformer_2D':'PRG-Gformer',
        'SecGNN_matched_128':'SecGNN','LittleSecGNN':'Little-SecGNN'}

    for (model_name,train_mode),rates in rate_dict.items():
        color = color_dict.get(model_name,'#7F7F7F'); marker = marker_dict.get(model_name,'o')
        linestyle = '-'
        markerfacecolor = color

        label = f'{label_dict.get(model_name,model_name)}'
        ax.plot(x_list,rates,linestyle=linestyle,marker=marker,markersize=7.5,linewidth=2.1,color=color,
                label=label,markerfacecolor=markerfacecolor,markeredgecolor=color,markeredgewidth=1.3)

    sdr_rates = parse_sdr_sca_baseline(x_list,sdr_sca_baseline)
    if sdr_rates is not None:
        color = '#4D4D4D'
        ax.plot(x_list,sdr_rates,linestyle='-.',marker='D',
            markersize=7.2,linewidth=2.0,color=color,label='SDR-SCA',
            markerfacecolor='white',markeredgecolor=color,markeredgewidth=1.1,zorder=2
        )

    if metric_mode == 'worst_sum':
        ylabel = 'Worst-case Sum Secrecy Rate (bps/Hz)'; file_tag = 'WorstSumRate'
    elif metric_mode == 'true_sum':
        ylabel = 'True-channel Sum Secrecy Rate (bps/Hz)'; file_tag = 'TrueSumRate'
    elif metric_mode == 'worst_per_user':
        ylabel = 'Worst-case Per-user Secrecy Rate (bps/Hz)'; file_tag = 'WorstPerUserRate'
    elif metric_mode == 'true_per_user':
        ylabel = 'True-channel Per-user Secrecy Rate (bps/Hz)'; file_tag = 'TruePerUserRate'
    else:
        raise ValueError('Invalid metric_mode')

    boundary_x = (max_pair_train+x_list[x_list.index(max_pair_train)+1])/2
    ax.axvline(boundary_x,linestyle='--',linewidth=1.2,color='#7F7F7F',alpha=0.8)
    ax.set_xlabel('Number of LU-Eve Pairs (K = Eve)',fontsize=18)
    ax.set_ylabel(ylabel,fontsize=20)

    ax.set_xticks(x_list)
    ax.tick_params(axis='both',labelsize=18,direction='in')
    ax.grid(True,linestyle='-',linewidth=0.6,color='#BFBFBF',alpha=0.45)
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    ax.legend(fontsize=18, loc='lower right',frameon=True,framealpha=1.0,edgecolor='#777777',fancybox=False)

    ax.set_ylim(0, 30)
    ax.margins(x=0.03,y=0.10)
    ymin,ymax = ax.get_ylim()
    ax.text((x_list[0]+max_pair_train)/2,ymax-0.04*(ymax-ymin),'ID',fontsize=18,ha='center',va='top')
    ax.text((x_list[x_list.index(max_pair_train)+1]+x_list[-1])/2,ymax-0.04*(ymax-ymin),'OOD',fontsize=18,ha='center',va='top')
    fig.tight_layout()
    if pair_train_distribution == 'exponential':
        save_dir = './graphs/exp_pair_generalization'
        save_path = (f'{save_dir}/{file_tag}_vs_PairNumber_exponential_'
                     f'Kmean{Kmean}_Evemean{Evemean}_Kmax{max_pair_train}_NT{NT}_'
                     f'KR{rician_K_dB}dB_SNRLU{model_SNR_LU}_SNREve{model_SNR_Eve}_mu{model_mu}.png')
    else:
        save_dir = './graphs/exp_pair_generalization_uniform'
        save_path = (f'{save_dir}/{file_tag}_vs_PairNumber_uniform_'
                     f'Kmin{min_pair_train}_Kmax{max_pair_train}_NT{NT}_'
                     f'KR{rician_K_dB}dB_SNRLU{model_SNR_LU}_SNREve{model_SNR_Eve}_mu{model_mu}.png')
    os.makedirs(save_dir,exist_ok=True)
    fig.savefig(save_path,dpi=400,bbox_inches='tight',facecolor='white')
    pdf_path = save_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Figure saved: {save_path}')
    print(f'PDF saved:  {pdf_path}')

if __name__=='__main__':
    with open('./config.yaml','r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cal_security_rates_func_dict = {
        'one':cal_dataset_security_rate_one_to_one}
    cal_security_rates_func = cal_security_rates_func_dict[config.get('loss','one')]

    pair_train_distribution = 'exponential'

    Kmean = 4; Evemean = 4
    min_pair_train = 1

    max_pair_train = 10
    pair_test_list = [2,4,6,8,10,12,14,16]
    NT = 16; model_mu = 0.1
    rician_K_dB = config['system'].get('rician_K_dB',10.0)
    model_SNR_LU = config['system']['SNR_dB_LU']; model_SNR_Eve = config['system']['SNR_dB_Eve']
    batch_size = config['training']['batch_size']; number_test = config['training'].get('number_test',10000)
    pgd_steps_eval = config['training'].get('pgd_steps_eval',10); pgd_lr = config['training'].get('pgd_lr',0.2)
    gpu_id = config.get('device',{}).get('gpu_id',0)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    channel_scale = float(config['training'].get('channel_scale',1.0))
    print(f'Device: {device}, channel_scale={channel_scale:.4e}')
    print(f'Pair training distribution: {pair_train_distribution}')

    model_setting_list = [
        ('PhysicalRiskGated_EA_Gformer_2D','Robust'),

        ('EA_Gformer_2D','NonRobust'),

        ('SecGNN_matched_128','NonRobust'),
    ]

    sdr_sca_worst_sum_baseline_200samples = {
        2:9.236062,4:14.269219,6:17.802045,8:20.333118,10:22.534388,
        12:24.501011,14:25.935008,16:27.164867,18:28.200122,20:29.053015}
    sdr_sca_true_sum_baseline = None
    sdr_sca_worst_user_baseline = {k:v/k for k,v in sdr_sca_worst_sum_baseline_200samples.items()}
    sdr_sca_true_user_baseline = None

    pair_result_dict = {}
    for model_name,train_mode in model_setting_list:
        pair_result_dict[(model_name,train_mode)] = cal_pair_to_rate(model_name,train_mode)

    worst_sum_rate_dict = {m:r['worst_sum'] for m,r in pair_result_dict.items()}
    true_sum_rate_dict = {m:r['true_sum'] for m,r in pair_result_dict.items()}
    worst_user_rate_dict = {m:r['worst_per_user'] for m,r in pair_result_dict.items()}
    true_user_rate_dict = {m:r['true_per_user'] for m,r in pair_result_dict.items()}

    plot_pair_rate(pair_test_list,worst_sum_rate_dict,'worst_sum',sdr_sca_worst_sum_baseline_200samples)

