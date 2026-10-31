# -*- coding: utf-8 -*-
import torch
import time,os,yaml,math
from torch.utils.data import Dataset,DataLoader,Sampler
from generate_geometry_rician_strong_eve import load_dataset
from generate_pair_generalization_rician_strong_eve import load_variable_pair_dataset
from networks import PhysicalRiskGated_EA_Gformer_2D
from function_cal_security_rate import cal_dataset_security_rate_one_to_one
from loss_function_PGD_relative_epsilon import cal_robust_PGD_adversarial_loss_relative

torch.cuda.empty_cache()

def set_seed(seed=44):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class VariablePairTrainDataset(Dataset):
    def __init__(self,dataset_K,dataset_Eve_est):
        super().__init__()

        if len(dataset_K) != len(dataset_Eve_est):
            raise ValueError('LU and Eve training datasets have different sample counts')

        if not torch.equal(dataset_K.pair_array,dataset_Eve_est.pair_array):
            raise ValueError('LU and Eve training datasets have different pair arrays')

        self.dataset_K = dataset_K
        self.dataset_Eve_est = dataset_Eve_est
        self.pair_array = dataset_K.pair_array.clone()

    def __len__(self):
        return len(self.dataset_K)

    def __getitem__(self,index):
        return self.dataset_K.data[index],self.dataset_Eve_est.data[index]

class SamePairBatchSampler(Sampler):
    def __init__(self,pair_array,batch_size,shuffle=True):
        self.pair_array = pair_array.cpu().tolist() if torch.is_tensor(pair_array) else list(pair_array)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.pair_to_indices = {}
        for index,pair_number in enumerate(self.pair_array):
            pair_number = int(pair_number)
            if pair_number not in self.pair_to_indices:
                self.pair_to_indices[pair_number] = []
            self.pair_to_indices[pair_number].append(index)

    def __iter__(self):
        batch_list = []

        for pair_number,index_list in self.pair_to_indices.items():
            indices = torch.tensor(index_list,dtype=torch.long)
            if self.shuffle:
                indices = indices[torch.randperm(len(indices))]
            for start in range(0,len(indices),self.batch_size):
                batch = indices[start:start+self.batch_size].tolist()
                batch_list.append(batch)

        if self.shuffle:
            order = torch.randperm(len(batch_list)).tolist()
            batch_list = [batch_list[i] for i in order]

        for batch in batch_list:
            yield batch

    def __len__(self):
        batch_number = 0
        for index_list in self.pair_to_indices.values():
            batch_number += (len(index_list)+self.batch_size-1)//self.batch_size
        return batch_number

if __name__=='__main__':
    with open('./config.yaml','r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(2026)

    cal_security_rates_func_dict = {
        'one':cal_dataset_security_rate_one_to_one
    }
    cal_func_mode = config.get('loss','one')
    cal_security_rates_func = cal_security_rates_func_dict[cal_func_mode]

    Kmean = 4
    Evemean = 4
    max_pair_train = 10
    pair_value_list = [2,4,6,8,10,12,14,16]
    selection_pair_list = [2,4,6,8,10]

    NT = 16
    mu = 0.1
    rician_K_dB = config['system'].get('rician_K_dB',10.0)
    SNR_dB_LU = config['system']['SNR_dB_LU']
    SNR_dB_Eve = config['system']['SNR_dB_Eve']

    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    max_epoch = config['training']['max_epoch']
    pgd_steps_train = config['training'].get('pgd_steps_train',10)
    pgd_steps_eval = config['training'].get('pgd_steps_eval',10)
    pgd_lr = config['training'].get('pgd_lr',0.2)
    eval_interval = config['training'].get('eval_interval',5)
    number_train = 500000
    number_val = config['training'].get('number_val',10000)
    layers = config['model']['layers']

    gpu_id = config.get('device',{}).get('gpu_id',0)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    root_path = './dataset_pair_generalization_rician_strong_eve'
    print('Loading pair-generalization training data')

    dataset_K_train = load_variable_pair_dataset(
        f'{root_path}/dataset_K_train/'
        f'dataset_train_Kmean={Kmean}_Kmax={max_pair_train}_NT={NT}_'
        f'KR={rician_K_dB}dB_number={number_train}.pt')

    dataset_Eve_train_est = load_variable_pair_dataset(
        f'{root_path}/dataset_Eve_train_est/'
        f'dataset_train_est_Evemean={Evemean}_Evemax={max_pair_train}_NT={NT}_'
        f'KR={rician_K_dB}dB_mu={mu}_number={number_train}.pt')

    dataset_train = VariablePairTrainDataset(dataset_K_train,dataset_Eve_train_est)

    pin_memory = torch.cuda.is_available()
    train_batch_sampler = SamePairBatchSampler(
        dataset_train.pair_array,batch_size=batch_size,shuffle=True)

    dataloader_train = DataLoader(
        dataset_train,batch_sampler=train_batch_sampler,pin_memory=pin_memory)

    print(f'Training samples: {len(dataset_train)}')
    print(f'Training pair range: {int(dataset_train.pair_array.min())}~{int(dataset_train.pair_array.max())}')
    print(f'Mean training pairs: {dataset_train.pair_array.float().mean().item():.4f}')
    print(f'Training batches per epoch: {len(dataloader_train)}')

    print('Loading fixed-pair validation data')
    dataloader_val_dict = {}

    for pair_number in pair_value_list:
        K_val = Eve_val = pair_number

        dataset_K_val = load_dataset(
            f'{root_path}/dataset_K_val/'
            f'dataset_val_K={K_val}_NT={NT}_KR={rician_K_dB}dB_number={number_val}.pt')

        dataset_Eve_val_true = load_dataset(
            f'{root_path}/dataset_Eve_val_true/'
            f'dataset_val_true_Eve={Eve_val}_K={K_val}_NT={NT}_'
            f'KR={rician_K_dB}dB_number={number_val}.pt')

        dataset_Eve_val_est = load_dataset(
            f'{root_path}/dataset_Eve_val_est/'
            f'dataset_val_est_Eve={Eve_val}_K={K_val}_NT={NT}_'
            f'KR={rician_K_dB}dB_mu={mu}_number={number_val}.pt')

        dataloader_K_val = DataLoader(dataset_K_val,batch_size=batch_size,shuffle=False,pin_memory=pin_memory)

        dataloader_Eve_val_true = DataLoader(dataset_Eve_val_true,batch_size=batch_size,shuffle=False,pin_memory=pin_memory)

        dataloader_Eve_val_est = DataLoader(dataset_Eve_val_est,batch_size=batch_size,shuffle=False,pin_memory=pin_memory)

        dataloader_val_dict[pair_number] = (dataloader_K_val,dataloader_Eve_val_true,dataloader_Eve_val_est)

        print(f'Loaded validation set: K=Eve={pair_number}, samples={len(dataset_K_val)}')

    model = PhysicalRiskGated_EA_Gformer_2D(input_dim=[2],hidden_layers=layers,output_dim=[2],n_heads=1)

    model_name = model.__class__.__name__
    model.to(device)

    learning_gamma = 0.3

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0)
    milestones = sorted(set([max(1,int(max_epoch*0.5)),max(1,int(max_epoch*0.85)),]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=learning_gamma)

    print(f'======Starting robust pair-generalization training======\n'
          f'model={model_name},Kmean={Kmean},Evemean={Evemean},Kmax={max_pair_train},'
          f'NT={NT},RicianK={rician_K_dB}dB,mu={mu},'
          f'SNR_dB_LU={SNR_dB_LU},SNR_dB_Eve={SNR_dB_Eve}')

    best_worst_sum = -float('inf')
    best_epoch = -1
    model_save_dir = './models/robust_models_pair_generalization'
    os.makedirs(model_save_dir,exist_ok=True)
    best_model_path = (
        f'{model_save_dir}/model_Robust_{model_name}_'
        f'Kmean{Kmean}_Evemean{Evemean}_Kmax{max_pair_train}_NT{NT}_'
        f'KR{rician_K_dB}dB_SNRLU{SNR_dB_LU}_SNREve{SNR_dB_Eve}_'
        f'mu{mu}_maxepoch{max_epoch}_numbertrain_{number_train}pth')
    print(f'Checkpoint selection pairs: K=Eve={selection_pair_list},metric=mean worst-case SSR across K')

    start_time = time.time()
    print(f'Hidden layers: {layers} | Learning rate: {learning_rate} | Max epochs: {max_epoch}')

    for epoch in range(max_epoch):
        model.train()
        train_rate_sum,train_user_rate_sum,train_number = 0.0,0.0,0

        for x_train_K,x_train_Eve_est in dataloader_train:
            x_train_K = x_train_K.to(device,non_blocking=True)
            x_train_Eve_est = x_train_Eve_est.to(device,non_blocking=True)

            real_batch_size = x_train_K.shape[0]
            current_K = x_train_K.shape[2]

            V_pred = model(x_train_K,x_train_Eve_est)

            robust_rate = cal_robust_PGD_adversarial_loss_relative(
                H_K=x_train_K,H_Eve_est=x_train_Eve_est,V=V_pred,
                cal_security_rates_func=cal_security_rates_func,batch_size=real_batch_size,
                SNR_dB_LU=SNR_dB_LU,SNR_dB_Eve=SNR_dB_Eve,mu=mu,
                pgd_steps=pgd_steps_train,pgd_lr=pgd_lr,
                use_conjugate=False,init_mode='zero',objective_mode='rate')

            loss = -robust_rate

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
            optimizer.step()

            train_rate_sum += robust_rate.detach().item()*real_batch_size
            train_user_rate_sum += robust_rate.detach().item()/current_K*real_batch_size
            train_number += real_batch_size

        scheduler.step()

        train_per_rate_worst = train_rate_sum/train_number
        train_per_user_rate_worst = train_user_rate_sum/train_number

        if (epoch%eval_interval==0) or (epoch==max_epoch-1):
            model.eval()

            print("\n"+"="*80)
            print(f'Epoch={epoch}')
            print(f'Train worst-case SSR: {train_per_rate_worst:.6f}')
            print(f'Train worst-case per-user rate: {train_per_user_rate_worst:.6f}')
            print("-"*80)

            selection_worst_rates = []
            for pair_number in pair_value_list:
                K_val = Eve_val = pair_number

                dataloader_K_val,\
                dataloader_Eve_val_true,\
                dataloader_Eve_val_est = dataloader_val_dict[pair_number]

                val_worst_sum,val_true_sum,val_number = 0.0,0.0,0

                for x_val_K,x_val_Eve_est,x_val_Eve_true in zip(
                    dataloader_K_val,dataloader_Eve_val_est,dataloader_Eve_val_true):

                    x_val_K = x_val_K.to(device,non_blocking=True)
                    x_val_Eve_est = x_val_Eve_est.to(device,non_blocking=True)
                    x_val_Eve_true = x_val_Eve_true.to(device,non_blocking=True)
                    real_batch_size = x_val_K.shape[0]

                    with torch.no_grad():
                        V_pred = model(x_val_K,x_val_Eve_est)

                    evaluation_worst = cal_robust_PGD_adversarial_loss_relative(
                        H_K=x_val_K,H_Eve_est=x_val_Eve_est,V=V_pred,
                        cal_security_rates_func=cal_security_rates_func,batch_size=real_batch_size,
                        SNR_dB_LU=SNR_dB_LU,SNR_dB_Eve=SNR_dB_Eve,mu=mu,
                        pgd_steps=pgd_steps_eval,pgd_lr=pgd_lr,
                        use_conjugate=False,init_mode='zero',objective_mode='rate')

                    with torch.no_grad():
                        evaluation_true = cal_security_rates_func(
                            H_K=x_val_K,H_Eve=x_val_Eve_true,V=V_pred,
                            batch_size=real_batch_size,
                            SNR_dB_LU=SNR_dB_LU,SNR_dB_Eve=SNR_dB_Eve)

                    val_worst_sum += evaluation_worst.detach().item()*real_batch_size
                    val_true_sum += evaluation_true.detach().item()*real_batch_size
                    val_number += real_batch_size

                val_per_rate_worst = val_worst_sum/val_number
                val_per_rate_true = val_true_sum/val_number
                val_per_user_rate_worst = val_per_rate_worst/K_val
                val_per_user_rate_true = val_per_rate_true/K_val

                generalization_mode = 'ID' if pair_number<=max_pair_train else 'OOD'

                print(f'K=Eve={pair_number:2d} [{generalization_mode}] | '
                      f'Worst Sum Rate:{val_per_rate_worst:.6f} | '
                      f'Worst Per-user Rate:{val_per_user_rate_worst:.6f} | '
                      f'True Sum Rate:{val_per_rate_true:.6f} | '
                      f'True Per-user Rate:{val_per_user_rate_true:.6f}')

                if pair_number in selection_pair_list:
                    if not math.isfinite(val_per_rate_worst):
                        raise RuntimeError(f'Epoch={epoch},K={pair_number} has invalid validation worst-case SSR')
                    selection_worst_rates.append(val_per_rate_worst)

            if len(selection_worst_rates)!=len(selection_pair_list):
                raise RuntimeError('Missing validation K values for checkpoint selection')
            selection_score = sum(selection_worst_rates)/len(selection_worst_rates)
            print(f'ID validation metric(K={selection_pair_list})mean worst-case SSR: {selection_score:.6f}')
            if selection_score>best_worst_sum:
                best_worst_sum = selection_score
                best_epoch = epoch
                tmp_path = best_model_path+'.tmp'
                torch.save(model.state_dict(),tmp_path)
                os.replace(tmp_path,best_model_path)
                print(f'Best checkpoint updated: Epoch={best_epoch},'
                      f'ID mean worst-case SSR={best_worst_sum:.6f}',flush=True)

            end_time = time.time()
            print("-"*80)
            print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.3e}')
            print(f'Train/evaluation time: {end_time-start_time:.2f}s')
            print("="*80+"\n",flush=True)
            start_time = end_time

    if best_epoch < 0:
        raise RuntimeError('No successful ID validation; no checkpoint saved')

    print(f'Best checkpoint: validation K={selection_pair_list},Epoch={best_epoch},ID mean worst-case SSR={best_worst_sum:.6f}')
    print(f'Best checkpoint saved: {best_model_path}')
    print("\n"+"="*50)
    print("TensorBoard:")
    print("="*50)
    print("1. Run from the project directory")
    print("2. Run: tensorboard --logdir=runs")
    print("3. Open: http://localhost:6006")
