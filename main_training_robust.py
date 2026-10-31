import torch
import time,os,yaml
from generate_geometry_rician_strong_eve import load_dataset
from torch.utils.data import DataLoader,TensorDataset
from networks import PhysicalRiskGated_EA_Gformer_2D
from function_cal_security_rate import cal_dataset_security_rate_one_to_one
from loss_function_PGD_relative_epsilon import cal_robust_PGD_adversarial_loss_relative

torch.cuda.empty_cache()

def set_seed(seed=44):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__=='__main__':
    with open('./config.yaml','r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    seed = 2026
    set_seed(seed)

    cal_security_rates_func_dict = {
        'one':cal_dataset_security_rate_one_to_one,
    }
    cal_func_mode = config.get('loss','one')
    cal_security_rates_func = cal_security_rates_func_dict[cal_func_mode]

    K = config['system']['K']
    Eve = config['system']['Eve']
    NT = config['system']['NT']
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
    number_train = 100000
    number_test = config['training'].get('number_test',10000)
    layers = config['model']['layers']

    gpu_id = config.get('device',{}).get('gpu_id',0)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    root_path = './dataset_exp01_scenario'
    print('Loading training and test datasets...')
    dataset_K_train = load_dataset(
        f'{root_path}/dataset_K_train/dataset_train_K={K}_NT={NT}_KR={rician_K_dB}dB_number={number_train}.pt')
    dataset_K_test = load_dataset(
        f'{root_path}/dataset_K_test/dataset_test_K={K}_NT={NT}_KR={rician_K_dB}dB_number={number_test}.pt')
    dataset_Eve_train_est = load_dataset(
        f'{root_path}/dataset_Eve_train_est/dataset_train_est_Eve={Eve}_NT={NT}_KR={rician_K_dB}dB_mu={mu}_number={number_train}.pt')
    dataset_Eve_test_true = load_dataset(
        f'{root_path}/dataset_Eve_test_true/dataset_test_true_Eve={Eve}_NT={NT}_KR={rician_K_dB}dB_number={number_test}.pt')
    dataset_Eve_test_est = load_dataset(
        f'{root_path}/dataset_Eve_test_est/dataset_test_est_Eve={Eve}_NT={NT}_KR={rician_K_dB}dB_mu={mu}_number={number_test}.pt')

    dataset_train = TensorDataset(dataset_K_train.data,dataset_Eve_train_est.data)
    pin_memory = torch.cuda.is_available()
    dataloader_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,pin_memory=pin_memory)
    dataloader_K_test = DataLoader(dataset_K_test,batch_size=batch_size,shuffle=False,pin_memory=pin_memory)
    dataloader_Eve_test_true = DataLoader(dataset_Eve_test_true,batch_size=batch_size,shuffle=False,pin_memory=pin_memory)
    dataloader_Eve_test_est = DataLoader(dataset_Eve_test_est,batch_size=batch_size,shuffle=False,pin_memory=pin_memory)

    model = PhysicalRiskGated_EA_Gformer_2D(input_dim=[2],hidden_layers=layers,output_dim=[2],n_heads=1)

    model_name = model.__class__.__name__
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0)
    milestones = sorted(set([max(1,int(max_epoch*0.4)),max(1,int(max_epoch*0.65)),max(1,int(max_epoch*0.85))]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=0.3)

    print(f'Training {model_name} | seed={seed} | K={K} | Eve={Eve} | NT={NT} | '
          f'Rician K={rician_K_dB} dB | mu={mu} | LU SNR={SNR_dB_LU} dB | '
          f'Eve SNR={SNR_dB_Eve} dB')
    start_time = time.time()
    print(f'Layers: {layers} | learning rate: {learning_rate} | epochs: {max_epoch}')

    for epoch in range(max_epoch):
        model.train()
        train_rate_sum,train_number = 0.0,0

        for x_train_K,x_train_Eve_est in dataloader_train:
            x_train_K = x_train_K.to(device,non_blocking=True)
            x_train_Eve_est = x_train_Eve_est.to(device,non_blocking=True)
            real_batch_size = x_train_K.shape[0]

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
            train_number += real_batch_size

        scheduler.step()
        train_per_rate_worst = train_rate_sum/train_number

        if (epoch%eval_interval==0) or (epoch==max_epoch-1):
            model.eval()
            test_worst_sum,test_true_sum,test_number = 0.0,0.0,0

            for x_test_K,x_test_Eve_est,x_test_Eve_true in zip(
                dataloader_K_test,dataloader_Eve_test_est,dataloader_Eve_test_true):

                x_test_K = x_test_K.to(device,non_blocking=True)
                x_test_Eve_est = x_test_Eve_est.to(device,non_blocking=True)
                x_test_Eve_true = x_test_Eve_true.to(device,non_blocking=True)
                real_batch_size = x_test_K.shape[0]

                with torch.no_grad():
                    V_pred = model(x_test_K,x_test_Eve_est)

                evaluation_worst = cal_robust_PGD_adversarial_loss_relative(
                    H_K=x_test_K,H_Eve_est=x_test_Eve_est,V=V_pred,
                    cal_security_rates_func=cal_security_rates_func,batch_size=real_batch_size,
                    SNR_dB_LU=SNR_dB_LU,SNR_dB_Eve=SNR_dB_Eve,mu=mu,
                    pgd_steps=pgd_steps_eval,pgd_lr=pgd_lr,
                    use_conjugate=False,init_mode='zero',objective_mode='rate')

                with torch.no_grad():
                    evaluation_true = cal_security_rates_func(
                        H_K=x_test_K,H_Eve=x_test_Eve_true,V=V_pred,batch_size=real_batch_size,
                        SNR_dB_LU=SNR_dB_LU,SNR_dB_Eve=SNR_dB_Eve)

                test_worst_sum += evaluation_worst.detach().item()*real_batch_size
                test_true_sum += evaluation_true.detach().item()*real_batch_size
                test_number += real_batch_size

            test_per_rate_worst = test_worst_sum/test_number
            test_per_rate_true = test_true_sum/test_number

            end_time = time.time()
            print(f'Epoch {epoch} | train worst SSR: {train_per_rate_worst:.6f} | '
                  f'test worst SSR: {test_per_rate_worst:.6f} | '
                  f'test true-CSI SSR: {test_per_rate_true:.6f} | '
                  f'lr: {optimizer.param_groups[0]["lr"]:.3e} | '
                  f'elapsed: {end_time-start_time:.2f}s',flush=True)
            start_time = end_time

    model_save_dir = './models/robust_models'
    os.makedirs(model_save_dir,exist_ok=True)
    model_save_path = (
        f'{model_save_dir}/model_Robust_{model_name}_K{K}_Eve{Eve}_NT{NT}_'
        f'KR{rician_K_dB}dB_SNRLU{SNR_dB_LU}_SNREve{SNR_dB_Eve}_'
        f'mu{mu}_maxepoch{max_epoch}.pth')
    torch.save(model.state_dict(),model_save_path)

    print(f'Saved checkpoint: {model_save_path}')
