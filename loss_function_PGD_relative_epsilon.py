import torch

def _project_relative_l2_ball(E,epsilon_Eve):
    """Project each Eve error vector onto its relative L2 uncertainty ball."""
    eps_broadcast = epsilon_Eve[:,None,:,None]
    e_norm = torch.sqrt(torch.sum(E**2,dim=(1,3),keepdim=True)).clamp_min(1e-12)
    return E*torch.clamp(eps_broadcast/e_norm,max=1.0)

def cal_robust_PGD_adversarial_loss_relative(
    H_K,H_Eve_est,V,cal_security_rates_func,batch_size,SNR_dB_LU,SNR_dB_Eve,
    mu=0.1,pgd_steps=10,pgd_lr=0.2,use_conjugate=False,
    init_mode="zero",objective_mode="rate",return_worst_eve=False
):
    """PGD Eve-channel attack for one-to-one secrecy under relative CSI error."""
    if mu < 0:
        raise ValueError("mu must be nonnegative")
    if objective_mode not in ["rate","sinr"]:
        raise ValueError("objective_mode must be 'rate' or 'sinr'")

    device = H_K.device
    B,_,K,NT = H_K.shape
    _,_,Eve,NT_eve = H_Eve_est.shape
    assert NT_eve == NT
    assert Eve == K,"One-to-one mode requires Eve == K"

    eps_num = 1e-12
    real_batch_size = H_K.shape[0]
    sigma2_Eve = 10.0**(-SNR_dB_Eve/10.0)

    with torch.no_grad():
        norm_G_est = torch.sqrt(torch.sum(H_Eve_est**2,dim=(1,3))).clamp_min(eps_num)
        epsilon_Eve = float(mu)*norm_G_est
        eps_broadcast = epsilon_Eve[:,None,:,None]

    if mu == 0 or pgd_steps <= 0:
        robust_rate = cal_security_rates_func(
            H_K=H_K,H_Eve=H_Eve_est,V=V,batch_size=real_batch_size,
            SNR_dB_LU=SNR_dB_LU,SNR_dB_Eve=SNR_dB_Eve)
        if return_worst_eve:
            return robust_rate,H_Eve_est
        return robust_rate

    V_fixed = V.detach()
    V_fixed_complex = torch.complex(V_fixed[:,0],V_fixed[:,1])
    H_Eve_est_attack = H_Eve_est.detach()

    with torch.no_grad():
        if init_mode == "zero":
            E_worst = torch.zeros_like(H_Eve_est_attack)
        elif init_mode == "random":
            E_worst = torch.randn_like(H_Eve_est_attack)
            e_norm = torch.sqrt(torch.sum(E_worst**2,dim=(1,3),keepdim=True)).clamp_min(eps_num)
            E_worst = E_worst/e_norm
            real_dim = 2*NT
            radius = torch.rand(B,1,Eve,1,device=device,dtype=H_Eve_est.dtype).pow(1.0/real_dim)
            E_worst = E_worst*(eps_broadcast*radius)
        else:
            raise ValueError("init_mode must be 'zero' or 'random'")

    step_size = float(pgd_lr)*eps_broadcast

    for _ in range(pgd_steps):
        E_worst = E_worst.detach().requires_grad_(True)
        H_Eve_adv = H_Eve_est_attack+E_worst
        G_adv_complex = torch.complex(H_Eve_adv[:,0],H_Eve_adv[:,1])

        if use_conjugate:
            GW = torch.einsum("bkn,bni->bki",G_adv_complex.conj(),V_fixed_complex)
        else:
            GW = torch.einsum("bkn,bni->bki",G_adv_complex,V_fixed_complex)

        GW_abs2 = torch.abs(GW)**2
        sig_Eve = torch.diagonal(GW_abs2,dim1=1,dim2=2)
        intf_Eve = torch.sum(GW_abs2,dim=2)-sig_Eve+sigma2_Eve
        sinr_Eve = sig_Eve/(intf_Eve+eps_num)
        R_Eve = torch.log2(1.0+sinr_Eve)

        objective = torch.sum(R_Eve) if objective_mode=="rate" else torch.sum(sinr_Eve)
        grad = torch.autograd.grad(objective,E_worst,retain_graph=False,create_graph=False)[0]

        with torch.no_grad():
            grad_norm = torch.sqrt(torch.sum(grad**2,dim=(1,3),keepdim=True)).clamp_min(eps_num)
            E_worst = E_worst+step_size*grad/grad_norm
            E_worst = _project_relative_l2_ball(E_worst,epsilon_Eve)

    H_Eve_worst_case = H_Eve_est+E_worst.detach()
    robust_security_rate = cal_security_rates_func(
        H_K=H_K,H_Eve=H_Eve_worst_case,V=V,batch_size=real_batch_size,
        SNR_dB_LU=SNR_dB_LU,SNR_dB_Eve=SNR_dB_Eve)

    if return_worst_eve:
        return robust_security_rate,H_Eve_worst_case
    return robust_security_rate
