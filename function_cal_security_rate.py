import torch

def cal_dataset_rate(H,V,batch_size,SNR_dB):

    sigma2 = 10**(SNR_dB/(-10))
    H = torch.complex(H[:,0],H[:,1])
    V = torch.complex(V[:,0],V[:,1])
    HV = torch.matmul(H,V)
    desired_signal = torch.diagonal(HV,dim1=1,dim2=2)
    desired_signal_diag = torch.diag_embed(desired_signal)
    desired_power = torch.abs(desired_signal)**2
    interference_signal = HV - desired_signal_diag
    interference_power = torch.sum(torch.abs(interference_signal)**2,dim=2)
    SINR = desired_power/(interference_power+sigma2)
    total_rate = torch.sum(torch.log2(1+SINR))

    return total_rate/batch_size

def cal_dataset_security_rate_one_to_one(H_K, H_Eve, V, batch_size, SNR_dB_LU, SNR_dB_Eve):
    """Mean sum secrecy rate with one eavesdropper assigned to each LU."""

    sigma2_LU = 10 ** (SNR_dB_LU / (-10))
    sigma2_Eve = 10 ** (SNR_dB_Eve / (-10))

    H_K = torch.complex(H_K[:, 0], H_K[:, 1])
    H_Eve = torch.complex(H_Eve[:, 0], H_Eve[:, 1])
    V = torch.complex(V[:, 0], V[:, 1])

    HV = torch.matmul(H_K, V)
    desired_signal = torch.diagonal(HV, dim1=1, dim2=2)
    desired_power = torch.abs(desired_signal) ** 2
    interference = HV - torch.diag_embed(desired_signal)
    interference_power = torch.sum(torch.abs(interference) ** 2, dim=2)

    SINR_LU = desired_power / (interference_power + sigma2_LU)
    R_LU = torch.log2(1 + SINR_LU)

    GV = torch.matmul(H_Eve, V)

    desired_eve = torch.diagonal(GV, dim1=1, dim2=2)
    desired_p_eve = torch.abs(desired_eve) ** 2

    total_p_eve = torch.sum(torch.abs(GV) ** 2, dim=2)
    interference_p_eve = total_p_eve - desired_p_eve

    SINR_Eve = desired_p_eve / (interference_p_eve + sigma2_Eve)
    R_Eve = torch.log2(1 + SINR_Eve)

    security_rate_per_user = R_LU - R_Eve
    security_rate_per_user = torch.clamp(security_rate_per_user, min=0.0)

    total_security_rate = torch.sum(security_rate_per_user)
    avg_security_rate = total_security_rate / batch_size

    return avg_security_rate
