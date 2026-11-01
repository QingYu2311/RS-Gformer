import torch
import torch.nn as nn
import torch.nn.functional as F
from common_function.NN_function import complex_matrix_matmul

class FFN_2DPE_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        ini = 1 / torch.sqrt(torch.FloatTensor([input_dim*output_dim]))
        self.P1 = nn.Parameter(torch.rand([input_dim,output_dim])*2*ini-ini,requires_grad=True)
        self.P2 = nn.Parameter(torch.rand([input_dim,output_dim])*2*ini-ini,requires_grad=True)
        self.P3 = nn.Parameter(torch.rand([input_dim,output_dim])*2*ini-ini,requires_grad=True)

    def forward(self, D, aggr_function = lambda x,dim:torch.mean(x,dim,keepdim=True)):

        D1 = torch.matmul(D,self.P1)
        D2 = torch.matmul(aggr_function(D,-2),self.P2)
        D3 = torch.matmul(aggr_function(D,-3),self.P3)
        D = D1+D2+D3
        return D

class MultiHeadAttention_layer(nn.Module):
    def __init__(self, input_dim, n_heads):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads

        ini = torch.sqrt(torch.FloatTensor([1.0/input_dim/input_dim]))
        self.UQ = nn.Parameter(torch.rand([input_dim,2*n_heads],requires_grad=True)*2*ini-ini)
        self.UK = nn.Parameter(torch.rand([input_dim,2*n_heads],requires_grad=True)*2*ini-ini)
        self.UV = nn.Parameter(torch.rand([input_dim,2*n_heads],requires_grad=True)*2*ini-ini)
        self.WQ = nn.Parameter(torch.rand([input_dim,2*n_heads],requires_grad=True)*2*ini-ini)
        self.WK = nn.Parameter(torch.rand([2,2*n_heads],requires_grad=True)*2*ini-ini)
        self.WV = nn.Parameter(torch.rand([2,2*n_heads],requires_grad=True)*2*ini-ini)
        self.fcn1 = nn.Linear(2*self.n_heads,input_dim)
        self.fcn2 = nn.Linear(2*self.n_heads,input_dim)

    def forward(self,D_LU,H_Eve):

        batch_size, K, N, input_dim = D_LU.shape
        _ , J, _ , _ = H_Eve.shape

        Query_K_to_K = torch.matmul(D_LU,self.UQ).reshape(batch_size, K, N, self.n_heads, -1).permute(0,3,1,2,4)
        Key_K_to_K   = torch.matmul(D_LU,self.UK).reshape(batch_size, K, N, self.n_heads, -1).permute(0,3,1,2,4)
        Value_K      = torch.matmul(D_LU,self.UV).reshape(batch_size, K, N, self.n_heads, -1).permute(0,3,1,2,4)
        attention_scores_K_to_K = complex_matrix_matmul(Query_K_to_K,Key_K_to_K.transpose(-2,-3))
        context1 = complex_matrix_matmul(attention_scores_K_to_K,Value_K)
        context1 = context1.permute(0,2,3,1,4).reshape(batch_size,K,N,-1)
        context1 = self.fcn1(context1)

        Query_K_to_J = torch.matmul(D_LU,self.WQ). reshape(batch_size, K, N, self.n_heads, -1).permute(0,3,1,2,4)
        Key_K_to_J   = torch.matmul(H_Eve,self.WK).reshape(batch_size, J, N, self.n_heads, -1).permute(0,3,1,2,4)
        Value_J      = torch.matmul(H_Eve,self.WV).reshape(batch_size, J, N, self.n_heads, -1).permute(0,3,1,2,4)
        attention_scores_K_to_J = complex_matrix_matmul(Query_K_to_J,Key_K_to_J.transpose(-2,-3))
        context2 = complex_matrix_matmul(attention_scores_K_to_J,Value_J)
        context2 = context2.permute(0,2,3,1,4).reshape(batch_size,K,N,-1)
        context2 = self.fcn2(context2)

        context = torch.concat([context1,context2],dim=-1)
        return context

class EncoderLayer_2D_PE(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, is_BN=True, is_activation=True):
        super().__init__()
        self.MultiHeadAttention_layer = MultiHeadAttention_layer(input_dim,n_heads)
        self.FFN_layer = FFN_2DPE_layer(3*input_dim,output_dim)
        self.is_BN = is_BN
        self.is_activation = is_activation
        self.batch_norms = nn.BatchNorm1d(1)
        self.transfer = nn.LeakyReLU()

    def forward(self,D,H_Eve):
        context = self.MultiHeadAttention_layer(D, H_Eve)

        D = torch.concat([D,context],dim=-1)
        D = self.FFN_layer(D)

        D_shape = D.shape
        if self.is_BN:
            D = self.batch_norms(D.reshape(-1,1)).reshape(D_shape)
        if self.is_activation:
            D = self.transfer(D)
        return D

class EA_Gformer_2D(nn.Module):
    def __init__(self,input_dim, hidden_layers, output_dim, n_heads):
        super().__init__()
        self.dim = input_dim + hidden_layers + output_dim
        self.layers = nn.ModuleList()
        self.n_heads = n_heads

        for i in range(len(self.dim)-1):
            if i!= len(self.dim)-2:
                self.layers.append(EncoderLayer_2D_PE(input_dim=self.dim[i],output_dim=self.dim[i+1],n_heads=n_heads))
            else:
                self.layers.append(EncoderLayer_2D_PE(input_dim=self.dim[i],output_dim=self.dim[i+1],n_heads=n_heads,is_BN=False,is_activation=False))

    def forward(self,H_LU,H_Eve):

        D = H_LU.permute(0,2,3,1)
        H_Eve = H_Eve.permute(0,2,3,1)

        for layers in self.layers:
            D = layers(D,H_Eve)

        V = D.permute(0,3,2,1)
        magnitude_V = torch.sqrt(torch.sum(V**2,dim=[1,2,3],keepdim=True)) + 1e-8
        V = V/magnitude_V
        return V

class PhysicalRiskGatedMultiHeadAttention_layer(nn.Module):
    def __init__(self, input_dim, n_heads, sigma_Eve=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.sigma_Eve = sigma_Eve

        ini = torch.sqrt(torch.FloatTensor([1.0/input_dim/input_dim]))
        self.UQ = nn.Parameter(torch.rand([input_dim, 2*n_heads], requires_grad=True)*2*ini-ini)
        self.UK = nn.Parameter(torch.rand([input_dim, 2*n_heads], requires_grad=True)*2*ini-ini)
        self.UV = nn.Parameter(torch.rand([input_dim, 2*n_heads], requires_grad=True)*2*ini-ini)

        self.WQ = nn.Parameter(torch.rand([input_dim, 2*n_heads], requires_grad=True)*2*ini-ini)
        self.WK = nn.Parameter(torch.rand([2, 2*n_heads], requires_grad=True)*2*ini-ini)
        self.WV = nn.Parameter(torch.rand([2, 2*n_heads], requires_grad=True)*2*ini-ini)
        self.R  = nn.Linear(input_dim, 2)

        self.fcn1 = nn.Linear(2*self.n_heads, input_dim)
        self.fcn2 = nn.Linear(2*self.n_heads, input_dim)

    def forward(self, D_LU, H_Eve):
        batch_size, K, N, input_dim = D_LU.shape
        _ , J, _ , _ = H_Eve.shape
        D_temp = self.R(D_LU).permute(0,2,1,3)

        Query_K_to_K = torch.matmul(D_LU, self.UQ).reshape(batch_size, K, N, self.n_heads, -1).permute(0,3,1,2,4)
        Key_K_to_K   = torch.matmul(D_LU, self.UK).reshape(batch_size, K, N, self.n_heads, -1).permute(0,3,1,2,4)
        Value_K      = torch.matmul(D_LU, self.UV).reshape(batch_size, K, N, self.n_heads, -1).permute(0,3,1,2,4)

        attention_scores_K_to_K = complex_matrix_matmul(Query_K_to_K, Key_K_to_K.transpose(-2,-3))
        context1 = complex_matrix_matmul(attention_scores_K_to_K, Value_K)
        context1 = context1.permute(0,2,3,1,4).reshape(batch_size, K, N, -1)
        context1 = self.fcn1(context1)

        Query_K_to_J = torch.matmul(D_LU, self.WQ).reshape(batch_size, K, N, self.n_heads, -1).permute(0,3,1,2,4)
        Key_K_to_J   = torch.matmul(H_Eve, self.WK).reshape(batch_size, J, N, self.n_heads, -1).permute(0,3,1,2,4)
        Value_J      = torch.matmul(H_Eve, self.WV).reshape(batch_size, J, N, self.n_heads, -1).permute(0,3,1,2,4)

        attention_scores_K_to_J = complex_matrix_matmul(Query_K_to_J, Key_K_to_J.transpose(-2,-3))

        V_complex = D_temp[..., 0] + 1j * D_temp[..., 1]
        power_norm = torch.sqrt(torch.sum(torch.abs(V_complex) ** 2, dim=(1, 2), keepdim=True)) + 1e-8
        V_complex = V_complex / power_norm
        G_complex = H_Eve[..., 0] + 1j * H_Eve[..., 1]

        GW = torch.einsum('bnk,bjn->bkj', V_complex,G_complex)
        power_K_J_physical = torch.abs(GW)**2

        signal = power_K_J_physical

        total_power = torch.sum(power_K_J_physical, dim=1, keepdim=True)
        interference = total_power - power_K_J_physical

        sinr = signal / (interference+self.sigma_Eve)
        rho = torch.log2(1.0 + sinr)
        omega_J_K = F.softmax(rho, dim=-1)

        omega_J_K = omega_J_K.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        gated_attention_scores = attention_scores_K_to_J * omega_J_K.unsqueeze(-1)

        context2 = complex_matrix_matmul(gated_attention_scores, Value_J)
        context2 = context2.permute(0,2,3,1,4).reshape(batch_size, K, N, -1)
        context2 = self.fcn2(context2)

        context = torch.concat([context1, context2], dim=-1)
        return context

class PhysicalRiskGatedEncoderLayer_2D_PE(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, is_BN=True, is_activation=True):
        super().__init__()
        self.MultiHeadAttention_layer = PhysicalRiskGatedMultiHeadAttention_layer(input_dim,n_heads)
        self.FFN_layer = FFN_2DPE_layer(3*input_dim,output_dim)
        self.is_BN = is_BN
        self.is_activation = is_activation
        self.batch_norms = nn.BatchNorm1d(1)
        self.transfer = nn.LeakyReLU()

    def forward(self,D,H_Eve):
        context = self.MultiHeadAttention_layer(D, H_Eve)

        D = torch.concat([D,context],dim=-1)
        D = self.FFN_layer(D)

        D_shape = D.shape
        if self.is_BN:
            D = self.batch_norms(D.reshape(-1,1)).reshape(D_shape)
        if self.is_activation:
            D = self.transfer(D)
        return D

class PhysicalRiskGated_EA_Gformer_2D(nn.Module):
    def __init__(self,input_dim, hidden_layers, output_dim, n_heads):
        super().__init__()
        self.dim = input_dim + hidden_layers + output_dim
        self.layers = nn.ModuleList()
        self.n_heads = n_heads

        for i in range(len(self.dim)-1):
            if i!= len(self.dim)-2:
                self.layers.append(PhysicalRiskGatedEncoderLayer_2D_PE(input_dim=self.dim[i],output_dim=self.dim[i+1],n_heads=n_heads))
            else:
                self.layers.append(PhysicalRiskGatedEncoderLayer_2D_PE(input_dim=self.dim[i],output_dim=self.dim[i+1],n_heads=n_heads,is_BN=False,is_activation=False))

    def forward(self,H_LU,H_Eve):

        D = H_LU.permute(0,2,3,1)
        H_Eve = H_Eve.permute(0,2,3,1)

        for layers in self.layers:
            D = layers(D,H_Eve)

        V = D.permute(0,3,2,1)
        magnitude_V = torch.sqrt(torch.sum(V**2,dim=[1,2,3],keepdim=True)) + 1e-8
        V = V/magnitude_V
        return V