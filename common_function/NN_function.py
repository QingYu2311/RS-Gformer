import torch
def complex_matrix_matmul(A,B):
    return torch.view_as_real(torch.matmul(torch.view_as_complex(A),torch.view_as_complex(B)))