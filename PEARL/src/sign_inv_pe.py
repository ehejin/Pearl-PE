import torch
from torch import nn
from typing import List, Callable
    

class SignInvPe_PEARL(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module) -> None:
        super(SignInvPe_PEARL, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(
            self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = V
        x = self.phi(x, edge_index) 
        x = x.mean(dim=1)
        x = self.rho(x) 

        return x

    @property
    def out_dims(self) -> int:
        return self.rho.out_dims


class MaskedSignInvPe(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module) -> None:
        super(MaskedSignInvPe, self).__init__()
        self.phi = phi #GIN
        self.rho = rho #MLP

    def forward(
            self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = V.unsqueeze(-1) 
        x = self.phi(x, edge_index) + self.phi(-x, edge_index) 
        pe_dim, N = x.size(1), x.size(0)
        num_nodes = [torch.sum(batch == i) for i in range(batch[-1]+1)]
        a = torch.arange(0, pe_dim).to(x.device)
        mask = torch.cat([(a < num).unsqueeze(0).repeat([num, 1]) for num in num_nodes], dim=0) 
        x = (x*mask.unsqueeze(-1)).sum(dim=1) # [N, hidden_dims]
        x = self.rho(x)  # [N, D_pe]
        return x


    @property
    def out_dims(self) -> int:
        return self.rho.out_dims

class MaskedSignInvPe_PEARL(nn.Module):
    # pe = rho(mask-sum(phi(V)+phi(-V)))
    def __init__(self, phi: nn.Module, rho: nn.Module) -> None:
        super(MaskedSignInvPe, self).__init__()
        self.phi = phi #GIN
        self.rho = rho #MLP

    def forward(
            self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.phi(V, edge_index) 
        x = x.mean(dim=1)
        x = self.rho(x) 
        return x


    @property
    def out_dims(self) -> int:
        return self.rho.out_dims
    


