import torch
import torch.nn as nn
import torch.distributed as dist
from .gather import GatherLayer

class Custom_InfoNCE(nn.Module):
    def __init__(self, batch_size, bound, simclr_compatibility):
        super(Custom_InfoNCE, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = lambda x1, x2: custom_similarity(x1.unsqueeze(-1),x2.unsqueeze(-2))
        self.simclr_compatibility=simclr_compatibility
        self.symetric=True
        self.bound = bound

    def forward(self, anchor_rec, positive_rec):

        sim11 = self.similarity_f(anchor_rec,anchor_rec)
        sim22 = self.similarity_f(positive_rec,positive_rec)
        sim12 = self.similarity_f(anchor_rec,positive_rec)

        # removal of 1:1 pairs
        sim11 = sim11.flatten()[1:].view(sim11.shape[0]-1, sim11.shape[0]+1)[:,:-1].reshape(sim11.shape[0], sim11.shape[0]-1)
        sim22 = sim22.flatten()[1:].view(sim22.shape[0]-1, sim22.shape[0]+1)[:,:-1].reshape(sim22.shape[0], sim22.shape[0]-1)

        d = sim12.shape[-1]

        if not self.simclr_compatibility:
            pos = sim12[..., range(d), range(d)]
            neg = torch.logsumexp(torch.cat([sim11,pos.unsqueeze(1)],dim=1),dim=1)

            total_loss_value = torch.mean(- pos + neg)
        elif self.bound:
            
            num = - torch.mean(torch.log(sim12[..., range(d), range(d)]),dim=-1)
            deno = torch.cat([sim12, sim11], dim=-1)
            deno = torch.log(torch.sum(torch.mean(deno,dim=-1),dim=1))
            total_loss_value = torch.mean(num + deno)
        else:
            # diagonal - targets where the values should be the highest
            raw_scores1 = torch.cat([sim12, sim11], dim=-1)
            targets1 = torch.arange(d, dtype=torch.long, device=raw_scores1.device)
            total_loss_value = torch.nn.CrossEntropyLoss()(raw_scores1, targets1)

        if self.symetric: 
            sim12 = self.similarity_f(positive_rec,anchor_rec)
            if not self.simclr_compatibility:
                pos = sim12[..., range(d), range(d)]
                neg = torch.logsumexp(torch.cat([sim11,pos.unsqueeze(1)],dim=1),dim=1)
                total_loss_value += torch.mean(- pos + neg)
            elif self.bound:
                num = - torch.mean(torch.log(sim12[..., range(d), range(d)]),dim=-1)
                deno = torch.cat([sim12, sim22], dim=-1)
                deno = torch.log(torch.sum(torch.mean(deno,dim=-1),dim=1))
                total_loss_value += torch.mean(num + deno)
            else:
                # creating matrix with all similarities
                raw_scores1 = torch.cat([sim12, sim22], dim=-1)
                total_loss_value += torch.nn.CrossEntropyLoss()(raw_scores1,targets1)
            total_loss_value *= 0.5

        losses_value = total_loss_value

        return losses_value

def custom_similarity(p_z_zrec,p_zpos_zrecpos):
    p_z_zrec = (p_z_zrec+1e-8) # / p_z
    p_zpos_zrecpos = (p_zpos_zrecpos+1e-8) # / p_zpos
    return torch.log(torch.sum(torch.matmul(p_z_zrec,p_zpos_zrecpos),dim=0)) + torch.log(torch.tensor(1/p_z_zrec.shape[1])) # log because cross entropy adds an exp


