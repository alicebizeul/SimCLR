import torch
import torch.nn as nn
import torch.distributed as dist
from .gather import GatherLayer
import numpy as np
import random

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Custom_InfoNCE(nn.Module):
    def __init__(self, batch_size, bound, simclr_compatibility,subsample):
        super(Custom_InfoNCE, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = lambda x1, x2: custom_similarity(x1.unsqueeze(-1),torch.transpose(x2.unsqueeze(-1),0,-1),bound,subsample)
        self.simclr_compatibility=simclr_compatibility
        self.symetric=True
        self.subsample=subsample
        self.bound = bound

    def forward(self, anchor_rec, positive_rec):

        sim11 = self.similarity_f(anchor_rec,anchor_rec)
        sim22 = self.similarity_f(positive_rec,positive_rec)
        sim12 = self.similarity_f(anchor_rec,positive_rec)

        d = sim12.shape[-1]

        # removal of 1:1 pairs
        if not self.bound:
            sim11 = sim11.flatten()[1:].view(sim11.shape[0]-1, sim11.shape[0]+1)[:,:-1].reshape(sim11.shape[0], sim11.shape[0]-1)
            sim22 = sim22.flatten()[1:].view(sim22.shape[0]-1, sim22.shape[0]+1)[:,:-1].reshape(sim22.shape[0], sim22.shape[0]-1)
        else:
            list_index=[idx for idx in range(sim11.shape[0]*sim11.shape[1]*sim11.shape[2]) if (idx-((idx//(sim11.shape[0]*sim11.shape[-1]))*(sim11.shape[0]*sim11.shape[-1])))%sim11.shape[0] != ((idx-((idx//(sim11.shape[0]*sim11.shape[-1]))*(sim11.shape[0]*sim11.shape[-1])))//sim11.shape[0]) ]  #.remove(-1)
            sim11= torch.transpose(torch.transpose(sim11,0,1).flatten()[list_index].view(sim11.shape[1],sim11.shape[0],sim11.shape[-1]-1),0,1)
            sim22= torch.transpose(torch.transpose(sim22,0,1).flatten()[list_index].view(sim22.shape[1],sim22.shape[0],sim22.shape[-1]-1),0,1)

        # if not self.simclr_compatibility:
        #     pos = sim12[..., range(d), range(d)]
        #     neg = torch.logsumexp(torch.cat([sim11,pos.unsqueeze(1)],dim=1),dim=1)

        #     total_loss_value = torch.mean(- pos + neg)
        if self.bound:
            if self.subsample:
                keep = list(np.arange(int(anchor_rec.shape[1])))
                random.shuffle(keep)
                keep = keep[:int(len(keep)/2)]
                sim11=sim11[:,keep,:]
                sim12=sim12[:,keep,:]
                sim22=sim22[:,keep,:]
            deno = torch.cat([sim12, sim11], dim=-1)
            num = - torch.mean(torch.log(sim12[..., range(d), :, range(d)]),dim=1)
            deno = torch.log(torch.sum(torch.mean(deno,dim=1),dim=1))
            total_loss_value = torch.mean(num + deno)
        else:
            # diagonal - targets where the values should be the highest
            raw_scores1 = torch.cat([sim12, sim11], dim=-1)
            targets1 = torch.arange(d, dtype=torch.long, device=raw_scores1.device)
            total_loss_value = torch.nn.CrossEntropyLoss()(raw_scores1, targets1)
        if self.symetric: 
            del sim12
            sim12 = self.similarity_f(positive_rec,anchor_rec)
            # if not self.simclr_compatibility:
            #     pos = sim12[..., range(d), range(d)]
            #     neg = torch.logsumexp(torch.cat([sim11,pos.unsqueeze(1)],dim=1),dim=1)
            #     total_loss_value += torch.mean(- pos + neg)
            if self.bound:
                if self.subsample:
                    sim12=sim12[:,keep,:]
                num = - torch.mean(torch.log(sim12[..., range(d), :, range(d)]),dim=1)
                deno = torch.cat([sim12, sim22], dim=-1)
                deno = torch.log(torch.sum(torch.mean(deno,dim=1),dim=1))
                total_loss_value += torch.mean(num + deno)
            else:
                # creating matrix with all similarities
                raw_scores1 = torch.cat([sim12, sim22], dim=-1)
                total_loss_value += torch.nn.CrossEntropyLoss()(raw_scores1,targets1)
            total_loss_value *= 0.5

        losses_value = total_loss_value

        return losses_value

def custom_similarity(p_z_zrec,p_zpos_zrecpos,bound,subsample):
    p_z_zrec = (p_z_zrec+1e-8) # / p_z
    p_zpos_zrecpos = (p_zpos_zrecpos+1e-8) # / p_zpos
    if not bound:
        if subsample: 
            keep=random.shuffle(list(np.range(p_z_zrec.shape[1])))
            p_z_zrec = p_z_zrec[:,keep,:]
            p_zpos_zrecpos = p_zpos_zrecpos[:,keep,:]
        else: return torch.log(torch.sum(p_z_zrec*p_zpos_zrecpos,dim=1))  # log because cross entropy adds an exp
    else:
        return p_z_zrec*p_zpos_zrecpos


