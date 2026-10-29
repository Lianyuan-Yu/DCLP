import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys, logging
from tqdm import tqdm
import argparse
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='la') #, required=True)
parser.add_argument('--exp', type=str, default='diff_cnn_cl_la')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_0.1')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_0.1')
parser.add_argument('-se', '--split_eval', type=str, default='test')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
parser.add_argument('-ep', '--max_epoch', type=int, default=300)
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--unsup_loss', type=str, default='w_ce+dice')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--mu', type=float, default=2.0)
parser.add_argument('-s', '--ema_w', type=float, default=0.99)
parser.add_argument('-r', '--mu_rampup', action='store_true', default=True) 
parser.add_argument('-cr', '--rampup_epoch', type=float, default=None) 
parser.add_argument('--sample_num', type=int, default=2000, help='num of embeddings per class in memory bank') 
parser.add_argument('--num_filtered', type=int, default=5500,
                    help='num of unlabeled embeddings to calculate similarity')  
parser.add_argument('--tau', type=float, default=0.1, help='temperature')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from DiffVNet.diff_vnet import DiffVNet
from DiffVNet.mamba_vnet import VNet
from utils import EMA, maybe_mkdir, get_lr, fetch_data, GaussianSmoothing, seed_worker, poly_lr, print_func, sigmoid_rampup, feature_memory
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.data_loaders import DatasetAllTasks
from utils.config import Config
from data.StrongAug import get_StrongAug, ToTensor, CenterCrop
from data.brats2024 import (RandomCrop,RandomRotFlip, ToTensor)

config = Config(args.task)


def get_current_mu(epoch):
    if args.mu_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.rampup_epoch is None:
            args.rampup_epoch = args.max_epoch
        return args.mu * sigmoid_rampup(epoch, args.rampup_epoch)
    else:
        return args.mu


def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)


def make_loader(split, dst_cls=DatasetAllTasks, repeat=None, is_training=True, unlabeled=False, task="", transforms_tr=None):
    if is_training:
        dst = dst_cls(
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            transform=transforms_tr,
            task=task,
            num_cls=config.num_cls
        )
        return DataLoader(
            dst,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            drop_last=True
        )
    else:
        dst = dst_cls(
            split=split,
            is_val=True,
            task=task,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)


def make_model_all():

    model_1 = VNet(n_channels=config.num_channels,
                   n_classes=config.num_cls,
                   n_filters=config.n_filters,
                   normalization='batchnorm', has_dropout=True).cuda()

    model_2 = DiffVNet(n_channels=config.num_channels,
                       n_classes=config.num_cls,
                       n_filters=config.n_filters,
                       normalization='batchnorm', has_dropout=True).cuda()

    optimizer_1 = optim.SGD(model_1.parameters(),
                            lr=args.base_lr,
                            momentum=0.9,
                            weight_decay=3e-5,
                            nesterov=True
                            )

    optimizer_2 = optim.SGD(model_2.parameters(),
                            lr=args.base_lr,
                            momentum=0.9,
                            weight_decay=3e-5,
                            nesterov=True
                            )

    return model_1, optimizer_1, model_2, optimizer_2
    
def unlabeled_feature_choose(features, label, num_classes=4, num_filtered=64):
    indices = torch.randperm(features.size(0))[:num_filtered]
    choice_feature = features[indices]
    
    return indices, choice_feature, num_filtered

class GumbelTopK(nn.Module):
    """
    Perform top-k or Gumble top-k on given data
    """
    def __init__(self, k: int, dim: int = -1, gumble: bool = False):
        """
        Args:
            k: int, number of chosen
            dim: the dimension to perform top-k
            gumble: bool, whether to introduce Gumble noise when sampling
        """
        super().__init__()
        self.k = k
        self.dim = dim
        self.gumble = gumble

    def forward(self, logits):
        # logits shape: [B, N], B denotes batch size, and N denotes the multiplication of channel and spatial dim
        if self.gumble:
            u = torch.rand(size=logits.shape, device=logits.device)
            z = - torch.log(- torch.log(u))
            return torch.topk(logits + z, self.k, dim=self.dim)
        else:
            return torch.topk(logits, self.k, dim=self.dim)

if __name__ == '__main__':
    
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    vis_path = os.path.join(snapshot_path, 'vis')
    maybe_mkdir(vis_path)

    # make logger
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S', force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # make data loader
    transforms_train_labeled = get_StrongAug(config.patch_size, 3, 0.7)
    transforms_train_unlabeled = get_StrongAug(config.patch_size, 3, 0.7) 

    if "mmwhs" not in args.task:
        unlabeled_loader = make_loader(args.split_unlabeled, transforms_tr=transforms_train_unlabeled, task=args.task, unlabeled=True)
        labeled_loader = make_loader(args.split_labeled, transforms_tr=transforms_train_labeled, task=args.task, repeat=len(unlabeled_loader.dataset))
    else:
        labeled_loader = make_loader(args.split_labeled, transforms_tr=transforms_train_labeled, task=args.task)
        unlabeled_loader = make_loader(args.split_unlabeled, transforms_tr=transforms_train_unlabeled, task=args.task, repeat=len(labeled_loader.dataset), unlabeled=True)
    eval_loader = make_loader(args.split_eval, task=args.task, is_training=False)

    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')
    
    memory_bank = feature_memory.MemoryBank(num_labeled_samples=config.labeled_num, num_cls=config.num_cls)
    # make model, optimizer, and lr scheduler
    model_1, optimizer_1, model_2, optimizer_2 = make_model_all()

    # make loss function
    deno_loss  = make_loss_function(args.sup_loss)
    topk = GumbelTopK(k=args.sample_num, dim=0)
    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    mu = get_current_mu(0)

    best_eval = 0.0
    best_epoch = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_1_list = []
        loss_2_list = []
        loss_sup_1_list = []
        loss_unsup_1_list = []
        loss_cl_list = []

        model_1.train()
        model_2.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            image_l, label_l, idx_l = fetch_data(batch_l)
            label_l = label_l.long()
            image_u, idx_u = fetch_data(batch_u, labeled=False)

            if args.mixed_precision:
                with autocast():
                    # model_1
                    image = torch.cat([image_l, image_u], dim=0)
                    o, o_ft = model_1(image)
                    o_l, o_ft_l, o_u, o_ft_u = o[:config.batch_size], o_ft[:config.batch_size], o[config.batch_size:], o_ft[config.batch_size:] 
                    o_u_soft = torch.softmax(o_u, dim=1)
                    o_u_pseu = torch.argmax(o_u_soft, dim=1)
                    
                    # model_2
                    shp = (config.batch_size, config.num_cls)+config.patch_size 

                    label_l_onehot = torch.zeros(shp).cuda()
                    label_l_onehot.scatter_(1, label_l, 1)
                    xl_start = label_l_onehot * 2 - 1

                    label_u_onehot = torch.zeros(shp).cuda()
                    label_u_onehot.scatter_(1, o_u_pseu.unsqueeze(dim=1), 1)
                    xu_start = label_u_onehot * 2 - 1

                    xl_t, t_l, noise_l = model_2(x=xl_start, pred_type="q_sample")
                    p_l_xi, ft_l_xi = model_2(x=xl_t, step=t_l, image=image_l, pred_type="D_xi_l")

                    xu_t, t_u, noise_u = model_2(x=xu_start, pred_type="q_sample")
                    p_u_xi, ft_u_xi = model_2(x=xu_t, step=t_u, image=image_u, pred_type="D_xi_l")

                    with torch.no_grad():
                        smoothing = GaussianSmoothing(config.num_cls, 3, 1)
                        p_l_xi_soft = smoothing(F.gumbel_softmax(p_l_xi, dim=1))
                        pseudo_label_l = torch.argmax(p_l_xi_soft, dim=1, keepdim=True)
                        p_u_xi_soft = smoothing(F.gumbel_softmax(p_u_xi, dim=1))
                        pseudo_label = torch.argmax(p_u_xi_soft, dim=1, keepdim=True)
                        
                    # contrative loss
                    with torch.no_grad():
                        labeled_features = ft_l_xi.permute(0, 2, 3, 4, 1).contiguous()
                        b, h, w, d, labeled_features_dim = labeled_features.shape
            
                        # get ema projected features               
                        proj_labeled_features = model_2.projection_head(labeled_features.view(-1, labeled_features_dim))
                        proj_labeled_features = proj_labeled_features.view(b, h, w, d, -1)
                        mask_prediction_correctly = ((pseudo_label_l == label_l).float() == 1)
                        
                        labels_correct_list = []
                        labeled_features_correct_list = []
                        labeled_index_list = []
                        for i in range(config.batch_size):
                            labels_correct_list.append(label_l[i][mask_prediction_correctly[i]])
                            labeled_features_correct_list.append(proj_labeled_features[i][mask_prediction_correctly[i].squeeze(0)])
                            labeled_index_list.append(idx_l[i])
            
                        # updated memory bank
                        memory_bank.update_labeled_features(labeled_features_correct_list, labels_correct_list,
                                                            labeled_index_list)
                        memory, memory_label = memory_bank.sample_labeled_features(args.num_filtered) # (num class,N,C)
        
                    # choose model's unlabeled feature
                    unlabeled_features = o_ft_u.permute(0, 2, 3, 4, 1)  # [num_consist, feat_dim]
                    projected_feature = model_1.projection_head(
                        unlabeled_features.contiguous().view(-1, labeled_features_dim))
                    predicted_feature = projected_feature.view(b, h, w, d, -1)
        
                    # mask_prediction_correctly = predicted_unlabel == predicted_unlabel_ema
                    mask_prediction_fuzzy = (o_u_soft[:, 1, ...] >= 0.1)
                    unlabeled_features = predicted_feature[mask_prediction_fuzzy]
        
                    predicted_unlabel_prob = o_u_soft.permute(1, 0, 2, 3, 4)[:,mask_prediction_fuzzy]
                    # prob.shape=[class_num, chosen]
                    predicted_unlabel = o_u_pseu[mask_prediction_fuzzy]
        
                    pixel_positions, unlabeled_feature, unlabeled_num = unlabeled_feature_choose(unlabeled_features,
                                                                                                 predicted_unlabel,
                                                                                                 config.num_cls,
                                                                                                 num_filtered=args.num_filtered)
                    unlabeled_prob = predicted_unlabel_prob[:, pixel_positions]
        
                    # sample memory bank size labeled features from memory bank
                    L_contra = torch.tensor(0.0).cuda()
                    for i in range(config.num_cls):
                        if memory[i] is not None and len(unlabeled_feature)>=4000:
                            norm_memory = F.normalize(memory[i].cuda()).cuda() # (N, C)
                            norm_unlabeled_feature = F.normalize(unlabeled_feature) # (N, C)
                            cos_distance = F.cosine_similarity(norm_unlabeled_feature.unsqueeze(1), norm_memory.unsqueeze(0), dim=-1) # (N * C, N * C)
                            cos_dis = cos_distance.sum(dim=1)/args.num_filtered                           
                            positive_indices = topk(cos_dis).indices  # K, 
                            positive = cos_dis[positive_indices]                           
                            negtive_indices = topk(1-cos_dis).indices
                            negative = cos_dis[negtive_indices]                            
                            nominator = torch.exp(positive / args.tau)
                            denominator = torch.exp(negative / args.tau).sum(dim=0) + nominator
                            L_contra += 0.003*(-torch.log(nominator / (denominator + 1e-8)).mean())

                    # loss_1
                    L_sup_1 = deno_loss(o_l, label_l)
                    L_unsup_1 = deno_loss(o_u, pseudo_label.detach())
                    loss_1 = L_sup_1 + 0.1*mu * (L_unsup_1 + L_contra)
                    
                    # loss_2
                    L_sup_2 = deno_loss(p_l_xi, label_l)
                    L_unsup_2 = deno_loss(p_u_xi, o_u_pseu.unsqueeze(dim=1))
                    loss_2 = L_sup_2 + 0.1*mu * L_unsup_2

                    loss = loss_1 + loss_2

                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer_1)
                amp_grad_scaler.step(optimizer_2)
                amp_grad_scaler.update()

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_1_list.append(loss_1.item())
            loss_2_list.append(loss_2.item())
            loss_sup_1_list.append(L_sup_1.item())
            loss_unsup_1_list.append(L_unsup_1.item())
            loss_cl_list.append(L_contra.item())


        logging.info(f'epoch {epoch_num} : loss_1 : {np.mean(loss_1_list)} | loss_sup_1 : {np.mean(loss_sup_1_list)} | loss_unsup_1 : {np.mean(loss_unsup_1_list)} |loss_cl : {np.mean(loss_cl_list)} | lr : {get_lr(optimizer_1)} | mu : {mu}')
        logging.info(f'epoch {epoch_num} : loss_2 : {np.mean(loss_2_list)} | lr : {get_lr(optimizer_2)} | mu : {mu}')
        optimizer_1.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_2.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)

        mu = get_current_mu(epoch_num)


        # =======================================================================================
        # Validation
        # =======================================================================================
        
        if epoch_num % 1 == 0:

            dice_list = [[] for _ in range(config.num_cls-1)]
            model_1.eval()
            dice_func = SoftDiceLoss(smooth=1e-8, do_bg=False)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt, idx_val = fetch_data(batch)
                    p_u_theta, ft = model_1(image)
                    del image

                    shp = (p_u_theta.shape[0], config.num_cls) + p_u_theta.shape[2:]
                    gt = gt.long()

                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    p_u_theta = torch.argmax(p_u_theta, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, p_u_theta, 1)


                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            
            save_path = os.path.join(snapshot_path, f'ckpts/epoch={epoch_num}_model_1.pth')
            torch.save({'state_dict': model_1.state_dict(), }, save_path)
            
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path1 = os.path.join(snapshot_path, 'ckpts/iter_{}_dice_{}_model_1.pth'.format(best_epoch, round(best_eval, 4)))
                torch.save({'state_dict': model_1.state_dict(),}, save_path1)
                save_path2 = os.path.join(snapshot_path, f'ckpts/best_model_2.pth')
                torch.save({'state_dict': model_2.state_dict(),}, save_path2)
                logging.info(f'saving best model to {save_path1}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            
            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break
                