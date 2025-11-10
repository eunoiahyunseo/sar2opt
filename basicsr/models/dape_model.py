import torch
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
import os
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


from ram.models.ram import ram as ram_fix
from ram.models.ram_lora import ram as ram

import numpy as np
import copy
import loralib as lora
import time
from typing import Dict

import torch.nn as nn

### SAR2OPT MODIFICATION ###
def _gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = _gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss
### END SAR2OPT MODIFICATION ###


@MODEL_REGISTRY.register()
class DAPEModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(DAPEModel, self).__init__(opt)

        # print(opt)

        # degradation-aware prompt extractor
        self.net_g = ram(pretrained=opt['ram_model_path'], image_size=384, vit='swin_l')
        self.net_g = self.model_to_device(self.net_g)

        # original ram  
        self.net_g_fix = ram_fix(pretrained=opt['ram_model_path'], image_size=384, vit='swin_l')
        self.net_g_fix = self.model_to_device(self.net_g_fix)
        self.net_g_fix.eval()

        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()


        stage_dims = [192, 384, 768, 1536] # Swin-L 기준. 사용하는 모델에 맞게 수정
        common_dim = 256
        self.mmd_proj_layers = nn.ModuleList()
        for dim in stage_dims:
            self.mmd_proj_layers.append(nn.Linear(dim, common_dim))
        self.mmd_proj_layers = self.model_to_device(self.mmd_proj_layers)

    def init_training_settings(self):
        self.net_g.train()

        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('cri_feature_opt'):
            self.cri_feature = build_loss(train_opt['cri_feature_opt']).to(self.device)
        else:
            self.cri_feature = None


        ### SAR2OPT MODIFICATION ###
        # L_tag: loss for logits distribution (BCE)
        if train_opt.get('cri_logits_opt'):
            self.cri_logits = build_loss(train_opt['cri_logits_opt']).to(self.device)
        else:
            # Fallback to manual calculation if not defined
            self.cri_logits = None
            
        # L_MMD: loss for intermediate feature distribution (MMD)
        # MMD is calculated manually, so no builder needed, but we need weights from opt
        self.lambda_rep = train_opt.get('lambda_rep', 1.0)
        self.lambda_tag = train_opt.get('lambda_tag', 1.0)
        self.lambda_mmd = train_opt.get('lambda_mmd', 1.0)

        # Curriculum learning settings
        self.warmup_iter = train_opt.get('warmup_iter', 0)
        self.total_iter = self.opt.get('num_iter', 200000)
        self.temp_start = train_opt.get('temp_start', 5.0)
        self.temp_end = train_opt.get('temp_end', 1.0)
        ### END SAR2OPT MODIFICATION ###


        

        # lora setting
        lora.mark_only_lora_as_trainable(self.net_g)
        
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()


    def setup_optimizers(self):

        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')


        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    # def feed_data(self, data):
    #     self.lq = data['lq'].to(self.device)
    #     self.gt = data['gt'].to(self.device)
    #     self.lq_ram = data['lq_ram'].to(self.device)
    #     self.gt_ram = data['gt_ram'].to(self.device)


    def feed_data(self, data):
        ### SAR2OPT MODIFICATION ###
        # Use more descriptive names for SAR2OPT task
        self.sar = data['lq'].to(self.device)  # SAR image
        self.opt = data['gt'].to(self.device)  # Paired OPT image
        self.sar_ram = data['lq_ram'].to(self.device) # Resized SAR
        self.opt_ram = data['gt_ram'].to(self.device) # Resized OPT
        ### END SAR2OPT MODIFICATION ###

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()


        # single gpu training
        # with torch.no_grad():
        #     feature_gt, logits_gt, _, hierarchical_feats_opt = self.net_g_fix.condition_forward(self.gt_ram, only_feature=False)
        # feature_lq, logits_lq, _, hierarchical_feats_sar = self.net_g.condition_forward(self.lq_ram, only_feature=False)
        
        ## multi-gpus training
        # with torch.no_grad():
        #     feature_gt, logits_gt, _ = self.net_g_fix.module.condition_forward(self.gt_ram, only_feature=False)
        # feature_lq, logits_lq, _ = self.net_g.module.condition_forward(self.lq_ram, only_feature=False) 
        
        l_total = 0
        loss_dict = OrderedDict()

        ## feature loss
        # l_fea = self.cri_feature(feature_lq, feature_gt)
        # l_total += l_fea
        # loss_dict['l_fea'] = l_fea

        # ## logits loss
        # sigmoid_lq = torch.sigmoid(logits_lq)
        # sigmoid_gt = torch.sigmoid(logits_gt)
        # l_logits = -(sigmoid_gt*torch.log(sigmoid_lq) + (1-sigmoid_gt)*torch.log(1-sigmoid_lq))
        # l_logits = 1.0 * l_logits.mean()
        # l_total += l_logits
        # loss_dict['l_logits'] = l_logits

        with torch.no_grad():
            # final_rep_opt, logits_opt, _, inter_feat_opt = self.net_g_fix.condition_forward(self.opt_ram, only_feature=False)
            
            # For multi-gpus training, you might need .module
            final_rep_opt, logits_opt, _, hierarchical_feats_opt = self.net_g_fix.module.condition_forward(self.opt_ram, only_feature=False)

        # Student forward pass (SAR data)
        # final_rep_sar, logits_sar, _, inter_feat_sar = self.net_g.condition_forward(self.sar_ram, only_feature=False)
        
        # For multi-gpus training
        final_rep_sar, logits_sar, _, hierarchical_feats_sar = self.net_g.module.condition_forward(self.sar_ram, only_feature=False)


        if current_iter < self.warmup_iter:
            # During warmup, keep temp high
            current_temp = self.temp_start
            # During warmup, MMD and Rep loss are more important
            current_lambda_tag = self.lambda_tag * (current_iter / self.warmup_iter) # Gradually increase tag loss weight
            current_lambda_mmd = self.lambda_mmd
        else:
            # After warmup, anneal temp and use full tag loss weight
            progress = (current_iter - self.warmup_iter) / (self.total_iter - self.warmup_iter)
            current_temp = self.temp_start + (self.temp_end - self.temp_start) * progress
            current_lambda_tag = self.lambda_tag
            current_lambda_mmd = self.lambda_mmd # Keep MMD weight or slightly decrease if needed

        loss_dict['current_temp'] = current_temp

        if self.cri_feature:
            l_rep = self.cri_feature(final_rep_sar, final_rep_opt)
            l_total += self.lambda_rep * l_rep
            loss_dict['l_rep'] = l_rep
        if self.lambda_mmd > 0:
            l_mmd_hierarchical = 0
            
            # 각 Stage의 특징에 대해 MMD Loss를 계산하여 합산
            num_stages = len(hierarchical_feats_sar)
            # DOGAN처럼 각 스케일에 다른 가중치를 줄 수 있음
            stage_weights = [1.0, 1.0, 1.0, 1.0] # 예시: 모든 스테이지에 동일한 가중치

            for i in range(num_stages):
                inter_feat_sar = hierarchical_feats_sar[i]
                inter_feat_opt = hierarchical_feats_opt[i]

                inter_feat_sar_proj = self.mmd_proj_layers[i](inter_feat_sar)
                inter_feat_opt_proj = self.mmd_proj_layers[i](inter_feat_opt)
            
                l_mmd_stage = mmd_loss(inter_feat_sar_proj, inter_feat_opt_proj)
                
                l_mmd_hierarchical += l_mmd_stage * stage_weights[i]
            
            l_mmd_hierarchical /= num_stages # 평균
            
            l_total += current_lambda_mmd * l_mmd_hierarchical
            loss_dict['l_mmd_h'] = l_mmd_hierarchical
        if self.lambda_tag > 0:
            prob_sar = torch.sigmoid(logits_sar / current_temp)
            with torch.no_grad():
                prob_opt = torch.sigmoid(logits_opt / current_temp)
            
            # Using built-in loss for stability
            if self.cri_logits:
                l_tag = self.cri_logits(prob_sar, prob_opt)
            else:
                # Manual calculation (be careful with numerical stability)
                l_tag = -(prob_opt * torch.log(prob_sar.clamp(min=1e-8)) + \
                          (1 - prob_opt) * torch.log((1 - prob_sar).clamp(min=1e-8)))
                l_tag = l_tag.mean()

            l_total += current_lambda_tag * l_tag
            loss_dict['l_tag'] = l_tag


        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


        del self.lq, self.gt, feature_gt
        del feature_lq

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            self.lq_enhancer.eval()
            with torch.no_grad():
                self.feature_gt, self.logits_gt, self.targets_gt = self.net_g_fix.condition_forward(self.gt, only_feature=False)
                self.feature_lq, self.logits_lq, self.targets_lq = self.net_g.condition_forward(self.lq, only_feature=False)

            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
  
        if use_pbar:
            pbar.close()

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network_lora(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def save_network_lora(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                # torch.save(save_dict, save_path)
                torch.save(self.lora_state_dict(save_dict['params']), save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

    def lora_state_dict(self, my_state_dict , bias: str = 'none') -> Dict[str, torch.Tensor]:

        if bias == 'none':
            return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
        elif bias == 'all':
            return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
        elif bias == 'lora_only':
            to_return = {}
            for k in my_state_dict:
                if 'lora_' in k:
                    to_return[k] = my_state_dict[k]
                    bias_name = k.split('lora_')[0]+'bias'
                    if bias_name in my_state_dict:
                        to_return[bias_name] = my_state_dict[bias_name]
            return to_return
        else:
            raise NotImplementedError
