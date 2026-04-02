import torch
from ignite.engine import Engine, Events
from typing import Sequence
import numpy as np

import math
from ignite.contrib.handlers import CosineAnnealingScheduler
from ema_pytorch import EMA
from torch.optim import Adam
from .utils.scheduler import LRScheduler, setup_scheduler_step

If_NDEBUG = False

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

class AutoEncoderTrainer(object):
    def __init__(
        self, model: torch.nn.Module,
        lr: float,
        iter_per_epoch: int,
        curriculum_it: int = 0,
        clip_grad_norm: float = 1.0,
        use_lr_scheduler: bool = False,
        **config,
        ):
        self.model = model
        self.config = config
        self.clip_grad_norm = clip_grad_norm
        self.iter_per_epoch = iter_per_epoch

            # Define optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
        # assert _check_lr_scheduler_params(lr_scheduler_warmup, lr_scheduler_update_every)
        self.lr_scheduler = LRScheduler(lr=lr, optimizer=self.optimizer, **config['lr_scheduler_kwargs']) if use_lr_scheduler else None

        class CurriculumLearning:
            def __init__(self):
                self.param_groups = [{'curriculum_factor': 0.}]

        self.curriculum = CurriculumLearning()
        self.curriculum_it = curriculum_it
        self.curriculum_scheduler = None
        if curriculum_it is not None and curriculum_it > 0.0:
            curriculum_scheduler = CosineAnnealingScheduler(self.curriculum,
                                                        'curriculum_factor',
                                                        start_value=1.0,
                                                        end_value=0.,
                                                        cycle_size=curriculum_it*iter_per_epoch,
                                                        start_value_mult=0.0,
                                                        save_history=True,
                                                        )
            self.curriculum_scheduler = curriculum_scheduler

    def objects_to_checkpoint(self):
        # Define objecs to save in checkpoint
        if self.curriculum_it is not None and self.curriculum_it > 0.0:
            objects_to_checkpoint = {'optimizer': self.optimizer, 'curriculum_scheduler': self.curriculum_scheduler} #'lr_scheduler': lr_scheduler, 
        else: 
            objects_to_checkpoint = { 'optimizer': self.optimizer,} #'lr_scheduler': lr_scheduler, 
            print("no curriculum scheduler")
        if self.lr_scheduler is not None:
            objects_to_checkpoint = {**objects_to_checkpoint, **{'lr_scheduler': self.lr_scheduler.lr_scheduler}}
        return objects_to_checkpoint

    def get_random_ph(self, engine):
        if engine.state.epoch >= self.config["prediction_horizon_train_min_from_epoch"]:
            ph_min = self.config["prediction_horizon_train_min"]
        else: 
            ph_min_per_epoch = torch.linspace(start=1, end=self.config["prediction_horizon_train_min"],steps=self.config["prediction_horizon_train_min_from_epoch"]*self.iter_per_epoch, dtype=int)
            ph_min = ph_min_per_epoch[engine.state.iteration]
        # ph_range = prediction_horizon_train - ph_min        
        ph = max(int(np.rint((1. - self.curriculum.param_groups[0]['curriculum_factor']) * self.config['prediction_horizon_train'])), ph_min)
        if ph > ph_min and self.config['random_prediction_horizon']:
            ph = np.random.randint(ph_min, ph)
        return ph
    
    # Define process function which is called during every training step
    def train_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = batch

        ph = self.get_random_ph(engine)
        y = y[:, :ph]

        pred, _, _ = self.model.autoencode(y,past=x, ph=ph)
 
        loss = self.model.loss(pred, y) 
        loss_pose_unscaled,  loss_hip_unscaled = loss, 0.

        loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        return loss, loss_pose_unscaled, (loss_hip_unscaled, _), pred, y, ph
    
    def validation_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        self.model.eval()
        with torch.no_grad():
            x, y = batch
            model_out, z_past, z = self.model.autoencode(y,past=x, ph=self.config['prediction_horizon_eval']) #model(x, y, prediction_horizon_eval, None)
        return model_out, y, x, z
        
        
class TrainerDiffusion(object):
    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        *,
        device: "cpu",
        lr = 1e-4,
        weight_decay=0., 
        train_pick_best_sample_among_k = 1,
        similarity_space = 'latent_space', # 'latent_space' or 'input_space' or 'metric_space
        if_use_ema = True,
        ema_update_every = 10,
        ema_decay = 0.995,
        ema_power = 2 / 3,
        ema_min_value = 0.0,
        step_start_ema: 100,
        adam_betas = (0.9, 0.99),
        use_lr_scheduler: bool = False,
        num_samples = 25,
        max_grad_norm = 1.,
        skeleton=None,
        autoencoder=None,
        **config,

    ):
        # model
        self.device = device
        self.model = diffusion_model
        assert self.model.condition
        # self.channels = diffusion_model.channels
        self.train_pick_best_sample_among_k = train_pick_best_sample_among_k
        self.similarity_space = similarity_space
        assert similarity_space in ['input_space', 'metric_space', 'latent_space'], f"Similarity space must be either 'input_space' or 'metric_space' or 'latent_space' but is {similarity_space}"
        
        self.skeleton = skeleton
        self.autoencoder = autoencoder
        self.config = config

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples

        # self.batch_size = train_batch_size
        self.max_grad_norm = max_grad_norm

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = lr, betas = adam_betas, weight_decay=weight_decay)

        # for logging results in a folder periodically

        self.if_use_ema = if_use_ema
        if if_use_ema:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every, update_after_step=step_start_ema, power=ema_power, min_value=ema_min_value)
            self.ema.to(self.device)
        self.lr_scheduler = LRScheduler(lr=lr, optimizer=self.opt, **config['lr_scheduler_kwargs']) if use_lr_scheduler else None

        # step counter state
        self.step = 0
        self.statistics_pred = None
        self.statistics_obs = None
    
    def get_checkpoint_object(self):
            from denoising_diffusion_pytorch.version import __version__
            data = {
                'model': self.model,
                'opt': self.opt,
            }
            if self.if_use_ema:
                data = {**data, **{'ema': self.ema}}
            if self.lr_scheduler is not None:
                data = {**data, **{'lr_scheduler': self.lr_scheduler.lr_scheduler}}

            return data
    
    
    def to_comparison_space_train(self, samples:torch.Tensor, diff_input:torch.Tensor, past_seq:torch.Tensor, autoencoder:torch.nn.Module, fut_seq:torch.Tensor, 
                                  space='latent_space', x_cond:torch.Tensor=None):
        # check dimensions of samples & other stuff
        num_samples = samples.shape[0]//past_seq.shape[0]
        x_cond = x_cond.repeat_interleave(num_samples, dim=0)
        if space == 'input_space' or space == 'metric_space':
            out, _ = self.decode_diffusion_sample(samples, obs=past_seq, autoencoder=autoencoder, x_cond=x_cond)
        # where do we want to compute similarity? input space(dct, or normaliezed)? metric space (in m)?
        if space == 'input_space':
            out_comparespace = out
            fut_seq_comparespace = fut_seq.unsqueeze(1).repeat_interleave(num_samples, dim=1)
        elif space == 'metric_space':
            # metric space we would need the skeleton
            out_comparespace = self.skeleton.transform_to_metric_space(out)
            fut_seq_comparespace = self.skeleton.transform_to_metric_space(fut_seq)
            out_comparespace = out_comparespace.flatten(start_dim=3)
            fut_seq_comparespace = fut_seq_comparespace.unsqueeze(1).flatten(start_dim=3).repeat_interleave(num_samples, dim=1)
        elif space == 'latent_space':
            out_comparespace = samples.view(diff_input.shape[0], -1, *samples.shape[1:])
            fut_seq_comparespace = diff_input.unsqueeze(1).repeat_interleave(num_samples, dim=1)
        else: 
            assert 0, "Not implemented"
        assert out_comparespace.shape == fut_seq_comparespace.shape
        return out_comparespace, fut_seq_comparespace
    
    def get_ksimilarity_loss(self, diffusion_loss:torch.Tensor, out_comparespace:torch.Tensor, fut_seq_comparespace:torch.Tensor, autoencoder:torch.nn.Module=None, **kwargs):
        b = out_comparespace.shape[0]
        with torch.no_grad():
            # where do we want to compute similarity? input space(dct, or normaliezed)? metric space (in m)?
            if self.similarity_space == 'input_space':
                loss_similarity = autoencoder.loss(out_comparespace, fut_seq_comparespace, reduction="none")
            elif self.similarity_space == 'metric_space':
                loss_similarity = torch.linalg.norm(out_comparespace-fut_seq_comparespace, axis=-1).mean(axis=-1)
            elif self.similarity_space == 'latent_space':
                loss_similarity = diffusion_loss
            else: 
                assert 0, "Not implemented"
            closest2gt_idx = loss_similarity.view(b, -1).min(axis=-1).indices
        loss = torch.gather(diffusion_loss.view(b, -1), dim=1, index=closest2gt_idx.unsqueeze(1)).squeeze(-1)
        assert len(loss.shape) == 1 and loss.shape[0] == b
        return loss, closest2gt_idx
    
    def loss(self, data:torch.Tensor, x_cond:torch.Tensor=None, current_epoch=None, **kwargs):
        b = data.shape[0]
        k = self.train_pick_best_sample_among_k

        if k <= 1:
            loss, diff_weights, _ = self.model(data, x_cond=x_cond, n_train_samples=1)
            return (loss * diff_weights).mean()

        # --- no_grad selection + single recompute ---
        # Run all k candidates in no_grad (no activation storage, no backward paths).
        # Then recompute only the b selected passes with full gradient tracking.
        # The key: p_losses accepts noise as an explicit argument, so we pre-generate
        # noise_k once and reuse the winning slice in the gradient pass.

        # Sample one timestep per sample, shared across all k candidates.
        t   = torch.randint(0, self.model.num_timesteps, (b,), device=data.device).long()
        t_k = t.repeat_interleave(k, 0)  # (b*k,)

        # Pre-expand inputs and generate k noise tensors (leaf tensors, no grad).
        data_k   = data.repeat_interleave(k, 0)
        x_cond_k = x_cond.repeat_interleave(k, 0) if x_cond is not None else None
        noise_k  = self.model.get_white_noise(data_k)  # (b*k, ...)

        # No-grad forward: run all k denoising passes, select the best per sample.
        with torch.no_grad():
            loss_k, diff_weigths_k, samples_k = self.model.p_losses(
                data_k, t_k, noise=noise_k, x_cond=x_cond_k, n_train_samples=1
            )
            out_similarityspace, fut_seq_similarityspace = self.to_comparison_space_train(
                samples_k, diff_input=data, x_cond=x_cond, space=self.similarity_space, **kwargs
            )
            _, best_idx = self.get_ksimilarity_loss(loss_k, out_similarityspace, fut_seq_similarityspace, **kwargs)

        # Extract the winning noise and diffusion weight for each sample.
        noise_sel = noise_k.view(b, k, *noise_k.shape[1:])[torch.arange(b, device=data.device), best_idx]
        diff_weigths_sel     = diff_weigths_k.view(b, k)[torch.arange(b, device=data.device), best_idx]

        # Recompute ONLY the b selected forward passes with gradient tracking.
        sim_loss, _, _ = self.model.p_losses(
            data, t, noise=noise_sel, x_cond=x_cond, n_train_samples=1
        )
        sim_loss = sim_loss * diff_weigths_sel

        return sim_loss.mean()
    
    def train_step(self, engine: Engine, data: Sequence[torch.Tensor]):
        autoencoder = self.autoencoder
        if self.if_use_ema:
            self.ema.ema_model.train()
        self.model.train()   
        self.opt.zero_grad()

        with torch.no_grad():
            autoencoder.eval()
            x, y = data
            z_past, z = autoencoder.get_train_embeddings(y, past=x, state=None)

        # Define diffusion objective
        diffusion_gt, x_cond = z, z_past       
            
        loss  = self.loss(diffusion_gt, x_cond=x_cond, autoencoder=autoencoder, past_seq=x, fut_seq=y, current_epoch=engine.state.epoch) if self.model.condition else self.loss(diffusion_gt, current_epoch=engine.state.epoch)
        total_loss = loss.item()
        if If_NDEBUG:
            if torch.isnan(loss).any():
                import numpy as np
                import os
                mode = 'wb' 
                myfile =  f"debug/latest_run/diffusion/{engine.state.iteration}"
                os.makedirs(myfile)
                torch.save(self.model.state_dict(), os.path.join(myfile, f"checkpoint.pth.tar"))
                myfile = [myfile+"loss.npy", myfile+"target.npy", myfile+"h.npy", myfile+"z.npy", myfile+"x.npy"]
                myouts = [loss, y, z, z_past, x]
                for fpath, outensor in zip(myfile, myouts):
                    with open(fpath, mode) as f:
                        np.save( f, outensor.detach().cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.opt.step()
        self.opt.zero_grad()
        self.step += 1

        if self.if_use_ema:
            self.ema.update()
        return total_loss
    
    def decode_diffusion_sample(self, samples:torch.Tensor, obs:torch.Tensor, autoencoder:torch.nn.Module, x_cond:torch.Tensor=None):
            
        x_t = obs.repeat_interleave(samples.shape[0]//obs.shape[0], dim=0) 
            
        z_past, z = x_cond, samples
        out = autoencoder.decode(x_t, z, z_past, self.config['prediction_horizon_eval'])
        
        out =out.view(obs.shape[0], -1, *out.shape[1:]) # (batch, num_samples, T, J, 3)
        samples = samples.view(obs.shape[0], -1, *samples.shape[1:])
        return out, samples

    def validation_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        autoencoder = self.autoencoder
        if self.if_use_ema:
            self.ema.ema_model.eval()
        self.model.eval()  
        with torch.no_grad():
            x, y = batch
            autoencoder.eval()
            if self.model.condition:
                past_embedding = autoencoder.get_past_embedding(x)
                x_cond = past_embedding
                x_cond = x_cond.repeat_interleave(self.config['num_prob_samples'], dim=0) 
            else: 
                x_cond = None

            if self.if_use_ema:
                samples, _ = self.ema.ema_model.sample(batch_size=self.config['num_prob_samples']*x.shape[0], x_cond=x_cond)
            else: 
                samples, _ = self.model.sample(batch_size=self.config['num_prob_samples']*x.shape[0], x_cond=x_cond)


            out, samples = self.decode_diffusion_sample(samples, obs=x, autoencoder=autoencoder, x_cond=x_cond)
            
            return out, y, samples, x

