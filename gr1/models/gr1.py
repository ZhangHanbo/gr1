# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GR-1 model."""
import torch
import torch.nn as nn
import copy
import numpy as np
from torch.autograd import Variable
from einops import rearrange

import transformers
from flamingo_pytorch import PerceiverResampler

from gr1.models.trajectory_gpt2 import GPT2Model
from gr1.models.vision_transformer import Block
from gr1.models.transformer_utils import get_2d_sincos_pos_embed
from .configuration_gpt2 import GPT2Config


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class MultiBlock(nn.Module):
    def __init__(self, layer, num_layer, norm, pos_embed):
        super().__init__()
        assert isinstance(layer, Block)
        self.layers = _get_clones(layer, num_layer)
        self.norm = norm
        self.pos_embed = pos_embed

    def forward(self, x):
        # x: (bs, seq_len, hidden_size)
        x += self.pos_embed[:x.shape[1]][None, ...].to(x.device)
        for blk in self.layers:
            x = blk(x)
        x = self.norm(x)
        return x


class FcActHead(nn.Module):
    def __init__(self, hidden_size, arm_act_dim, gripper_act_dim, chunk_size):
        super().__init__()
        # Action prediction
        self.chunk_size = chunk_size
        self.arm_act_dim = arm_act_dim
        self.gripper_act_dim = gripper_act_dim
        self.pred_act_mlps = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size//2),
            nn.Linear(hidden_size//2, hidden_size//2)])
        self.pred_arm_act = nn.Linear(hidden_size//2, self.arm_act_dim * self.chunk_size) # arm action
        self.pred_gripper_act = nn.Linear(hidden_size//2, self.gripper_act_dim * chunk_size) # gripper action (binary)

    def forward(self, action_embedding, **kwargs):
        """
        action_embedding: (b, l, h)
        """
        for pred_act_mlp in self.pred_act_mlps:
            action_embedding = pred_act_mlp(action_embedding)
        
        bs, seq_len, _ = action_embedding.shape
        arm_action_preds = self.pred_arm_act(action_embedding)  # (b, l, arm_dim * chunk_size)
        gripper_action_preds = self.pred_gripper_act(action_embedding)  # (b, l, arm_dim * chunk_size)
        arm_action_preds = arm_action_preds.view(bs, seq_len, self.chunk_size, self.arm_act_dim)
        gripper_action_preds = gripper_action_preds.view(bs, seq_len, self.chunk_size, self.gripper_act_dim)
        return {
            'arm_action_preds': arm_action_preds,
            'gripper_action_preds': gripper_action_preds
        }
    

class VAEPolicy(nn.Module):
    # modified from: https://github.com/tonyzhaozh/act/blob/main/policy.py
    def __init__(self, hidden_size, arm_act_dim, gripper_act_dim, chunk_size, vae_latent_size=256, d_enc=3, d_dec=3, down_sample='none', **kwargs):
        """
        hidden_size: hidden size of the input embeddings
        act_dim: action dimension
        fwd_pred_next_n: number of future steps to predict, also known as chunk_size
        vae_latent_size: size of the latent variable for 
        d_enc: number of encoder layers
        d_dec: number of decoder layers
        down_sample: 'pooling' or 'none'. If 'pooling', use global pooling to down-sample the action embeddings, 
            else 'none' to keep the original size.
        """
        
        super().__init__()
        fwd_pred_next_n = chunk_size
        self.hidden_size = hidden_size
        self.arm_act_dim = arm_act_dim
        self.gripper_act_dim = gripper_act_dim
        self.down_sample = down_sample

        self.kl_weight = kwargs.get('kl_weight', 1.0)
        print(f'KL Weight {self.kl_weight}')

        self.fwd_pred_next_n = fwd_pred_next_n
        self.latent_dim = vae_latent_size # final size of latent z # TODO tune

        # encoder: (ENC_CLS, ACT) -> ENC_CLS -> latents
        self.cls_embed = nn.Embedding(1, self.hidden_size) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(self.arm_act_dim + self.gripper_act_dim, self.hidden_size) # project action to embedding
        self.register_buffer('enc_pos_embeddings', get_sinusoid_encoding_table(1+self.fwd_pred_next_n, self.hidden_size))
        self.encoder = MultiBlock(
            Block(self.hidden_size, 16, 4, qkv_bias=True, norm_layer=nn.LayerNorm),
            d_enc, nn.LayerNorm(self.hidden_size),
            self.enc_pos_embeddings
        )
        self.latent_proj = nn.Linear(self.hidden_size, self.latent_dim*2) # project hidden state to latent std, var

        # decoder: latents -> ENC_CLS + conditions: (OBS, QUERY) -> QUERY
        self.query_embed = nn.Embedding(fwd_pred_next_n, self.hidden_size)
        self.latent_out_proj = nn.Linear(self.latent_dim, self.hidden_size) # project latent sample to embedding
        self.global_1d_pool = nn.AdaptiveAvgPool1d(1) # global pooling for down-sampling, typically latent is 1.
        self.register_buffer('dec_pos_embeddings', get_sinusoid_encoding_table(1024, self.hidden_size))
        self.decoder = MultiBlock(
            Block(self.hidden_size, 16, 4, qkv_bias=True, norm_layer=nn.LayerNorm),
            d_dec, nn.LayerNorm(self.hidden_size),
            self.dec_pos_embeddings
        )

        self.act_head = FcActHead(self.hidden_size, self.arm_act_dim, self.gripper_act_dim, 1)
        self.cached_variables = {}

    def forward(self, action_embeddings, action=None, **kwargs):
        """
        action_embeddings: b x l x d or b x l x n x d
        action: only for training the auto-encoder, b x l x chunk_size x act_dim
        """
        # adapt to traditional input names
        actions = action

        if action_embeddings.dim() == 3:
            action_embeddings = action_embeddings[:, :, None, :]
        assert action_embeddings.dim() == 4
        # HACK: for now, actions may not be None during testing due to legacy code. Fix it when possible to keep codes clean.
        if not self.training:
            actions = None

        bs, seq_len, _, _  = action_embeddings.shape

        if self.training:
            bs_a, seq_len_a, chunk_size, _ = actions.shape

            assert chunk_size == self.fwd_pred_next_n
            assert bs_a == bs and seq_len_a == seq_len

            # project action sequence to embedding dim, and concat with a CLS token
            actions = rearrange(actions, 'b l n d -> (b l) n d', b=bs, l=seq_len)
            action_embed = self.encoder_action_proj(actions) # (bs*seq, chunk_size, hidden_size)

            cls_embed = self.cls_embed.weight # (1, hidden_size)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs * seq_len, 1, 1) # (bs*seq, 1, hidden_size)

            encoder_input = torch.cat([cls_embed, action_embed], axis=1) # (bs*seq, chunk_size+1, hidden_size)

            # query model, output: (bs*seq, chunk_size + 1, hidden_size)
            encoder_output = self.encoder(encoder_input)
            encoder_output = encoder_output[:, 0] # take cls output only, (bs*seq, hidden_size)

            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample) # (bs*seq, hidden_size)

        else:
            # here for testing, we only use zero-vector as the latent variale.
            # if one wants multi-modal, this vector should be sampled from normalized gaussian distribution.
            mu = logvar = None
            latent_sample = torch.zeros([bs * seq_len, self.latent_dim], dtype=torch.float32).to(action_embeddings.device)
            latent_input = self.latent_out_proj(latent_sample) # (bs*seq, hidden_size)

        if self.down_sample == 'pooling':
            action_embeddings = rearrange(action_embeddings, 'b l n d-> (b l) n d')
            action_embeddings = self.global_1d_pool(action_embeddings.permute(0, 2, 1))
            action_embeddings = rearrange(action_embeddings, 'b d n-> b n d')
            # tok_seq = rearrange(tok_seq, '(b l) d n -> b l n d', b=bs, l=seq_len)
        elif self.down_sample == 'none':
            action_embeddings = rearrange(action_embeddings, 'b l n d-> (b l) n d')
        else:
            raise NotImplementedError

        assert action_embeddings.dim() == 3 and action_embeddings.shape[0] == bs * seq_len

        latent_input = latent_input[:, None] # (bs*seq, 1, hidden_size)
        # (bs*seq, 1+latent+chunk_size, hidden_size)
        dec_input = torch.cat([latent_input, action_embeddings, self.query_embed.weight[None, ...].repeat(bs * seq_len, 1, 1)], dim=1)
        # take the embeddings of queries only
        dec_output = self.decoder(dec_input)[:, -self.fwd_pred_next_n:, :]

        arm_and_gripper = self.act_head(dec_output)
        actions = arm_and_gripper['arm_action_preds'].squeeze()
        gripper = arm_and_gripper['gripper_action_preds'].squeeze()
        
        actions = rearrange(actions, '(b l) n d -> b l n d', b=bs, l=seq_len, n=self.fwd_pred_next_n)
        gripper = rearrange(gripper, '(b l) n d -> b l n d', b=bs, l=seq_len, n=self.fwd_pred_next_n).squeeze(-1)

        outputs = {}
        outputs['arm_action_preds'] = actions
        outputs['gripper_action_preds'] = gripper
        outputs['mu'] = mu
        outputs['logvar'] = logvar

        return outputs
    

class GR1(nn.Module):
    def __init__(
            self,
            model_clip,
            model_mae,
            state_dim,
            act_dim,
            chunk_size,
            hidden_size,
            sequence_length,
            training_target,
            img_feat_dim,
            patch_feat_dim,
            lang_feat_dim,
            resampler_params,
            fwd_pred_next_n = 1,
            without_norm_pixel_loss=False,
            use_hand_rgb=True,
            act_head='FC', # from FC or VAE
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.fwd_pred_next_n = fwd_pred_next_n
        if self.state_dim == 7:
            # ee pose
            self.arm_dim = 6
            self.gripper_dim = 1
        else:
            # joint states
            assert self.state_dim == 9
            self.arm_dim = 7
            self.gripper_dim = 2
            
        self.act_dim = act_dim
        self.sequence_length = sequence_length

        # GPT
        self.hidden_size = hidden_size
        config = GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        # Perciever resampler
        self.n_patch_latents = resampler_params['num_latents']
        self.perceiver_resampler = PerceiverResampler(
            dim=patch_feat_dim,
            depth=resampler_params['depth'],
            dim_head=resampler_params['dim_head'],
            heads=resampler_params['heads'],
            num_latents=self.n_patch_latents,
            num_media_embeds=resampler_params['num_media_embeds'])        

        # CLIP for language encoding
        self.model_clip = model_clip
        for _, param in self.model_clip.named_parameters():
            param.requires_grad = False

        # MAE for image encoding
        self.model_mae = model_mae
        for _, param in self.model_mae.named_parameters():
            param.requires_grad = False
        
        self.n_patches = 49
        self.patch_size = 16
        self.image_size = 224
        self.img_feat_dim = img_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.patch_feat_dim = patch_feat_dim
        self.use_hand_rgb = use_hand_rgb

        self.act_pred = False
        self.fwd_pred = False
        self.fwd_pred_hand = False
        if 'act_pred' in training_target:
            self.act_pred = True
        if 'fwd_pred' in training_target:
            self.fwd_pred = True
        if 'fwd_pred_hand' in training_target:
            self.fwd_pred_hand = True
        
        self.without_norm_pixel_loss = without_norm_pixel_loss

        # Embedding functions for states
        self.embed_arm_state = torch.nn.Linear(self.arm_dim, hidden_size)
        self.embed_gripper_state = torch.nn.Linear(self.gripper_dim, hidden_size) # one-hot gripper state
        self.embed_state = torch.nn.Linear(2*hidden_size, hidden_size)

        # Relative timestep embedding
        self.embed_timestep = nn.Embedding(self.sequence_length, hidden_size)

        # Embedding function for languages
        self.embed_lang = torch.nn.Linear(self.lang_feat_dim, hidden_size)

        # Embedding functions for images
        self.embed_hand_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_hand_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size) 
        self.embed_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size)

        # Layer norm
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Action query token
        self.action_queries = nn.Embedding(1, hidden_size) # arm + gripper

        # Observation query token
        self.obs_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)
        self.obs_hand_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)

        # Action head
        if act_head == 'FC':
            self.pred_act_head = FcActHead(hidden_size, self.arm_dim, self.gripper_dim, chunk_size)
        else:
            assert act_head == 'VAE', f"act_head should be 'FC' or 'VAE', got {act_head}"
            # TODO: make VAE Policy more flexible
            self.pred_act_head = VAEPolicy(hidden_size, self.arm_dim, self.gripper_dim, chunk_size, 
                                           vae_latent_size=64, d_enc=3, d_dec=3, down_sample='none')
        
        # Forward prediction
        self.decoder_embed = nn.Linear(hidden_size, hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))
        decoder_depth = 2
        self.decoder_blocks = nn.ModuleList([
            Block(hidden_size, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder_pred = nn.Linear(hidden_size, self.patch_size**2 * 3, bias=True) # decoder to patch
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (self.image_size//self.patch_size)**2,
            hidden_size), requires_grad=False)  # (1, n_patch, h)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.image_size//self.patch_size))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    def forward(self, 
                rgb, 
                state, 
                text, 
                attention_mask,
                hand_rgb=None, 
                action=None, # to train VAE, we need take actions as input
                action_mask=None,
                **kwargs
    ):
        language = text
        if self.use_hand_rgb:
            assert hand_rgb is not None

        obs_preds = None
        obs_hand_preds = None
        obs_targets = None
        obs_hand_targets = None
        arm_action_preds = None
        gripper_action_preds = None

        batch_size, sequence_length, c, h, w = rgb.shape
        
        # Embed state
        arm_state = state['arm']
        gripper_state = state['gripper']
        arm_state_embeddings = self.embed_arm_state(arm_state.view(batch_size, sequence_length, self.arm_dim))
        gripper_state_embeddings = self.embed_gripper_state(gripper_state)
        state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2)
        state_embeddings = self.embed_state(state_embeddings)  # (b, l, h)

        # Embed language
        lang_embeddings = self.model_clip.encode_text(language)
        lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6) # normalization 
        lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, h)
    
        # Get obs and patch feature from MAE
        obs_embeddings, patch_embeddings = self.model_mae(
            rgb.view(batch_size*sequence_length, c, h, w))  # (b * l, img_feat_dim), (b * l, n_patches, patch_feat_dim)
        obs_embeddings = obs_embeddings.view(batch_size, sequence_length, -1)  # (b, l, img_feat_dim)
        if self.use_hand_rgb:
            hand_obs_embeddings, hand_patch_embeddings = self.model_mae(
                hand_rgb.view(batch_size*sequence_length, c, h, w))
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, -1)  # (b, l, img_feat_dim)

        # Forward prediction
        obs_loss_mask = None
        if self.fwd_pred:
            obs_loss_mask = attention_mask[:, self.fwd_pred_next_n:]

            p = self.patch_size
            h_p = h // p
            w_p = w // p
            rgb = rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p)) 
            obs_targets = rgb.permute(0, 1, 3, 5, 4, 6, 2)
            obs_targets = obs_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2) * 3))  # (b, l, n_patches, p*p*3)
            if not self.without_norm_pixel_loss:
                # norm the target 
                obs_targets = (obs_targets - obs_targets.mean(dim=-1, keepdim=True)
                    ) / (obs_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
            obs_targets = obs_targets[:, self.fwd_pred_next_n:]  # (b, l-fwd_pred_next_n, n_patches, p*p*3)

            if self.fwd_pred_hand:
                hand_rgb = hand_rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p))
                obs_hand_targets = hand_rgb.permute(0, 1, 3, 5, 4, 6, 2)
                obs_hand_targets = obs_hand_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2)*3))  # (b, l, n_patches, p*p*3)
                if not self.without_norm_pixel_loss:
                    # norm the target 
                    obs_hand_targets = (obs_hand_targets - obs_hand_targets.mean(dim=-1, keepdim=True)
                        ) / (obs_hand_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)   
                obs_hand_targets = obs_hand_targets[:, self.fwd_pred_next_n:]  # (b, l-fwd_pred_next_n, n_patches, p*p*3)         

        # Use resampler to process patch embeddings
        patch_embeddings = patch_embeddings.unsqueeze(1)  # (b * l, 1, n_patches, patch_feat_dim)
        patch_embeddings = self.perceiver_resampler(patch_embeddings)  # (b * l, 1, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.squeeze(1)  # (b * l, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, l, n_patch_latents, patch_feat_dim)
        if self.use_hand_rgb:
            hand_patch_embeddings = hand_patch_embeddings.unsqueeze(1)
            hand_patch_embeddings = self.perceiver_resampler(hand_patch_embeddings)
            hand_patch_embeddings = hand_patch_embeddings.squeeze(1)
            hand_patch_embeddings = hand_patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, l, n_patch_latents, patch_feat_dim)
        
        # Embed images and patches
        obs_embeddings = self.embed_img(obs_embeddings.float())  # (b, l, h)
        patch_embeddings = self.embed_patch(patch_embeddings.float())  # (b, l, n_patch_latents, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = self.embed_hand_img(hand_obs_embeddings.float())  # (b, l, h)
            hand_patch_embeddings = self.embed_hand_patch(hand_patch_embeddings.float())  # (b, l, n_patch_latents, h)
        
        # Add timestep embeddings
        time_embeddings = self.embed_timestep.weight  # (l, h)
        lang_embeddings = lang_embeddings.view(batch_size, 1, -1) + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        patch_embeddings = patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)
        obs_embeddings = obs_embeddings + time_embeddings
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings + time_embeddings
            hand_patch_embeddings = hand_patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)

        # Format sequence: lang, state, patch, obs, hand_patch, hand_obs, [ACT], [OBS], [OBS_HAND]
        lang_embeddings = lang_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        state_embeddings = state_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        obs_embeddings = obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        stacked_inputs = torch.cat(
                (lang_embeddings, 
                 state_embeddings, 
                 patch_embeddings, 
                 obs_embeddings), dim=2)  # (b, l, n_tokens, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
            stacked_inputs = torch.cat(
                (stacked_inputs,
                 hand_patch_embeddings, 
                 hand_obs_embeddings), dim=2)  # (b, l, n_tokens, h)
        if self.act_pred:
            action_queries = self.action_queries.weight  # (1, h)
            action_queries = action_queries.view(1, 1, 1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, l, 1, h)
            stacked_inputs = torch.cat((stacked_inputs, action_queries), dim=2)  # (b, l, n_tokens, h)
        
        if self.fwd_pred:
            obs_queries = self.obs_queries.weight  # (n_patch_latents + 1, h)
            obs_queries = obs_queries.view(1, 1, self.n_patch_latents + 1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, l, n_patch_latents + 1, h)
            stacked_inputs = torch.cat((stacked_inputs, obs_queries), dim=2)
            if self.fwd_pred_hand:
                obs_hand_queries = self.obs_hand_queries.weight # 10, h
                obs_hand_queries = obs_hand_queries.view(1, 1, self.n_patch_latents+1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)
                stacked_inputs = torch.cat((stacked_inputs, obs_hand_queries), dim=2)
        
        # Number of tokens
        n_lang_tokens = 1
        n_state_tokens = 1
        n_patch_tokens = self.n_patch_latents
        n_obs_tokens = 1
        n_hand_patch_tokens = self.n_patch_latents
        n_hand_obs_tokens = 1
        n_tokens = n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens
        if self.use_hand_rgb:
            n_tokens += n_hand_obs_tokens
            n_tokens += n_hand_patch_tokens
        n_act_pred_tokens = 1
        if self.act_pred:
            act_query_token_start_i = n_tokens
            n_tokens += 1
        if self.fwd_pred:
            obs_query_token_start_i = n_tokens
            n_tokens += (n_patch_tokens + n_obs_tokens)
            if self.fwd_pred_hand:
                obs_hand_query_token_start_i = n_tokens
                n_tokens += (n_patch_tokens + n_obs_tokens) 

        # Layer norm
        stacked_inputs = stacked_inputs.reshape(batch_size, n_tokens * sequence_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Attention mask
        stacked_attention_mask = attention_mask.view(batch_size, sequence_length, 1)
        if self.use_hand_rgb:
            stacked_attention_mask = stacked_attention_mask.repeat(
                1, 1, n_lang_tokens + n_state_tokens + n_hand_patch_tokens + n_hand_obs_tokens + n_patch_tokens + n_obs_tokens)
        else:
            stacked_attention_mask = stacked_attention_mask.repeat(
                1, 1, n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens)
        if self.act_pred:
            act_query_attention_mask = torch.zeros((batch_size, sequence_length, n_act_pred_tokens), dtype=torch.long).cuda()
            stacked_attention_mask = torch.cat((stacked_attention_mask, act_query_attention_mask), dim=2)
        if self.fwd_pred:
            obs_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long).cuda()
            stacked_attention_mask = torch.cat((stacked_attention_mask, obs_query_attention_mask), dim=2)
            if self.fwd_pred_hand:
                obs_hand_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long).cuda()
                stacked_attention_mask = torch.cat((stacked_attention_mask, obs_hand_query_attention_mask), dim=2)
        stacked_attention_mask = stacked_attention_mask.reshape(batch_size, n_tokens * sequence_length)

        # GPT forward pass
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, sequence_length, n_tokens, self.hidden_size)

        # Action prediction
        if self.act_pred:
            action_embedding = x[:, :, act_query_token_start_i]
            prediction = self.pred_act_head(action_embedding, action=action)
            arm_action_preds = prediction['arm_action_preds']
            gripper_action_preds = prediction['gripper_action_preds']
            mu = prediction.get('mu', None)
            logvar = prediction.get('logvar', None)


        if self.fwd_pred:
            mask_token = self.mask_token  # (1, 1, 1, h)
            mask_tokens = mask_token.repeat(batch_size, sequence_length, (self.image_size//self.patch_size)**2, 1)  # (b, l, n_patches, h)
            mask_tokens = mask_tokens + self.decoder_pos_embed.unsqueeze(0).repeat(batch_size, sequence_length, 1, 1)  # (b, l, n_patches, h)

            obs_pred = self.decoder_embed(x[:, :, obs_query_token_start_i:(obs_query_token_start_i + self.n_patch_latents + n_obs_tokens)])  # (b, l, n_patch_latents + 1, h)
            obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=2)  # (b, l, n_patches + n_patch_latens + 1, h)
            obs_pred_ = obs_pred_.reshape(-1, obs_pred_.shape[-2], obs_pred_.shape[-1])  # (b * l, n_patches + n_patch_latens + 1, h)
            for blk in self.decoder_blocks:
                obs_pred_ = blk(obs_pred_)
            obs_pred_ = self.decoder_norm(obs_pred_)
            obs_preds = self.decoder_pred(obs_pred_)  # (b * l, n_patches + n_patch_latens + 1, h)
            obs_preds = obs_preds.reshape(batch_size, sequence_length, -1, obs_preds.shape[-1])  # (b, l, n_patches + n_patch_latens + 1, h)
            obs_preds = obs_preds[:, :, (self.n_patch_latents+n_obs_tokens):]  # (b, l, n_patches, h)
            obs_preds = obs_preds[:, :-self.fwd_pred_next_n]  # (b, l-fwd_pred_next_n, n_patches, h)

            if self.fwd_pred_hand:
                obs_pred_hand = self.decoder_embed(x[:, :, obs_hand_query_token_start_i:(obs_hand_query_token_start_i + self.n_patch_latents + n_obs_tokens)])
                obs_pred_hand_ = torch.cat([obs_pred_hand, mask_tokens], dim=2)
                obs_pred_hand_ = obs_pred_hand_.reshape(-1, obs_pred_hand_.shape[-2], obs_pred_hand_.shape[-1])
                for blk in self.decoder_blocks:
                    obs_pred_hand_ = blk(obs_pred_hand_)
                obs_pred_hand_ = self.decoder_norm(obs_pred_hand_)
                obs_hand_preds = self.decoder_pred(obs_pred_hand_)
                obs_hand_preds = obs_hand_preds.reshape(batch_size, sequence_length, -1, obs_hand_preds.shape[-1])
                obs_hand_preds = obs_hand_preds[:, :, (self.n_patch_latents+n_obs_tokens):]
                obs_hand_preds = obs_hand_preds[:, :-self.fwd_pred_next_n]
        
        prediction = {
            'obs_preds': obs_preds, # (b, l, n_patches, p*p*3)
            'obs_targets': obs_targets, # (b, l, n_patches, p*p*3)
            'obs_hand_preds': obs_hand_preds, # (b, l, n_patches, p*p*3)
            'obs_hand_targets': obs_hand_targets, # (b, l, n_patches, p*p*3)
            'obs_loss_mask': obs_loss_mask,
            'arm_action_preds': arm_action_preds, # (b, l, act_dim - 1)
            'arm_action_targets': action[:, :, :, :self.arm_dim], # (b, l, chunk_size, arm_dim)
            'gripper_action_preds': gripper_action_preds, # (b, l, 1)
            'gripper_action_targets': action[:, :, :, -self.gripper_dim:], # b, l, chunk_size, gripper_dim)
            'action_loss_mask': action_mask,
            'mu': mu, 
            'logvar': logvar,
        }
        return prediction

