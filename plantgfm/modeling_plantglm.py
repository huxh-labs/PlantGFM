# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .configuration_plantglm import PlantGLMConfig
from transformers import PreTrainedModel
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutput, SequenceClassifierOutput, BaseModelOutputWithNoAttention


def fftconv(u, k, D):
    """
    We apply a convolution through the fourier domain (from the Convolution Theorem)

    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k.to(torch.float32), n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=torch.float32), n=fft_size)

    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class HyenaSin(nn.Module):
    """The Sin activation function for the Hyena Filter function."""
    def __init__(self, config):
        super().__init__()
        self.freq = nn.Parameter(config.activation_freq * torch.ones(1, config.filter_order)) if config.train_freq else config.activation_freq * torch.ones(1, config.filter_order)

    def forward(self, x):
        return torch.sin(self.freq * x)


class HyenaPositionalEmbedding(nn.Module):
    def __init__(self, config):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = config.max_seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # 1, L, 1

        if config.emb_dim > 1:
            bands = (config.emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, self.seq_len - 1, self.seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / self.seq_len # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]

        z = torch.cat([t, torch.cos(-f * w), torch.sin(-f * w)], dim=-1)

        self.register_buffer("z", z)
        self.register_buffer("t", t)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class HyenaExponentialModulation(nn.Module):
    """The window function applied to the output of the (MLP) filter function."""
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulate: bool=True,
        shift: float = 0.05,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register_buffer("deltas", deltas)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    def __init__(
            self,
            config,
            **kwargs
        ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()

        self.d_model = config.d_model * (config.hyena_order - 1)
        self.use_bias = config.use_bias
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(config.hyena_filter_dropout)

        act = HyenaSin(config)
        self.emb_dim = config.emb_dim
        assert self.emb_dim % 2 != 0 and self.emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = config.max_seq_len

        self.pos_emb = HyenaPositionalEmbedding(config)

        self.implicit_filter = nn.Sequential(
            nn.Linear(self.emb_dim, config.filter_order),
            act,
        )
        for i in range(config.num_inner_mlps):
            self.implicit_filter.append(nn.Linear(config.filter_order, config.filter_order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(config.filter_order, config.d_model, bias=False))

        self.modulation = HyenaExponentialModulation(config.d_model)

        self.normalized = False

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z.to(dtype=self.implicit_filter[0].weight.dtype))
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        y = fftconv(x, k, bias)
        return y


class HyenaOperator(nn.Module):
    def __init__(
            self,
            config,
            **filter_args,
        ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()

        self.d_model = config.d_model
        self.l_max = config.max_seq_len
        self.order = config.hyena_order
        inner_width = config.d_model * (self.order + 1)
        self.dropout = nn.Dropout(config.hyena_dropout)
        self.in_proj = nn.Linear(self.d_model, inner_width)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            config.short_filter_order,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(config)

    def forward(self, u):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u).transpose(1, 2)

        uc = self.short_filter(u)[...,:l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = k.transpose(0, 1).reshape(self.order - 1, self.d_model, l_filter)
        bias = self.filter_fn.bias.reshape(self.order - 1, self.d_model)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = (v * x[0]).transpose(1, 2)

        y = self.out_proj(y)
        return y

class PlantGLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        From Llama RMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)

class PlantGLMSwiGLU(nn.Module):

    def __init__(self, config):
        """
        From Llama SwiGLU
        """
        super().__init__()
        in_features = config.d_model
        hidden_features = config.d_inner
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, in_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x):
        y = F.silu(self.gate_proj(x)) * self.up_proj(x)
        y = self.down_proj(y)

        return y

class PlantGLMBlock(nn.Module):

    def __init__(self, config):
        """
        Adapted from Llama Block, replace the Masked Multi-Head Attention (MHA) to Hyena Operator
        """
        
        super().__init__()

        self.input_layernorm = PlantGLMRMSNorm(hidden_size=config.d_model, eps=config.rms_norm_epsilon)
        self.mixer = HyenaOperator(config)
        self.post_attention_layernorm = PlantGLMRMSNorm(hidden_size=config.d_model, eps=config.rms_norm_epsilon)
        self.mlp = PlantGLMSwiGLU(config)

    def forward(self, hidden_states):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states.to(dtype=self.input_layernorm.weight.dtype))
        hidden_states = self.mixer(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states.to(dtype=self.post_attention_layernorm.weight.dtype))
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states


class PlantGLMEmbeddings(nn.Module):

    def __init__(self, config, padding_idx=None):
        """
            If max_position_embeddings <= 0, there's no position embeddings
            If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
                the project up to embed_dim
        """
        super().__init__()
        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (vocab_size % config.pad_vocab_size_multiple)
        self.word_embeddings = nn.Embedding(vocab_size, config.d_model, padding_idx=padding_idx)

    def forward(self, input_ids):
        """
            input_ids: (batch, seqlen)
        """
        embeddings = self.word_embeddings(input_ids)
        return embeddings

class PlantGLMBackbone(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        # note max_position_embeddings is 0 for Hyena, and therefore isn't used
        self.embeddings = PlantGLMEmbeddings(config)
        self.dropout = nn.Dropout(config.embed_dropout)

        self.layers = nn.ModuleList([PlantGLMBlock(config) for _ in range(config.n_layer)])

        self.rn_f = PlantGLMRMSNorm(hidden_size=config.d_model, eps=config.rms_norm_epsilon)
        self.gradient_checkpointing = False

    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=False):
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(layer.__call__, hidden_states)
            else:
                hidden_states = layer(hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.rn_f(hidden_states.to(dtype=self.rn_f.weight.dtype))
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states


class PlantGLMPreTrainedModel(PreTrainedModel):
    config_class = PlantGLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PlantGLMBlock"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_missing = [r"freq"] 

    def _init_weights(self, module, initializer_range=0.02):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)


class PlantGLMModel(PlantGLMPreTrainedModel):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.backbone = PlantGLMBackbone(config)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=None, return_dict=None):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, all_hidden_states = self.backbone(input_ids, inputs_embeds=inputs_embeds, output_hidden_states=output_hidden_states)
        if return_dict:
            return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states,
                                                  hidden_states=all_hidden_states if output_hidden_states else None)
        elif output_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states


class PlantGLMForCausalLM(PlantGLMPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.glm = PlantGLMModel(config)
        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (vocab_size % config.pad_vocab_size_multiple)
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.glm.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.glm.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.glm = decoder

    def get_decoder(self):
        return self.glm

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.glm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
    
    def prepare_inputs_for_generation(
            self, 
            input_ids: torch.LongTensor = None, 
            past=None, 
            **kwargs
        ):
        if past:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids}


class PlantGLMForSequenceClassification(PlantGLMPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.glm = PlantGLMModel(config)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.glm.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.glm.backbone.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.glm(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
        )