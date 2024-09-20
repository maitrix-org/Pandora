import torch
from torch import nn
import transformers
from transformers.utils import logging
from transformers import PreTrainedModel, Blip2QFormerModel, AutoModelForCausalLM
from transformers import  CLIPTextConfig, CLIPTextModel
from transformers.modeling_outputs import  BaseModelOutputWithPooling

from transformers.models.clip.modeling_clip import CLIPTextTransformer
import sys
from einops import rearrange
from configuration import WorldModelConfig
from typing import Optional, Tuple, Union, List, Dict


from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, CLIPTokenizer

from ChatUniVi.constants import DEFAULT_IMAGE_TOKEN
from ChatUniVi.model import ChatUniViLlamaForCausalLM, ChatUniViConfig

sys.path.append('./DynamiCrafter')
from DynamiCrafter.scripts.evaluation.inference import load_model_checkpoint, instantiate_from_config
from DynamiCrafter.lvdm.models.samplers.ddim import DDIMSampler
from DynamiCrafter.lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from omegaconf import OmegaConf
from einops import repeat
from transformers import logging
logging.set_verbosity_error()
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
_make_causal_mask = AttentionMaskConverter._make_causal_mask

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
IMAGE_PREFIX_TOKEN = "[IMG_P]"


logger = logging.get_logger(__name__)

def freeze_sub_models(function):
    def wrapper(*args, **kwargs):
        model = function(*args, **kwargs)
        if model.config.freeze_video_model:
            for param in model.video_model.parameters():
                param.requires_grad = False

        if model.config.use_diffusion_text_encoder and model.config.freeze_diffusion_text_encoder:
            for param in model.diffusion_text_encoder.parameters():
                param.requires_grad = False
        if model.config.use_image_callbacks:
            for param in model.diffusion_original_text_encoder.parameters():
                param.requires_grad = False
        if model.config.freeze_diffusion_qformer:
            for param in model.diffusion_qformer.parameters():
                param.requires_grad = False
            for param in model.diffusion_qformer_proj.parameters():
                param.requires_grad = False
            model.diffusion_query_tokens.requires_grad = False

            for param in model.diffusion_proj.parameters():
                param.requires_grad = False

        return model
    return wrapper



class WorldModel(PreTrainedModel):
    config_class = WorldModelConfig
    sub_models = ['video_model']
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)

        if config.use_image_prefix:
            self.image_prefix = nn.Linear(config.video_model_config.hidden_size, config.image_prefix_length, bias=False)

        if config.use_flash_attn:
            video_model_config = self._check_and_enable_flash_attn_2(config=config.video_model_config)
        else:
            video_model_config = config.video_model_config
            video_model_config._flash_attn_2_enabled = False

        if config.use_image_tokenizer:
            self.image_embeddings = nn.Embedding(config.image_vocab_size, config.video_model_config.hidden_size)
        self.diffusion_qformer_proj = nn.Linear(config.video_model_config.hidden_size, config.diffusion_qformer_config.hidden_size)
        self.diffusion_qformer = Blip2QFormerModel(config.diffusion_qformer_config)
        
        self.diffusion_query_tokens = nn.Parameter(torch.zeros(config.diffusion_text_encoder_config.max_position_embeddings, config.diffusion_qformer_config.hidden_size))
        
        self.diffusion_proj = nn.Linear(config.diffusion_qformer_config.hidden_size, config.diffusion_proj_out_dim)

        if config.use_image_callbacks:
            self.diffusion_original_text_encoder = CLIPTextModel.from_pretrained(config.diffusion_model_name_or_path, subfolder="text_encoder")
            self.diffusion_tokenizer = CLIPTokenizer.from_pretrained(config.diffusion_model_name_or_path,subfolder='tokenizer')
        if config.use_diffusion_text_encoder:
            self.diffusion_text_encoder = CLIPTextEmbeddingModel(config.diffusion_text_encoder_config)

        self.post_init()
        
        self.video_model = AutoModelForCausalLM.from_pretrained(config.video_model_name_or_path, config=video_model_config)
        for module in self.video_model.modules():
            module._is_hf_initialized = True

        if not config.do_alignment:
            model_config = OmegaConf.load(config.dynamicrafter)
            model_config = model_config.pop("model", OmegaConf.create())
            model_config['params']['unet_config']['params']['use_checkpoint'] = False
            self.diffusion_model = instantiate_from_config(model_config)
            self.diffusion_model.perframe_ae = True
            # load_model_checkpoint(self.diffusion_model, config.dynamicrafter_ckpt)
            for module in self.diffusion_model.modules():
                module._is_hf_initialized = True

        if config.use_image_tokenizer:
            self.image_embeddings.weight.data.normal_(mean=0.0, std=0.5)



    @classmethod
    @freeze_sub_models
    def from_pretrained(cls, *args, **kwargs):
        return super(WorldModel, cls).from_pretrained(*args, **kwargs)


    def get_diffusion_conditioning(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        diffusion_tgt_mask: Optional[torch.LongTensor] = None,
    ):
        # print(f'11 normal {pixel_values.shape=}', flush=True)
        # Copy and modify from ChatUniVi forward function ------------------------------------------------------------------------------------------------------------------
        video_model = self.video_model
        output_attentions = output_attentions if output_attentions is not None else video_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else video_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else video_model.config.use_return_dict

        # Use labels to keep track of the location of image prefix tokens since image features will change the length of the sequence
        image_prefix_token_id = self.video_model.config.vocab_size + 1 # ugly hardcode here
        labels = input_ids.clone()
        input_ids[input_ids.eq(image_prefix_token_id)] = 0
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = video_model.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, None, labels, pixel_values)
        
        # print('11 normal', flush=True)
        if self.config.use_image_prefix:
            bs, seq_len = labels.shape
            labels = labels.reshape(-1)
            image_prefix_mask = labels.eq(image_prefix_token_id)
            inputs_embeds = inputs_embeds.reshape(bs * seq_len, -1)
            
 
            image_num = image_prefix_mask.sum().item() / self.config.image_prefix_length
            assert int(image_num) == image_num
            image_prefix_embeddings = self.image_prefix.weight.repeat(int(image_num), 1)
            
            inputs_embeds[image_prefix_mask] = image_prefix_embeddings
            inputs_embeds = inputs_embeds.reshape(bs, seq_len, -1)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        # print(f'12 normal {inputs_embeds.shape=}', flush=True)
        outputs = video_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # print(f'13 normal {outputs[0].shape}', flush=True)

        output_hidden_states = outputs[0]
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
        output_hidden_states = output_hidden_states.reshape(bs * seq_len, -1)
        image_outputs_embeds = output_hidden_states[image_prefix_mask][diffusion_tgt_mask]
        diffusion_loss = None
        img_feat_num = self.config.image_prefix_length # if self.config.use_image_prefix else self.config.num_query_tokens
        diffusion_conditioning = image_outputs_embeds.view(-1, img_feat_num, self.config.video_model_config.hidden_size)
        diffusion_conditioning = self.diffusion_qformer_proj(diffusion_conditioning)
        
        diffusion_query_tokens = self.diffusion_query_tokens.expand(diffusion_conditioning.shape[0], -1, -1)
        diffusion_conditioning = self.diffusion_qformer(
            query_embeds=diffusion_query_tokens,
            encoder_hidden_states=diffusion_conditioning,
        )[0]
        
        diffusion_conditioning = self.diffusion_proj(diffusion_conditioning)
        return diffusion_conditioning


    @staticmethod
    def get_latent_z(model, videos):
        b, c, t, h, w = videos.shape
        x = rearrange(videos, 'b c t h w -> (b t) c h w')
        z = model.encode_first_stage(x)
        z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
        if t == 1:
            zero_pad = repeat(torch.zeros_like(z), 'b c t h w -> b c (repeat t) h w', repeat=3)
            z = torch.cat([z, zero_pad], dim=2)
        z = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=4)
        return z

    def image_guided_synthesis(self, diffusion_conditioning, videos, diffusion_cond_image, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                               unconditional_guidance_scale=1.0, cfg_img=None, fs=None, multiple_cond_cfg=False, loop=False, gfi=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
        ddim_sampler = DDIMSampler(self.diffusion_model) if not multiple_cond_cfg else DDIMSampler_multicond(self.diffusion_model)
        batch_size = noise_shape[0]
        fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=self.diffusion_model.device)

        # img = videos[:,:,0] #bchw
        img = diffusion_cond_image
        img_emb = self.diffusion_model.embedder(img) ## blc
        img_emb = self.diffusion_model.image_proj_model(img_emb)

        # cond_emb = self.diffusion_model.get_learned_conditioning(prompts)
        cond_emb = diffusion_conditioning
        cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
        if self.diffusion_model.model.conditioning_key == 'hybrid':
            z = self.get_latent_z(self.diffusion_model, videos) # b c t h w
            img_cat_cond = z
            cond["c_concat"] = [img_cat_cond] # b c 1 h w
        
        if unconditional_guidance_scale != 1.0:
            if self.diffusion_model.uncond_type == "empty_seq":
                prompts = batch_size * [""]
                uc_emb = self.diffusion_model.get_learned_conditioning(prompts)
            elif self.diffusion_model.uncond_type == "zero_embed":
                uc_emb = torch.zeros_like(cond_emb)
            uc_img_emb = self.diffusion_model.embedder(torch.zeros_like(img)) ## b l c
            uc_img_emb = self.diffusion_model.image_proj_model(uc_img_emb)
            uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
            if self.diffusion_model.model.conditioning_key == 'hybrid':
                uc["c_concat"] = [img_cat_cond]
        else:
            uc = None

        ## we need one more unconditioning image=yes, text=""
        if multiple_cond_cfg and cfg_img != 1.0:
            uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
            if self.diffusion_model.model.conditioning_key == 'hybrid':
                uc_2["c_concat"] = [img_cat_cond]
            kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
        else:
            kwargs.update({"unconditional_conditioning_img_nonetext": None})

        z0 = None
        cond_mask = None

        batch_variants = []
        for _ in range(n_samples):

            if z0 is not None:
                cond_z0 = z0.clone()
                kwargs.update({"clean_cond": True})
            else:
                cond_z0 = None
            if ddim_sampler is not None:

                samples, _ = ddim_sampler.sample(S=ddim_steps,
                                                 conditioning=cond,
                                                 batch_size=batch_size,
                                                 shape=noise_shape[1:],
                                                 verbose=True,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta,
                                                 cfg_img=cfg_img, 
                                                 mask=cond_mask,
                                                 x0=cond_z0,
                                                 fs=fs,
                                                 timestep_spacing=timestep_spacing,
                                                 guidance_rescale=guidance_rescale,
                                                 precision=diffusion_conditioning.dtype,
                                                 **kwargs
                                                 )

            ## reconstruct from latent to pixel space
            batch_images = self.diffusion_model.decode_first_stage(samples)
            batch_variants.append(batch_images)
        ## variants, batch, c, t, h, w
        batch_variants = torch.stack(batch_variants)
        return batch_variants.permute(1, 0, 2, 3, 4, 5)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: torch.FloatTensor = None,
        diffusion_pixel_values: torch.FloatTensor = None,
        diffusion_cond_image: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        **generate_kwargs, # max_new_tokens, guidance_scale
    ):
        assert input_ids.size(0) == 1, "Currently only support batch size 1"
        past_key_values = None
        output_sequence = input_ids
        gen_images = []
        
        img_feat_num = self.config.image_prefix_length if self.config.use_image_prefix else self.config.num_query_tokens

        assert input_ids[0][-1] == tokenizer.image_prefix_token_id
            
        diffusion_conditioning = self.get_diffusion_conditioning(input_ids, pixel_values, attention_mask, True, None, None)
        diffusion_conditioning = diffusion_conditioning[-1:] # Only generate last video

        h, w = diffusion_pixel_values.shape[-2:]
        samples = self.image_guided_synthesis(diffusion_conditioning=diffusion_conditioning,
                                              videos=diffusion_pixel_values[None, ...],
                                              diffusion_cond_image=diffusion_cond_image,
                                              noise_shape=[1, 4, self.diffusion_model.temporal_length, h//8, w//8],
                                              **generate_kwargs)
        return samples
        

class CLIPTextEmbeddingTransformer(CLIPTextTransformer):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            # raise ValueError("You have to specify input_ids")

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        else:
            assert inputs_embeds is not None

            input_shape = inputs_embeds.size()[:-1]
            hidden_states = inputs_embeds

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)


        if not return_dict:
            return (last_hidden_state) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPTextEmbeddingModel(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CLIPTextEmbeddingTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
