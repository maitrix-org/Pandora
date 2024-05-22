import copy
from transformers import PretrainedConfig, Blip2QFormerConfig, CLIPVisionConfig, CLIPTextConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

class WorldModelConfig(PretrainedConfig):
    model_type = "world_model"
    is_composition = True

    def __init__(
            self, 
            qformer_config=None, 
            video_model_config=None,
            diffusion_text_encoder_config=None,
            video_model_name_or_path="Chat-UniVi/Chat-UniVi",
            diffusion_model_name_or_path="stabilityai/stable-diffusion-2-base",
            freeze_video_model=True,
            use_diffusion_text_encoder=True,
            freeze_diffusion_text_encoder=True,
            freeze_diffusion_model=True,
            initializer_range=0.02,
            initializer_factor=1.0,
            use_image_prefix=False,
            image_prefix_length=10,
            train_diffusion_cross_attn=False,
            do_alignment=False,
            use_image_callbacks=True,
            use_image_tokenizer=False,
            image_vocab_size=1024,
            debug_mode=False,
            freeze_diffusion_qformer=False,
            train_cfg_ratio=0.,
            use_controlnet=False,
            use_instructpix2pix=False,
            use_flash_attn=True,
            **kwargs):
        super().__init__(**kwargs)

        if qformer_config is None:
            qformer_config = {}

        if video_model_config is None:
            video_model_config = {}

        if diffusion_text_encoder_config is None:
            diffusion_text_encoder_config = {}

        self.diffusion_model_name_or_path = diffusion_model_name_or_path


        self.qformer_config = Blip2QFormerConfig(**qformer_config)
        self.video_model_config = CONFIG_MAPPING["ChatUniVi"](**video_model_config)

        self.use_diffusion_text_encoder = use_diffusion_text_encoder
        
        self.diffusion_text_encoder_config = CLIPTextConfig(**diffusion_text_encoder_config)
            
        self.do_alignment = do_alignment
        if do_alignment:
            if not use_diffusion_text_encoder:
                raise ValueError("do_alignment requires use_diffusion_text_encoder to be True")

        self.tie_word_embeddings = self.video_model_config.tie_word_embeddings
        self.is_encoder_decoder = self.video_model_config.is_encoder_decoder


        diffsuion_qformer_config = copy.deepcopy(self.qformer_config)
        diffsuion_qformer_config.encoder_hidden_size = diffsuion_qformer_config.hidden_size
        self.diffusion_qformer_config = diffsuion_qformer_config
        if use_diffusion_text_encoder:
            self.diffusion_proj_out_dim = self.diffusion_text_encoder_config.hidden_size
        else:
            self.diffusion_proj_out_dim = PretrainedConfig.get_config_dict(diffusion_model_name_or_path, subfolder="unet")[0]["cross_attention_dim"]

        self.use_decoder_only_language_model = True

        self.video_model_name_or_path = video_model_name_or_path
        self.freeze_video_model = freeze_video_model

        self.diffusion_model_name_or_path = diffusion_model_name_or_path
        self.freeze_diffusion_model = freeze_diffusion_model

        self.freeze_diffusion_text_encoder = freeze_diffusion_text_encoder

        self.vocab_size = self.video_model_config.vocab_size

        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        
        self.use_image_prefix = use_image_prefix
        self.image_prefix_length = image_prefix_length
        self.train_diffusion_cross_attn = train_diffusion_cross_attn
        
        self.use_diffusion_text_encoder = use_diffusion_text_encoder
        self.use_image_callbacks = use_image_callbacks
        
        self.use_image_tokenizer = use_image_tokenizer
        self.image_vocab_size = image_vocab_size
        
        self.debug_mode = debug_mode
        
        self.train_cfg_ratio = train_cfg_ratio
        
        self.use_controlnet = use_controlnet
        self.use_instructpix2pix = use_instructpix2pix
        self.freeze_diffusion_qformer = freeze_diffusion_qformer


        self.use_flash_attn = use_flash_attn

    def reset_training_args(
        self,
        freeze_video_model=None,
        freeze_diffusion_model=None,
        freeze_diffusion_text_encoder=None,
        use_diffusion_text_encoder=None,
        train_diffusion_cross_attn=None,
        do_alignment=None,
        train_cfg_ratio=None,
        use_controlnet=None,
        use_instructpix2pix=None,
        freeze_diffusion_qformer=None,
        use_dynamicrafter=None,
        dynamicrafter_ckpt=None,
        dynamicrafter=None,
        use_flash_attn=None,
    ):
        if freeze_video_model is not None:
            self.freeze_video_model = freeze_video_model
        if freeze_diffusion_model is not None:
            self.freeze_diffusion_model = freeze_diffusion_model
        if freeze_diffusion_text_encoder is not None:
            self.freeze_diffusion_text_encoder = freeze_diffusion_text_encoder
        if use_diffusion_text_encoder is not None:
            self.use_diffusion_text_encoder = use_diffusion_text_encoder
        if train_diffusion_cross_attn is not None:
            self.train_diffusion_cross_attn = train_diffusion_cross_attn
        if train_cfg_ratio is not None:
            self.train_cfg_ratio = train_cfg_ratio
        if use_controlnet is not None:
            self.use_controlnet = use_controlnet
        if use_instructpix2pix is not None:
            self.use_instructpix2pix = use_instructpix2pix
        if freeze_diffusion_qformer is not None:
            self.freeze_diffusion_qformer = freeze_diffusion_qformer
        if use_dynamicrafter is not None:
            self.use_dynamicrafter = use_dynamicrafter
        if dynamicrafter_ckpt is not None:
            self.dynamicrafter_ckpt = dynamicrafter_ckpt
        if dynamicrafter is not None:
            self.dynamicrafter = dynamicrafter
        if self.do_alignment and not do_alignment:
            self.diffusion_model_hf_initialized = True
            self.do_alignment = do_alignment
            
        if use_flash_attn is not None:
            self.use_flash_attn = use_flash_attn

    @classmethod
    def from_sub_configs(
        cls, 
        qformer_config: PretrainedConfig,
        video_model_config: PretrainedConfig,
        diffusion_text_encoder_config: PretrainedConfig = None, 
        **kwargs
    ):
        return cls(
            qformer_config=qformer_config.to_dict(),
            video_model_config=video_model_config.to_dict(),
            diffusion_text_encoder_config=diffusion_text_encoder_config.to_dict(),
            **kwargs,
        )
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["qformer_config"] = self.qformer_config.to_dict()
        output["video_model_config"] = self.video_model_config.to_dict()
        if hasattr(self, "diffusion_text_encoder_config") and self.diffusion_text_encoder_config is not None:
            output["diffusion_text_encoder_config"] = self.diffusion_text_encoder_config.to_dict()
        output["model_type"] = self.__class__.model_type
        del output["diffusion_qformer_config"]
        return output
