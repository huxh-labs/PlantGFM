from transformers import PretrainedConfig
import json


class PlantGFMConfig(PretrainedConfig):
    model_type = "plantgfm"
    def __init__(
        self,
        vocab_size=24,
        d_model=256,
        d_inner=None,
        use_bias=True,
        train_freq=True,
        max_seq_len=1024,
        emb_dim=3,
        n_layer=12,
        num_inner_mlps=2,
        hyena_order=2,
        short_filter_order=3,
        filter_order=64,
        activation_freq=1,
        embed_dropout=0.1,
        hyena_dropout=0.0,
        hyena_filter_dropout=0.0,
        rms_norm_epsilon=1e-6,
        initializer_range=0.02,
        pad_vocab_size_multiple=8,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        if d_inner is None:
            self.d_inner = 3 * d_model
        else:
            self.d_inner = d_inner
        self.use_bias = use_bias
        self.train_freq = train_freq
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.n_layer = n_layer
        self.hyena_order = hyena_order
        self.filter_order = filter_order
        self.short_filter_order = short_filter_order
        self.activation_freq = activation_freq
        self.num_inner_mlps = num_inner_mlps
        self.embed_dropout = embed_dropout
        self.hyena_dropout = hyena_dropout
        self.hyena_filter_dropout = hyena_filter_dropout
        self.rms_norm_epsilon = rms_norm_epsilon
        self.initializer_range = initializer_range
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        super().__init__(**kwargs)

    @classmethod
    def from_original_config(cls, config_path, **kwargs):
        with open(config_path, "r") as f:
            config = json.load(f)

        vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        d_inner = config["d_inner"]
        max_seq_len = config["layer"]["l_max"]
        emb_dim = config["layer"]["emb_dim"]
        filter_order = config["layer"]["filter_order"]
        if "local_order" in config["layer"]:
            short_filter_order = config["layer"]["local_order"]
        elif "short_filter_order" in config["layer"]:
            short_filter_order = config["layer"]["short_filter_order"]
        else:
            short_filter_order = 3
        n_layer = config["n_layer"]
        activation_freq = config["layer"]["w"]
        embed_dropout = config["embed_dropout"]
        pad_vocab_size_multiple = config["pad_vocab_size_multiple"]
        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            d_inner=d_inner,
            max_seq_len=max_seq_len,
            emb_dim=emb_dim,
            filter_order=filter_order,
            short_filter_order=short_filter_order,
            n_layer=n_layer,
            activation_freq=activation_freq,
            embed_dropout=embed_dropout,
            pad_vocab_size_multiple=pad_vocab_size_multiple,
            tie_word_embeddings=False,
            **kwargs
        )