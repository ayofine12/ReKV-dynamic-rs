import torch
from logzero import logger


class Abstract_ReKV:
    processor = None
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.n_local = n_local
        self.topk = topk
        self.chunk_size = chunk_size
        self.last_retrieval_logits = None

    def clear_cache(self):
        self.kv_cache = None
        self.last_retrieval_logits = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def capture_retrieval_logits(self):
        layers = []
        if self.kv_cache is None:
            self.last_retrieval_logits = {"retrieval_available": False, "layers": []}
            return self.last_retrieval_logits

        for layer_idx, layer_kv in enumerate(self.kv_cache):
            similarity = getattr(layer_kv, "similarity", None)
            if similarity is None:
                logits = None
            else:
                logits = similarity.detach().float().cpu()

            retrieved_block_indices = getattr(layer_kv, "retrieved_block_indices", None)
            actual_topk = None
            if retrieved_block_indices:
                actual_topk = len(retrieved_block_indices[0])

            layers.append({
                "layer_idx": layer_idx,
                "topk": int(getattr(layer_kv, "topk", self.topk)),
                "actual_topk": actual_topk,
                "chunk_size": int(getattr(layer_kv, "chunk_size", self.chunk_size)),
                "block_size": int(getattr(layer_kv, "block_size", self.n_frame_tokens)),
                "num_global_block": int(getattr(layer_kv, "num_global_block", 0)),
                "retrieved_block_indices": retrieved_block_indices,
                "score_logits": logits,
                "retrieval_policy": getattr(layer_kv, "retrieve_policy", "fixed"),
                "dynamic_alpha": getattr(layer_kv, "dynamic_alpha", None),
                "dynamic_normalize": getattr(layer_kv, "dynamic_normalize", None),
                "dynamic_min_topk": getattr(layer_kv, "dynamic_min_topk", None),
                "dynamic_max_topk": getattr(layer_kv, "dynamic_max_topk", None),
                "selected_topk_per_unit": getattr(layer_kv, "last_selected_topk", None),
                "selected_mass_per_unit": getattr(layer_kv, "last_selected_mass", None),
            })

        self.last_retrieval_logits = {
            "retrieval_available": any(layer["score_logits"] is not None for layer in layers),
            "layers": layers,
        }
        return self.last_retrieval_logits

    def get_last_retrieval_logits(self):
        return self.last_retrieval_logits

    def set_dynamic_retrieval_alpha(self, alpha):
        alpha = float(alpha)
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f'dynamic alpha must be in (0, 1], got {alpha}.')
        if isinstance(self.topk, dict):
            self.topk['alpha'] = alpha
        if self.kv_cache is None:
            return
        for layer_kv in self.kv_cache:
            if not hasattr(layer_kv, 'set_dynamic_retrieval_alpha'):
                raise AttributeError('KV cache layer does not support dynamic alpha updates.')
            layer_kv.set_dynamic_retrieval_alpha(alpha)

    def set_fixed_retrieve_size(self, retrieve_size):
        retrieve_size = int(retrieve_size)
        if retrieve_size <= 0:
            raise ValueError(f'retrieve_size must be positive, got {retrieve_size}.')
        if retrieve_size % self.chunk_size != 0:
            raise ValueError(
                f'retrieve_size={retrieve_size} must be divisible by chunk_size={self.chunk_size}.'
            )
        self.topk = retrieve_size
        if self.kv_cache is None:
            return
        for layer_kv in self.kv_cache:
            if not hasattr(layer_kv, 'set_fixed_retrieve_size'):
                raise AttributeError('KV cache layer does not support fixed retrieve-size updates.')
            layer_kv.set_fixed_retrieve_size(retrieve_size)

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    def _get_video_features(self, pixel_values_videos):
        pass

    def _encode_video_chunk(self, video_chunk):
        pixel_values_videos = self.processor.video_processor(video_chunk, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*196, D)
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'

        output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=64):  # video: (Nv, H, W, 3)
        # encode chunk by chunk
        num_frames = video.shape[0]
        num_chunks = num_frames // encode_chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            self._encode_video_chunk(chunk_video)
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            self._encode_video_chunk(remaining_video)
        
        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory
