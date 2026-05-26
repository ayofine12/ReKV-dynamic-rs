import warnings
import random
import json
import os
import math
import argparse
import csv

import pandas as pd
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import (
    logging,
    LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor,
    VideoLlavaForConditionalGeneration, VideoLlavaProcessor
)
import logzero
from logzero import logger

from model import llava_onevision_rekv, video_llava_rekv, qwen2_5_vl_rekv

try:
    from model import longva_rekv
except ImportError as exc:
    longva_rekv = None
    LONGVA_IMPORT_ERROR = exc
else:
    LONGVA_IMPORT_ERROR = None


MODELS = {
    'llava_ov_0.5b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-0.5b-ov-hf',
    },
    'llava_ov_7b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': '/mnt/models/llava_ov_7b-hf',
    },
    'qwen2_5_vl_7b': {
        'load_func': qwen2_5_vl_rekv.load_model,
        'model_path': '/mnt/models/qwen/Qwen2.5-VL-7B-Instruct',
    },
    'llava_ov_72b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-72b-ov-hf',
    },
    'video_llava_7b': {
        'load_func': video_llava_rekv.load_model,
        'model_class': VideoLlavaForConditionalGeneration,
        'processor_class': VideoLlavaProcessor,
        'model_path': 'model_zoo/Video-LLaVA-7B-hf',
    },
    'longva_7b': {
        'load_func': (longva_rekv.load_model if longva_rekv is not None else None),
        'model_path': 'model_zoo/LongVA-7B',
    },
}


class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None,
                 retrieve_size=64, chunk_size=1,
                 layer_retrieve_sizes=None,
                 dynamic_retrieve_alpha=None,
                 dynamic_retrieve_min_size=None,
                 dynamic_retrieve_max_size=None,
                 dynamic_retrieve_normalize='zscore_softmax',
                 dynamic_retrieve_alphas=None,
                 retrieve_sizes=None,
                 save_retrieval_logits=False) -> None:
        
        self.sample_fps = sample_fps

        self.qa_model = qa_model
        self.qa_processor = qa_processor

        # Retrieval Hyperparams
        assert chunk_size <= retrieve_size, f'chunk_size: {chunk_size}, retrieve_size: {retrieve_size}'
        self.retrieve_size = retrieve_size
        self.chunk_size = chunk_size
        self.layer_retrieve_sizes = layer_retrieve_sizes or ''
        self.dynamic_retrieve_alpha = dynamic_retrieve_alpha
        self.dynamic_retrieve_min_size = dynamic_retrieve_min_size
        self.dynamic_retrieve_max_size = dynamic_retrieve_max_size
        self.dynamic_retrieve_normalize = dynamic_retrieve_normalize
        self.dynamic_retrieve_alphas = dynamic_retrieve_alphas or []
        self.retrieve_sizes = retrieve_sizes or []

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        self.anno = anno
        self.eval_grounding = 'temporal_windows' in anno[0]['conversations'][0]

        self.save_dir = save_dir
        self._active_save_dir = save_dir
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.record = {(self.retrieve_size, self.chunk_size): []}
        self.save_retrieval_logits = save_retrieval_logits
        self.output_csv_path = os.path.join(self.save_dir, f'{self.num_chunks}_{self.chunk_idx}.csv')
        self.stream_result_columns = [
            'video_id', 'uid', 'question', 'choices', 'answer', 'correct_choice',
            'pred_answer', 'pred_choice', 'qa_acc', 'task', 'retrieval_logits_path',
            'retrieve_size', 'chunk_size', 'layer_retrieve_sizes',
            'dynamic_retrieve_alpha', 'dynamic_retrieve_min_size',
            'dynamic_retrieve_max_size', 'dynamic_retrieve_normalize',
        ]

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def load_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
        video = vr.get_batch(frame_idx).asnumpy()
        logger.debug(f'video shape: {video.shape}')
        return video

    def _alpha_label(self, alpha):
        return str(alpha).replace('.', 'p')

    def get_alpha_save_dir(self, alpha):
        return os.path.join(self.save_dir, f'alpha{self._alpha_label(alpha)}-{self.sample_fps}')

    def get_retrieve_size_save_dir(self, retrieve_size):
        return os.path.join(self.save_dir, f'rs{int(retrieve_size)}-{self.sample_fps}')

    def set_active_dynamic_alpha(self, alpha):
        alpha = float(alpha)
        self.dynamic_retrieve_alpha = alpha
        if hasattr(self.qa_model, 'set_dynamic_retrieval_alpha'):
            self.qa_model.set_dynamic_retrieval_alpha(alpha)
        if self.dynamic_retrieve_alphas:
            self._active_save_dir = self.get_alpha_save_dir(alpha)
            os.makedirs(self._active_save_dir, exist_ok=True)
            self.output_csv_path = os.path.join(self._active_save_dir, f'{self.num_chunks}_{self.chunk_idx}.csv')

    def set_active_retrieve_size(self, retrieve_size):
        retrieve_size = int(retrieve_size)
        self.retrieve_size = retrieve_size
        if hasattr(self.qa_model, 'set_fixed_retrieve_size'):
            self.qa_model.set_fixed_retrieve_size(retrieve_size)
        if self.retrieve_sizes:
            self._active_save_dir = self.get_retrieve_size_save_dir(retrieve_size)
            os.makedirs(self._active_save_dir, exist_ok=True)
            self.output_csv_path = os.path.join(self._active_save_dir, f'{self.num_chunks}_{self.chunk_idx}.csv')

    def prepare_output_files(self):
        if self.retrieve_sizes:
            for retrieve_size in self.retrieve_sizes:
                run_dir = self.get_retrieve_size_save_dir(retrieve_size)
                os.makedirs(run_dir, exist_ok=True)
                csv_path = os.path.join(run_dir, f'{self.num_chunks}_{self.chunk_idx}.csv')
                if os.path.exists(csv_path):
                    os.remove(csv_path)
            return
        if self.dynamic_retrieve_alphas:
            for alpha in self.dynamic_retrieve_alphas:
                run_dir = self.get_alpha_save_dir(alpha)
                os.makedirs(run_dir, exist_ok=True)
                csv_path = os.path.join(run_dir, f'{self.num_chunks}_{self.chunk_idx}.csv')
                if os.path.exists(csv_path):
                    os.remove(csv_path)
            return
        if os.path.exists(self.output_csv_path):
            os.remove(self.output_csv_path)
    
    def calc_recall_precision(self, gt_temporal_windows, retrieved_mask):
        total_intersection_length = 0.0
    
        for (start_sec, end_sec) in gt_temporal_windows:
            start = math.floor(start_sec)
            end = math.ceil(end_sec)
            for i in range(start, end):
                if i < len(retrieved_mask) and retrieved_mask[i]:
                    intersection_start = max(start_sec, i)
                    intersection_end = min(end_sec, i + 1)
                    total_intersection_length += intersection_end - intersection_start

        gt_len = sum([end_sec - start_sec for start_sec, end_sec in gt_temporal_windows])
        retrieved_len = sum(retrieved_mask).item()

        recall = total_intersection_length / gt_len if gt_len > 0 else 0
        precision = total_intersection_length / retrieved_len if retrieved_len > 0 else 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return recall, precision, f1
    
    def format_mcqa_prompt(self, question, candidates):
        assert len(question) > 0, f"Q: {question}"

        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."

        return {
            "question": f"{question}",
            "formatted_question": formatted_question,
            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
        }

    def extract_characters_regex(self, s):
        s = s.strip()
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            return s[0]

    def append_record(self, row):
        row['layer_retrieve_sizes'] = self.layer_retrieve_sizes
        row['dynamic_retrieve_alpha'] = self.dynamic_retrieve_alpha
        row['dynamic_retrieve_min_size'] = self.dynamic_retrieve_min_size
        row['dynamic_retrieve_max_size'] = self.dynamic_retrieve_max_size
        row['dynamic_retrieve_normalize'] = self.dynamic_retrieve_normalize
        self.record.setdefault((self.retrieve_size, self.chunk_size), []).append(row)

        stream_row = {col: row.get(col, '') for col in self.stream_result_columns}
        stream_row['retrieve_size'] = self.retrieve_size
        stream_row['chunk_size'] = self.chunk_size
        stream_row['layer_retrieve_sizes'] = self.layer_retrieve_sizes
        stream_row['dynamic_retrieve_alpha'] = self.dynamic_retrieve_alpha
        stream_row['dynamic_retrieve_min_size'] = self.dynamic_retrieve_min_size
        stream_row['dynamic_retrieve_max_size'] = self.dynamic_retrieve_max_size
        stream_row['dynamic_retrieve_normalize'] = self.dynamic_retrieve_normalize

        file_exists = os.path.exists(self.output_csv_path)
        with open(self.output_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.stream_result_columns)
            if not file_exists:
                writer.writeheader()
            writer.writerow(stream_row)

    def video_open_qa(self, question, max_new_tokens=1024):
        pass

    def video_close_qa(self, question, candidates, correct_choice):
        pass

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        pass

    def analyze(self, debug=False):
        self.prepare_output_files()
        video_annos = self.anno[:1] if debug else self.anno
        for video_sample in tqdm(video_annos):
            logger.debug(f'video_id: {video_sample["video_id"]}')
            self.analyze_a_video(video_sample)

        dfs = []
        for (retrieve_size, chunk_size), dict_list in self.record.items():
            df = pd.DataFrame(dict_list)
            df['retrieve_size'] = retrieve_size
            df['chunk_size'] = chunk_size
            if 'layer_retrieve_sizes' not in df.columns:
                df['layer_retrieve_sizes'] = self.layer_retrieve_sizes
            if 'dynamic_retrieve_alpha' not in df.columns:
                df['dynamic_retrieve_alpha'] = self.dynamic_retrieve_alpha
            if 'dynamic_retrieve_min_size' not in df.columns:
                df['dynamic_retrieve_min_size'] = self.dynamic_retrieve_min_size
            if 'dynamic_retrieve_max_size' not in df.columns:
                df['dynamic_retrieve_max_size'] = self.dynamic_retrieve_max_size
            if 'dynamic_retrieve_normalize' not in df.columns:
                df['dynamic_retrieve_normalize'] = self.dynamic_retrieve_normalize
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv', index=False)


def parse_layer_retrieve_sizes(spec, default_retrieve_size):
    if spec is None or str(spec).strip() == '':
        return None

    topk = {'default': int(default_retrieve_size)}
    for item in str(spec).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid layer retrieve size item {item!r}; expected LAYER:RS or START-END:RS."
            )
        layer_part, value_part = item.split(':', 1)
        value = int(value_part)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"retrieve size must be positive, got {value}.")
        if '-' in layer_part:
            start, end = [int(piece) for piece in layer_part.split('-', 1)]
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid layer range {layer_part!r}.")
            for layer_idx in range(start, end + 1):
                topk[layer_idx] = value
        else:
            topk[int(layer_part)] = value
    return topk


def parse_dynamic_retrieve_alphas(spec):
    if spec is None or str(spec).strip() == '':
        return []
    pieces = str(spec).replace(',', ' ').split()
    alphas = [float(piece) for piece in pieces]
    for alpha in alphas:
        if not (0.0 < alpha <= 1.0):
            raise argparse.ArgumentTypeError(f"dynamic alpha must be in (0, 1], got {alpha}.")
    return alphas


def parse_retrieve_sizes(spec):
    if spec is None or str(spec).strip() == '':
        return []
    pieces = str(spec).replace(',', ' ').split()
    values = [int(piece) for piece in pieces]
    for value in values:
        if value <= 0:
            raise argparse.ArgumentTypeError(f"retrieve size must be positive, got {value}.")
    return values


def build_dynamic_retrieve_policy(args):
    if args.dynamic_retrieve_alpha is None:
        return None

    min_size = args.dynamic_retrieve_min_size
    if min_size is None:
        min_size = args.retrieve_chunk_size
    max_size = args.dynamic_retrieve_max_size
    if max_size is None:
        max_size = args.retrieve_size

    min_size = int(min_size)
    max_size = int(max_size)
    alpha = float(args.dynamic_retrieve_alpha)
    if min_size <= 0 or max_size <= 0:
        raise ValueError(f"dynamic retrieve sizes must be positive, got min={min_size}, max={max_size}.")
    if min_size > max_size:
        raise ValueError(f"dynamic min size {min_size} must be <= max size {max_size}.")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"dynamic alpha must be in (0, 1], got {alpha}.")
    if min_size % args.retrieve_chunk_size != 0 or max_size % args.retrieve_chunk_size != 0:
        raise ValueError(
            f"dynamic min/max sizes must be divisible by retrieve_chunk_size={args.retrieve_chunk_size}."
        )

    return {
        'policy': 'mass_threshold',
        'alpha': alpha,
        'min_topk': min_size,
        'max_topk': max_size,
        'normalize': args.dynamic_retrieve_normalize,
    }


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def work(QA_CLASS):
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="llava_ov_7b")
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_sizes", type=str, default=None,
                        help="Run multiple fixed retrieve sizes in one video-encoding pass, e.g. '16 64'.")
    parser.add_argument("--layer_retrieve_sizes", type=str, default=None,
                        help="Optional comma-separated per-layer overrides, e.g. '5:64' or '4-5:64'.")
    parser.add_argument("--dynamic_retrieve_alpha", type=float, default=None,
                        help="Enable cumulative-mass dynamic retrieval with this alpha threshold.")
    parser.add_argument("--dynamic_retrieve_alphas", type=str, default=None,
                        help="Run multiple dynamic alpha values in one video-encoding pass, e.g. '0.2 0.25 0.3 0.35'.")
    parser.add_argument("--dynamic_retrieve_min_size", type=int, default=None,
                        help="Minimum number of blocks retrieved per layer when dynamic retrieval is enabled.")
    parser.add_argument("--dynamic_retrieve_max_size", type=int, default=None,
                        help="Maximum number of blocks retrieved per layer when dynamic retrieval is enabled. Defaults to --retrieve_size.")
    parser.add_argument("--dynamic_retrieve_normalize", type=str, default='zscore_softmax',
                        choices=['zscore_softmax', 'softmax', 'minmax_l1', 'relu_l1'],
                        help="Normalization used before cumulative-mass dynamic retrieval.")
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--save_retrieval_logits", type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)
    layer_retrieve_sizes = parse_layer_retrieve_sizes(args.layer_retrieve_sizes, args.retrieve_size)
    try:
        dynamic_retrieve_alphas = parse_dynamic_retrieve_alphas(args.dynamic_retrieve_alphas)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    try:
        retrieve_sizes = parse_retrieve_sizes(args.retrieve_sizes)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if retrieve_sizes and dynamic_retrieve_alphas:
        parser.error("Use either --retrieve_sizes or --dynamic_retrieve_alphas, not both.")
    if retrieve_sizes and args.dynamic_retrieve_alpha is not None:
        parser.error("Use either --retrieve_sizes or --dynamic_retrieve_alpha, not both.")
    if dynamic_retrieve_alphas:
        if args.dynamic_retrieve_alpha is not None:
            parser.error("Use either --dynamic_retrieve_alpha or --dynamic_retrieve_alphas, not both.")
        args.dynamic_retrieve_alpha = dynamic_retrieve_alphas[0]
    try:
        dynamic_retrieve_policy = build_dynamic_retrieve_policy(args)
    except ValueError as exc:
        parser.error(str(exc))
    if dynamic_retrieve_policy is not None and layer_retrieve_sizes is not None:
        parser.error("--dynamic_retrieve_alpha and --layer_retrieve_sizes are currently mutually exclusive.")
    retrieval_topk = dynamic_retrieve_policy or layer_retrieve_sizes or (max(retrieve_sizes) if retrieve_sizes else args.retrieve_size)

    # fix random seed
    random.seed(2024)
    logger.info('seed: 2024')

    # VideoQA model
    model_path = MODELS[args.model]['model_path']
    load_func = MODELS[args.model]['load_func']
    if load_func is None:
        raise ImportError(f"Failed to import model backend for {args.model}: {LONGVA_IMPORT_ERROR}")
    logger.info(f"Loading VideoQA model: {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        n_local=args.n_local,
        topk=retrieval_topk,
        chunk_size=args.retrieve_chunk_size,
    )

    # Load ground truth file
    anno = json.load(open(args.anno_path))

    retrieve_analyzer = QA_CLASS(
        anno=anno,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        retrieve_size=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        layer_retrieve_sizes=args.layer_retrieve_sizes,
        dynamic_retrieve_alpha=args.dynamic_retrieve_alpha,
        dynamic_retrieve_min_size=(dynamic_retrieve_policy or {}).get('min_topk'),
        dynamic_retrieve_max_size=(dynamic_retrieve_policy or {}).get('max_topk'),
        dynamic_retrieve_normalize=args.dynamic_retrieve_normalize,
        dynamic_retrieve_alphas=dynamic_retrieve_alphas,
        retrieve_sizes=retrieve_sizes,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        save_dir=args.save_dir,
        save_retrieval_logits=args.save_retrieval_logits,
    )

    retrieve_analyzer.analyze(debug=args.debug)
