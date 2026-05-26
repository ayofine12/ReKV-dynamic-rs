import re
from pathlib import Path

import torch
from logzero import logger

from video_qa.base import BaseVQA, work


class ReKVOfflineVQA(BaseVQA):
    def _safe_path_part(self, value):
        value = '' if value is None else str(value)
        value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')
        return value or 'unknown'

    def save_last_retrieval_logits(self, video_id, sample):
        if not self.save_retrieval_logits:
            return ''
        if not hasattr(self.qa_model, 'get_last_retrieval_logits'):
            return ''

        payload = self.qa_model.get_last_retrieval_logits()
        if not payload or not payload.get('retrieval_available'):
            return ''

        uid = sample.get('uid')
        if uid is None:
            uid = f"q{getattr(self, '_logits_counter', 0):06d}"
        self._logits_counter = getattr(self, '_logits_counter', 0) + 1

        rs_label = f'rs{self.retrieve_size}_cs{self.chunk_size}'
        if getattr(self, 'layer_retrieve_sizes', ''):
            rs_label += f"_layers_{self._safe_path_part(self.layer_retrieve_sizes)}"
        if getattr(self, 'dynamic_retrieve_alpha', None) is not None:
            rs_label += (
                f"_dynmass_a{self.dynamic_retrieve_alpha}"
                f"_min{self.dynamic_retrieve_min_size}"
                f"_max{self.dynamic_retrieve_max_size}"
            )
        if getattr(self, 'relative_retrieve_beta', None) is not None:
            rs_label += (
                f"_relmass_b{self.relative_retrieve_beta}"
                f"_min{self.relative_retrieve_min_size}"
                f"_max{self.relative_retrieve_max_size}"
            )
        out_dir = Path(getattr(self, '_active_save_dir', self.save_dir)) / 'retrieval_logits' / rs_label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self._safe_path_part(video_id)}_{self._safe_path_part(uid)}.pt"

        torch.save({
            'video_id': video_id,
            'uid': None if uid is None else str(uid),
            'question': sample.get('question'),
            'question_type': sample.get('question_type'),
            'time_reference': sample.get('time_reference'),
            'retrieve_size': self.retrieve_size,
            'chunk_size': self.chunk_size,
            'layer_retrieve_sizes': getattr(self, 'layer_retrieve_sizes', ''),
            'dynamic_retrieve_alpha': getattr(self, 'dynamic_retrieve_alpha', None),
            'dynamic_retrieve_min_size': getattr(self, 'dynamic_retrieve_min_size', None),
            'dynamic_retrieve_max_size': getattr(self, 'dynamic_retrieve_max_size', None),
            'dynamic_retrieve_normalize': getattr(self, 'dynamic_retrieve_normalize', None),
            'relative_retrieve_beta': getattr(self, 'relative_retrieve_beta', None),
            'relative_retrieve_min_size': getattr(self, 'relative_retrieve_min_size', None),
            'relative_retrieve_max_size': getattr(self, 'relative_retrieve_max_size', None),
            'relative_retrieve_normalize': getattr(self, 'relative_retrieve_normalize', None),
            'retrieval': payload,
        }, out_path)
        return str(out_path)

    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }

        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens, retrieved_indices=retrieved_indices)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    def _answer_sample(self, video_sample, sample):
        logger.debug(f'sample: {sample}')
        question = sample['question']
        answer = sample['answer']

        # QA
        if 'choices' in sample:  # CloseQA
            choices = sample['choices']
            if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
                answer = choices[0]
            correct_choice = self.choice_letters[choices.index(answer)]
            qa_results = self.video_close_qa(question, choices, correct_choice)
            record = {
                'video_id': video_sample['video_id'],
                'uid': sample.get('uid'),
                'question': question,
                'choices': choices,
                'answer': answer,
                'correct_choice': correct_choice,
                'pred_answer': qa_results['pred_answer'],
                'pred_choice': qa_results['pred_choice'],
                'qa_acc': qa_results['acc'] * 100,
            }
        else:  # OpenQA
            qa_results = self.video_open_qa(question)
            record = {
                'video_id': video_sample['video_id'],
                'uid': sample.get('uid'),
                'question': question,
                'answer': answer,
                'pred_answer': qa_results['pred_answer'],
            }

        if 'question_type' in sample:
            record['task'] = sample['question_type']
        record['retrieval_logits_path'] = self.save_last_retrieval_logits(video_sample['video_id'], sample)
        self.append_record(record)

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # load and preprocess video frames for QA
        video_path = video_sample['video_path']
        video = self.load_video(video_path)
        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self.qa_model.encode_video(video_tensor)

        for sample in video_sample['conversations']:
            if self.retrieve_sizes:
                for retrieve_size in self.retrieve_sizes:
                    self.set_active_retrieve_size(retrieve_size)
                    self._answer_sample(video_sample, sample)
                continue

            if self.relative_retrieve_betas:
                for beta in self.relative_retrieve_betas:
                    self.set_active_relative_beta(beta)
                    self._answer_sample(video_sample, sample)
                continue

            alpha_values = self.dynamic_retrieve_alphas or [None]
            for alpha in alpha_values:
                if alpha is not None:
                    self.set_active_dynamic_alpha(alpha)
                self._answer_sample(video_sample, sample)


if __name__ == "__main__":
    work(ReKVOfflineVQA)
