import json
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

    def save_last_retrieval_logits(self, video_id, sample, payload=None, label_suffix=''):
        if not self.save_retrieval_logits:
            return ''
        if payload is None:
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
        if label_suffix:
            rs_label += f"_{self._safe_path_part(label_suffix)}"
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
            'confidence_label_suffix': label_suffix,
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

    def _choice_pred_answer(self, pred_choice, candidates):
        if pred_choice in self.choice_letters:
            idx = self.choice_letters.index(pred_choice)
            if idx < len(candidates):
                return f"{pred_choice}) {candidates[idx]}"
        return str(pred_choice)

    def _choice_score_record(self, scores, prefix='choice'):
        entropy_key = f'{prefix}_entropy'
        normalized_entropy_key = f'{prefix}_normalized_entropy'
        if prefix == 'initial':
            entropy_key = 'initial_choice_entropy'
            normalized_entropy_key = 'initial_normalized_choice_entropy'
        return {
            f'{prefix}_top1_prob': scores.get('top1_prob'),
            f'{prefix}_top2_prob': scores.get('top2_prob'),
            f'{prefix}_prob_margin': scores.get('prob_margin'),
            f'{prefix}_logit_margin': scores.get('logit_margin'),
            entropy_key: scores.get('choice_entropy'),
            normalized_entropy_key: scores.get('normalized_choice_entropy'),
        }

    def _choice_score_json_record(self, scores):
        return {
            'choice_logits_json': json.dumps(scores.get('choice_logits', {}), sort_keys=True),
            'choice_logprobs_json': json.dumps(scores.get('choice_logprobs', {}), sort_keys=True),
            'choice_probs_json': json.dumps(scores.get('choice_probs', {}), sort_keys=True),
        }

    def _confidence_value(self, scores):
        metric = self.confidence_fallback_metric
        if metric == 'prob_margin':
            return scores.get('prob_margin')
        if metric == 'logit_margin':
            return scores.get('logit_margin')
        if metric == 'top1_prob':
            return scores.get('top1_prob')
        if metric == 'normalized_choice_entropy':
            return scores.get('normalized_choice_entropy')
        raise ValueError(f'Unknown confidence fallback metric: {metric}')

    def _should_confidence_fallback(self, scores):
        if not self.confidence_fallback_enabled:
            return False, None
        value = self._confidence_value(scores)
        if value is None:
            return False, value
        threshold = float(self.confidence_fallback_threshold)
        if self.confidence_fallback_metric == 'normalized_choice_entropy':
            return float(value) > threshold, value
        return float(value) < threshold, value

    def _multiple_choice_by_logits(self, input_text, candidates, correct_choice, video_id=None, sample=None, retrieved_indices=None):
        if not hasattr(self.qa_model, 'multiple_choice_answering'):
            raise AttributeError('The current QA model does not expose multiple_choice_answering().')

        base_retrieve_size = int(self.retrieve_size)
        scores = self.qa_model.multiple_choice_answering(
            input_text,
            num_choices=len(candidates),
            retrieved_indices=retrieved_indices,
            return_scores=True,
        )
        fallback_triggered, confidence_value = self._should_confidence_fallback(scores)
        initial_payload = None
        if fallback_triggered and hasattr(self.qa_model, 'get_last_retrieval_logits'):
            initial_payload = self.qa_model.get_last_retrieval_logits()

        initial_path = ''
        final_path = ''
        final_scores = scores
        final_retrieve_size = base_retrieve_size

        if fallback_triggered:
            if video_id is not None and sample is not None:
                initial_path = self.save_last_retrieval_logits(
                    video_id, sample, payload=initial_payload, label_suffix='initial'
                )
            fallback_size = int(self.confidence_fallback_retrieve_size)
            self.set_active_retrieve_size(fallback_size)
            try:
                final_scores = self.qa_model.multiple_choice_answering(
                    input_text,
                    num_choices=len(candidates),
                    retrieved_indices=retrieved_indices,
                    return_scores=True,
                )
                final_retrieve_size = fallback_size
                if video_id is not None and sample is not None:
                    final_path = self.save_last_retrieval_logits(video_id, sample, label_suffix='fallback')
            finally:
                self.set_active_retrieve_size(base_retrieve_size)
        elif video_id is not None and sample is not None:
            final_path = self.save_last_retrieval_logits(video_id, sample)

        pred_choice = final_scores['pred_choice']
        result = {
            'pred_answer': self._choice_pred_answer(pred_choice, candidates),
            'pred_choice': pred_choice,
            'acc': float(pred_choice == correct_choice),
            'retrieval_logits_path': final_path,
            'confidence_fallback_triggered': bool(fallback_triggered),
            'confidence_fallback_value': confidence_value,
            'confidence_initial_retrieve_size': base_retrieve_size,
            'confidence_final_retrieve_size': final_retrieve_size,
            'confidence_effective_retrieve_size': (
                base_retrieve_size + final_retrieve_size if fallback_triggered else base_retrieve_size
            ),
            'initial_pred_choice': scores.get('pred_choice'),
            'initial_retrieval_logits_path': initial_path,
        }
        result.update(self._choice_score_record(final_scores, prefix='choice'))
        result.update(self._choice_score_json_record(final_scores))
        result.update(self._choice_score_record(scores, prefix='initial'))
        return result

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None, video_id=None, sample=None):
        input_text = self.format_mcqa_prompt(question, candidates)
        use_choice_logits = self.mc_answer_mode == 'choice_logits' or self.confidence_fallback_enabled
        if use_choice_logits and hasattr(self.qa_model, 'multiple_choice_answering'):
            return self._multiple_choice_by_logits(
                input_text, candidates, correct_choice,
                video_id=video_id, sample=sample, retrieved_indices=retrieved_indices,
            )
        if self.confidence_fallback_enabled:
            raise AttributeError('Confidence fallback requires a model with multiple_choice_answering().')

        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
            'confidence_fallback_triggered': False,
            'confidence_initial_retrieve_size': self.retrieve_size,
            'confidence_final_retrieve_size': self.retrieve_size,
            'confidence_effective_retrieve_size': self.retrieve_size,
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
            qa_results = self.video_close_qa(
                question, choices, correct_choice,
                video_id=video_sample['video_id'], sample=sample,
            )
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
            for key in (
                'choice_top1_prob', 'choice_top2_prob', 'choice_prob_margin',
                'choice_logit_margin', 'choice_entropy', 'choice_normalized_entropy',
                'choice_logits_json', 'choice_logprobs_json', 'choice_probs_json',
                'confidence_fallback_triggered', 'confidence_fallback_value',
                'confidence_initial_retrieve_size', 'confidence_final_retrieve_size',
                'confidence_effective_retrieve_size', 'initial_pred_choice',
                'initial_top1_prob', 'initial_top2_prob', 'initial_prob_margin',
                'initial_logit_margin', 'initial_choice_entropy',
                'initial_normalized_choice_entropy', 'initial_retrieval_logits_path',
                'retrieval_logits_path',
            ):
                if key in qa_results:
                    record[key] = qa_results[key]
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
        if 'retrieval_logits_path' not in record:
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
        if self.confidence_fallback_enabled:
            self.set_active_retrieve_size(self.confidence_base_retrieve_size)

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
