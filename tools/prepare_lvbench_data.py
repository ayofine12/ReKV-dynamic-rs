import argparse
import json
import re
from pathlib import Path

OPTION_RE = re.compile(r'(?:^|\n)\(([A-H])\)\s*(.*?)(?=\n\([A-H]\)\s*|\Z)', re.S)


def parse_question_options(text):
    matches = list(OPTION_RE.finditer(text))
    if not matches:
        raise ValueError(f'No options found in question: {text[:120]!r}')
    question = text[:matches[0].start()].strip()
    labels = []
    choices = []
    for match in matches:
        labels.append(match.group(1))
        choices.append(' '.join(match.group(2).strip().split()))
    return question, labels, choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/mnt/ssd1/mwnoh/LVBench/data/video_info.json')
    parser.add_argument('--output', default='data/lvbench/full_mc.json')
    parser.add_argument('--skip_missing_videos', action='store_true')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    records = json.loads(input_path.read_text())

    converted = []
    skipped_videos = 0
    total_questions = 0
    for video in records:
        video_path = Path(video.get('downloaded_video_path', ''))
        if args.skip_missing_videos and not video_path.exists():
            skipped_videos += 1
            continue

        conversations = []
        for qa in video['qa']:
            question, labels, choices = parse_question_options(qa['question'])
            answer_label = qa['answer']
            if answer_label not in labels:
                raise ValueError(f'Answer {answer_label!r} not in labels {labels!r} for uid={qa.get("uid")}')
            answer = choices[labels.index(answer_label)]
            conversations.append({
                'uid': str(qa['uid']),
                'question': question,
                'choices': choices,
                'answer': answer,
                'answer_label': answer_label,
                'question_type': qa.get('question_type', []),
                'time_reference': qa.get('time_reference'),
            })
            total_questions += 1

        converted.append({
            'video_id': video['key'],
            'video_path': str(video_path),
            'video_type': video.get('type'),
            'video_info': video.get('video_info', {}),
            'conversations': conversations,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, indent=2))
    print(f'Wrote {len(converted)} videos / {total_questions} questions to {output_path}')
    if skipped_videos:
        print(f'Skipped {skipped_videos} videos with missing files')


if __name__ == '__main__':
    main()
