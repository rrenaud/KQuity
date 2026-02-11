#!/usr/bin/env python3
"""Merge prediction files + chapter JSON → augmented chapter JSON for the viewer.

Converts game-relative timestamps to video-relative timestamps and writes
model_timelines into each chapter.

Usage:
    python assemble_comparison.py \
        --chapters ~/kq_stream_highlights/chapters/league_nights/sf-2026-01-06.json \
        --predictions predictions/baseline.json predictions/experiment.json \
        --output ~/kq_stream_highlights/chapters/league_nights/sf-2026-01-06.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description='Merge prediction files into chapter JSON for the viewer')
    parser.add_argument('--chapters', required=True,
                        help='Path to input chapter JSON file')
    parser.add_argument('--predictions', required=True, nargs='+',
                        help='Prediction JSON files to merge')
    parser.add_argument('--output', '-o', required=True,
                        help='Output chapter JSON file path')
    args = parser.parse_args()

    # Load chapter data
    with open(args.chapters) as f:
        chapter_data = json.load(f)

    # Load prediction files
    prediction_sets = []
    for pred_path in args.predictions:
        with open(pred_path) as f:
            prediction_sets.append(json.load(f))
        print(f"Loaded predictions: {prediction_sets[-1]['name']} "
              f"({len(prediction_sets[-1]['games'])} games)")

    # Augment each chapter with model_timelines
    augmented = 0
    for ch in chapter_data['chapters']:
        game_id = str(ch['game_id'])
        # video_time = game_relative_t + chapter['start_time'] + 1
        # because chapter['start_time'] = max(0, game_start_video_seconds - 1)
        video_offset = ch['start_time'] + 1

        model_timelines = {}
        for pred_set in prediction_sets:
            game_preds = pred_set['games'].get(game_id)
            if game_preds is None:
                continue
            # Convert game-relative → video-relative timestamps
            timeline = [{'t': round(pt['t'] + video_offset, 2), 'p': pt['p']}
                        for pt in game_preds]
            model_timelines[pred_set['name']] = timeline

        if model_timelines:
            ch['model_timelines'] = model_timelines
            augmented += 1

    print(f"Augmented {augmented}/{len(chapter_data['chapters'])} chapters "
          f"with {len(prediction_sets)} model timelines")

    with open(args.output, 'w') as f:
        json.dump(chapter_data, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
