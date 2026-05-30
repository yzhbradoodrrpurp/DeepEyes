#!/usr/bin/env python3
"""Convert this project's VQA parquet into the DeepEyes visual-toolbox interface."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEEPEYES_SYSTEM_PROMPT_V2 = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""

DEEPEYES_USER_PROMPT_V2 = (
    "\nThink first, call **image_zoom_in_tool** if needed, then answer. "
    "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> "
    "(if tools needed)  <answer>...</answer> "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source parquet.")
    parser.add_argument("--output", type=Path, required=True, help="Converted parquet.")
    parser.add_argument("--limit", type=int, default=0, help="Rows to keep. 0 means all rows.")
    parser.add_argument("--env-name", default="visual_toolbox_v2")
    parser.add_argument("--data-source", default="vl_agent")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_question(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("question"):
        return str(extra_info["question"])

    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content.replace("<image>", "").strip()
    return ""


def convert_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    question = extract_question(row)
    user_content = f"<image>\n{question}{DEEPEYES_USER_PROMPT_V2}"
    return [
        {"role": "system", "content": DEEPEYES_SYSTEM_PROMPT_V2},
        {"role": "user", "content": user_content},
    ]


def convert_row(row: dict[str, Any], index: int, env_name: str, data_source: str) -> dict[str, Any]:
    row = copy.deepcopy(row)
    question = extract_question(row)
    original_data_source = row.get("data_source")
    row["prompt"] = convert_prompt(row)
    row["env_name"] = env_name
    row["data_source"] = data_source

    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        extra_info = {}
    extra_info.setdefault("index", index)
    extra_info.setdefault("question", question)
    extra_info.setdefault("original_data_source_before_deepeyes", original_data_source)
    row["extra_info"] = extra_info

    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        reward_model = {}
    reward_model.setdefault("ground_truth", extra_info.get("answer", ""))
    row["reward_model"] = reward_model
    return row


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0.")
    if not args.input.is_file():
        raise SystemExit(f"Input parquet does not exist: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output}. Use --overwrite.")

    table = pq.read_table(args.input)
    rows = table.to_pylist()
    if args.limit:
        rows = rows[: args.limit]

    converted = [
        convert_row(row, index=i, env_name=args.env_name, data_source=args.data_source)
        for i, row in enumerate(rows)
    ]
    if not converted:
        raise SystemExit("No rows selected.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted), args.output)
    print(f"Wrote {len(converted)} rows to {args.output}")


if __name__ == "__main__":
    main()
