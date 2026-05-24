# OpenCompass OpenAI-compatible model template.
#
# dcu_regression.py will export these environment variables before running
# OpenCompass:
#   OPENCOMPASS_API_BASE
#   OPENCOMPASS_MODEL_NAME
#   OPENCOMPASS_API_KEY
#
# Adjust this file to match the OpenCompass version installed on the DCU test
# host. The exact OpenCompass API-model class names can differ by release.

# import os

# from opencompass.models import OpenAI

# models = [
#     dict(
#         type=OpenAI,
#         path=os.environ["OPENCOMPASS_MODEL_NAME"],
#         key=os.environ.get("OPENCOMPASS_API_KEY", "EMPTY"),
#         openai_api_base=os.environ["OPENCOMPASS_API_BASE"],
#         query_per_second=1,
#         max_out_len=2048,
#         temperature=0,
#         batch_size=1,
#     )
# ]

import os
from mmengine.config import read_base

with read_base():
    # datasets
#    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
#    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
#    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    from opencompass.configs.summarizers.example import summarizer

datasets = sum(
    [v for k, v in locals().items() if k.endswith("_datasets") or k == "datasets"],
    [],
)

# 输出目录（按你习惯改）
work_dir = "/mnt/nmz/0430/397_int8-mtp3-marlin"

from opencompass.models import OpenAISDK

# chat / instruct 形式（和你给的模板一致）
api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
)

models = [
    dict(
        abbr="397_int8-mtp3-marlin",
        type=OpenAISDK,
        # 这里的 path 是 OpenAI 接口里用的 model 字段：
        # - 如果你 server 启了 --served-model-name，就填 served-model-name
        # - 没设置的话，通常填模型路径 basename（DeepSeek-R1-0528-W4A8-V2）
        path=os.environ["OPENCOMPASS_MODEL_NAME"],
        openai_api_base=os.environ["OPENCOMPASS_API_BASE"],
        tokenizer_path=os.environ["OPENCOMPASS_MODEL_NAME"],
        key="EMPTY",
        meta_template=api_meta_template,
        temperature=0,
        query_per_second=64,
        max_out_len=32768,
        max_seq_len=32768,
        pred_postprocessor=dict(
            type="opencompass.utils.text_postprocessors.extract_non_reasoning_content"
        ),
        batch_size=64,
    ),
]
