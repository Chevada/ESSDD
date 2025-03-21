import os
import json
import requests
import numpy as np
from tqdm import tqdm
from time import sleep
# from your_eval_utils import _bleu, calc_code_bleu  # 替换为你的评估工具
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from openai import OpenAI

def build_code_prompt(prompt):
    # 结构化模板
    structured_template = f"""
你是一个专业的Python代码生成器，请严格按照以下约束生成类代码：
1. 输入格式：[描述][className][Method列表][Attribute列表]
2. 输出要求：
   - 类名必须完全匹配className
   - 方法必须包含所有Method项，顺序保持一致，实现描述的类代码功能
   - 属性必须包含所有Attribute项
   - 不要包含任何注释或文档字符串
   - 不要使用代码块标记

当前需求：
{prompt}
"""
    return [
        {
            "role": "system",
            "content": "你是一个严格遵循规范的Python代码生成器，只会输出符合要求的类代码，不包含任何额外内容"
        },
        {
            "role": "user",
            "content": structured_template
        }
    ]

# 配置参数 (根据实际需求修改)
CONFIG = {
    # API配置
    "api_key": "xxx",
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "model_name": "deepseek-coder-33b-instruct",
    "temperature": 0.3,
    "max_tokens": 2048,

    # 路径配置
    # "input_file": "test_set.json",
    "input_file": "xxxx/test.json",
    "output_dir": "evaluation_results",
    "result_files": {
        "pred": "test_output.txt",
        "gold": "test_gold.txt",
        "src": "test_src.txt"
    },

    # 评估参数
    "task_type": "gen_class",
    "lang": "python"
}


def generate_code(prompt: str) -> str:
    """调用DeepSeek API生成代码"""
    client = OpenAI(api_key=CONFIG["api_key"], base_url="https://api.deepseek.com")

    messages = build_code_prompt(prompt)

    for _ in range(3):  # 重试机制
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",  # 使用 DeepSeek-V3 模型
                # model="deepseek-reasoner",  # 使用 DeepSeek-V3 模型
                messages=messages,
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            sleep(2)

    return ""


def format_code(text: str) -> str:
    """统一代码格式（保持与原评估一致）"""
    return text.replace('\n', '\\n').strip()


def run_full_pipeline():
    # 初始化输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    pred_path = os.path.join(CONFIG["output_dir"], CONFIG["result_files"]["pred"])
    gold_path = os.path.join(CONFIG["output_dir"], CONFIG["result_files"]["gold"])
    src_path = os.path.join(CONFIG["output_dir"], CONFIG["result_files"]["src"])

    # 加载测试数据
    dataset = []
    with open(CONFIG["input_file"],encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            dataset.append(data)

    count_path = os.path.join(CONFIG["output_dir"], 'count.txt')
    count = 0
    # 生成代码并保存结果
    dev_accs = []
    with open(pred_path, 'w',encoding="utf-8") as f_pred, \
            open(gold_path, 'w',encoding="utf-8") as f_gold, \
            open(src_path, 'w',encoding="utf-8") as f_src, \
            open(count_path, 'w',encoding="utf-8") as f_count:

        for item in tqdm(dataset, desc="Generating & Evaluating"):
            # 生成代码
            generated_code = generate_code(item["input"])
            # print(generated_code)

            if generate_code:
                # 格式处理（与原评估逻辑完全一致）
                formatted_pred = format_code(generated_code)
                formatted_gold = format_code(item["label"])
                formatted_src = format_code(item["input"])

                # 写入文件
                f_pred.write(formatted_pred + '\n')
                f_gold.write(formatted_gold + '\n')
                f_src.write(formatted_src + '\n')

                count += 1

                # 计算EM
                dev_accs.append(formatted_pred == formatted_gold)
        f_count.write(f'成功生成的样本数：{count}')

    # 计算评估指标
    bleu = round(_bleu(gold_path, pred_path), 2)
    codebleu = calc_code_bleu.get_codebleu(gold_path, pred_path, CONFIG["lang"])

    # 输出结果
    results = {
        "em": np.mean(dev_accs) * 100,
        "bleu": bleu
    }
    if codebleu:
        results["codebleu"] = codebleu * 100

    print("\nFinal Evaluation Results:")
    # for k, v in results.items():
    #     print(f"{k.upper():<10}: {v:.2f}")
    for key in sorted(results.keys()):
        print("  %s = %s", key, str(round(results[key], 4)))


if __name__ == "__main__":
    run_full_pipeline()