import os
import json
import requests
import numpy as np
from tqdm import tqdm
from time import sleep
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
    # OpenAI API配置
    "api_key": "xxx",
    "api_base": "https://api.chatanywhere.tech",
    "model_name": "gpt-3.5-turbo-0125",
    # "model_name": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 2048,

    # 路径配置
    "input_file": "xxxx/test.json",
    "output_dir": "gpt-3.5-turbo_evaluation_results",
    # "output_dir": "gpt-4o_evaluation_results",
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
    """调用GPT-4 API生成代码"""
    client = OpenAI(
        api_key=CONFIG["api_key"],
        base_url=CONFIG["api_base"]
    )

    messages = build_code_prompt(prompt)

    # messages = [
    #     {
    #         "role": "system",
    #         "content": "你是一个专业的Python开发助手，请严格按照要求生成类实现代码"
    #     },
    #     {
    #         "role": "user",
    #         "content": f"仅需要生成该类及其方法实现，输出时不包含任何代码块标记，不需要任何注释、文档字符串或多余信息，接下来是生成类的要求，包括类的描述、类名(在className之后)、方法（在Method之后）以及属性（在Attribute之后）：' + {prompt}"
    #     }
    # ]

    for attempt in range(3):  # 重试机制
        try:
            response = client.chat.completions.create(
                model=CONFIG["model_name"],
                messages=messages,
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"],
                # stop=["\n\n"]  # 停止生成标记
            )

            # 清理代码块标记
            raw_code = response.choices[0].message.content.strip()
            # cleaned_code = raw_code.replace("```python", "").replace("```", "").strip()
            return raw_code

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            sleep(2 ** attempt)  # 指数退避

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
    # with open(CONFIG["input_file"]) as f:
    with open(CONFIG["input_file"], encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))

    # 生成代码并保存结果
    success_count = 0
    dev_accs = []
    with open(pred_path, 'w',encoding="utf-8", errors="ignore") as f_pred, \
            open(gold_path, 'w',encoding="utf-8", errors="ignore") as f_gold, \
            open(src_path, 'w',encoding="utf-8", errors="ignore") as f_src:

        for item in tqdm(dataset, desc="Generating Codes"):
            generated_code = generate_code(item["input"])
            # print(generated_code)

            if generated_code:
                formatted_pred = format_code(generated_code)
                formatted_gold = format_code(item["label"])
                formatted_src = format_code(item["input"])

                f_pred.write(formatted_pred + '\n')
                f_gold.write(formatted_gold + '\n')
                f_src.write(formatted_src + '\n')

                dev_accs.append(formatted_pred == formatted_gold)
                success_count += 1

    # 保存成功计数
    with open(os.path.join(CONFIG["output_dir"], 'count.txt'), 'w') as f:
        f.write(f"成功生成样本数: {success_count}")

    # 计算评估指标
    bleu = round(_bleu(gold_path, pred_path), 2)
    codebleu = calc_code_bleu.get_codebleu(gold_path, pred_path, CONFIG["lang"])

    # 输出结果
    results = {
        "em": np.mean(dev_accs) * 100,
        "bleu": bleu,
        "codebleu": codebleu * 100
    }

    print("\nFinal Evaluation Results:")
    for key in sorted(results.keys()):
        print(f"  {key} = {results[key]:.2f}")


if __name__ == "__main__":
    run_full_pipeline()