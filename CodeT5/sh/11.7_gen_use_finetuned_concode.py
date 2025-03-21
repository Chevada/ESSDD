import torch
from transformers import RobertaTokenizer,T5ForConditionalGeneration
# 指定模型和标记器
tokenizer = RobertaTokenizer.from_pretrained("./Salesforce/codet5-base")
model = T5ForConditionalGeneration.from_pretrained("./Salesforce/codet5-base")

model.load_state_dict(torch.load("./saved_models/gen_class/codet5_base_all_lr3_bs16_src128_trg512_pat3_e30/checkpoint-best-ppl/pytorch_model.bin"))

# 输入自然语言文本
input_text = "Generate a Python class.  This is a class allows to add words to a list and find the longest word in a given sentence by comparing the words with the ones in the word list." \
             "class_name: LongestWord " \
             "method_ __init__: Initialize a list of word." \
             "method_ add_word:append the input word into self.word_list" \
             "method_ find_longest_word:Remove punctuation marks and split a sentence into a list of word. Find the longest splited word that is in the self.word_list."


# input_text = '[method] calculate the area of a circle.'

# 使用模型生成代码
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output_ids = model.generate(input_ids, max_length=1024, num_beams=4, length_penalty=2.0, early_stopping=True)
# 解码生成的代码
generated_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_code)