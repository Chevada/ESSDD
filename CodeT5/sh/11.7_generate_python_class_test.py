from transformers import RobertaTokenizer, T5ForConditionalGeneration
tokenizer = RobertaTokenizer.from_pretrained('./Salesforce/codet5-base')
model = T5ForConditionalGeneration.from_pretrained('./Salesforce/codet5-base')
text ="Generate a Java class ,this is a class for calculating the area of different shapes, including circle, sphere, cylinder, sector and annulus."
input_ids = tokenizer(text, return_tensors="pt").input_ids
# simply generate one code span
generated_ids = model.generate(input_ids, max_length=500, num_beams=4, no_repeat_ngram_size=2, length_penalty=2.0, early_stopping=True)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))