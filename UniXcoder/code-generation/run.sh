#pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html > log.txt 2>&1
#pip install sklearn scipy transformers tqdm > log.txt 2>&1
#CUDA_VISIBLE_DEVICES=15,12,13,14
#lang=java #programming language
#lr=5e-5
#batch_size=32
#accm_steps=1
#beam_size=3
#source_length=512
#target_length=150
#data_dir=../../dataset
#output_dir=saved_models/$lang
#train_file=$data_dir/train.json
#dev_file=$data_dir/dev.json
#epochs=30
#pretrained_model=../../../pretrained-model/UniXcoder-base/
#
#mkdir -p $output_dir
#python run.py \
#--do_train \
#--do_eval \
#--model_name_or_path $pretrained_model \
#--train_filename $train_file \
#--dev_filename $dev_file \
#--tokenizer_name roberta-base \
#--output_dir $output_dir \
#--max_source_length $source_length \
#--max_target_length $target_length \
#--beam_size $beam_size \
#--train_batch_size $batch_size \
#--eval_batch_size $batch_size \
#--learning_rate $lr \
#--gradient_accumulation_steps $accm_steps \
#--num_train_epochs $epochs 2>&1| tee $output_dir/train.log
#
#
#batch_size=64
#dev_file=$data_dir/dev.json
#test_file=$data_dir/test.json
#test_model=$output_dir/checkpoint-best-score/pytorch_model.bin #checkpoint for test
#
#python run.py \
#--do_test \
#--model_name_or_path $pretrained_model \
#--load_model_path $test_model \
#--dev_filename $dev_file \
#--test_filename $test_file \
#--output_dir $output_dir \
#--max_source_length $source_length \
#--max_target_length $target_length \
#--beam_size $beam_size \
#--gradient_accumulation_steps $accm_steps \
#--eval_batch_size $batch_size 2>&1| tee $output_dir/test.log




#run_gen
output_dir=saved_models/gen_class
log_dir=${output_dir}/train.log
python run.py \
--do_train --do_eval --do_test \
--model_name_or_path microsoft/unixcoder-base \
--train_filename dataset/train.json \
--dev_filename dataset/dev.json \
--test_filename dataset/test.json \
--output_dir ${output_dir} \
--max_source_length 64 --max_target_length 380 \
--beam_size 3 --train_batch_size 24 --eval_batch_size 24 \
--learning_rate 5e-5 \
--gradient_accumulation_steps 1 \
--num_train_epochs 40 \
2>&1 | tee $log_dir





