WORKDIR="xxxxxxxxx"
export PYTHONPATH=$WORKDIR

TASK=${1}
SUB_TASK=${2}
MODEL_TAG=${3}
GPU=${4}
DATA_NUM=${5}
BS=${6}
LR=${7}
SRC_LEN=${8}
TRG_LEN=${9}
PATIENCE=${10}
EPOCH=${11}
WARMUP=${12}
MODEL_DIR=${13}
SUMMARY_DIR=${14}
RES_FN=${15}
LOSS_FN=${16}

if [[ $DATA_NUM == -1 ]]; then
  DATA_TAG='all'
else
  DATA_TAG=$DATA_NUM
  EPOCH=1
fi

if [[ ${TASK} == 'multi_task' ]]; then
  FULL_MODEL_TAG=${MODEL_TAG}_${DATA_TAG}_lr${LR}_s${16}
elif [[ ${TASK} == 'siamese' || ${TASK} == 'sia_gen' ]]; then
  CURRENT_TIME=$(date +"%Y%m%d")
  FULL_MODEL_TAG=${MODEL_TAG}_${DATA_TAG}_use${LOSS_FN}_lr${LR}_bs${BS}_src${SRC_LEN}_trg${TRG_LEN}_pat${PATIENCE}_e${EPOCH}_${CURRENT_TIME}
else
  CURRENT_TIME=$(date +"%Y%m%d_%H%M")
  FULL_MODEL_TAG=${MODEL_TAG}_${DATA_TAG}_lr${LR}_bs${BS}_src${SRC_LEN}_trg${TRG_LEN}_pat${PATIENCE}_e${EPOCH}_${CURRENT_TIME}
fi


if [[ ${SUB_TASK} == none ]]; then
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${FULL_MODEL_TAG}
else
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
fi

CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

if [[ $MODEL_TAG == roberta ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=roberta-base
elif [[ $MODEL_TAG == codebert ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=microsoft/codebert-base
elif [[ $MODEL_TAG == bart_base ]]; then
  MODEL_TYPE=bart
  TOKENIZER=facebook/bart-base
  MODEL_PATH=facebook/bart-base
elif [[ $MODEL_TAG == codet5_small ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-small
  MODEL_PATH=Salesforce/codet5-small
elif [[ $MODEL_TAG == codet5_base ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-base
  MODEL_PATH=Salesforce/codet5-base
elif [[ $MODEL_TAG == codet5_large ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-large
  MODEL_PATH=Salesforce/codet5-large
fi


if [[ ${TASK} == 'multi_task' ]]; then
  RUN_FN=${WORKDIR}/run_multi_gen.py
  MULTI_TASK_AUG='--max_steps '${16}' --save_steps '${17}' --log_steps '${18}
elif [[ ${TASK} == 'clone' ]]; then
  RUN_FN=${WORKDIR}/run_clone.py
elif [[ ${TASK} == 'defect' ]] && [[ ${MODEL_TYPE} == 'roberta' ||  ${MODEL_TYPE} == 'bart' ]]; then
  RUN_FN=${WORKDIR}/run_defect.py
elif [[ ${TASK} == 'siamese' ]]; then
  RUN_FN=${WORKDIR}/run_siamese.py
elif [[ ${TASK} == 'sia_gen' ]]; then
  RUN_FN=${WORKDIR}/run_gen_use_sia_model.py
else
  RUN_FN=${WORKDIR}/run_gen.py
fi
# 实际调用的文件是：${WORKDIR}/run_gen.py，也即"your_CodeT5_path/CodeT5/run_gen.py"


# 此处要去掉--do_test，记得同步到服务器
# 每个epoch对验证集评价bleu指标太耗费时间，此处把--do_eval_bleu参数去掉  --do_eval_bleu
CUDA_VISIBLE_DEVICES=${GPU} \
  python ${RUN_FN}  ${MULTI_TASK_AUG}   \
  --do_train --do_eval --do_test  \
  --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
  --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
  --tokenizer_name=${TOKENIZER}  --model_name_or_path=${MODEL_PATH} --data_dir ${WORKDIR}/data  \
  --cache_path ${CACHE_DIR}  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
  --save_last_checkpoints --always_save_model --res_dir ${RES_DIR} --res_fn ${RES_FN} \
  --train_batch_size ${BS} --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
  2>&1 | tee ${LOG}

#CUDA_VISIBLE_DEVICES=${GPU} \
#python ${RUN_FN}  ${MULTI_TASK_AUG}   \
#--do_test  \
#--task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
#--num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
#--tokenizer_name=${TOKENIZER}  --model_name_or_path=${MODEL_PATH} --data_dir ${WORKDIR}/data  \
#--cache_path ${CACHE_DIR}  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
#--save_last_checkpoints --always_save_model --res_dir ${RES_DIR} --res_fn ${RES_FN} \
#--train_batch_size ${BS} --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
#2>&1 | tee ${LOG}


#CUDA_VISIBLE_DEVICES=${GPU} \
#  python ${RUN_FN}  ${MULTI_TASK_AUG}   \
#  --do_train --do_eval\
#  --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
#  --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
#  --tokenizer_name=${TOKENIZER}  --model_name_or_path=${MODEL_PATH} --data_dir ${WORKDIR}/data  \
#  --cache_path ${CACHE_DIR}  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
#  --save_last_checkpoints --always_save_model --res_dir ${RES_DIR} --res_fn ${RES_FN} \
#  --train_batch_size ${BS} --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
#  2>&1 | tee ${LOG}
