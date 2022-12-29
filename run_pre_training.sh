nohup python -m torch.distributed.launch --nproc_per_node 4 --nnodes=1 --master_port=23455 run_pre_training.py \
--output_dir spanmask_sbo \
--model_name_or_path hfl/chinese-bert-wwm-ext \
--do_train \
--save_steps 200 \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--fp16 \
--warmup_ratio 0.1 \
--learning_rate 1e-4 \
--num_train_epochs 32 \
--overwrite_output_dir \
--dataloader_num_workers 16 \
--max_seq_length 128 \
--train_dir pretrain \
--weight_decay 0.01 \
--late_mlm \
--use_sbo \
> run.log 2>&1 &