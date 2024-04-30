python main.py \
    -t \
    --base configs/sd-scalar-flow-finetune-c_concat-256_aml.yaml \
    --gpus 0,1,2,3 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from /mnt/blob/Dynamics/zero123-weights/zero123-xl.ckpt \
