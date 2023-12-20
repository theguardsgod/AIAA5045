cd /home/zhuzhengjie/root/CODE/AIAA5045

set -e
set -x
export PYTHONPATH='./src'

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}

CUDA_VISIBLE_DEVICES=2 python -u src/trainer.py \
    --experiment_index=$experiment_index \
    --cudas=0 \
    --resume=/home/zhuzhengjie/root/CODE/AIAA5045/SimCLR/runs/Dec20_22-07-24_jhcpu4/checkpoint_0100.pth.tar \
    --n_epochs=800 \
    --batch_size=160 \
    --server=zzj \
    --eval_frequency=5 \
    --backbone=dense121 \
    --learning_rate=1e-4 \
    --optimizer=Adam \
    --initialization=default \
    --num_classes=7 \
    --num_worker=4 \
    --input_channel=3 \
    --seed 32 \
    --loss_fn CE \
    2>&1 | tee $log_file

