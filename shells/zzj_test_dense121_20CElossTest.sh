cd /home/zhuzhengjie/root/CODE/AIAA5045

set -e
set -x
export PYTHONPATH='./src'

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}

CUDA_VISIBLE_DEVICES=1 python -u src/test.py \
    --experiment_index=$experiment_index \
    --cudas=0 \
    --n_epochs=800 \
    --batch_size=80 \
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

