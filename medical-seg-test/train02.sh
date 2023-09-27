# Inception TransUnet >> {model_name}_d{hidden_size}_p{num_path}_f{pixshuf_factor}_{concat}
python train.py --epoch 160 -s btcv_v2 -m inception_transunet_d192_p3_f2_concat\
 --valid-freq -1 --print-freq 50 --cuda 2 --use-wandb

python train.py --epoch 160 -s btcv_v2 -m inception_transunet_d192_p3_f2_concat\
 --valid-freq -1 --print-freq 50 --cuda 2 --use-wandb

python train.py --epoch 160 -s btcv_v2 -m inception_transunet_d192_p3_f2\
 --valid-freq -1 --print-freq 50 --cuda 2 --use-wandb

python train.py --epoch 160 -s btcv_v2 -m inception_transunet_d192_p3_f2\
 --valid-freq -1 --print-freq 50 --cuda 2 --use-wandb


## Multi GPU
# torchrun --nproc_per_node=2 --master_port=12345 train.py --epoch 100 -s messidor2_v1 eyepacs_v1 -m swin_small_patch4_window7_224\
#  --metric-names accuracy auroc f1_score precision recall specificity -c 1,2\
 
# torchrun --nproc_per_node=2 --master_port=12345 train.py \
# -s setting1 setting2 -m model1 model2 -c 0,1

### Test multi GPU