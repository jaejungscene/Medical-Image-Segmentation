#### TransUnet
# python train.py --epoch 150 -s acdc_2d -m transunet\
#  --valid-freq -1 --print-freq 50 --cuda 1

# python train.py --epoch 150 -s btcv_2d -m transunet\
#  --valid-freq -1 --print-freq 50 --cuda 1 --use-wandb

python train.py --epoch 150 -s btcv_2d -m inception-transunet_d192_p3_d0_f2_concat\
 --valid-freq -1 --print-freq 50 --cuda 1 --use-wandb
python train.py --epoch 150 -s btcv_2d -m inception-transunet_d192_p2_d1_f2_concat\
 --valid-freq -1 --print-freq 50 --cuda 1 --use-wandb
python train.py --epoch 150 -s btcv_2d -m inception-transunet_d192_p2_d2_f2_concat\
 --valid-freq -1 --print-freq 50 --cuda 1 --use-wandb


## Multi GPU
# torchrun --nproc_per_node=2 --master_port=12345 train.py --epoch 100 -s messidor2_v1 eyepacs_v1 -m swin_small_patch4_window7_224\
#  --metric-names accuracy auroc f1_score precision recall specificity -c 1,2\
 
# torchrun --nproc_per_node=2 --master_port=12345 train.py \
# -s setting1 setting2 -m model1 model2 -c 0,1

### Test multi GPU