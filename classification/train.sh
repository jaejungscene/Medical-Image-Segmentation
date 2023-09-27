## Single GPU
# python3 train.py -s setting1 setting2 -m model1 model2 -c 0


# # 01
# python3 train.py --epoch 100 -s messidor2_v1 eyepacs_v1 -m densenet121\
#  --metric-names accuracy auroc f1_score precision recall specificity\
#  --cuda 1 --use-wandb

# # 02
# python3 train.py --epoch 100 -s messidor2_v1 eyepacs_v1 -m resnet50\
#  --metric-names accuracy auroc f1_score precision recall specificity\
#  --cuda 1 --use-wandb

# # 03
# python3 train.py --epoch 100 -s messidor2_v1 eyepacs_v1 -m swin_tiny_patch4_window7_224\
#  --metric-names accuracy auroc f1_score precision recall specificity\
#  --cuda 3 --use-wandb

# 04
python3 train.py --epoch 100 -s messidor2_v1 eyepacs_v1 -m swin_small_patch4_window7_224\
 --metric-names accuracy auroc f1_score precision recall specificity\
 --cuda 2 --use-wandb


## Multi GPU
# torchrun --nproc_per_node=2 --master_port=12345 train.py \
# -s setting1 setting2 -m model1 model2 -c 0,1

### Test multi GPU