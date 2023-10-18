# python -W ignore train_synapse.py -m multiway-TransCASCADE-d192-d0-p3-concat --cuda 1,4 --use-wandb
# python -W ignore train_synapse.py -m multiway-TransCASCADE-d192-d0-p3-add --cuda 1,4 --use-wandb

# python -W ignore train_polyp.py --cuda 0 --use-wandb
# python -W ignore train_synapse.py -m PVT-CASCADE --cuda 0 --use-wandb
# python -W ignore train_ACDC.py --cuda 0 --use-wandb

python -W ignore train_synapse.py -m TransCASCADE --cuda 1 --use-wandb
python -W ignore train_synapse.py -m TransCASCADE-pt --cuda 1 --use-wandb