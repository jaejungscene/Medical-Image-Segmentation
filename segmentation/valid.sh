# python3 valid.py -s btcv_v2 -b 1 -m inception_transunet_d192_p1_f2_d0_add\
#           --cuda 2 -weights btcv_v2_inception_transunet_d192_p1_f2_d0_add_v2

# python3 valid.py -s btcv_v2 -b 1 -m inception_transunet_d192_p2_f2_d2_add_ptres\
#           --cuda 4 -weights btcv_v2_inception_transunet_d192_p2_f2_d2_add_ptres_v0

# python3 valid.py -s btcv_v2 -b 1 -m inception_transunet_d192_p1_f2_d0_add_ptres\
#           --cuda 4 -weights btcv_v2_inception_transunet_d192_p1_f2_d0_add_ptres_v0

python3 valid.py -s btcv_v2 -b 1 -m transunet_d768_pt\
          --cuda 4 -weights btcv_v2_transunet_d768_pt_v0

# python3 valid.py -s isic2018_v1 isic2019_v1 -c 2 --mode test 