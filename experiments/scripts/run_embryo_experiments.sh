########################################################################################################################
########################## MD-nets Source to Target Domain Adaption ####################################################
########################################################################################################################

export CUDA_VISIBLE_DEVICES=0
export aim_database_type=MySql
# lambda_adv=0.001
# lambda_reg=0.01
lr=0.001
seed=0
lambda_advs=(0.0001 0.001 0.01 0.1)
lambda_regs=(0.0001 0.001 0.01 0.1)
# lambda_adv=0.001
# lambda_reg=0.001
# loss_modes=(proposed)
# archs=(Xception ResNet50 Inception)
# arch=Xception
# python -u experiments/python_scripts/train_md_nets.py --mode train \
#                   --seed 0 --num_iterations 50000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
#                   --dset embryo  --s_dset ed4 --t_dset ed3  --lr 0.01\
#                   --s_dset_txt "data/embryo/ed4/ed4_source_2-class.txt" \
#                   --sv_dset_txt "data/embryo/ed4/ed4_validation_2-class.txt" \
#                   --t_dset_txt "data/embryo/ed3/ed3_target.txt" \
#                   --no_of_classes 2 --output_dir "experiments"  --gpu_id 0 --arch Xception\
#                             --crop_size 224 --image_size 256
for lambda_adv in "${lambda_advs[@]}"
do
    for lambda_reg in "${lambda_regs[@]}"
    do
        for arch in Xception ResNet50 Inception
        do
            # run embryo same domain settings
            # ED4 to ED4 adaption
            for loss_mode in default proposed
            do
                echo "$arch $loss_mode $lambda_adv $lambda_reg" 
                echo "MedicalDomain-reg_$lambda_reg-adv_$lambda_adv-lr_0.001-seed_0"
                python -u experiments/python_scripts/train_md_nets.py --mode train \
                                --seed $seed --num_iterations 50000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                                --dset embryo  --s_dset ed4 --t_dset ed4 --lr $lr \
                                --lambda_reg $lambda_reg --lambda_adv $lambda_adv \
                                --s_dset_txt "data/embryo/ed4/ed4_source_same_domain.txt" \
                                --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
                                --t_dset_txt "data/embryo/ed4/ed4_target_same_domain.txt" --loss_mode $loss_mode \
                                --no_of_classes 5 --output_dir "experiments" --gpu_id 0 --arch $arch\
                                            --crop_size 224 --image_size 256
                # Embryo cross domain settings
                # ED4 to ED3 adaption
                python -u experiments/python_scripts/train_md_nets.py --mode train \
                                --seed $seed --num_iterations 50000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                                --dset embryo  --s_dset ed4 --t_dset ed3  --lr $lr \
                                --lambda_reg $lambda_reg --lambda_adv $lambda_adv \
                                --s_dset_txt "data/embryo/ed4/ed4_source_2-class.txt" \
                                --sv_dset_txt "data/embryo/ed4/ed4_validation_2-class.txt" \
                                --t_dset_txt "data/embryo/ed3/ed3_target.txt" --loss_mode $loss_mode \
                                --no_of_classes 2 --output_dir "experiments"  --gpu_id 0 --arch $arch\
                                            --crop_size 224 --image_size 256



                # Embryo cross domain settings
                # ED4 to ED2 and ED1 adaption
                for target in ed1 ed2
                do
                    python -u experiments/python_scripts/train_md_nets.py --mode train \
                                --seed $seed --num_iterations 50000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                                --dset embryo --s_dset ed4 --t_dset $target  --lr $lr \
                                --lambda_reg $lambda_reg --lambda_adv $lambda_adv \
                                --s_dset_txt "data/embryo/ed4/ed4_source_same_domain.txt" --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
                                --t_dset_txt "data/embryo/${target}/${target}_target.txt" --loss_mode $loss_mode \
                                --no_of_classes 5 --output_dir "experiments"  --gpu_id 0 --arch $arch\
                                --crop_size 224 --image_size 256 

                done
            done
        done
    done
done