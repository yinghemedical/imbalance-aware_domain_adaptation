########################################################################################################################
########################## MD-nets Source to Target Domain Adaption ####################################################
########################################################################################################################
#初次提交实验 archs=(Xception ResNet50 Inception) 
#          optimizers=(SGD) 
#            max_iterations=50000 
#            loss_modes=(default proposed)  
#            use_bottlenecks=(true)
#补充实验1 archs=(Xception ResNet50 Inception) 
#          optimizers=(SGD) 
#            max_iterations=50000 
#            loss_modes=(default proposed)  
#            use_bottlenecks=(false)
#补充实验2 archs=(Xception ResNet50 Inception) 
#          optimizers=(Adam) 
#            max_iterations=50000 
#            loss_modes=(default proposed)  
#            use_bottlenecks=(true false)

#补充实验3 archs=(Xception ResNet50 Inception) 
#          optimizers=(SGD,Adam) 
#            max_iterations=50000 
#            loss_modes=(CB)  
#            use_bottlenecks=(true,false)

gpu_id=0
export aim_database_type=MySql
# lambda_adv=0.001
# lambda_reg=0.01
lr=0.001
seed=0
lambda_advs=(0.01)
lambda_regs=(0.0001)
#--use_multitask
archs=(Inception)
use_multitask=false
#default (SGD)
optimizers=(SGD)
max_iterations=50000
#default (default proposed)
loss_modes=(default proposed)
#default (false)
use_bottlenecks=(false)
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
#1 ED4-ED3 Inception SDG
python -u experiments/python_scripts/train_md_nets.py --mode train \
                                        --seed $seed --num_iterations $max_iterations --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                                        --dset embryo  --s_dset ed4 --t_dset ed3  --lr $lr \
                                        --lambda_reg 0.0001 --lambda_adv 0.001 \
                                        --s_dset_txt "data/embryo/ed4/ed4_source_2-class.txt" \
                                        --sv_dset_txt "data/embryo/ed4/ed4_validation_2-class.txt" \
                                        --t_dset_txt "data/embryo/ed3/ed3_target.txt" --loss_mode proposed \
                                        --no_of_classes 2 --output_dir "experiments"  --gpu_id $gpu_id --arch Inception\
                                                    --crop_size 224 --image_size 256

python -u experiments/python_scripts/train_md_nets.py --mode train \
                                        --seed $seed --num_iterations $max_iterations --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                                        --dset embryo  --s_dset ed4 --t_dset ed3  --lr $lr \
                                        --lambda_reg 0.0001 --lambda_adv 0.001 \
                                        --s_dset_txt "data/embryo/ed4/ed4_source_2-class.txt" \
                                        --sv_dset_txt "data/embryo/ed4/ed4_validation_2-class.txt" \
                                        --t_dset_txt "data/embryo/ed3/ed3_target.txt" --loss_mode default \
                                        --no_of_classes 2 --output_dir "experiments"  --gpu_id $gpu_id --arch Inception\
                                                    --crop_size 224 --image_size 256


#2 ED4-ED2 ED1 Inception SDG
for target in ed1 ed2
do
    python -u experiments/python_scripts/train_md_nets.py --mode train \
                --seed $seed --num_iterations $max_iterations --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                --dset embryo --s_dset ed4 --t_dset $target  --lr $lr \
                --lambda_reg 0.001 --lambda_adv 0.1 \
                --s_dset_txt "data/embryo/ed4/ed4_source_same_domain.txt" --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
                --t_dset_txt "data/embryo/${target}/${target}_target.txt" --loss_mode default \
                --no_of_classes 5 --output_dir "experiments"  --gpu_id $gpu_id --arch Inception\
                --crop_size 224 --image_size 256 

done

for target in ed1 ed2
do
    python -u experiments/python_scripts/train_md_nets.py --mode train \
                --seed $seed --num_iterations $max_iterations --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                --dset embryo --s_dset ed4 --t_dset $target  --lr $lr \
                --lambda_reg 0.001 --lambda_adv 0.1 \
                --s_dset_txt "data/embryo/ed4/ed4_source_same_domain.txt" --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
                --t_dset_txt "data/embryo/${target}/${target}_target.txt" --loss_mode proposed \
                --no_of_classes 5 --output_dir "experiments"  --gpu_id $gpu_id --arch Inception\
                --crop_size 224 --image_size 256 

done

#3 ED4-ED3 ResNet50 SDG
python -u experiments/python_scripts/train_md_nets.py --mode train \
                                        --seed $seed --num_iterations $max_iterations --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                                        --dset embryo  --s_dset ed4 --t_dset ed3  --lr $lr \
                                        --lambda_reg 0.001 --lambda_adv 0.0001 \
                                        --s_dset_txt "data/embryo/ed4/ed4_source_2-class.txt" \
                                        --sv_dset_txt "data/embryo/ed4/ed4_validation_2-class.txt" \
                                        --t_dset_txt "data/embryo/ed3/ed3_target.txt" --loss_mode proposed \
                                        --no_of_classes 2 --output_dir "experiments"  --gpu_id $gpu_id --arch ResNet50\
                                                    --crop_size 224 --image_size 256

python -u experiments/python_scripts/train_md_nets.py --mode train \
                                        --seed $seed --num_iterations $max_iterations --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                                        --dset embryo  --s_dset ed4 --t_dset ed3  --lr $lr \
                                        --lambda_reg 0.001 --lambda_adv 0.0001 \
                                        --s_dset_txt "data/embryo/ed4/ed4_source_2-class.txt" \
                                        --sv_dset_txt "data/embryo/ed4/ed4_validation_2-class.txt" \
                                        --t_dset_txt "data/embryo/ed3/ed3_target.txt" --loss_mode default \
                                        --no_of_classes 2 --output_dir "experiments"  --gpu_id $gpu_id --arch ResNet50\
                                                    --crop_size 224 --image_size 256

#4 ED4-ED2 ED1 ResNet50 SDG
for target in ed2
do
    python -u experiments/python_scripts/train_md_nets.py --mode train \
                --seed $seed --num_iterations $max_iterations --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                --dset embryo --s_dset ed4 --t_dset $target  --lr $lr \
                --lambda_reg 0.0001 --lambda_adv 0.001 \
                --s_dset_txt "data/embryo/ed4/ed4_source_same_domain.txt" --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
                --t_dset_txt "data/embryo/${target}/${target}_target.txt" --loss_mode default \
                --no_of_classes 5 --output_dir "experiments"  --gpu_id $gpu_id --arch ResNet50\
                --crop_size 224 --image_size 256 

done

for target in ed2
do
    python -u experiments/python_scripts/train_md_nets.py --mode train \
                --seed $seed --num_iterations $max_iterations --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                --dset embryo --s_dset ed4 --t_dset $target  --lr $lr \
                --lambda_reg 0.0001 --lambda_adv 0.001 \
                --s_dset_txt "data/embryo/ed4/ed4_source_same_domain.txt" --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
                --t_dset_txt "data/embryo/${target}/${target}_target.txt" --loss_mode proposed \
                --no_of_classes 5 --output_dir "experiments"  --gpu_id $gpu_id --arch ResNet50\
                --crop_size 224 --image_size 256 

done