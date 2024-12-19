
该项目在 [Medical Domain Adaptive Neural Networks](https://github.com/shafieelab/Medical-Domain-Adaptive-Neural-Networks)代码的基础上做了一些修改




## System Requirements
- Linux (Tested on Ubuntu 18.04.05)
- NVIDIA GPU (Tested on Nvidia GeForce GTX 1080 Ti x 4 on local workstations, and Nvidia V100 GPUs on Cloud)
- Python (Tested with v3.6)

## Python Requirements
- Pytorch 1.4.0
- PyYAML 5.3.1
- scikit-image 0.14.0
- scikit-learn 0.20.0
- SciPy 1.1.0
- opencv-python 4.2.0.34
- Matplotlib 3.0.0
- NumPy 1.15.2
- aim 

## Dataset
### Download
.txt files are lists for source and target domains 

The Embryo are available online [here](https://osf.io/3kc2d/)  Once they are downloaded and extracted into your data directory, create .TXT files with filepaths and numeric annotation with space delimited.

The data used for training and testing are suggested to be organized as follows:

```bash
DATA_ROOT_DIR/
└── DATA_SET_NAME
    ├── DISTRIBUTION_SET_NAME
            ├── CLASS_NUMBER
            .   ├── file_name.png
            .   └── file_name.png
            .   . 
            ├── CLASS_NUMBER
            .   ├── file_name.png
            .   └── file_name.png
    └── DISTRIBUTION_SET_NAME
            ├── CLASS_NUMBER
            .   ├── file_name.png
            .   └── file_name.png
            .   . 
            ├── CLASS_NUMBER
            .   ├── file_name.png
            .   └── file_name.png
```

 
 
## Training

You can train MD-nets as follows
```python
python -u experiments/python_scripts/train_md_nets.py --mode train \  
              --seed seed 
              --num_iterations 100000 --patience 5000 --test_interval 500 --snapshot_interval 1000 \  
              --dset dataset_name --s_dset source_name --t_dset target_name \  
              --lr "0.0001"  --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256
              --s_dset_txt "source_training.txt" 
              --sv_dset_txt "source_validation.txt" \  
              --t_dset_txt "target.txt" \  
              --no_of_classes 5 \
              --output_dir "experiments" \
              --gpu_id 1 \
              --arch Xception\  
              --crop_size 224 --image_size 256
              --target_labelled true  --trade_off 1.0 \  
              --trained_model_path ""
```
##### For MD-nets (Nos) as follows
```python
python -u experiments/python_scripts/train_md_nets_nos.py --mode train \  
              --seed seed 
              --num_iterations 10000 --patience 5000 --test_interval 500 --snapshot_interval 1000 \  
              --dset embryo --s_dset source_name --t_dset target_name \  
              --lr "0.0001"  --batch_size 32 --optimizer SGD --use_bottleneck true --batch_size_test 256
              --tv_dset_txt "target_validation.txt" \  
              --t_dset_txt "target.txt" \  
              --no_of_classes 31 \
              --output_dir "experiments" \
              --gpu_id 0 \
              --arch ResNet50\  
              --crop_size 224 --image_size 256
              --source_model_path "model.pth.tar"
              --trade_off_cls 0.3 
```

### Automated execution
To run all the experiments reported in the paper
```
./experiments/scripts/run_DATA_SET_NAME_experiments.sh 
```

## License  

© [Shafiee Lab](https://shafieelab.bwh.harvard.edu/) - This code is made available under the GNU GPLv3 License and is available for non-commercial academic purposes.

## Contact

If you have any questions, please contact us via hshafiee[at]bwh.harvard.edu.
