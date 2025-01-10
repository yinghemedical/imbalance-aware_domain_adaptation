# [Addressing Domain Shift via Imbalance-Aware Domain Adaptation in Embryo Development Assessment](https://arxiv.org/abs/2501.04958)

**Authors**  
Lei Li, Xinglin Zhang, Jun Liang, and Tao Chen

---

## Table of Contents
1. [Introduction](#introduction)  
2. [System Requirements](#system-requirements)  
3. [Python Requirements](#python-requirements)  
4. [Dataset](#dataset)  
   - [Download](#download)  
5. [Training](#training)  
   - [Baseline](#baseline)  
   - [Proposed](#proposed)  
   - [Automated Execution](#automated-execution)  
6. [License](#license)  
7. [Contact](#contact)  
8. [Acknowledgements](#acknowledgements)

---

## Introduction
This repository provides code for our work on **“Addressing Domain Shift via Imbalance-Aware Domain Adaptation in Embryo Development Assessment.”** We explore methods to handle domain shift under class imbalance in embryo developmental stage classification tasks.

---

## System Requirements
- **Linux** (Tested on Ubuntu 18.04.05)  
- **NVIDIA GPU** (Tested on Nvidia GeForce GTX 1080 Ti x 4 locally, and Nvidia V100 GPUs on Cloud)  
- **Python** (Tested with v3.6)

---

## Python Requirements
- **PyTorch** 1.4.0  
- **PyYAML** 5.3.1  
- **scikit-image** 0.14.0  
- **scikit-learn** 0.20.0  
- **SciPy** 1.1.0  
- **opencv-python** 4.2.0.34  
- **Matplotlib** 3.0.0  
- **NumPy** 1.15.2  
- **aim** (optional, for experiment tracking)

You can install these via:
```bash
pip install -r requirements.txt
```
*(It is recommended to use a virtual environment.)*

---

## Dataset

### Download
- The `.txt` files in this repository provide file path lists for source and target domains.
- The **Embryo** dataset is available online [here](https://osf.io/3kc2d/). Once downloaded and extracted into your data directory, create `.txt` files containing file paths and numeric annotations (space-delimited).

**Suggested data organization**:
```bash
DATA_ROOT_DIR/
└── DATA_SET_NAME
    ├── DISTRIBUTION_SET_NAME
        ├── CLASS_NUMBER
        │   ├── file_name.png
        │   └── file_name.png
        .
        .
        ├── CLASS_NUMBER
        │   ├── file_name.png
        │   └── file_name.png
    └── DISTRIBUTION_SET_NAME
        ├── CLASS_NUMBER
        │   ├── file_name.png
        │   └── file_name.png
        .
        .
        ├── CLASS_NUMBER
        │   ├── file_name.png
        │   └── file_name.png
```

---

## Training

### Baseline
Train the baseline model using:
```bash
python -u experiments/python_scripts/train_md_nets.py --mode train \
                                --seed $seed --num_iterations 50000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                                --dset embryo --s_dset ed4 --t_dset ed4 --lr 0.001 \
                                --lambda_reg 0.001 --lambda_adv 0.001 \
                                --s_dset_txt "data/embryo/ed4/ed4_source_same_domain.txt" \
                                --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
                                --t_dset_txt "data/embryo/ed4/ed4_target_same_domain.txt" --loss_mode default \
                                --no_of_classes 5 --output_dir "experiments" --gpu_id 0 --arch ResNet50 \
                                --crop_size 224 --image_size 256
```

### Proposed
Train the **proposed** model using:
```bash
python -u experiments/python_scripts/train_md_nets.py --mode train \
                                --seed $seed --num_iterations 50000 --patience 2000 --test_interval 50 --snapshot_interval 1000 \
                                --dset embryo --s_dset ed4 --t_dset ed4 --lr 0.001 \
                                --lambda_reg 0.001 --lambda_adv 0.001 \
                                --s_dset_txt "data/embryo/ed4/ed4_source_same_domain.txt" \
                                --sv_dset_txt "data/embryo/ed4/ed4_validation.txt" \
                                --t_dset_txt "data/embryo/ed4/ed4_target_same_domain.txt" --loss_mode proposed \
                                --no_of_classes 5 --output_dir "experiments" --gpu_id 0 --arch ResNet50 \
                                --crop_size 224 --image_size 256
```

### Automated Execution
To replicate all experiments reported in the paper, run:
```bash
./experiments/scripts/run_embryo_experiments.sh
```

---

## License
This code is released under the **GNU GPLv3 License** and is intended for non-commercial academic purposes. For more details, please see the [LICENSE](LICENSE) file.

---

## Contact
For any inquiries or questions:
- **Lei Li**: [lilei@di.ku.dk](mailto:lilei@di.ku.dk)  


---

## References
You can cite this work using the following BibTeX entry:

@misc{li2025addressingdomainshiftimbalanceaware,
      title={Addressing Domain Shift via Imbalance-Aware Domain Adaptation in Embryo Development Assessment}, 
      author={Lei Li and Xinglin Zhang and Jun Liang and Tao Chen},
      year={2025},
      eprint={2501.04958},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.04958}, 
}

---

## Acknowledgements
This project is built upon the code from [Medical Domain Adaptive Neural Networks](https://github.com/shafieelab/Medical-Domain-Adaptive-Neural-Networks). We thank the original authors for making their code publicly available, and we have made some modifications based on their work.
