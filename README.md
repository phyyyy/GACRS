# Generative Attack in Complex Real-World Scenarios

This repository provides the **official source code** of the paper:

**"Generative Attack in Complex Real-World Scenarios"**  
Hongyu Peng, Gong Cheng, Xuxiang Sun  
Published in *Pattern Recognition*, 2026  
📄 [[Paper on ScienceDirect](https://doi.org/10.1016/j.patcog.2025.111893)]

## 🚀 Usage

1. Download the necessary pretrained models from [here](https://drive.google.com/drive/folders/1pmsNESi4ofKGJw19yPZNHeRx9aGxxNUg?usp=sharing) and place them in ```classifer_models``` folders.
2. Install the packages listed in ```requirements.txt```.
3. To train a perturbation generator, run the following command:
```
python train.py --surr_model_type <surrogate name> --data_name <voc/coco> --train_dir <path to dataset> --eps <l_infty> --batch_size 8 --epochs 20 --save_folder <path to trained models folder> --clip_backbone <clip model type>
```
4. To evaluate a trained perturbation generator on multi-label classification (e.g., VOC/COCO), run the following command:
```
python test_multicls.py --data_name <voc/coco> --gen_path <path to trained generator file (.pth)> 
```
5. To evaluate a trained perturbation generator on single-label classification (e.g., ImageNet), run the following command:
```
python test_singlecls.py --log-dir <path to save the results> --gen_path <path to trained generator file (.pth)> 
```

## 📌 Citation 
If you find this work useful in your research, please cite:

@article{peng2026generative,
  title={Generative attack in complex real-world scenarios},
  author={Peng, Hongyu and Cheng, Gong and Sun, Xuxiang},
  journal={Pattern Recognition},
  volume = {169},
  pages = {111893},
  year = {2026}
}

## 🧠 Acknowledgement
We thank the authors of the following repositories for making their code open-source.  
1. https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack
2. https://github.com/abhishekaich27/GAMA-pytorch

## 📬 Contact
If you have any questions, please contact:
Hongyu Peng: hongyupeng@mail.nwpu.edu.cn
