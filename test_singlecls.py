import torch
import torch.nn as nn
import argparse
import logging
import torchvision.transforms as T
import tqdm
import os,csv
import PIL.Image as Image
from torch.utils.data import Dataset
import timm 
from generator import GeneratorResnet

parser = argparse.ArgumentParser(description='Transfer towards Black-box Domain')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--log-dir', type=str, default='results/result.log')
parser.add_argument('--model-name', type=str, default='res101,vgg16,den121,incv4,wdr50,mobv2,seresn101-324d,covnxtb,vitb,swinb,deit3,beitb,mlpmixb')
parser.add_argument('--batch-size', type=int, default=12)
parser.add_argument('--gen-model-path', type=str, required=True, help='Path to the generator model file')
args = parser.parse_args()

abbr_full = {
    'wres101': 'wide_resnet101_2.tv2_in1k', 'maevitb': 'vit_base_patch16_224.mae',
    'res18':'resnet18.tv_in1k', 'res34':'resnet34.gluon_in1k', 'res101': 'resnet101.tv2_in1k',
    'vgg19':'vgg19.tv_in1k', 'res152':'resnet152.tv_in1k','seresn101-324d':'seresnext101_32x4d.gluon_in1k',
    "incv3":"inception_v3.tf_in1k", "incv4":"inception_v4.tf_in1k", "incresv2":"inception_resnet_v2.tf_in1k",
    "vitb":"vit_base_patch32_clip_224.laion2b_ft_in12k_in1k", "swinb":"swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "vgg16":"vgg16.tv_in1k", "vgg19bn":"vgg19_bn.tv_in1k", "res50":"resnet50.tv_in1k", "seres50":"seresnet50.ra2_in1k",
    "deit3":"deit3_medium_patch16_224.fb_in22k_ft_in1k", "den121":"densenet121.ra_in1k", "mobv2":"mobilenetv2_120d.ra_in1k",
    "beitb":"beit_base_patch16_224.in22k_ft_in22k_in1k", "mlpmixb":"mixer_b16_224.miil_in21k_ft_in1k", "covnxtb": "convnext_base.clip_laion2b_augreg_ft_in12k_in1k",
    "wdr50":"wide_resnet50_2.tv2_in1k", "advincv3":"inception_v3.tf_adv_in1k", "advincres":"inception_resnet_v2.tf_ens_adv_in1k", "den169": "densenet169.tv_in1k","pnanet":"pnasnet5large.tf_in1k"
}

def build_model(model_name):
    timm_root = 'huggingface/hub/pytorch_models'

    model = timm.create_model(abbr_full[model_name], pretrained=False)

    pretrained_path = os.path.join(timm_root, abbr_full[model_name]+'.bin')

    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
    else:
        print(f"Pretrained weights not found for model {model_name} at {pretrained_path}")

    data_config = model.default_cfg

    model = nn.Sequential(T.Normalize(data_config["mean"], 
                                      data_config["std"]), 
                          model)
    model = nn.DataParallel(model)
    model.eval()
    model.cuda()
    return model, data_config

class SubImg(Dataset):
    def __init__(self, transform, data_csv_dir, data_dir):
        self.transform=transform
        self.data_dir = data_dir
        label_csv = open('imagenet_label.csv', 'r')
        label_reader = csv.reader(label_csv)
        label_ls = list(label_reader)
        self.label_ls = label_ls
        label = {} 
        index=0
        for i in label_ls:          
            label[i[0]] = (i[1:],index)
            index+=1
        self.label=label                # Utilize self.label to record all labels
        data_csv = open(data_csv_dir, 'r')
        csvreader = csv.reader(data_csv)
        data_ls = list(csvreader)
        self.imgs = self.img_ls(data_ls)
        size = len(self.imgs)
        print(f'Datasize of ImageNet:{size}')
        
    def img_ls(self, data_ls): 
        total_imgs = [[data_ls[label_ind][i+1]]+[self.label[data_ls[label_ind][0]][1]]+[data_ls[label_ind][0]]\
            for label_ind in range(len(data_ls)) for i in range(len(data_ls[label_ind])-1)]
        return total_imgs
 
    def __getitem__(self, item):
        label = self.imgs[item][1]
        path = os.path.join(self.data_dir, self.imgs[item][2], self.imgs[item][0])
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if not isinstance(self.transform, list):
            imgs = self.transform(img)
        elif isinstance(self.transform, list):
            imgs = [unit(img) for unit in self.transform]
        return imgs, label, path
    

    def __len__(self):
        return len(self.imgs)
    
# Ensure log directory exists
log_dir = os.path.dirname(args.log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Logger Configuration
logging.basicConfig(filename=args.log_dir, format='%(message)s', level=logging.WARNING)
logging.warning(args.log_dir)

def evaluate(model, dataloader, netG):
    n_img, n_correct, n_success = 0, 0, 0
    for img, label, _ in dataloader:
        label, img = label.cuda(), img.cuda()
        advs = netG(img)
        advs = torch.min(torch.max(advs, img - args.eps/255), img + args.eps/255)
        advs = torch.clamp(advs, min=0, max=1)

        with torch.no_grad():
            pred = torch.argmax(model(img), dim=1).view(1,-1)
            advpred = torch.argmax(model(advs), dim=1).view(1,-1)
        n_correct += (label == pred.squeeze(0)).sum().item()
        n_success += ((label == pred.squeeze(0))*(label != advpred.squeeze(0))).sum().item()
        n_img += len(label)
    return round(100. * n_correct / n_img, 2), round(100. * n_success / n_correct, 2)

trans = T.Compose([
        T.Resize(size=(256, 256), interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor()
    ])

dataset = SubImg(transform=trans,
                data_csv_dir='allcls_val_5img_seed101.csv',
                data_dir='ilsvrc2012/val/')
                               
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=args.batch_size, 
                                         num_workers=4, 
                                         pin_memory=True)

# Load Generator Model
netG = GeneratorResnet()
netG.load_state_dict(torch.load(args.gen_model_path))
netG = nn.DataParallel(netG).cuda().eval()

# Evaluate Models
args.model_name = args.model_name.split(",")
with tqdm.tqdm(args.model_name, colour='GREEN', total=len(args.model_name)) as miter:
    for mname in miter:
        model, data_config = build_model(mname)
        accuracy, successrate = evaluate(model, dataloader, netG)
        miter.set_postfix(acc=f'{accuracy}', uasr=f'{successrate}')
        logging.warning("{}: ACC@{}, UASR@{}".format(mname, accuracy, successrate))
