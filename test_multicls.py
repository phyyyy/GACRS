import argparse
import torch
import torchvision
from tqdm import tqdm
from generator import GeneratorResnet
from classifier_models.res_models import resnet50, resnet152
from ml_dataset import CocoClsDataset, VOCClsDataset
from torch.utils.data import DataLoader

from surrogate_model_utils import *


def get_ml_model(model_type, num_classes):
    if model_type == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif model_type == 'res152':
        model = resnet152(pretrained=True, num_classes=num_classes, use_conv_fc=True)
    elif model_type == 'dense169':
        model = torchvision.models.densenet169(pretrained=True)
        model.classifier = torch.nn.Linear(1664, num_classes)
    else:
        raise Exception('Please check the model_type')
    
    return model

def get_checkpoint_path(args, attack_model_type):
    path = 'classifier_models/{0}_checkpoints/{1}/best_model.pth'.format(args.data_name, attack_model_type)
    
    return path


def get_predict(out):
    out = out.sigmoid()
    max_pred, _ = out.max(dim=1, keepdim=True)
    positive_thresh = max_pred * (1/2)
    predict = (out > positive_thresh)

    return predict


def hamming_score(y_true, y_pred):
    return (
        (y_true & y_pred).sum(axis=1) / (y_true | y_pred).sum(axis=1)
    ).mean()

def evaluate_ml(args, eps, num_classes, dataloader, gen_state_dict, device):
    all_results = []
    for attack_model_type in ['vgg16', 'vgg19', 'res50', 'res152', 'dense169', 'dense121']:
        print('Attack model type: {} pretrained on {}'.format(attack_model_type, args.data_name))
        # Load attack model
        model = get_ml_model(attack_model_type, num_classes)
        model_checkpoint = get_checkpoint_path(args, attack_model_type)
        model = model.to(device)
        model.load_state_dict(torch.load(model_checkpoint))
        model = model.eval()
        
        # Load trained generator
        netG = GeneratorResnet() 
        netG.load_state_dict(gen_state_dict)
        netG = netG.to(device).eval()

        # Initialize lists for predictions and targets
        total_adv_predict, total_clean_predict, total_target = [], [], []

        for img, label in tqdm(dataloader):
            img, label = img.to(device), label.to(device)
            adv = netG(img)
        
            adv = torch.min(torch.max(adv, img - eps), img + eps)
            adv = torch.clamp(adv, 0.0, 1.0)

            with torch.no_grad():
                adv_out = model(normalize(adv))
                clean_out = model(normalize(img))

            adv_predict = get_predict(adv_out).long()
            clean_predict = get_predict(clean_out).long()
            label = label.type_as(adv_predict)

            total_adv_predict.append(adv_predict)
            total_clean_predict.append(clean_predict)
            total_target.append(label)

            del img, adv, label, clean_predict, adv_predict
        
        total_adv_predict = torch.cat(total_adv_predict, 0).cpu().detach().numpy()
        total_clean_predict = torch.cat(total_clean_predict, 0).cpu().detach().numpy()
        total_target = torch.cat(total_target, 0).cpu().detach().numpy()

        # Compute hamming scores
        adv_hamming_score = hamming_score(y_true=total_target, y_pred=total_adv_predict) 
        clean_hamming_score = hamming_score(y_true=total_target, y_pred=total_clean_predict) 

        print('adv HS (%):', adv_hamming_score*100, ', clean HS (%):', clean_hamming_score*100)

        # Store results for this attack model
        result = {
            'attack_model_type': attack_model_type,
            'adv_hamming_score': adv_hamming_score,
            'clean_hamming_score': clean_hamming_score
        }
        all_results.append(result)
    # After the loop, print the overall results
    print("\nFinal results for all models:")
    for result in all_results:
        print('Model: {}, adv HS (%): {:.2f}, clean HS (%): {:.2f}'.format(
            result['attack_model_type'], result['adv_hamming_score']*100, result['clean_hamming_score']*100))

    return all_results


if __name__ == "__main__":
    # options
    parser = argparse.ArgumentParser(description='Multilabel-Classifier Attacker Evaluation')
    parser.add_argument('--bs', type=int, default=16, help='Number of trainig samples/batch')
    parser.add_argument('--loss_type', default='contrastive', help='contrastive, triplet')
    parser.add_argument('--data_name', default='coco', help='coco, voc')
    parser.add_argument('--eps', type=int, default=10, help='Perturbation budget')
    parser.add_argument('--gen_path', type=str, help='Path to trained generator')
    args = parser.parse_args()
    print(args)
    
    # dataloader
    if args.data_name == 'coco':
        test_dataset = CocoClsDataset(root_dir='coco2017/',
                                    ann_file='annotations/instances_val2017.json',
                                    img_dir='val2017',
                                    phase='test')
        num_classes = len(test_dataset.coco.dataset['categories'])
    
    elif args.data_name == 'voc':
        test_dataset = VOCClsDataset(root_dir='VOCdevkit/', # path to dataset. update this
                                    ann_file='VOC2007/ImageSets/Main/test.txt',
                                    img_dir=['VOC2007'],
                                    phase='test')
        num_classes = len(test_dataset.CLASSES)

    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=2)
    eps = args.eps/255.0
    gen_state_dict = torch.load(args.gen_path)

    # GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluate_ml(args, eps, num_classes, test_dataloader, gen_state_dict, device)