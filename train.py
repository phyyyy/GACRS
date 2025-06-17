from __future__ import annotations

import argparse
import os
import pprint
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import clip
import pickle
from sklearn.cluster import KMeans

from classifier_models import Vgg16_all_layer, Res152_all_layer, Dense169_all_layer
from generator import GeneratorResnet
from ml_dataset import CocoClsDataset, VOCClsDataset
from losses import ContrastiveLoss
from test_multicls import evaluate_ml
from surrogate_model_utils import normalize, clip_normalize
from utils import choose_input, get_dynamic_p, TRAINING_CFG, build_text_prompts

parser = argparse.ArgumentParser(description="Classifier attack training")

parser.add_argument('--train_dir', default='VOCdevkit/', help='Path for training data')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--eps', type=int, default=10, help='Perturbation budget (in 1/255)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--margin', type=float, default=1.0, help='Contrastive loss margin')
parser.add_argument('--surr_model_type', type=str, default='res152',
                    help='Surrogate model: vgg16 / res152 / dense169')
parser.add_argument('--attack_model_type', type=str, default='vgg16',
                    help='Target model to attack')
parser.add_argument('--clip_backbone', type=str, default='ViT-B/16',
                    help='CLIP backbone')
parser.add_argument('--data_name', default='voc', help='voc or coco')
parser.add_argument('--save_folder', type=str, default='checkpoints/',
                    help='Folder to save checkpoints')
parser.add_argument('--loss_type', type=str, default='contrastive',
                    help='Loss type')

args = parser.parse_args()
pprint.pprint(vars(args), width=120)

# --------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------- #
def setup_seed(seed: int) -> None:
    import numpy as np, random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------- #
# Output folder
# --------------------------------------------------------------------- #
save_root = Path(args.save_folder) / args.data_name / args.surr_model_type
exp_name = f"{args.loss_type}_{args.surr_model_type}_{args.clip_backbone.replace('-', '').replace('/', '')}"
save_root = save_root / exp_name
save_root.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Checkpoints → {save_root}")

# --------------------------------------------------------------------- #
# Dataset & dataloader
# --------------------------------------------------------------------- #
if args.data_name == "coco":
    train_dataset = CocoClsDataset(
        root_dir='coco2017/',
        ann_file='annotations/instances_train2017.json',
        img_dir='train2017', phase='train')
    test_dataset = CocoClsDataset(
        root_dir='coco2017/',
        ann_file='annotations/instances_val2017.json',
        img_dir='val2017', phase='test')
    class_list = [cat['name'] for cat in train_dataset.coco.dataset['categories']]
elif args.data_name == "voc":
    train_dataset = VOCClsDataset(
        root_dir=args.train_dir,
        ann_file=['VOC2007/ImageSets/Main/trainval.txt',
                  'VOC2012/ImageSets/Main/trainval.txt'],
        img_dir=['VOC2007', 'VOC2012'], phase='train')
    test_dataset = VOCClsDataset(
        root_dir=args.train_dir,
        ann_file='VOC2007/ImageSets/Main/test.txt',
        img_dir=['VOC2007'], phase='test')
    class_list = train_dataset.CLASSES
else:
    raise ValueError("--data_name must be 'voc' or 'coco'")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=2, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=max(1, args.batch_size // 2),
    shuffle=False, num_workers=2)

num_classes = len(class_list)
print(f"[INFO] Train={len(train_dataset)} | Test={len(test_dataset)}")

# --------------------------------------------------------------------- #
# CLIP model & prompt features
# --------------------------------------------------------------------- #
model_clip, _ = clip.load(args.clip_backbone, device=device)
model_clip.eval()

adj_path = f"ml_data/{args.data_name}/{args.data_name}_adj.pkl"
with open(adj_path, 'rb') as f:
    adj = pickle.load(f)['adj']
adj[adj > 1] = 1.0  # binarize

_templates = [
    "A photo of a {} in the center of the image.",
    "An image featuring a {} prominently in the middle.",
    "A photo of a {} filling the entire image.",
    "A picture with a {} dominating the entire frame.",
    "An image filled with numerous {} spanning across the whole scene.",
    "A photo of multiple {} covering the entire image.",
]
min_c, max_c = (2, 4) if args.data_name == "coco" else (2, 5)

text_prompts = build_text_prompts(
    adj, class_list, min_c, max_c, _templates,
    max_prompts=TRAINING_CFG["max_text_combinations"],
    shuffle=(args.data_name == "coco"))
print(f"[INFO] Text prompts: {len(text_prompts)}")

embed_dim_map = {"RN50": 1024, "RN101": 512, "ViTB32": 512,
                 "ViTB16": 512, "ViTL14": 768}
embed_dim = embed_dim_map[args.clip_backbone.replace("-", "").replace("/", "")]
prompt_feats = torch.zeros(len(text_prompts), embed_dim, device=device)

with torch.no_grad():
    for i, (_, sent) in enumerate(text_prompts):
        token = clip.tokenize(sent).to(device)
        feat = model_clip.encode_text(token).float()
        prompt_feats[i] = feat / feat.norm(dim=-1, keepdim=True)
print(f"[INFO] Prompt feature matrix {prompt_feats.shape}")

# --------------------------------------------------------------------- #
# Surrogate CNN and generator
# --------------------------------------------------------------------- #
if args.surr_model_type == "vgg16":
    model_surr = Vgg16_all_layer.Vgg16(num_classes, args.data_name)
    layer_idx, layer_bia = [15, 16, 17, 18], 16
elif args.surr_model_type == "res152":
    model_surr = Res152_all_layer.Resnet152(num_classes, args.data_name)
    layer_idx, layer_bia = [3, 4, 5], 5
elif args.surr_model_type == "dense169":
    model_surr = Dense169_all_layer.Dense169(num_classes, args.data_name)
    layer_idx, layer_bia = [5, 6], 6
else:
    raise ValueError("--surr_model_type unsupported")

model_surr = nn.DataParallel(model_surr.eval().to(device))
netG = nn.DataParallel(GeneratorResnet().to(device))

criterion = ContrastiveLoss(args.margin)
criterion_mse = nn.MSELoss().to(device)
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

# --------------------------------------------------------------------- #
# K-means clustering on prompt features
# --------------------------------------------------------------------- #
K = TRAINING_CFG["num_clusters"]
kmeans = KMeans(n_clusters=K, random_state=42).fit(prompt_feats.cpu().numpy())
cluster_labels = torch.tensor(kmeans.labels_, device=device)

cluster_to_indices = defaultdict(list)
for idx, lab in enumerate(cluster_labels):
    cluster_to_indices[lab.item()].append(idx)

# --------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------- #
eps_float = args.eps / 255.0
all_eval = []

for epoch in range(args.epochs):
    p_orig = get_dynamic_p(epoch, args.epochs - 1)
    print(f"\n[Epoch {epoch}]  p(original image)={p_orig:.3f}")

    total_loss, steps = 0.0, 0
    for img, _ in tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100):
        img = img.to(device)
        optimG.zero_grad()

        # ---------- affine transform ----------
        deg = torch.randint(TRAINING_CFG["affine_deg_min"],
                            TRAINING_CFG["affine_deg_max"],
                            (1,), device=device).item()
        img_affine = transforms.functional.affine(
            img, angle=deg,
            translate=TRAINING_CFG["affine_translate"],
            scale=TRAINING_CFG["affine_scale"],
            shear=TRAINING_CFG["affine_shear"])
        img_in = choose_input(img, img_affine, p_orig)

        # ---------- generator ----------
        adv = netG(img_in)
        adv = torch.min(torch.max(adv, img - eps_float), img + eps_float)
        adv.clamp_(0.0, 1.0)

        # ---------- features ----------
        feat_img = model_surr(normalize(img_in))[layer_idx[-1]].mean((-1, -2))
        feat_adv = model_surr(normalize(adv))[layer_idx[-1]].mean((-1, -2))

        feat_clip = model_clip.encode_image(clip_normalize(img_in)).float()
        feat_clip = feat_clip / feat_clip.norm(dim=-1, keepdim=True)

        # ---------- sample text feats ----------
        n_pc = TRAINING_CFG["samples_per_cluster"]
        chosen = []
        for cid in range(K):
            pool = cluster_to_indices[cid]
            chosen.extend(random.sample(pool, n_pc) if len(pool) >= n_pc
                          else random.choices(pool, k=n_pc))
        chosen = torch.tensor(chosen, device=device)
        batch_txt = prompt_feats[chosen]
        batch_txt = batch_txt / batch_txt.norm(dim=-1, keepdim=True)

        sim = (batch_txt @ feat_clip.T).abs().cpu().numpy()
        worst = np.argmin(sim, axis=0)
        batch_txt = batch_txt[worst]

        # ---------- loss ----------
        loss = criterion(feat_adv, feat_img, batch_txt)
        loss -= criterion_mse(feat_adv / feat_adv.norm(dim=-1, keepdim=True), feat_clip)

        img_t = model_surr(normalize(img_in))[layer_bia]
        adv_t = model_surr(normalize(adv))[layer_bia]
        attention = img_t.mean(1, keepdim=True).detach()
        loss += torch.cosine_similarity(
            (adv_t * attention).flatten(1),
            (img_t * attention).flatten(1)
        ).mean()
        loss = loss / 3.0

        loss.backward()
        optimG.step()

        total_loss += loss.item()
        steps += 1

    print(f"[Epoch {epoch}] Mean loss = {total_loss/steps:.6f}")

    if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
        metrics = evaluate_ml(args, eps_float, num_classes,
                              test_loader, netG.module.state_dict(), device)
        all_eval.append(dict(epoch=epoch, result=metrics))
        torch.save(netG.module.state_dict(), save_root / f"netG_epoch{epoch}.pth")
        print(f"[Eval] Epoch {epoch}: {metrics}")

print("\n=== Training completed ===")
for rec in all_eval:
    print(f"Epoch {rec['epoch']:02d} → {rec['result']}")
