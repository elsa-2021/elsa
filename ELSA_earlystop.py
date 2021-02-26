"""
1. Pretraining SimCLR & Proto-typing 
2. Training OOD (one-class classification)
3. Evaluation (eval.py?)
python3 train_main_ssl.py --save_dir semi --known_normal 0 --load_path ./pretrained/model_cifar10_0.pth --lr .0001 --dataset cifar10 --optimizer adam --ratio_known_normal 0.05 --ratio_known_outlier 0.05
"""
import sys, os
import utils,json
import torch.nn as nn
import transform_layers as TL
import torch.nn.functional as F
import torchvision.transforms as tr

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import model_csi as C
from dataloader_es import *
from parser import * 

#for kmeans++ to cluster the prototypes..
from soyclustering import SphericalKMeans
from scipy import sparse

from randaugment_without_rotation import *

import random,numpy as np
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
### helper functions
def checkpoint(f,  tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "prototypes": f.module.prototypes # add model prototype save
    }
    torch.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)
def generate_prototypes(model, valid_loader, n_cluster=100):
    first = True
    with torch.no_grad():
        normal_distance = []
        anomal_distance = []
        first = True
        for idx, (pos_1, _, _, semi_target, _, _) in enumerate(valid_loader):
            pos_1 = pos_1.cuda(non_blocking=True)
            #feature = model(pos_1) # normalized prototypes 
            _, outputs_aux = model(pos_1, simclr=True, penultimate=False, shift=False)
            out = outputs_aux['simclr']
            feature = F.normalize(out, dim=-1)

            true_feature = feature[semi_target != -1,:]
            
            if first:
                totalembed = true_feature
                first = False
            else:
                totalembed = torch.cat((totalembed, true_feature), dim = 0)           

    # Set prototypes (k-means++)
    all_out_numpy = totalembed.cpu().numpy() # T * 4 * D
    proto_list = []

    all_out = all_out_numpy.reshape(-1, all_out_numpy.shape[1])
    all_out_sp = sparse.csr_matrix(all_out)
    while True:
        try:
            spherical_kmeans = SphericalKMeans(
                n_clusters=n_cluster,
                max_iter=10,
                verbose=1,
                init='similar_cut'
            )

            spherical_kmeans.fit(all_out_sp)
            break
        except KeyboardInterrupt:
            assert 0
        except:
            print("K-means failure... Retrying")
            continue
        
    protos = spherical_kmeans.cluster_centers_
    protos = F.normalize(torch.Tensor(protos), dim = -1)
    return protos.to(device)

def get_simclr_augmentation(image_size):

    # parameter for resizecrop
    resize_scale = (0.54, 1.0) # resize scaling factor
    if True: # if resize_fix is True, use same scale
        resize_scale = (0.54, 0.54)

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    # Transform define #
    if args.dataset == 'imagenet': # Using RandomResizedCrop at PIL transform
        transform = nn.Sequential(
            color_jitter,
            color_gray,
        )
    else:
        transform = nn.Sequential(
            color_jitter,
            color_gray,
            resize_crop,
        )

    return transform

def energy_score(img,model):
    #z = model(img)
    _, outputs_aux = model(img, simclr=True, penultimate=False, shift=False)
    out = outputs_aux['simclr']
    z = F.normalize(out, dim=-1)

    zp = model.module.prototypes
    logits = torch.matmul(z, zp.t()) / args.temperature
    Le = torch.log(torch.exp(logits).sum(dim=1))
    return Le, logits

def cal_class_auroc(nd1,nd2,and1,and2,ndsum,andsum,ndmul,andmul,cls_list):
    # Class AUROC
    normal_class = args.known_normal
    anomaly_classes = [i for i in range(args.n_classes)]
    anomaly_classes.remove(normal_class)
    
    tosum_average = 0
    tomul_average = 0
    tod1_average = 0
    tod2_average = 0
    tod3_average = 0
    for anomaly in anomaly_classes:
        tosum = ndsum + np.array(andsum)[np.array(cls_list) == anomaly].tolist()
        tomul = ndmul + np.array(andmul)[np.array(cls_list) == anomaly].tolist()
        tod1 = nd1 + np.array(and1)[np.array(cls_list) == anomaly].tolist()
        tod2 = nd2 + np.array(and2)[np.array(cls_list) == anomaly].tolist()
#         tod3 = nd3 + np.array(and3)[np.array(cls_list) == anomaly].tolist()
        
        total_label = [1 for i in range(len(ndsum))] + [0 for i in range(len(tosum) - len(ndsum))]
        print('---------------------- Evaluation class: {} --------------------------'.format(anomaly))
        print(len(ndsum), len(tosum) - len(ndsum))
        print("sum\t", roc_auc_score(total_label, tosum))
        print("mul\t", roc_auc_score(total_label, tomul))
        print("px\t", roc_auc_score(total_label, tod1))
        print("pyx\t", roc_auc_score(total_label, tod2))
#         print("pshi\t", roc_auc_score(total_label, tod3))
        print('----------------------------------------------------------------------')
        print()
        
        tosum_average += roc_auc_score(total_label, tosum)
        tomul_average += roc_auc_score(total_label, tomul)
        tod1_average  += roc_auc_score(total_label, tod1)
        tod2_average  += roc_auc_score(total_label, tod2)
#         tod3_average  += roc_auc_score(total_label, tod3)
    
    tosum_average /= len(anomaly_classes)
    tomul_average /= len(anomaly_classes)
    tod1_average /= len(anomaly_classes)
    tod2_average /= len(anomaly_classes)
    tod3_average /= len(anomaly_classes)      
    
    print('------------------- Evaluation class average --------------------')
    print(len(ndsum), len(tosum) - len(ndsum))
    print("sum\t", tosum_average)
    print("mul\t", tomul_average)
    print("px\t", tod1_average)
    print("pyx\t", tod2_average)
#     print("pshi\t", tod3_average)
    print('----------------------------------------------------------------------')
    print()
    return 

def earlystop_score(model,valid_loader):
    rot_num = 4
    weighted_aucscores,aucscores = [],[]
    zp = model.module.prototypes
    for pos, pos2, _, semi_target, _, raw in valid_loader:
        prob,prob2, label_list = [] , [], []
        weighted_prob, weighted_prob2 = [], []
        Px_mean,Px_mean2 = 0, 0  
        images1 = pos.to(device)
        images2 = pos2.to(device)
        images1, images2 = simclr_aug(images1), simclr_aug(images2)
        images1, images2 = normalize(images1), normalize(images2)
        all_semi_targets = torch.cat([semi_target,semi_target+1])
        
#         _, out = model(images1,need_feat = True)
        _, outputs_aux = model(images1, simclr=True, penultimate=False, shift=False)
        out = outputs_aux['simclr']
        norm_out = F.normalize(out,dim=-1)
        zp = model.module.prototypes
        logits = torch.matmul(norm_out, zp.t())
        Le = torch.log(torch.exp(logits).sum(dim=1))
        prob.extend(Le.tolist())
        
#         _, out = model(images2,need_feat = True)
        _, outputs_aux = model(images2, simclr=True, penultimate=False, shift=False)
        out = outputs_aux['simclr']
        norm_out = F.normalize(out,dim=-1)
        zp = model.module.prototypes
        logits = torch.matmul(norm_out, zp.t())
        Le = torch.log(torch.exp(logits).sum(dim=1))
        prob2.extend(Le.tolist())


        label_list.extend(all_semi_targets)
        aucscores.append(roc_auc_score(label_list, prob2+prob))
#         weighted_aucscores.append(roc_auc_score(label_list, weighted_prob2+weighted_prob))
    print("earlystop_score:",np.mean(aucscores))
    return np.mean(aucscores)

def test(model, test_loader, train_loader, epoch):
    model.eval()
    with torch.no_grad():
        nd1,nd2,ndsum,ndmul = [],[],[],[]
        and1,and2,andsum,andmul = [],[],[],[]
        cls_list = []
        first = True
        for idx, (pos_1, _, _, semi_target, _, _) in enumerate(train_loader):
            pos_1 = pos_1.cuda(non_blocking=True)
            pos_1 = simclr_aug(pos_1)
            #pos_1 = normalize(pos_1) 
            # feature = model(pos_1)
            _, outputs_aux = model(pos_1, simclr=True, penultimate=False, shift=False)
            out = outputs_aux['simclr']
            feature = F.normalize(out, dim=-1)
            true_feature = feature[semi_target != -1,:]

            if first:
                totalembed = true_feature
                first = False
            else:
                totalembed = torch.cat((totalembed, true_feature), dim = 0)
        for idx, (pos_1, _, target, _, cls, image) in enumerate(test_loader):
            

            negative_target = (target == 1).nonzero().squeeze()
            positive_target = (target != 1).nonzero().squeeze()
#             pos_1 = pos_1.cuda(non_blocking=True)
            image = pos_1.cuda(non_blocking=True)
            out_ensemble = []
            
            for seed in range(args.sample_num):
                set_random_seed(seed) # random seed setting
                
                pos_1 = simclr_aug(image)
#                 pos_1 = normalize(pos_1) 
                pos_1 = pos_1.cuda(non_blocking=True)
                
#                 _ , feature = model(pos_1,need_feat = True)              
                _, outputs_aux = model(pos_1, simclr=True, penultimate=False, shift=False)
                out = outputs_aux['simclr']
                out_ensemble.append(feature) 
            
            
            out = torch.stack(out_ensemble,dim=1).mean(dim=1)
            norm_out = F.normalize(out,dim=-1)
            
            zp = model.module.prototypes
            logits = torch.matmul(norm_out, zp.t())
            Le = torch.log(torch.exp(logits).sum(dim=1))



            totalsim, _ = torch.matmul(norm_out, totalembed.t()).max(dim = 1)

            Psum = Le + totalsim
            Pmul = Le * totalsim

            cls_list.extend(cls[negative_target])
            if len(positive_target.shape) != 0:
                nd1.extend(Le[positive_target].tolist())
                nd2.extend(totalsim[positive_target].tolist())
                ndsum.extend(Psum[positive_target].tolist())
                ndmul.extend(Pmul[positive_target].tolist())

            if len(negative_target.shape) != 0:
                and1.extend(Le[negative_target].tolist())
                and2.extend(totalsim[negative_target].tolist())
                andsum.extend(Psum[negative_target].tolist())
                andmul.extend(Pmul[negative_target].tolist())
    cal_class_auroc(nd1,nd2,and1,and2,ndsum,andsum,ndmul,andmul,cls_list)


## 0) setting 
seed = args.seed
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

utils.makedirs(args.save_dir)
with open(f'{args.save_dir}/params.txt', 'w') as f: # training setting saving
    json.dump(args.__dict__, f)
if args.print_to_log: # 
    sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

args.device = device
    
## 1) pretraining & prototyping
"""
if args.backbone == 'wide_resnet':
    model = Wide_ResNet(28, 2, args.feature_dim)
elif args.backbone == 'resnet':
    model = ResNet(args.feature_dim)
"""
model = C.get_classifier('resnet18', n_classes=10).to(device)
model = C.get_shift_classifer(model, 1).to(device)

if args.dataset == 'cifar10':
    args.image_size = (32, 32, 3)
else:
    raise

if args.load_path != None: # pretrained model loading
    ckpt_dict = torch.load(args.load_path)
    model.load_state_dict(ckpt_dict,strict=True)
else:
    assert False , "Not implemented error: you should give pretrained and prototyped model"
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)    
model.to(args.device)

#model.module.prototypes = torch.rand(100, 128) - 0.5
#model.module.prototypes = F.normalize(model.module.prototypes, dim = -1)
# model.module.prototypes = model.module.prototypes.to(args.device)


# print(model.module.prototypes)

train_transform = transforms.Compose([
    transforms.Resize((args.image_size[0], args.image_size[1])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((args.image_size[0], args.image_size[1])),
    transforms.ToTensor(),
])

strong_aug = RandAugmentMC(n=12,m=5)
# dataset loader
total_dataset = load_dataset("./data", normal_class=[args.known_normal], known_outlier_class=args.known_outlier,
                             n_known_outlier_classes=args.n_known_outlier, ratio_known_normal=args.ratio_known_normal,
                             ratio_known_outlier=args.ratio_known_outlier, ratio_pollution=args.ratio_pollution, random_state=None,
                             train_transform=train_transform, test_transform=test_transform, 
                            valid_transform=strong_aug)

train_loader, false_valid_loader, valid_loader, test_loader = total_dataset.loaders(batch_size = args.batch_size)

simclr_aug = get_simclr_augmentation(image_size=(32, 32, 3)).to(device)
normalize = TL.NormalizeLayer()

if args.set_initial_kmeanspp:
    print("Prototype: initialize kmeans pp")
    model.module.prototypes = generate_prototypes(model, false_valid_loader, n_cluster=100)
else:
    print("Prototype: initialize random")
    raise
#     model.module.prototypes = model.module.prototypes.to(args.device)
    model.module.prototypes = torch.rand(args.n_cluster, 128) - 0.5
    model.module.prototypes = F.normalize(model.module.prototypes, dim = -1)
    model.module.prototypes = model.module.prototypes.to(args.device)

params = model.parameters()
if args.optimizer == "adam":
    optim = torch.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
elif args.optimizer =="SGD":
    optim = torch.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)
import copy
    
# Evaluation before training
test(model, test_loader, train_loader, -1)
earlystop_trace = []
end_train = False
max_earlystop_auroc = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)
for epoch in range(args.n_epochs):
    model.train()
    
    # adjust learning rate 
#     if epoch in args.decay_epochs:
#         for param_group in optim.param_groups:
#             new_lr = param_group['lr'] * args.decay_rate
#             param_group['lr'] = new_lr
#         print("Decaying lr to {}".format(new_lr))

    
    # training
    losses = []
    for i, (pos, pos2, _, semi_target, _, _) in tqdm(enumerate(train_loader)):
        #pos1, pos2 = pos1.to(args.device), pos2.to(args.device)
        semi_target = semi_target.to(args.device)
        semi_target = semi_target.repeat(2)
        #La = similarity_loss(pos1, pos2, model)

        pos = pos.to(args.device)
        pos_1, pos_2 = hflip(pos.repeat(2, 1, 1, 1)).chunk(2)
        pos = torch.cat([pos_1,pos_2],dim=0)
        pos = simclr_aug(pos)
#         pos = normalize(pos)
        score, logits1 = energy_score(pos, model)
#         _, logits2 = energy_score(pos2, model)
        C = (torch.log(torch.Tensor([args.n_cluster])) + 1/args.temperature).to(device)
        Le = torch.where(semi_target == -1, (C - score) ** -1, score ** -1).mean()   ## Le inverse
#         Le = torch.where(semi_target == -1, score, score ** -1).mean()  
        
        #x_out = F.softmax(logits1, dim=-1)
        #x_tf_out = F.softmax(logits2, dim=-1)
        #La = entropyloss(x_out,x_tf_out)        
        
        L = Le
        optim.zero_grad()
        L.backward()
        optim.step()
        losses.append(L.cpu().detach())
    
    model.eval()
    with torch.no_grad():
        earlystop_auroc = earlystop_score(model,valid_loader)
    earlystop_trace.append(earlystop_auroc)
    print('[{}]epoch loss:'.format(epoch), np.mean(losses))
    print('[{}]earlystop loss:'.format(epoch),earlystop_auroc)
    if epoch % args.ckpt_every == 0 or epoch == args.n_epochs - 1: 
        checkpoint(model,  f'ckpt_ssl_{epoch}.pt', args, args.device)
        
    if max_earlystop_auroc < earlystop_auroc:
        max_earlystop_auroc = earlystop_auroc
        best_epoch = epoch
        checkpoint(model,  f'ckpt_ssl_{epoch}.pt', args, args.device)
        best_model = copy.deepcopy(model)
    if epoch>50:
        if earlystop_trace[-4] < max_earlystop_auroc and earlystop_trace[-3] < max_earlystop_auroc and earlystop_trace[-2] < max_earlystop_auroc:
            end_train = True
    
    if end_train:
        checkpoint(model,  f'ckpt_ssl_{epoch}.pt', args, args.device)
        print("trainin ended")
        break

print("best epoch:",best_epoch,"best auroc:",max_earlystop_auroc)
test(best_model, test_loader, train_loader, epoch) # we do not test them

