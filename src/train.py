#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.spatial.distance import cdist
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import multiprocessing
from datasets.base_dataset import RandomIdSampler
#from datasets.market1501 import Market1501 as Dataset
#from triplet import TripletSemihardLoss
from triplet import TripletLoss
from models.pcb_plus_dropout_pyramid import PCB_plus_dropout_pyramid
from __init__ import DEVICE, cmc, mean_ap, save_ckpt, load_ckpt
from custom_transforms import RandomErasing

from config import parse_args
# root = os.path.dirname(os.path.realpath(__file__)) + '/../../Market-1501-v15.09.15'
args = parse_args()

root = args.root
print("root = {}".format(root))
#root = os.path.dirname(os.path.realpath(__file__)) + '/../DukeMTMC-reID'
num_workers = multiprocessing.cpu_count() / 2
GPUID = args.GPUID
print("GPUID = {}".format(GPUID))
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

if args.data_loader == 'Duke':
    from datasets.duke import Duke as Dataset
elif args.data_loader == 'cuhk03':
    from datasets.cuhk03 import CUHK03 as Dataset
else:
    from datasets.market1501 import Market1501 as Dataset


def run():
    batch_id = args.batch_id  # 8
    batch_image = args.batch_image  # 8
    batch_train = args.batch_train  # 64
    batch_test = args.batch_test  # 32

    # ==============================================================
    trple_margin = args.trple_margin  # 0.1 increment
    para_balance = args.para_balance  # 0.01 increment
    # ==============================================================

    print("trple_margin:{}".format(trple_margin))
    print("para_balance:{}".format(para_balance))

    train_transform = transforms.Compose([
        transforms.Resize(args.transform_imsize, interpolation=3),
        #transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.transform_norm_mean,
                             std=args.transform_norm_std),
        RandomErasing(probability=args.transform_random_erase_p,
                      mean=args.transform_random_erase_mean)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(args.transform_imsize, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.transform_norm_mean,
                             std=args.transform_norm_std)
    ])

    train_dataset = Dataset(root + '/bounding_box_train',
                            transform=train_transform)
    train_loader_tri = DataLoader(train_dataset,
                                  sampler=RandomIdSampler(
                                      train_dataset, batch_image=batch_image),
                                  batch_size=batch_id * batch_image)
    # num_workers=num_workers)

    query_dataset = Dataset(root + '/query', transform=test_transform)
    query_loader = DataLoader(
        query_dataset, batch_size=batch_test, shuffle=False)

    test_dataset = Dataset(root + '/bounding_box_test',
                           transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_test, shuffle=False)

    model = PCB_plus_dropout_pyramid(num_classes=len(train_dataset.unique_ids))
    model_w = nn.DataParallel(model).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    triplet_loss = TripletLoss(margin=trple_margin)  # original margin: 1.2

    finetuned_params = list(model.base.parameters())
    # To train from scratch
    new_params = [p for n, p in model.named_parameters()
                  if not n.startswith('base.')]
    param_groups = [{'params': finetuned_params, 'lr': args.lr_finetune},
                    {'params': new_params, 'lr': args.lr_new}]
    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)

    modules_optims = [model, optimizer]

    #resume_ep, scores = load_ckpt(modules_optims, 'logs/pcb/ckpt_ep59.pth')
    #print('Resume from EP: {}'.format(resume_ep))
    print(optimizer)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_schedule, gamma=0.5)

    refine_ep = 10
    epochs = args.n_epoch

    # ==============================================================
    print("epochs:{}".format(epochs))
    print("refine_ep:{}".format(refine_ep))
    # ==============================================================

    max_mAP = 0
    m_ap = 0

    k_id = 0.0
    k_tri = 0.0
    loss_alpha = 0.25
    loss_sigma = 0.16
    loss_gamma = 2.0
    first_step = True

    for epoch in range(epochs):
        model_w.train()
        scheduler.step()
        train_loader = train_loader_tri

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            feats_list, logits_list = model_w(inputs)

            loss1 = torch.stack([criterion(logits, labels)
                                 for logits in logits_list], dim=0).sum()

            loss2 = torch.stack([triplet_loss(feats, labels)
                                 for feats in feats_list]).sum()

            new_k_id = loss_alpha * loss1.item() + (1.0 - loss_alpha) * k_id
            new_k_tri = loss_alpha * loss2.item() + (1.0 - loss_alpha) * k_tri

            if first_step:
                p_id = 0.5
                p_tri = 0.5
                first_step = False
            else:
                p_id = min(new_k_id, k_id) / k_id
                p_tri = min(new_k_tri, k_tri) / k_tri

            fl_id = - ( 1.0 - p_id) ** loss_gamma * np.log(p_id)
            fl_tri = - ( 1.0 - p_tri) ** loss_gamma * np.log(p_tri)
            k_id = new_k_id
            k_tri = new_k_tri

            if fl_tri / fl_id < loss_sigma:
                loss = loss1
            else:
                loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 0:
                print('%d/%d - %d/%d - loss: %f (%f, %f)' % (epoch, epochs, i, len(train_loader), loss.item(),
                                                             loss1.item(), loss2.item()))
        print('epoch: %d/%d - loss: %f' %
              (epoch, epochs, running_loss / len(train_loader)))

        if (epoch == 0 or epoch > 95) and ((epoch % 4 == 0) or (epoch == epochs-1)):
            model_w.eval()
            query = np.concatenate([torch.cat(model_w(inputs.to(DEVICE))[0], dim=1).detach().cpu().numpy()
                                    for i, (inputs, _) in enumerate(query_loader)])

            test = np.concatenate([torch.cat(model_w(inputs.to(DEVICE))[0], dim=1).detach().cpu().numpy()
                                   for i, (inputs, _) in enumerate(test_loader)])

            dist = cdist(query, test)
            r = cmc(dist, query_dataset.ids, test_dataset.ids, query_dataset.cameras, test_dataset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, query_dataset.ids, test_dataset.ids,
                           query_dataset.cameras, test_dataset.cameras)
            print('epoch[%d]: mAP=%f, r@1=%f, r@5=%f, r@10=%f' %
                  (epoch + 1, m_ap, r[0], r[4], r[9]))

        if epoch > 50 and max_mAP < m_ap:
            max_mAP = m_ap
            save_ckpt(modules_optims, epoch, 0,
                      'logs/ckpt_ep{}_re02_bs64_dropout02_GPU{}_mAP{}_market.pth'.format(epoch, GPUID, m_ap))


if __name__ == '__main__':
    run()
