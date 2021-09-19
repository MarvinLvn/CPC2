# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from os import remove
from os.path import join, exists
from time import time

import progressbar
import torch
import torch.nn as nn


class kMeanCluster(nn.Module):

    def __init__(self, Ck):

        super(kMeanCluster, self).__init__()
        self.register_buffer('Ck', Ck)
        self.k = Ck.size(1)

    def forward(self, features):
        B, S, D = features.size()
        features = features.contiguous().view(B*S, 1, -1)
        return ((features - self.Ck)**2).sum(dim=2).view(-1, S, self.k)


class kMeanClusterStep(torch.nn.Module):

    def __init__(self, k, D):

        super(kMeanClusterStep, self).__init__()
        self.k = k
        self.register_buffer('Ck', torch.zeros(1, k, D))

    def forward(self, locF):

        index = ((locF - self.Ck)**2).mean(dim=2).min(dim=1)[1]
        Ck1 = torch.cat([locF[index == p].sum(dim=0, keepdim=True)
                         for p in range(self.k)], dim=1)
        nItems = torch.cat([(index == p).sum(dim=0, keepdim=True)
                            for p in range(self.k)], dim=0).view(1, -1)
        return Ck1, nItems


def kMeanGPU(dataLoader, featureMaker, k, n_group=1,
             MAX_ITER=100, EPSILON=1e-4,
             perIterSize=-1, start_clusters=None,
             save=False, load=False, save_dir=None,
             save_last=5):

    print(f"Start Kmean clustering with {k} clusters and {n_group} groups...")

    if save or load:
        assert save_dir is not None

    if start_clusters is None:
        if load and exists(join(save_dir, "checkpoint_last.pt")):
            print("Loading from last checkpoint")
            state_dict = torch.load(join(save_dir, "checkpoint_last.pt"))
            Ck = state_dict["state_dict"]["Ck"]
            D = Ck.size(2)
        else:
            Ck = []
            with torch.no_grad():
                for index, data in enumerate(dataLoader):
                    # Hack to check if the features have been loaded
                    # with the code integrating the data augmentation
                    # In which case, each audio chunk is loaded twice (past and future)
                    # with or without data augmentation.
                    # We just take the future sequence so that everything is compatible
                    if len(list(data[0].size())) == 4:
                        data[0] = data[0][:, 1, ...]
                    cFeature = featureMaker(data)
                    cFeature = cFeature.contiguous().view(-1, cFeature.size(2)//n_group)
                    Ck.append(cFeature)
                    if index > k:
                        break
            Ck = torch.cat(Ck, dim=0)
            N, D = Ck.size()
            indexes = torch.randperm(N)[:k]
            Ck = Ck[indexes].view(1, k, D)
    else:
        Ck = start_clusters
        D = Ck.size(2)

    if perIterSize < 0:
        perIterSize = len(dataLoader)

    clusterStep = kMeanClusterStep(k, D).cuda()
    clusterStep = torch.nn.DataParallel(clusterStep)
    clusterStep.module.Ck.copy_(Ck)

    bar = progressbar.ProgressBar(maxval=MAX_ITER)
    bar.start()
    iter, stored = 0, 0
    sum_seen = 0
    print("perIterSize = %.f" % perIterSize)
    if load and start_clusters is None and exists(join(save_dir, "checkpoint_last.pt")):
        iter = state_dict["iteration"]
        lastDiff = state_dict["lastDiff"]
        print(f"Continuing training from iteration {iter}. lastDiff: {lastDiff}")
    with torch.no_grad():
        while iter < MAX_ITER:
            start_time = time()
            Ck1 = torch.zeros(Ck.size()).cuda()
            nItemsClusters = torch.zeros(Ck.size(1),
                                         dtype=torch.long).cuda()
            for index, data in enumerate(dataLoader):
                # same dirty hack see above
                if len(list(data[0].size())) == 4:
                    data[0] = data[0][:, 1, ...]
                cFeature = featureMaker(data).contiguous().view(-1, 1, D)
                locC, locN = clusterStep(cFeature)
                Ck1 += locC.sum(dim=0, keepdim=True)
                nItemsClusters += locN.sum(dim=0)
                ### If the training set is too big and we want to redude the number of item per iteration
                stored += 1
                sum_seen += data[0].shape[0] * data[0].shape[-1] / 16000

                if stored >= perIterSize:
                    break

            if stored < perIterSize:
                # We haven't seen enough batches to save the checkpoint
                # Let's get data rolling!
                continue

            stored = 0
            iter += 1
            bar.update(iter)
            print("I've seen %.2f hours in %d epochs :) More data more data more data!" % (sum_seen / 3600, iter))

            nItemsClusters = nItemsClusters.float().view(1, -1, 1) + 1e-8
            Ck1 /= nItemsClusters
            lastDiff = (clusterStep.module.Ck - Ck1).norm(dim=2).max().item()
            nItems = int(nItemsClusters.sum().cpu().detach().item())
            info=f"ITER {iter} done in {time()-start_time:.2f} seconds. nItems: {nItems}. Difference with last checkpoint: {lastDiff}"
            print(info)
            with open(join(save_dir, "training_logs.txt"), "a") as f:
                f.write(info+"\n")
            if save:
                info=f"Saving last checkpoint to {join(save_dir, 'checkpoint_last.pt')}"
                print(info)
                with open(join(save_dir, "training_logs.txt"), "a") as f:
                    f.write(info+"\n")
                out_state_dict = {}
                clusterModule = kMeanCluster(Ck1)
                out_state_dict["state_dict"] = clusterModule.state_dict()
                out_state_dict["n_clusters"] = Ck1.size(1)
                out_state_dict['dim'] = Ck1.size(2)
                out_state_dict["iteration"] = iter
                out_state_dict["lastDiff"] = lastDiff
                torch.save(out_state_dict, join(save_dir, "checkpoint_last.pt"))
                torch.save(out_state_dict, join(save_dir, f"checkpoint_{iter}.pt"))
                if exists(join(save_dir, f"checkpoint_{iter-save_last}.pt")):
                    remove(join(save_dir, f"checkpoint_{iter-save_last}.pt"))
            if lastDiff < EPSILON:
                print(
                    f"Clustering ended in {iter} iterations out of {MAX_ITER}")
                break
            clusterStep.module.Ck.copy_(Ck1)

    bar.finish()

    print(f"Clustering ended in {MAX_ITER} iterations out of {MAX_ITER}")
    print(f"Last diff {lastDiff}")
    if start_clusters is not None:
        nEmptyClusters = (nItemsClusters < 1).sum().item()
        print(f"{nEmptyClusters} empty clusters out of {k}")
    return clusterStep.module.Ck
