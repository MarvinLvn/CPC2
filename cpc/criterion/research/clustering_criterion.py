# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from cpc.clustering.clustering import (
    kMeanCluster,
    kMeanGPU,
    fastDPMean,
    distanceEstimation,
)
from cpc.criterion.criterion import CTCPhoneCriterion


class ClusteringLoss(nn.Module):
    def __init__(self, k, d, delay, clusterIter, clusteringUpdate):

        super(ClusteringLoss, self).__init__()
        self.clusters = kMeanCluster(torch.zeros(1, k, d))
        self.k = k
        self.d = d
        self.init = False
        self.delay = delay
        self.step = 0
        self.clusterIter = clusterIter

        self.TARGET_QUANTILE = 0.05
        availableUpdates = ["kmean", "dpmean"]
        if clusteringUpdate not in availableUpdates:
            raise ValueError(
                f"{clusteringUpdate} is an invalid clustering \
                            update option. Must be in {availableUpdates}"
            )

        print(f"Clustering update mode is {clusteringUpdate}")
        self.DP_MEAN = clusteringUpdate == "dpmean"

    def canRun(self):

        return self.step > self.delay

    def getOPtimalLambda(self, dataLoader, model, MAX_ITER=10):

        distData = distanceEstimation(
            model, dataLoader, maxIndex=MAX_ITER, maxSizeGroup=300
        )
        nData = len(distData)
        print(f"{nData} samples analyzed")
        index = int(self.TARGET_QUANTILE * nData)
        return distData[index]

    def updateCLusters(self, dataLoader, featureMaker, MAX_ITER=20, EPSILON=1e-4):

        self.step += 1
        if not self.canRun():
            return

        featureMaker = featureMaker.cuda()
        if not isinstance(featureMaker, nn.DataParallel):
            featureMaker = nn.DataParallel(featureMaker)

        if self.DP_MEAN:
            l_ = self.getOPtimalLambda(dataLoader, featureMaker)
            clusters = fastDPMean(
                dataLoader,
                featureMaker,
                l_,
                MAX_ITER=MAX_ITER,
                perIterSize=self.clusterIter,
            )
            self.k = clusters.size(1)
        else:
            start_clusters = None
            clusters = kMeanGPU(
                dataLoader,
                featureMaker,
                self.k,
                MAX_ITER=MAX_ITER,
                EPSILON=EPSILON,
                perIterSize=self.clusterIter,
                start_clusters=start_clusters,
            )
        self.clusters = kMeanCluster(clusters)
        self.init = True


class DeepClustering(ClusteringLoss):
    def __init__(self, *args):
        ClusteringLoss.__init__(self, *args)
        self.classifier = nn.Linear(self.d, self.k)
        self.lossCriterion = nn.CrossEntropyLoss()

    def forward(self, x, labels):

        if not self.canRun():
            return torch.zeros(1, 1, device=x.device)

        B, S, D = x.size()
        predictedLabels = self.classifier(x.view(-1, D))

        return self.lossCriterion(predictedLabels, labels.view(-1)).mean().view(-1, 1)


class CTCCLustering(ClusteringLoss):
    def __init__(self, *args):
        ClusteringLoss.__init__(self, *args)
        self.mainModule = CTCPhoneCriterion(self.d, self.k, False)

    def forward(self, cFeature, label):
        return self.mainModule(cFeature, None, label)[0]


class DeepEmbeddedClustering(ClusteringLoss):
    def __init__(self, lr, *args):

        self.lr = lr
        ClusteringLoss.__init__(self, *args)

    def forward(self, x):

        if not self.canRun():
            return torch.zeros(1, 1, device=x.device)

        B, S, D = x.size()
        clustersDist = self.clusters(x)
        clustersDist = clustersDist.view(B * S, -1)
        clustersDist = 1.0 / (1.0 + clustersDist)
        Qij = clustersDist / clustersDist.sum(dim=1, keepdim=True)

        qFactor = (Qij ** 2) / Qij.sum(dim=0, keepdim=True)
        Pij = qFactor / qFactor.sum(dim=1, keepdim=True)

        return (Pij * torch.log(Pij / Qij)).sum().view(1, 1)

    def updateCLusters(self, dataLoader, model):

        if not self.init:
            super(DeepEmbeddedClustering, self).updateCLusters(dataLoader, model)
            self.clusters.Ck.requires_grad = True
            self.init = True
            return

        self.step += 1
        if not self.canRun():
            return

        print("Updating the deep embedded clusters")
        optimizer = torch.optim.SGD([self.clusters.Ck], lr=self.lr)

        maxData = len(dataLoader) if self.clusterIter <= 0 else self.clusterIter

        for index, data in enumerate(dataLoader):
            if index > maxData:
                break

            optimizer.zero_grad()

            batchData, label = data
            batchData = batchData.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            with torch.no_grad():
                cFeature, _, _ = model(batchData, label)

            loss = self.forward(cFeature).sum()
            loss.backward()

            optimizer.step()
