from torch.utils.data import Sampler
import numpy as np

class BalancedSampler(Sampler):
    def __init__(self, labels, batch_size=12):
        """
        针对6:4比例的平衡采样器
        每个batch: 6正 + 6负 = 12样本
        负样本需要放回抽取
        """
        self.labels = np.array([ img[1] for img in labels.imgs])
        self.batch_size = batch_size
        
        # 分离正负样本索引
        self.positive_indices = np.where(self.labels == 1)[0]  # 多数类
        self.negative_indices = np.where(self.labels == 0)[0]  # 少数类
        
        self.num_positive = len(self.positive_indices)
        self.num_negative = len(self.negative_indices)
        
        # 每个batch中正负样本数量（1:1）
        self.positive_per_batch = batch_size // 2  # 6
        self.negative_per_batch = batch_size // 2  # 6
        
        print(f"每个batch: {self.positive_per_batch}正 + {self.negative_per_batch}负")
        print(f"原始数据 - 正样本: {self.num_positive}, 负样本: {self.num_negative}")
        
        # 计算epoch长度（基于负样本数量，因为负样本少）
        # 确保每个epoch看到足够多的负样本
        self.num_batches_per_epoch = max(
            self.num_positive // self.positive_per_batch,
            100  # 至少100个batch
        )
        
        self.length = self.num_batches_per_epoch * self.batch_size
    
    def __iter__(self):
        for _ in range(self.num_batches_per_epoch):
            # 正样本 - 无放回抽取（因为正样本多）
            positive_batch = np.random.choice(
                self.positive_indices, 
                self.positive_per_batch, 
                replace=False  # 无放回
            )
            
            # 负样本 - 放回抽取（因为负样本少）
            negative_batch = np.random.choice(
                self.negative_indices,
                self.negative_per_batch,
                replace=True  # 放回！
            )
            
            # 合并并打乱顺序
            batch = np.concatenate([positive_batch, negative_batch])
            np.random.shuffle(batch)
            
            yield from batch.tolist()
    
    def __len__(self):
        return self.length