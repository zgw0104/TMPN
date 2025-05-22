import torch
import torch.nn as nn
from hetsgg.modeling.utils import cat
import torch.nn.functional as F


class RobustMoERouter(nn.Module):
    def __init__(self, input_dim=2048, num_experts=3, balance_weight=0.1, epsilon=1e-6):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, 1024),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(1024,512),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(512,128),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(128,3)
                                    )
        self.num_experts = num_experts
        self.balance_weight = balance_weight
        self.epsilon = epsilon  # 平滑系数

    # def forward(self, x):
    #     n = x.shape[0]
    #     device = x.device

    #     # 门控计算
    #     logits = self.gate(x)
    #     probs = torch.softmax(logits, dim=1)
        
    #     # 添加平滑防止零概率
    #     probs = (probs + self.epsilon) / (1 + self.num_experts * self.epsilon)

    #     # 初始专家选择
    #     expert_choice = torch.argmax(probs, dim=1)
        
    #     # 强制分配机制 --------------------------------------------------
    #     # 检查是否有未分配的专家
    #     present_mask = torch.bincount(expert_choice, minlength=self.num_experts) > 0
        
    #     # 对未分配的专家进行强制分配
    #     for expert_idx in torch.where(~present_mask)[0]:
    #         # 找到该专家概率最大的样本
    #         expert_probs = probs[:, expert_idx]
    #         target_pos = torch.argmax(expert_probs)
            
    #         # 修改该样本的专家分配
    #         expert_choice[target_pos] = expert_idx
    #     # --------------------------------------------------------------

    #     # 分割数据和收集索引
    #     expert_data = []
    #     expert_indices = []
    #     for i in range(self.num_experts):
    #         mask = (expert_choice == i)
    #         indices = torch.where(mask)[0]
            
    #         # 二次验证确保至少有一个样本
    #         if len(indices) == 0:
    #             # 应急方案：随机选择一个样本
    #             indices = torch.randint(0, n, (1,), device=device)
            
    #         expert_indices.append(indices)
    #         expert_data.append(x[indices])

    #     # 负载均衡损失计算（改进版）
    #     mean_probs = torch.mean(probs, dim=0)
    #     counts = torch.bincount(expert_choice, minlength=self.num_experts)
    #     fractions = counts.float() / n
        
    #     # 双模式损失：Switch loss + 最小分配惩罚
    #     switch_loss = self.num_experts * torch.sum(mean_probs * fractions)
    #     min_alloc_loss = -torch.log(torch.min(fractions))  # 惩罚最小分配比例
        
    #     total_loss = switch_loss + min_alloc_loss
        
    #     return expert_data, total_loss * self.balance_weight, expert_indices


    # def forward(self, x):
    #     n = x.shape[0]
    #     device = x.device

    #     # 门控计算
    #     logits = self.gate(x)
    #     probs = torch.softmax(logits, dim=1)
    #     probs = (probs + self.epsilon) / (1 + self.num_experts * self.epsilon)

    #     # 初始化分配
    #     expert_choice = torch.argmax(probs, dim=1)
    #     allocated_mask = torch.zeros(n, dtype=torch.bool, device=device)  # 新增分配掩码

    #     # 强制分配机制（改进版）
    #     present_mask = torch.bincount(expert_choice, minlength=self.num_experts) > 0
        
    #     for expert_idx in torch.where(~present_mask)[0]:
    #         # 只考虑未分配的样本
    #         available = ~allocated_mask
    #         if available.any():
    #             expert_probs = probs[:, expert_idx]
    #             # 在未分配样本中选择概率最高的
    #             candidate_probs = expert_probs * available.float()
    #             target_pos = torch.argmax(candidate_probs)
                
    #             # 更新分配并标记
    #             expert_choice[target_pos] = expert_idx
    #             allocated_mask[target_pos] = True

    #     # 二次分配冲突解决 ----------------------------------------
    #     # 检测重复分配
    #     _, inverse, counts = torch.unique(expert_choice, return_inverse=True, return_counts=True)
    #     duplicates = torch.where(counts[inverse] > 1)[0]
        
    #     # 为重复分配的样本重新分配
    #     for dup_idx in duplicates:
    #         original_expert = expert_choice[dup_idx]
    #         # 选择次优的未饱和专家
    #         expert_probs = probs[dup_idx]
    #         sorted_experts = torch.argsort(expert_probs, descending=True)
    #         for e in sorted_experts:
    #             if e != original_expert and counts[e] < (n // self.num_experts + 1):
    #                 expert_choice[dup_idx] = e
    #                 counts[original_expert] -= 1
    #                 counts[e] += 1
    #                 break
    #     # ---------------------------------------------------------

    #     # 最终分配验证
    #     expert_data = []
    #     expert_indices = []
    #     total_allocated = 0
    #     for i in range(self.num_experts):
    #         mask = (expert_choice == i)
    #         indices = torch.where(mask)[0]
            
    #         # 最终应急机制：确保至少一个样本
    #         if len(indices) == 0:
    #             unassigned = torch.where(~allocated_mask)[0]
    #             if len(unassigned) > 0:
    #                 selected = unassigned[0]
    #             else:
    #                 selected = torch.argmin(allocated_mask.float())  # 选择分配次数最少的
    #             indices = torch.tensor([selected], device=device)
    #             expert_choice[selected] = i
    #             allocated_mask[selected] = True
                
    #         expert_indices.append(indices)
    #         expert_data.append(x[indices])
    #         total_allocated += len(indices)

    #     # 完整性验证
    #     assert total_allocated == n, f"分配总数错误：{total_allocated} != {n}"
    #     assert torch.all(torch.bincount(torch.cat(expert_indices))) == 1, "存在重复分配"

    #     # 损失计算（加入重复惩罚项）
    #     mean_probs = torch.mean(probs, dim=0)
    #     counts = torch.bincount(expert_choice, minlength=self.num_experts)
    #     fractions = counts.float() / n
        
    #     switch_loss = self.num_experts * torch.sum(mean_probs * fractions)
    #     min_alloc_loss = -torch.log(torch.min(fractions))
    #     conflict_loss = torch.sum(torch.bincount(torch.cat(expert_indices)) > 1).float()  # 重复样本计数
        
    #     total_loss = switch_loss + min_alloc_loss + conflict_loss * 10  # 强化冲突惩罚

    #     return expert_data, total_loss * self.balance_weight, expert_indices

    def forward(self, x):
        n = x.shape[0]
        device = x.device
        
        # 步骤1：计算理论分配基数
        base_size = n // self.num_experts
        remainder = n % self.num_experts
        target_counts = torch.tensor( #[n1,n2,n3]
            [base_size + 1 if i < remainder else base_size 
             for i in range(self.num_experts)],
            device=device
        )
        
        # 步骤2：门控计算
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=1)
        
        # 步骤3：初始分配
        expert_choice = torch.argmax(probs, dim=1)
        
        # 步骤4：动态平衡调整 -------------------------------------------------
        # 创建分配池
        allocation_pool = [[] for _ in range(self.num_experts)]
        for idx, expert in enumerate(expert_choice):
            allocation_pool[expert].append(idx)
        
        # 调整超额专家
        for expert in range(self.num_experts):
            current_count = len(allocation_pool[expert])
            if current_count <= target_counts[expert]:
                continue
                
            # 需要转移的样本数
            overflow = current_count - target_counts[expert]
            
            # 选择该专家中概率最低的样本进行转移
            expert_probs = probs[allocation_pool[expert], expert]
            sorted_indices = torch.argsort(expert_probs)[:overflow]
            to_redistribute = [allocation_pool[expert][i] for i in sorted_indices]
            
            # 从原专家移除
            allocation_pool[expert] = [idx for idx in allocation_pool[expert] 
                                      if idx not in to_redistribute]
            
            # 重新分配到其他专家
            for idx in to_redistribute:
                # 寻找最需要样本且概率次高的专家
                candidate_experts = torch.argsort(probs[idx], descending=True)[1:]
                for candidate in candidate_experts:
                    if len(allocation_pool[candidate]) < target_counts[candidate]:
                        allocation_pool[candidate].append(idx)
                        break

        # 步骤5：最终分配验证
        expert_data = []
        expert_indices = []
        total_allocated = 0
        for i in range(self.num_experts):
            indices = torch.tensor(allocation_pool[i], device=device)
            expert_indices.append(indices)
            expert_data.append(x[indices])
            total_allocated += len(indices)
        
        # 完整性检查
        assert total_allocated == n, f"分配错误：{total_allocated}/{n}"
        assert torch.all(torch.bincount(torch.cat(expert_indices)) == 1), "存在重复分配"

        # 损失计算（关注概率分布均衡）
        mean_probs = torch.mean(probs, dim=0)
        ideal_dist = target_counts.float() / n
        distribution_loss = torch.sum(mean_probs * ideal_dist)
        
        return expert_data, distribution_loss * self.balance_weight, expert_indices


def reconstruct_data(expert_data, expert_indices, original_shape):
    """
    根据专家分配结果重建原始顺序的数据
    
    参数：
    expert_data: List[Tensor] - 路由后的专家数据列表 
    expert_indices: List[Tensor] - 对应的原始索引列表
    original_shape: Tuple - 原始数据的形状 [n, 2048]
    
    返回：
    reconstructed: Tensor - 按原始顺序重建的数据
    mask: Tensor - 分配掩码矩阵 [n, num_experts]
    """
    num_experts = len(expert_data)
    n = original_shape[0]
    device = expert_data[0].device if expert_data else 'cpu'
    
    # 创建空容器（保留梯度）
    reconstructed = torch.zeros(original_shape, device=device)
    mask = torch.zeros((n, num_experts), device=device)
    
    # 逐专家填充数据
    for expert_idx, (data, indices) in enumerate(zip(expert_data, expert_indices)):
        if len(indices) > 0:
            # 写入数据到原始位置
            reconstructed[indices] = data
            
            # 构建分配掩码
            mask[indices, expert_idx] = 1.0
    
    # 验证完整性
    assigned_count = sum(len(idx) for idx in expert_indices)
    assert assigned_count == n, f"数据不完整！已分配{assigned_count}/{n}个样本"
    
    return reconstructed, mask

class UniformMoERouter(nn.Module):
    def __init__(self, input_dim=2048, num_experts=3, balance_weight=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, 1024),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(1024,512),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(512,128),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(128,3)
                                    )
        self.num_experts = num_experts
        self.balance_weight = balance_weight

    def forward(self, x):
        # 原始索引记录
        original_indices = torch.arange(x.size(0), device=x.device)  # [0, 1, ..., n-1]
        
        # 数据打乱
        shuffle_idx = torch.randperm(x.size(0), device=x.device)
        shuffled_x = x[shuffle_idx]  # 打乱后的数据
        
        # 理论分配基数计算
        n = x.size(0)
        base_size = n // self.num_experts
        remainder = n % self.num_experts
        target_counts = torch.tensor(
            [base_size + 1 if i < remainder else base_size 
             for i in range(self.num_experts)],
            device=x.device
        )
        
        # 门控计算（基于打乱后的数据）
        logits = self.gate(shuffled_x)
        probs = torch.softmax(logits, dim=1)
        
        # 初始分配
        expert_choice = torch.argmax(probs, dim=1)
        
        # 动态平衡调整 -------------------------------------------------
        allocation_pool = [[] for _ in range(self.num_experts)]
        for idx, expert in enumerate(expert_choice):
            allocation_pool[expert].append(idx)
        
        # 调整超额专家（使用打乱后的索引）
        for expert in range(self.num_experts):
            current_count = len(allocation_pool[expert])
            if current_count <= target_counts[expert]:
                continue
                
            overflow = current_count - target_counts[expert]
            expert_probs = probs[allocation_pool[expert], expert]
            sorted_indices = torch.argsort(expert_probs)[:overflow]
            to_redistribute = [allocation_pool[expert][i] for i in sorted_indices]
            
            allocation_pool[expert] = [idx for idx in allocation_pool[expert] 
                                      if idx not in to_redistribute]
            
            for idx in to_redistribute:
                candidate_experts = torch.argsort(probs[idx], descending=True)[1:]
                for candidate in candidate_experts:
                    if len(allocation_pool[candidate]) < target_counts[candidate]:
                        allocation_pool[candidate].append(idx)
                        break
        
        # 索引转换到原始顺序 --------------------------------------------
        expert_original_indices = []
        for expert_pool in allocation_pool:
            # 将打乱后的索引转换为原始索引
            restored = shuffle_idx[torch.tensor(expert_pool, device=x.device)]
            expert_original_indices.append(restored)
        
        # 最终验证
        total_allocated = sum(len(idx) for idx in expert_original_indices)
        assert total_allocated == n, f"分配错误：{total_allocated}/{n}"
        
        all_indices = torch.cat(expert_original_indices)
        assert torch.all(torch.bincount(all_indices) == 1), "存在重复分配"
        
        # 损失计算（基于打乱后的数据）
        counts = torch.tensor([len(pool) for pool in allocation_pool], device=x.device)
        fractions = counts.float() / n
        mean_probs = torch.mean(probs, dim=0)
        distribution_loss = torch.sum(mean_probs * fractions) * self.num_experts
        
        # 分割最终数据
        expert_data = []
        for expert_pool in allocation_pool:
            expert_data.append(shuffled_x[expert_pool])
        
        return expert_data, distribution_loss * self.balance_weight, expert_original_indices

class UniformMoERouter3(nn.Module):
    def __init__(self, input_dim=2048, num_experts=3, balance_weight=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, 1024),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(1024,512),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(512,128),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(128,3)
                                    )
        self.num_experts = num_experts
        self.balance_weight = balance_weight

    def forward(self, x):
        n = x.shape[0]
        device = x.device
        num_experts = self.num_experts  # 固定专家数为3

        ####################################################################
        # 修改点1：重新计算理论分配基数
        # 保证每个专家至少获得1/5的数据量（向上取整）
        min_per_expert = (n + 4) // 5  # ceil(n/5)
        remaining = n - min_per_expert * num_experts

        # 处理极小样本数的边界情况
        while remaining < 0 and min_per_expert > 0:
            min_per_expert -= 1
            remaining = n - min_per_expert * num_experts

        # 剩余数据按门控动态分配
        target_counts = [min_per_expert] * num_experts
        
        ####################################################################
        # 门控计算（保持不变）
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=1)
        expert_choice = torch.argmax(probs, dim=1)
        
        ####################################################################
        # 修改点2：动态平衡调整策略
        # 第一阶段：确保最低分配
        allocation_pool = [[] for _ in range(num_experts)]
        for idx, expert in enumerate(expert_choice):
            allocation_pool[expert].append(idx)
            
        # 强制满足最低分配要求
        for expert in range(num_experts):
            current = len(allocation_pool[expert])
            if current >= min_per_expert:
                continue
                
            # 需要补充的样本数
            need = min_per_expert - current
            candidates = []
            
            # 从其他专家处收集可转移样本
            for other_expert in range(num_experts):
                if other_expert == expert:
                    continue
                if len(allocation_pool[other_expert]) > min_per_expert:
                    # 收集其他专家超过最低分配的样本
                    overflow = len(allocation_pool[other_expert]) - min_per_expert
                    pool = allocation_pool[other_expert]
                    # 选择概率最低的样本
                    other_probs = probs[pool, other_expert]
                    sorted_indices = torch.argsort(other_probs)[:overflow]
                    candidates.extend([(pool[i], other_expert) for i in sorted_indices])
            
            # 按样本对新专家的适配度排序
            candidate_scores = [probs[idx, expert] for idx, _ in candidates]
            sorted_candidates = sorted(zip(candidates, candidate_scores),
                                    key=lambda x: x[1], reverse=True)
            
            # 转移最佳适配样本
            for (idx, src_expert), _ in sorted_candidates[:need]:
                allocation_pool[src_expert].remove(idx)
                allocation_pool[expert].append(idx)
                need -= 1
                if need == 0:
                    break

        ####################################################################
        # 第二阶段：动态分配剩余容量
        # 计算剩余可用容量
        capacities = [len(pool) - min_per_expert for pool in allocation_pool]
        total_remaining = sum(max(c,0) for c in capacities)
        
        # 需要分配的剩余样本（原始门控结果中未被强制分配的）
        remaining_indices = []
        for expert in range(num_experts):
            if len(allocation_pool[expert]) > min_per_expert:
                keep = allocation_pool[expert][:min_per_expert]
                remaining = allocation_pool[expert][min_per_expert:]
                allocation_pool[expert] = keep
                remaining_indices.extend(remaining)
        
        # 按门控概率重新分配剩余样本
        for idx in remaining_indices:
            # 排除已经满足最低分配的专家
            valid_experts = [e for e in range(num_experts)
                            if len(allocation_pool[e]) < min_per_expert + capacities[e]]
            if not valid_experts:
                valid_experts = list(range(num_experts))
                
            # 选择概率最高的可用专家
            scores = [(e, probs[idx, e]) for e in valid_experts]
            chosen_expert = max(scores, key=lambda x: x[1])[0]
            allocation_pool[chosen_expert].append(idx)
        
        ####################################################################
        # 最终验证与输出（保持不变）
        expert_data = []
        expert_indices = []
        total_allocated = 0
        for i in range(num_experts):
            indices = torch.tensor(allocation_pool[i], device=device)
            expert_indices.append(indices)
            expert_data.append(x[indices])
            total_allocated += len(indices)
        
        assert total_allocated == n, f"分配错误：{total_allocated}/{n}"
        assert torch.all(torch.bincount(torch.cat(expert_indices)) == 1), "重复分配"

        ####################################################################
        # 损失函数（保持之前设计）
        counts = torch.tensor([len(pool) for pool in allocation_pool], 
                            dtype=torch.float32, device=device)
        mean_gate_scores = torch.zeros(num_experts, device=device)
        for i in range(num_experts):
            pool = allocation_pool[i]
            if pool:
                mean_gate_scores[i] = probs[pool, i].mean()
        
        distribution_loss = (counts * mean_gate_scores).sum() / num_experts

        return expert_data, distribution_loss,expert_indices
    

class UniformMoERouter4(nn.Module):
    def __init__(self, input_dim=2048, num_experts=4, balance_weight=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, 1024),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(1024,512),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(512,128),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(128,4)
                                    )
        self.num_experts = num_experts
        self.balance_weight = balance_weight

    def forward(self, x):
        n = x.shape[0]
        device = x.device
        num_experts = self.num_experts  # 固定专家数为4

        ####################################################################
        # 修改点1：重新计算理论分配基数
        # 保证每个专家至少获得1/5的数据量（向上取整）
        min_per_expert = (n + 5) // 6  # ceil(n/5)
        remaining = n - min_per_expert * num_experts

        # 处理极小样本数的边界情况
        while remaining < 0 and min_per_expert > 0:
            min_per_expert -= 1
            remaining = n - min_per_expert * num_experts

        # 剩余数据按门控动态分配
        target_counts = [min_per_expert] * num_experts
        
        ####################################################################
        # 门控计算（保持不变）
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=1)
        expert_choice = torch.argmax(probs, dim=1)
        
        ####################################################################
        # 修改点2：动态平衡调整策略
        # 第一阶段：确保最低分配
        allocation_pool = [[] for _ in range(num_experts)]
        for idx, expert in enumerate(expert_choice):
            allocation_pool[expert].append(idx)
            
        # 强制满足最低分配要求
        for expert in range(num_experts):
            current = len(allocation_pool[expert])
            if current >= min_per_expert:
                continue
                
            # 需要补充的样本数
            need = min_per_expert - current
            candidates = []
            
            # 从其他专家处收集可转移样本
            for other_expert in range(num_experts):
                if other_expert == expert:
                    continue
                if len(allocation_pool[other_expert]) > min_per_expert:
                    # 收集其他专家超过最低分配的样本
                    overflow = len(allocation_pool[other_expert]) - min_per_expert
                    pool = allocation_pool[other_expert]
                    # 选择概率最低的样本
                    other_probs = probs[pool, other_expert]
                    sorted_indices = torch.argsort(other_probs)[:overflow]
                    candidates.extend([(pool[i], other_expert) for i in sorted_indices])
            
            # 按样本对新专家的适配度排序
            candidate_scores = [probs[idx, expert] for idx, _ in candidates]
            sorted_candidates = sorted(zip(candidates, candidate_scores),
                                    key=lambda x: x[1], reverse=True)
            
            # 转移最佳适配样本
            for (idx, src_expert), _ in sorted_candidates[:need]:
                allocation_pool[src_expert].remove(idx)
                allocation_pool[expert].append(idx)
                need -= 1
                if need == 0:
                    break

        ####################################################################
        # 第二阶段：动态分配剩余容量
        # 计算剩余可用容量
        capacities = [len(pool) - min_per_expert for pool in allocation_pool]
        total_remaining = sum(max(c,0) for c in capacities)
        
        # 需要分配的剩余样本（原始门控结果中未被强制分配的）
        remaining_indices = []
        for expert in range(num_experts):
            if len(allocation_pool[expert]) > min_per_expert:
                keep = allocation_pool[expert][:min_per_expert]
                remaining = allocation_pool[expert][min_per_expert:]
                allocation_pool[expert] = keep
                remaining_indices.extend(remaining)
        
        # 按门控概率重新分配剩余样本
        for idx in remaining_indices:
            # 排除已经满足最低分配的专家
            valid_experts = [e for e in range(num_experts)
                            if len(allocation_pool[e]) < min_per_expert + capacities[e]]
            if not valid_experts:
                valid_experts = list(range(num_experts))
                
            # 选择概率最高的可用专家
            scores = [(e, probs[idx, e]) for e in valid_experts]
            chosen_expert = max(scores, key=lambda x: x[1])[0]
            allocation_pool[chosen_expert].append(idx)
        
        ####################################################################
        # 最终验证与输出（保持不变）
        expert_data = []
        expert_indices = []
        total_allocated = 0
        for i in range(num_experts):
            indices = torch.tensor(allocation_pool[i], device=device)
            expert_indices.append(indices)
            expert_data.append(x[indices])
            total_allocated += len(indices)
        
        assert total_allocated == n, f"分配错误：{total_allocated}/{n}"
        assert torch.all(torch.bincount(torch.cat(expert_indices)) == 1), "重复分配"

        ####################################################################
        # 损失函数（保持之前设计）
        counts = torch.tensor([len(pool) for pool in allocation_pool], 
                            dtype=torch.float32, device=device)
        mean_gate_scores = torch.zeros(num_experts, device=device)
        for i in range(num_experts):
            pool = allocation_pool[i]
            if pool:
                mean_gate_scores[i] = probs[pool, i].mean()
        
        distribution_loss = (counts * mean_gate_scores).sum() / num_experts

        return expert_data, distribution_loss, expert_indices


class UniformMoERouter5(nn.Module):
    def __init__(self, input_dim=2048, num_experts=5, balance_weight=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, 1024),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(1024,512),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(512,128),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(128,5)
                                    )
        self.num_experts = num_experts
        self.balance_weight = balance_weight

    def forward(self, x):
        n = x.shape[0]
        device = x.device
        num_experts = self.num_experts  # 专家数改为5

        # 计算理论分配基数（更新为1/8保证分配）
        min_per_expert = (n + 7) // 8  # ceil(n/8)
        remaining = n - min_per_expert * num_experts
        
        # 处理极小样本情况
        while remaining < 0 and min_per_expert > 0:
            min_per_expert -= 1
            remaining = n - min_per_expert * num_experts

        # 门控计算（保持不变）
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=1)
        expert_choice = torch.argmax(probs, dim=1)

        # 动态平衡调整（适配5专家）-------------------------------------------------
        allocation_pool = [[] for _ in range(num_experts)]
        
        # 初始分配
        for idx, expert in enumerate(expert_choice):
            allocation_pool[expert].append(idx)

        # 第一阶段：强制最低分配（每个专家至少min_per_expert）
        for expert in range(num_experts):
            current = len(allocation_pool[expert])
            if current >= min_per_expert:
                continue
                
            need = min_per_expert - current
            candidates = []
            
            # 从其他专家收集可转移样本
            for other_expert in range(num_experts):
                if other_expert == expert:
                    continue
                overflow = len(allocation_pool[other_expert]) - min_per_expert
                if overflow > 0:
                    pool = allocation_pool[other_expert]
                    other_probs = probs[pool, other_expert]
                    sorted_indices = torch.argsort(other_probs)[:overflow]
                    candidates.extend([(pool[i], other_expert) for i in sorted_indices])
            
            # 按适配度排序候选样本
            candidate_scores = [probs[idx, expert] for idx, _ in candidates]
            sorted_candidates = sorted(zip(candidates, candidate_scores),
                                    key=lambda x: x[1], reverse=True)
            
            # 转移样本
            for (idx, src_expert), _ in sorted_candidates[:need]:
                allocation_pool[src_expert].remove(idx)
                allocation_pool[expert].append(idx)
                need -= 1
                if need == 0:
                    break

        # 第二阶段：分配剩余样本（占3/8）
        remaining_indices = []
        # 收集所有超额样本
        for expert in range(num_experts):
            if len(allocation_pool[expert]) > min_per_expert:
                keep = allocation_pool[expert][:min_per_expert]
                remaining = allocation_pool[expert][min_per_expert:]
                allocation_pool[expert] = keep
                remaining_indices.extend(remaining)
        
        # 按门控概率重新分配剩余样本
        for idx in remaining_indices:
            valid_experts = [e for e in range(num_experts)
                            if len(allocation_pool[e]) < (min_per_expert + remaining // num_experts + 1)]
            if not valid_experts:
                valid_experts = list(range(num_experts))
            
            # 选择概率最高的可用专家
            scores = [(e, probs[idx, e]) for e in valid_experts]
            chosen_expert = max(scores, key=lambda x: x[1])[0]
            allocation_pool[chosen_expert].append(idx)

        # 最终输出处理（保持结构）------------------------------------------------
        expert_data = []
        expert_indices = []
        total_allocated = 0
        
        for i in range(num_experts):
            indices = torch.tensor(allocation_pool[i], device=device, dtype=torch.long)
            expert_indices.append(indices)
            expert_data.append(x[indices])
            total_allocated += len(indices)

        # 验证增强
        assert total_allocated == n, f"Total allocated mismatch: {total_allocated}/{n}"
        all_indices = torch.cat(expert_indices)
        assert torch.unique(all_indices).numel() == n, "Duplicate indices found"
        assert torch.all(all_indices.sort().values == torch.arange(n, device=device)), "Index discontinuity"

        # 损失计算（适配5专家）
        counts = torch.tensor([len(pool) for pool in allocation_pool], 
                            dtype=torch.float32, device=device)
        mean_gate_scores = torch.zeros(num_experts, device=device)
        for i in range(num_experts):
            if expert_indices[i].numel() > 0:
                mean_gate_scores[i] = probs[expert_indices[i], i].mean()
        
        # distribution_loss = (counts + mean_gate_scores).sum() / num_experts
        distribution_loss = (counts * mean_gate_scores).sum() / num_experts

        return expert_data, distribution_loss, expert_indices

class UniformMoERouter3_tucker(nn.Module):
    def __init__(self, input_dim=2048, num_experts=3, balance_weight=0.1,
    d_rel = 512, d_graph = 512, ):
        super().__init__()

        self.num_experts = num_experts
        self.balance_weight = balance_weight

        self.d_r = d_rel + d_graph
        self.temperature = 0.1
        self.proj_rel = nn.Sequential(nn.Linear(input_dim, 1024),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(1024,d_rel),
                                    nn.ReLU(True))

        self.proj_graph = nn.Sequential(nn.Linear(input_dim, 1024),
                                    nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Linear(1024,d_graph),
                                    nn.ReLU(True))

        self.register_buffer(
            "routing_embeds",
            torch.randn(num_experts, self.d_r))
        nn.init.orthogonal_(self.routing_embeds)

        self.expert_emb = nn.Embedding(3,512)
        nn.init.kaiming_normal_(self.expert_emb.weight)
        self.relation_embeddings = nn.Parameter(torch.Tensor(512, 512))
        nn.init.kaiming_normal_(self.relation_embeddings)
        self.core_tensor = nn.Parameter(torch.Tensor(512, 512, 512))
        nn.init.normal_(self.core_tensor, mean=0, std=0.01)

    def forward(self, x, y):

        n = x.shape[0]
        device = x.device
        num_experts = self.num_experts  # 固定专家数为3

        ####################################################################
        # 修改点1：重新计算理论分配基数
        # 保证每个专家至少获得1/5的数据量（向上取整）
        min_per_expert = (n + 4) // 5  # ceil(n/5)
        remaining = n - min_per_expert * num_experts

        # 处理极小样本数的边界情况
        while remaining < 0 and min_per_expert > 0:
            min_per_expert -= 1
            remaining = n - min_per_expert * num_experts

        # 剩余数据按门控动态分配
        target_counts = [min_per_expert] * num_experts
        
        ####################################################################
        # 门控计算（保持不变）
        q_rel = self.proj_rel(x)
        q_graph = self.proj_graph(y)
        q = torch.cat([q_rel, q_graph], dim=-1)

        scores = torch.einsum("bd,nd->bn", q, self.routing_embeds)  # [batch, num_experts]
        scores = scores / self.temperature
        probs = F.softmax(scores, dim=-1)  # [batch, num_experts]



        # logits = self.gate(x)
        # probs = torch.softmax(logits, dim=1)
        expert_choice = torch.argmax(probs, dim=1)
        
        ####################################################################
        # 修改点2：动态平衡调整策略
        # 第一阶段：确保最低分配
        allocation_pool = [[] for _ in range(num_experts)]
        for idx, expert in enumerate(expert_choice):
            allocation_pool[expert].append(idx)
            
        # 强制满足最低分配要求
        for expert in range(num_experts):
            current = len(allocation_pool[expert])
            if current >= min_per_expert:
                continue
                
            # 需要补充的样本数
            need = min_per_expert - current
            candidates = []
            
            # 从其他专家处收集可转移样本
            for other_expert in range(num_experts):
                if other_expert == expert:
                    continue
                if len(allocation_pool[other_expert]) > min_per_expert:
                    # 收集其他专家超过最低分配的样本
                    overflow = len(allocation_pool[other_expert]) - min_per_expert
                    pool = allocation_pool[other_expert]
                    # 选择概率最低的样本
                    other_probs = probs[pool, other_expert]
                    sorted_indices = torch.argsort(other_probs)[:overflow]
                    candidates.extend([(pool[i], other_expert) for i in sorted_indices])
            
            # 按样本对新专家的适配度排序
            candidate_scores = [probs[idx, expert] for idx, _ in candidates]
            sorted_candidates = sorted(zip(candidates, candidate_scores),
                                    key=lambda x: x[1], reverse=True)
            
            # 转移最佳适配样本
            for (idx, src_expert), _ in sorted_candidates[:need]:
                allocation_pool[src_expert].remove(idx)
                allocation_pool[expert].append(idx)
                need -= 1
                if need == 0:
                    break

        ####################################################################
        # 第二阶段：动态分配剩余容量
        # 计算剩余可用容量
        capacities = [len(pool) - min_per_expert for pool in allocation_pool]
        total_remaining = sum(max(c,0) for c in capacities)
        
        # 需要分配的剩余样本（原始门控结果中未被强制分配的）
        remaining_indices = []
        for expert in range(num_experts):
            if len(allocation_pool[expert]) > min_per_expert:
                keep = allocation_pool[expert][:min_per_expert]
                remaining = allocation_pool[expert][min_per_expert:]
                allocation_pool[expert] = keep
                remaining_indices.extend(remaining)
        
        # 按门控概率重新分配剩余样本
        for idx in remaining_indices:
            # 排除已经满足最低分配的专家
            valid_experts = [e for e in range(num_experts)
                            if len(allocation_pool[e]) < min_per_expert + capacities[e]]
            if not valid_experts:
                valid_experts = list(range(num_experts))
                
            # 选择概率最高的可用专家
            scores = [(e, probs[idx, e]) for e in valid_experts]
            chosen_expert = max(scores, key=lambda x: x[1])[0]
            allocation_pool[chosen_expert].append(idx)
        
        ####################################################################
        # 最终验证与输出（保持不变）
        expert_data = []
        expert_indices = []
        total_allocated = 0
        for i in range(num_experts):
            indices = torch.tensor(allocation_pool[i], device=device)
            expert_indices.append(indices)
            expert_data.append(x[indices])
            total_allocated += len(indices)
        
        assert total_allocated == n, f"分配错误：{total_allocated}/{n}"
        assert torch.all(torch.bincount(torch.cat(expert_indices)) == 1), "重复分配"

        ####################################################################
        # 损失函数（保持之前设计）
        counts = torch.tensor([len(pool) for pool in allocation_pool], 
                            dtype=torch.float32, device=device)
        mean_gate_scores = torch.zeros(num_experts, device=device)
        for i in range(num_experts):
            pool = allocation_pool[i]
            if pool:
                mean_gate_scores[i] = probs[pool, i].mean()
        
        distribution_loss = (counts * mean_gate_scores).sum() / num_experts
        importance_loss = self.add_importance_loss(probs)
        loss = distribution_loss + importance_loss

        return expert_data, loss ,expert_indices

    def add_importance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        计算专家重要性损失（需在训练循环中调用）
        Args:
            routing_weights: 路由权重 [batch_size, num_experts]
        Returns:
            loss: 重要性损失标量
        """
        expert_importance = routing_weights.sum(dim=0)  # [num_experts]
        cv = expert_importance.std() / (expert_importance.mean() + 1e-6)
        loss = torch.clamp(cv, min=0.1)  # 仅当cv < 0.1时惩罚
        return loss
    


import torch
import torch.nn as nn
import torch.nn.functional as F

class TuckerRouting(nn.Module):
    def __init__(self, input_dim, num_experts, d_e, d_r):
        super(TuckerRouting, self).__init__()
        self.d_e = d_e  # 实体嵌入维度
        self.d_r = d_r  # 关系嵌入维度
        
        # 初始化实体嵌入矩阵 E ∈ R^{N_e × d_e}
        self.entity_embeddings = nn.Embedding(num_experts, d_e)
        # 初始化关系嵌入矩阵 R ∈ R^{d_r × d_r}
        self.relation_embeddings = nn.Parameter(torch.Tensor(d_r, d_r))
        
        # 初始化核心张量 W ∈ R^{d_e × d_r × d_e}
        self.core_tensor = nn.Parameter(torch.Tensor(d_e, d_r, d_e))
        
        # 初始化投影层
        self.proj_rel = nn.Linear(input_dim, d_e)
        self.proj_graph = nn.Linear(input_dim, d_r)
        
        # 温度参数
        self.temperature = nn.Parameter(torch.Tensor([1.0]))
        
        # 初始化权重
        nn.init.kaiming_normal_(self.entity_embeddings.weight)
        nn.init.kaiming_normal_(self.relation_embeddings)
        nn.init.normal_(self.core_tensor, mean=0, std=0.01)
        nn.init.xavier_normal_(self.proj_rel.weight)
        nn.init.xavier_normal_(self.proj_graph.weight)

    def forward(self, x, y):
        """
        Tucker Routing前向传播函数
        Args:
            x: 输入张量1，形状为 (batch_size, input_dim)
            y: 输入张量2，形状为 (batch_size, input_dim)
        Returns:
            概率分布，形状为 (batch_size, num_experts)
        """
        # 通过投影层得到关系和图的嵌入向量
        q_rel = self.proj_rel(x)  # (batch_size, d_e)
        q_graph = self.proj_graph(y)  # (batch_size, d_r)
        
        # 沿着最后一个维度连接两个嵌入向量
        # q = torch.cat([q_rel, q_graph], dim=-1)  # (batch_size, d_e + d_r)
        
        # 核心张量与关系嵌入的第二模态积，得到形状为 (batch_size, d_e, d_e)
        Wr = torch.einsum('ijk,br->bikj', self.core_tensor, q_graph)
        Wr = Wr.squeeze(1)  # (batch_size, d_e, d_e)
        
        # 核心张量与关系嵌入和实体嵌入的交互操作
        # 首先对关系嵌入进行变换
        transformed_rel = torch.einsum('bi,ij->bj', q_graph, self.relation_embeddings)  # (batch_size, d_r)
        
        # 然后与核心张量进行交互
        interaction = torch.einsum('ijk,bj->bik', self.core_tensor, transformed_rel)  # (batch_size, d_e, d_e)
        
        # 与主体实体嵌入进行交互
        Wr_es = torch.bmm(interaction, q_rel.unsqueeze(2)).squeeze(2)  # (batch_size, d_e)
        
        # 与所有客体实体嵌入计算点积，得到预测分数
        all_entities = self.entity_embeddings.weight  # (num_experts, d_e)
        scores = torch.einsum('bd,nd->bn', Wr_es, all_entities)  # (batch_size, num_experts)
        scores = scores / self.temperature
        
        # 计算概率分布
        probs = F.softmax(scores, dim=-1)  # (batch_size, num_experts)
        
        return probs

# 示例使用
if __name__ == "__main__":
    # 参数设置
    input_dim = 256
    num_experts = 10
    d_e = 128
    d_r = 64
    
    # 创建模型实例
    model = TuckerRouting(input_dim, num_experts, d_e, d_r)
    
    # 随机生成输入数据
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, input_dim)
    
    # 前向传播
    probs = model(x, y)
    
    print("Probs shape:", probs.shape)  # 输出概率分布的形状，应为 torch.Size([32, 10])