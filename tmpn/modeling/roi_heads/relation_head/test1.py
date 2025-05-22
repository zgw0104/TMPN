import torch
import torch.nn as nn
from hetsgg.modeling.utils import cat

class UniformMoERouter(nn.Module):
    def __init__(self, input_dim=2048, num_experts=3, balance_weight=0.1):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
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

# 验证测试
def test_uniform_shuffle():
    torch.manual_seed(42)
    data = torch.randn(856, 2048)
    print("原始数据特征均値:", data.mean(dim=1)[:4].tolist())
    
    router = UniformMoERouter()
    expert_data, loss, indices = router(data)
    
    print("\n专家分配数量:", [len(idx) for idx in indices])
    print("唯一索引验证:", torch.unique(torch.cat(indices)).size(0) == 856)
    
    # # 数据重建验证
    # reconstructed = torch.zeros_like(data)
    # for i, idx in enumerate(indices):
    #     reconstructed[idx] = expert_data[i]
    # print("数据一致性:", torch.equal(data, reconstructed))
    g1,g2,g3 = expert_data[0], expert_data[1], expert_data[2]
    idx = cat([indices[0], indices[1], indices[2]])

    a = cat([g1,g2,g3], dim=0)
    a = a[torch.argsort(idx)]

    print(torch.equal(data, a))
test_uniform_shuffle()