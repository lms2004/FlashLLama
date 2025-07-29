import torch

# ✅ 常量定义
NEG_INF = -1e10     # ❗用于初始化 m（最大值），模拟 -∞
EPSILON = 1e-10     # 🛡️ 防止除以 0
P_DROP = 0.2        # 🎯 Dropout 概率

# ✅ 模拟输入参数
Q_LEN, K_LEN = 6, 6       # 🧠 Q 与 K/V 序列长度
Q_BLOCK_SIZE = 3          # 📦 Q 分块大小 Br
KV_BLOCK_SIZE = 3         # 📦 K/V 分块大小 Bc
Tr = Q_LEN // Q_BLOCK_SIZE  # 🔁 Q 分为 Tr 块
Tc = K_LEN // KV_BLOCK_SIZE # 🔁 K 分为 Tc 块

# ✅ 输入张量：Q, K, V
Q = torch.randn(1, 1, Q_LEN, 4, requires_grad=True)
K = torch.randn(1, 1, K_LEN, 4, requires_grad=True)
V = torch.randn(1, 1, K_LEN, 4, requires_grad=True)

# ✅ 初始化输出和统计量
O = torch.zeros_like(Q, requires_grad=True)            # 🔧 最终输出
l = torch.zeros(Q.shape[:-1])[..., None]               # 📊 softmax 分母
m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF      # 🧱 softmax 最大值

# ✅ Step 4: 分块
Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)

# ✅ Step 5: 分块输出与中间变量
O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

# ✅ Step 6: 遍历每个 KV 块（Tc 次）
for j in range(Tc):
    Kj = K_BLOCKS[j]
    Vj = V_BLOCKS[j]

    # ✅ Step 8: 遍历每个 Q 块（Tr 次）
    for i in range(Tr):
        Qi = Q_BLOCKS[i]
        Oi = O_BLOCKS[i]
        li = l_BLOCKS[i]
        mi = m_BLOCKS[i]

        # ✅ Step 10: 点积计算 S_ij = Q_i K_j^T
        S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi, Kj)

        # ✅ Step 11: Softmax Mask（可选）
        mask = S_ij.ge(0.5)
        S_ij = torch.masked_fill(S_ij, mask, value=0)  # 可模拟 causal mask 或 attention mask

        # ✅ Step 12: Safe Softmax 计算
        m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)            # 最大值
        P_ij = torch.exp(S_ij - m_block_ij)                               # 平移后指数
        l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON    # 分母 ε 避免除 0
        P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)   # 注意力加权值

        # ✅ Step 13: 累积最大值 & 分母
        mi_new = torch.maximum(m_block_ij, mi)
        li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

        # ✅ Step 14: Dropout 应用于注意力加权输出
        dropout = torch.nn.Dropout(p=P_DROP)
        P_ij_Vj = dropout(P_ij_Vj)

        # ✅ Step 15: 累积输出
        O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi \
                      + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj

        print(f'🧠 Attention Block Q{i} × K{j} → O{i} shape: {O_BLOCKS[i].shape}')
        print('O[0] sample:\n', O_BLOCKS[0])
        print('O[1] sample:\n', O_BLOCKS[1])
        print('\n')

        # ✅ Step 16: 写回更新后的 l 与 m
        l_BLOCKS[i] = li_new
        m_BLOCKS[i] = mi_new

# ✅ Step 17: 拼接最终输出
O = torch.cat(O_BLOCKS, dim=2)
l = torch.cat(l_BLOCKS, dim=2)
m = torch.cat(m_BLOCKS, dim=2)
