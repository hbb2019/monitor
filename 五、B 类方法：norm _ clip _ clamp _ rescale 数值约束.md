本章讨论 **B 类方法**：通过对 gradient、activation、hidden state、logits、attention scores、weight、optimizer update、loss / reward 等对象施加 **归一化、裁剪、截断、缩放、平滑上界或投影约束**，从而降低 loss spike、gradient explosion、activation explosion、logits explosion、NaN / Inf、optimizer update 失控等风险。它沿用全文的三分类框架：B 类方法的核心功能是**数值约束、止血、缓解和保险丝**，而不是直接解释异常为什么发生。

需要强调的是，B 类方法不应被简单贬低为“无用的治标”。在大模型训练中，它们通常承担三类工程角色：

1. **保险丝**：当异常 batch、局部 outlier、fp16 overflow、optimizer update spike 出现时，防止单次异常污染整个训练状态。
2. **稳定化约束**：在训练 recipe 中持续限制某类信号的动态范围，例如 gradient norm clipping、update-to-weight ratio cap、logit soft-capping。
3. **诊断辅助工具**：通过观察 clip rate、saturation rate、overflow rate、clamp ratio 等指标，反向定位哪里正在失控。

但是，B 类方法的风险也很明确：如果长期高频触发，它们可能已经不再是“保护机制”，而是在悄悄重写训练动力学，甚至掩盖数据、mask、LR、optimizer state、precision 或 distributed consistency 的真实问题。

---

### 5.1 B 类方法的定义与边界
B 类方法可以定义为：

**B 类方法是对训练过程中某个张量、统计量或更新量施加显式数值边界，使其无法超过某个幅度、范数、比例、分布范围或动态区间的稳定化手段。其主要目标是防止异常数值扩散，而不是直接修复导致异常产生的根因。**

典型作用对象包括：

| 作用对象 | 典型约束形式 | 代表方法 |
| --- | --- | --- |
| gradient | norm clipping、value clipping、layer-wise clipping、adaptive clipping | global grad norm clipping、per-layer clipping、AGC |
| activation / hidden state | clamp、percentile clipping、RMS constraint、outlier clipping | hidden state clamp、MLP activation clipping |
| logits | hard clipping、temperature scaling、soft cap | output logits clipping、logit soft-capping |
| attention scores | score scaling、softcap、clamp、stable softmax | attention logits clamp、QK score scaling |
| Q/K vectors | normalization、learned scale、score range control | QK Norm |
| weight / parameter | max-norm、weight clipping、spectral constraint | weight clipping、spectral norm |
| optimizer update | update norm clipping、update-to-weight cap、trust ratio cap | Adam update clipping、U/W ratio constraint |
| loss / reward | loss clipping、reward clipping、reward normalization | RLHF reward clipping、PPO-style clipping |
| gradient / update direction | normalization、direction-only update、trust-region style scaling | gradient normalization、update normalization |


B 类方法与 A 类根因修复的区别在于：

| 判断问题 | 更像 B 类 | 更像 A 类 |
| --- | --- | --- |
| 是否解释异常来源 | 不解释，只限制异常扩散 | 找到具体来源，例如 mask 错、LR 错、state 错 |
| 是否改变数值幅度 | 是主要作用 | 可能不是主要作用 |
| 是否依赖阈值或缩放系数 | 通常依赖 | 不一定依赖 |
| 是否可能高频触发 | 可能 | 修复后应低频或不触发 |
| 是否可能掩盖问题 | 风险较高 | 风险较低 |
| 是否直接改变训练目标或梯度方向 | 经常会 | 一般不是目标 |


---

### 5.2 使用 B 类方法前必须明确的四个问题
在使用任何 norm / clip / clamp / rescale 之前，应先明确四件事。否则这些方法很容易从“保险丝”变成“遮羞布”。

#### 1. 被约束的对象是什么？
不同对象的约束含义完全不同。

| 被约束对象 | 约束含义 | 工程风险 |
| --- | --- | --- |
| gradient | 限制 backward signal | 可能改变梯度方向，削弱困难样本学习 |
| activation | 限制 forward signal | 可能丢失表达信息，掩盖上游尺度失控 |
| logits | 限制输出分布尖锐程度 | 可能影响概率校准、CE 梯度和生成分布 |
| attention scores | 限制 attention sharpness | 可能改变 head 的选择性和 long-context 行为 |
| weight | 限制参数空间 | 可能降低容量或破坏优化几何 |
| optimizer update | 限制实际参数变化 | 可能改变优化器算法行为 |
| reward / loss | 限制训练信号 | 可能改变 RLHF / preference learning 的目标 |


#### 2. 约束发生在 forward、backward 还是 update 阶段？
| 阶段 | 典型方法 | 主要影响 |
| --- | --- | --- |
| forward | activation clamp、attention score clamp、logit softcap | 改变模型函数本身 |
| loss 计算 | loss clipping、reward clipping、label smoothing 类约束 | 改变训练目标或样本权重 |
| backward | gradient clipping、gradient normalization | 改变反向传播信号 |
| optimizer | update clipping、trust ratio、update-to-weight cap | 改变参数更新动力学 |
| precision / numerical kernel | stable softmax、loss scaling、fp32 accumulation | 改变数值实现或动态范围 |


一般来说：

+ **forward 约束**更容易改变模型表达；
+ **loss / reward 约束**更容易改变优化目标；
+ **gradient 约束**更容易改变学习方向或学习速度；
+ **update 约束**更接近实际参数安全边界；
+ **precision 约束**若用于修复 overflow / underflow，通常副作用较小。

#### 3. 约束是偶发触发还是长期高频触发？
这是判断 B 类方法是否掩盖真实问题的关键。

| 触发模式 | 判断 |
| --- | --- |
| 偶发触发，例如 <1% step | 更像保险丝 |
| warmup 早期触发，随后下降 | 可能是正常稳定化机制 |
| 某些 batch 触发 | 更像数据 / packing / mask 问题 |
| 某些 layer / head / expert 长期触发 | 更像局部架构、初始化、LR group 或 MoE routing 问题 |
| 几乎每个 step 都触发 | 该约束已经成为主训练机制，必须重新审查 LR、初始化、optimizer 和数据 |
| clip 后 loss 稳定但 valid loss 变差 | 可能压制了有效学习信号 |
| clip 前后 gradient cosine 很低 | 训练方向被显著改写 |


#### 4. 约束是否改变训练目标、梯度方向或 update 尺度？
建议记录三个辅助指标：

| 指标 | 含义 | 异常判断 |
| --- | --- | --- |
| clip rate / clamp ratio | 有多少 step、层、token、元素触发约束 | 长期高频触发危险 |
| pre/post norm ratio | 约束前后范数比例 | 比例长期很小，说明约束很强 |
| direction cosine | 约束前后梯度或 update 方向夹角 | cosine 低说明方向被重写 |


例如 global grad norm clipping 不只是“把梯度变小”。当某个 batch 的梯度中存在极大 outlier 时，全局缩放会同时压小所有参数的梯度，导致正常层的学习也被削弱。因此，global clipping 的副作用通常不是局部的，而是全模型的。

---

### 5.3 B 类方法总览表
| 方法名称 | 作用对象 | 约束方式 | 主要缓解什么异常 | 为什么属于 B 类 | 是否解决根因 | 典型适用场景 | 推荐监控指标 | 可能副作用 | 是否可能掩盖真实问题 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Global grad norm clipping | 全模型梯度 | 当 ||g|| 超过阈值时整体缩放 | gradient explosion、loss spike 后的梯度扩散 | 只限制梯度总幅度，不解释梯度为什么变大 | 通常否 | 预训练、SFT、RLHF 中作为通用保险丝 | global grad norm、clip rate、pre/post norm ratio、grad cosine | 压小所有层梯度；可能削弱困难样本学习 | 高 |
| Per-layer grad clipping | 单层梯度 | 每层独立裁剪梯度范数 | 某层梯度异常放大 | 限制局部梯度幅度 | 通常否 | 某些层长期 grad norm 偏大 | layer grad norm、layer clip rate、layer update ratio | 破坏层间相对学习率 | 高 |
| Per-parameter grad clipping | 单个参数张量梯度 | 对每个 tensor 独立裁剪 | 某些矩阵梯度异常 | 限制 tensor-level 梯度 | 通常否 | embedding、lm_head、expert 参数异常 | tensor grad norm、param update ratio | 不同参数学习率被隐式重写 | 高 |
| Gradient value clipping | 梯度元素 | 按元素 clamp 到上下界 | 极端梯度 outlier | 直接截断元素值 | 否 | 调试或极端数值止血 | grad max、p99、outlier ratio | 严重改变梯度方向；破坏稀疏信号 | 很高 |
| Adaptive Gradient Clipping, AGC | 梯度相对参数范数 | 限制 ||g|| / ||w|| | 小权重层相对梯度过大 | 约束相对尺度 | 不一定 | 参数尺度差异大、局部层爆炸 | grad/weight ratio、AGC trigger rate | 对低范数参数敏感；可能限制新层学习 | 中高 |
| Gradient centralization / normalization | 梯度分布或方向 | 减均值、归一化或重缩放 | 梯度尺度不稳、方向噪声 | 改写梯度统计 | 不直接 | 特殊优化 recipe、实验性训练 | grad mean/std、grad cosine、loss trend | 可能改变优化器假设 | 中 |
| Hidden state clamp | residual / hidden state | 对 hidden value hard clamp | residual stream outlier、activation NaN | 直接限制 forward activation | 通常否 | 紧急防 NaN、调试定位 | hidden RMS、p99、max、clamp ratio | 丢失表达信息；掩盖上游失控 | 很高 |
| Residual stream norm cap | residual stream | 限制 residual RMS 或 norm | 深层 residual 累积过大 | 压制残差流幅度 | 通常否 | 极深模型早期不稳定 | layer-wise residual RMS、cap rate | 改变残差动力学；影响深层表示 | 高 |
| MLP activation clipping | FFN 中间激活 | 对 gate/up/down 激活裁剪 | MLP outlier、SwiGLU gate 过大 | 限制 MLP 数值 | 通常否 | MLP output 导致 residual RMS spike | gate/up/down RMS、p99、kurtosis | 削弱非线性表达；影响稀疏特征 | 高 |
| Activation percentile clipping | activation 分布尾部 | 按 p99 / p99.9 截断 | heavy-tail activation outlier | 限制尾部分布 | 通常否 | outlier 激活导致 fp16 overflow | activation histogram、tail ratio | 改变尾部特征；阈值依赖 batch | 高 |
| Output logits hard clipping | lm_head logits | 将 logits clamp 到固定范围 | CE overflow、softmax 饱和 | 事后压制输出 | 否 | 临时防 NaN，调试 logits 爆炸 | logit max/min、entropy、clamp ratio | 影响概率校准；改变 CE 梯度 | 高 |
| Logit soft-capping | lm_head logits | 使用平滑上界，例如 tanh 类 softcap | 极端 logits、softmax 过尖 | 限制输出幅度，但比 hard clip 平滑 | 不一定 | 长训中 logits 逐渐极端化 | logit RMS、max、entropy、saturation rate | 改变尾部概率；影响 calibration | 中 |
| Temperature scaling | logits | logits 除以温度或乘缩放因子 | 分布过尖或过平 | 调整输出尺度 | 通常否 | 推理校准、训练中控制熵 | entropy、top-1 prob、CE、temperature | 温度不当会过平滑或过尖锐 | 中 |
| Attention score clamp | QK scores | 对 softmax 前 scores hard clamp | attention softmax overflow、one-hot collapse | 直接限制 attention logits | 通常否 | attention score 爆炸时止血 | score p99/max、attention entropy、clamp ratio | 破坏 head 的选择性 | 高 |
| Attention score softcap | QK scores | 对 scores 施加平滑上界 | attention entropy collapse、score outlier | 平滑限制 score 幅度 | 不一定 | 长上下文、高 LR、RoPE scaling 敏感场景 | score histogram、entropy、saturation rate | attention 变钝或局部模式受损 | 中 |
| QK score scaling | QK scores | 调整 (1/\sqrt{d})、temperature 或 learned scale | attention scores 过大或过小 | 改变 score 尺度 | 部分 | attention entropy 异常 | Q/K norm、score std、entropy | 可能影响 attention sharpness | 中 |
| Stable softmax / logsumexp | softmax 输入 | 减 max、fp32 softmax、logsumexp | exp overflow、NaN | 通过数值重写避免 overflow | 对数值实现根因有效，但形式上是 rescale | softmax/CE 数值不稳定 | pre-softmax max、finite ratio、overflow count | 通常副作用小，可能增加开销 | 低 |
| QK Norm 作为数值约束 | Q/K 向量 | normalize Q/K 后再用 scale 控制 scores | QK norm 增长、attention score 爆炸 | 限制 score 生成范围 | 部分 | attention entropy collapse、score p99 过大 | Q norm、K norm、score p99、head entropy | 改变 attention 几何 | 中 |
| LayerNorm 临时插入 | hidden / residual | 对 hidden 做均值方差归一化 | hidden scale drift | 强制归一化激活 | 不一定 | 某层激活长期失控时的结构性改动 | LN input RMS、output RMS、gamma norm | 改变模型架构，需要重调 recipe | 中 |
| RMSNorm 临时插入 | hidden / residual | 按 RMS 归一化，不减均值 | hidden RMS drift | 强制控制 RMS | 不一定 | decoder-only LLM 中尺度稳定化 | RMSNorm input RMS、gamma norm | 改变 residual geometry | 中 |
| Weight clipping | 参数值 | 对 W 元素 hard clamp | weight value explosion | 直接限制参数值 | 通常否 | 特殊约束模型、紧急调试 | weight max、weight histogram、clip ratio | 严重破坏优化几何 | 高 |
| Weight max-norm | 参数向量 | 限制某维度或某行权重范数 | embedding row norm、lm_head row norm 过大 | 投影到范数球 | 不一定 | embedding/lm_head 局部 row 爆炸 | row norm、max row、projection rate | 限制表示容量 | 中 |
| Spectral norm | 权重矩阵 | 限制最大奇异值 | operator norm 过大、Lipschitz 失控 | 约束线性映射放大能力 | 部分 | 对稳定性强约束的模型或判别器 | spectral norm、power iteration residual | 计算开销高，限制表达 | 中 |
| Weight normalization | 参数重参数化 | 将方向和尺度分离 | 参数尺度不稳 | 通过重参数化控制尺度 | 不一定 | 小模型或特定模块 | weight scale、grad scale | 与 optimizer 交互复杂 | 中 |
| Optimizer update norm clipping | optimizer update | 对 (\Delta W) 范数裁剪 | Adam / Adafactor update explosion | 限制实际更新 | 不直接 | grad 不大但 update 大 | update norm、pre/post update ratio | 改变优化器行为 | 中高 |
| Update-to-weight ratio cap | 参数相对更新 | 限制 (|\Delta W| / |W|) | 小权重层相对更新过大 | 约束相对步长 | 不一定 | 大模型预训练、resume 风险控制 | U/W ratio、layer-wise cap rate | 小参数层学习变慢 | 中 |
| Trust ratio clipping | layer update / weight | 限制 trust ratio 上下界 | 层间 update 尺度不均 | 约束层级更新比例 | 不直接 | LAMB/LARS 类优化设置 | trust ratio histogram、layer update | 可能隐式改写层级 LR | 中 |
| Loss clipping | per-sample loss 或 batch loss | 对 loss 上界裁剪 | 极端样本 loss、噪声 label | 改变 loss 贡献 | 通常否 | 噪声数据、RL、preference learning | per-sample loss、loss clip rate | 改变训练目标；困难样本被忽略 | 高 |
| Reward clipping | reward | 将 reward 截断到范围内 | reward outlier、reward model 极端输出 | 改变 RL 信号 | 通常否 | RLHF / RL 中防 reward spike | reward mean/std/max、clip rate、KL | 掩盖 reward model bug；导致 reward hacking | 高 |
| Advantage clipping | advantage | 对 advantage 裁剪或归一化 | PPO / RLHF update spike | 限制 policy gradient 信号 | 不直接 | RLHF PPO 训练 | advantage std/max、policy KL | 改变样本权重；降低有效学习 | 中高 |
| Gradient normalization | gradient | 使用单位范数或固定范数梯度 | gradient scale 不稳定 | 去除梯度幅度信息 | 否 | 特殊优化器或实验性稳定化 | grad norm、direction cosine | 丢失曲率和难度信息 | 中 |
| Update normalization | update | 对 update 方向归一化 | update scale 不稳定 | 改写实际更新 | 否或部分 | trust-region 风格训练 | update norm、U/W ratio、cosine | 可能变成另一种优化算法 | 中 |
| EMA / moving average smoothing of metrics for triggers | 监控触发信号 | 用 EMA 平滑后触发 clip 或 rescale | 短期 spike 误触发 | 平滑触发机制 | 不解决根因 | 自动化保护系统 | raw/EMA 指标、trigger latency | 可能延迟保护 | 中 |
| Bad batch skip / skip step | batch / update | 遇到 NaN/overflow 跳过 step | 防止坏 batch 污染参数 | 跳过异常，不解释异常 | 否 | 混合精度 overflow、偶发坏 batch | skip rate、bad batch id、overflow count | 丢弃训练信号；掩盖数据问题 | 高 |


---

### 5.4 Gradient clipping：最常用但也最容易被误用的 B 类方法
#### 5.4.1 Global grad norm clipping
Global grad norm clipping 通常形式为：

[  
g \leftarrow g \cdot \min \left(1,\frac{\tau}{|g|_2 + \epsilon}\right)  
]

其中 (\tau) 是 clip threshold。

它的特点是：

| 维度 | 说明 |
| --- | --- |
| 优点 | 简单、鲁棒、实现成本低；能防止单次 backward 产生极端更新 |
| 缺点 | 全局缩放会影响所有层，不区分异常来源 |
| 适合 | 作为大模型预训练、SFT、RLHF 的默认保险丝 |
| 不适合 | 用来长期掩盖 LR 错误、数据异常、mask bug、optimizer state 错误 |


推荐监控：

| 指标 | 判断标准 |
| --- | --- |
| global grad norm pre-clip | 是否出现快速增长或 heavy-tail |
| global grad norm post-clip | 是否长期贴着阈值 |
| clip rate | 偶发正常；长期高频危险 |
| pre/post norm ratio | 长期过小说明 clip 过强 |
| grad cosine before/after clipping | cosine 低说明方向被改变 |
| layer-wise contribution to global norm | 判断是谁触发了 global clipping |


典型误区：

| 误区 | 问题 |
| --- | --- |
| 只记录 post-clip grad norm | 会看不到真实爆炸程度 |
| 只设置 clip，不记录 clip rate | 无法判断是否掩盖根因 |
| global clip 阈值过低 | 训练被长期限速 |
| global clip 阈值过高 | 保险丝形同虚设 |
| loss spike 后才记录梯度 | 已错过前置信号 |


#### 5.4.2 Per-layer / per-parameter grad clipping
Per-layer clipping 适合当异常长期集中在某些层时使用。例如：

+ 某个 attention block 的 grad norm 长期高于相邻层；
+ embedding 或 lm_head 梯度偶发极端；
+ MoE 某些 expert 梯度经常爆。

但它的副作用是会隐式改变层间学习率。原本某层梯度大，可能是因为该层确实承担了更多学习信号；直接 per-layer clip 可能削弱有效学习。因此应同时记录：

[  
\frac{|\Delta W_l|}{|W_l|}  
]

而不是只看：

[  
|g_l|  
]

推荐判断标准：

| 现象 | 处理建议 |
| --- | --- |
| 只有某层偶发触发 clipping | 可以作为保险丝 |
| 某层持续触发 clipping | 查该层输入 RMS、attention score、MLP output、LR group |
| 多层同时触发 clipping | 优先查 LR、loss spike、precision、optimizer state |
| clipping 后该层 update ratio 仍大 | 问题可能在 optimizer state 或参数范数过小 |
| clipping 后 valid loss 下降变慢 | clipping 可能过强 |


#### 5.4.3 Adaptive Gradient Clipping
Adaptive Gradient Clipping 的思想是限制梯度相对参数的比例：

[  
\frac{|g|}{|w|}  
]

它比固定阈值 clipping 更关心“相对更新风险”。适合参数尺度差异大的模型，但在 LLM 中使用时要注意：

| 风险点 | 说明 |
| --- | --- |
| 小范数参数容易被过度裁剪 | 例如新初始化层、norm 参数、某些 expert 参数 |
| embedding row 差异大 | row-wise 范数差异可能导致不均匀裁剪 |
| 与 Adam 自适应更新叠加 | grad/weight ratio 正常不代表 update/weight ratio 正常 |
| 不能替代 update-level 监控 | Adam 的实际 update 仍可能异常 |


因此，AGC 更适合与 update-to-weight ratio 监控配合，而不是单独使用。

---

### 5.5 Activation clipping / clamp：对 forward signal 的硬保护
Activation clipping 通常作用在 hidden state、residual stream、MLP activation 或 attention output 上。它比 gradient clipping 更敏感，因为它直接改变模型的 forward function。

#### 5.5.1 Hidden state clamp
Hidden state clamp 的典型形式是：

[  
h \leftarrow \text{clamp}(h, -c, c)  
]

它可以防止 hidden outlier 继续传递到后续层、lm_head 或 softmax，但副作用明显：

| 风险 | 具体表现 |
| --- | --- |
| 表示信息丢失 | 大幅值 feature 可能携带有用语义 |
| 梯度路径改变 | clamp 区域梯度可能被截断或变形 |
| 掩盖上游问题 | residual scaling、attention score、MLP gate、LR 问题可能仍在 |
| 分布不连续 | histogram 出现明显硬边界 |


推荐只在以下场景短期使用：

| 场景 | 使用方式 |
| --- | --- |
| 定位 NaN 来源 | 临时插入 clamp，观察 NaN 是否消失 |
| 防止一次异常污染训练 | 与 bad batch logging 联动 |
| 某模块输出出现极端 outlier | 短期保护，同时 hook 上游模块 |
| 混合精度 overflow 调试 | 对比 bf16/fp32 后再决定是否保留 |


长期使用 hidden clamp 时，必须监控：

+ clamp ratio；
+ hidden RMS pre/post clamp；
+ p99 / max 是否长期贴近阈值；
+ clamp token 是否集中在某些数据源、长度、位置或 token 类型；
+ valid loss、downstream eval、generation entropy 是否受损。

#### 5.5.2 MLP activation clipping
MLP activation clipping 主要用于 gate/up activation 或 activation function 输出异常的场景。对 SwiGLU / GeGLU 等门控结构尤其要谨慎，因为 gate outlier 可能导致 down_proj 输出突然放大。

推荐监控：

| 模块 | 指标 |
| --- | --- |
| gate_proj output | RMS、p99、max、kurtosis |
| up_proj output | RMS、p99、max |
| activation 后输出 | outlier ratio、histogram |
| down_proj output | residual contribution ratio |
| MLP grad norm | 是否与 activation outlier 同步 |


判断标准：

| 现象 | 更可能是什么问题 |
| --- | --- |
| gate activation 极端但输入 RMS 正常 | gate_proj 初始化、LR group 或激活函数问题 |
| gate/up 都正常但 down_proj 输出大 | down_proj scale 或权重范数问题 |
| 某数据源触发 MLP outlier | 数据分布或特殊 token 问题 |
| 所有层 MLP 同时 outlier | LR、precision、normalization 或 optimizer 问题 |


Activation clipping 可以止血，但如果 MLP activation 长期高频触发，优先检查初始化、residual scaling、activation function 实现、norm 位置和 LR，而不是继续调大 clamp 阈值。

---

### 5.6 Logit clipping / logit soft-capping：控制输出分布极端化
Logits 是 loss 计算前最后一道信号。Logits 极端化通常表现为：

+ logit max 持续增大；
+ softmax entropy 降低；
+ top-1 probability 过早接近 1；
+ CE loss 出现 overflow 或 NaN；
+ 某些 token 的 logit row 长期成为 outlier；
+ generation 过早变得低熵、重复或模式坍缩。

#### 5.6.1 Output logits hard clipping
Hard clipping 是最直接但副作用最大的方式：

[  
z \leftarrow \text{clamp}(z, -c, c)  
]

它适合作为临时防 NaN 工具，不适合作为长期训练 recipe 的默认选择。原因是它会直接改变 cross entropy 的梯度结构，尤其是在正确 token logit 或强负类 logit 被截断时。

推荐用途：

| 用途 | 是否推荐长期保留 |
| --- | --- |
| 定位 logits 是否导致 CE NaN | 可以短期使用 |
| 防止一次 bad batch 导致 NaN | 可以作为保险丝 |
| 长期控制输出分布 | 不优先推荐 |
| 替代修复 hidden/lm_head 尺度问题 | 不推荐 |


#### 5.6.2 Logit soft-capping
Logit soft-capping 用平滑函数限制 logits，例如：

[  
z' = c \cdot \tanh(z / c)  
]

相较 hard clipping，它的优点是连续、平滑，对梯度的破坏更小。但它仍然会改变输出分布和概率校准。

推荐监控：

| 指标 | 含义 |
| --- | --- |
| logit RMS | 整体输出尺度 |
| logit max / p99 | 尾部极端化 |
| saturation rate | 有多少 logits 进入 softcap 饱和区 |
| entropy | 输出分布尖锐程度 |
| top-1 probability | 是否过早过度自信 |
| calibration / NLL | 概率校准是否恶化 |
| vocab outlier row | 是否特定 token 异常 |


判断标准：

| 现象 | 结论 |
| --- | --- |
| saturation rate 很低，仅偶发 | softcap 更像保险丝 |
| saturation rate 长期高 | softcap 正在重写输出分布 |
| softcap 后 CE 稳定但 eval 变差 | 可能压制了有效 logit margin |
| softcap 后 logit RMS 仍持续增长 | 上游 hidden 或 lm_head 尺度仍未解决 |
| 某些 token 经常被 softcap | 检查 tokenizer、数据模板、lm_head row norm |


#### 5.6.3 Temperature scaling
Temperature scaling 通过调整 logits 尺度影响 softmax entropy：

[  
p_i = \text{softmax}(z_i / T)  
]

在训练中，temperature scaling 应谨慎使用。它不是修复 logits 爆炸的根因，而是改变 loss landscape 的尖锐程度。

| 温度变化 | 效果 | 风险 |
| --- | --- | --- |
| (T > 1) | 分布变平，梯度更分散 | 降低模型区分能力 |
| (T < 1) | 分布变尖，强化高 logit token | 加剧过度自信和 entropy collapse |


训练中如果需要长期 temperature scaling，应同时检查 hidden RMS、lm_head norm、label smoothing、数据难度和 learning rate。

---

### 5.7 Attention score clipping / scaling：控制 softmax 前的危险区域
Attention scores 是 Transformer 中最容易出现数值极端化的对象之一：

[  
S = \frac{QK^\top}{\sqrt{d}}  
]

当 Q/K norm 增长、RoPE scaling 不合理、position id 错误、attention mask 错误或 LR 过大时，attention scores 可能快速放大，导致 attention entropy collapse、softmax overflow 或某些 head 近似 one-hot。

#### 5.7.1 Attention score hard clamp
形式：

[  
S \leftarrow \text{clamp}(S, -c, c)  
]

它能直接防止 softmax 前 scores 过大，但副作用是改变 attention pattern。

适合短期使用：

+ 定位 NaN 是否来自 attention scores；
+ 对比 clamp 前后 entropy 是否恢复；
+ 防止某个 step 的 score outlier 污染训练。

不适合长期依赖：

+ 如果某些 head 长期被 clamp，说明 attention score 生成机制本身有问题；
+ 如果长上下文位置更容易触发 clamp，应检查 RoPE scaling、position id、mask 和 sequence length curriculum；
+ 如果所有 head 同时触发 clamp，应检查 LR、normalization、precision、optimizer update。

#### 5.7.2 QK score scaling
QK score scaling 可以通过调整 attention temperature、scale factor 或 learned scale 控制 score 标准差。

推荐监控：

| 指标 | 说明 |
| --- | --- |
| Q norm / K norm | 判断 score 放大来自向量范数 |
| score mean / std / p99 / max | 判断 score 分布是否失控 |
| attention entropy | 判断 softmax 是否坍缩 |
| max attention prob | 判断 head 是否 one-hot |
| per-head entropy histogram | 判断是否局部 head 异常 |
| illegal attention ratio | 判断 mask 是否错误 |
| score by position distance | 判断长上下文位置机制是否异常 |


判断标准：

| 现象 | 可能处理 |
| --- | --- |
| Q/K norm 增大导致 score 增大 | QK Norm、weight scale、LR 调整 |
| score 正常但 entropy 异常 | mask、temperature 或 softmax 实现 |
| 长位置 score 异常 | RoPE scaling、position id、long-context recipe |
| 某 head score 极端 | head-specific Wq/Wk、初始化或数据触发 |
| score overflow | fp32 softmax、stable softmax、score softcap |


#### 5.7.3 Stable softmax
Stable softmax 通常是低副作用的数值修复：

[  
\text{softmax}(x)_i = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}  
]

这类方法虽然也属于 rescale，但它与普通 clip 的性质不同：它没有改变 softmax 的数学结果，只是改变数值实现方式。因此，当问题是 exp overflow 或 logsumexp 不稳定时，stable softmax 更接近“数值实现修复”，掩盖根因的风险较低。

但如果 softmax 已经稳定实现，attention score 仍然极端，则不能把 stable softmax 当作完整解决方案，还需要继续检查 Q/K norm、mask、position 和 LR。

---

### 5.8 QK Norm：数值约束还是架构稳定性设计？
QK Norm 处在 B 类和 C 类边界上。本章从 B 类角度讨论它：当 QK Norm 的主要作用是限制 Q/K 范数、控制 attention score 动态范围、防止 softmax 饱和时，它可以被视为一种数值约束。

#### 什么时候 QK Norm 更像 B 类数值约束？
| 场景 | 判断 |
| --- | --- |
| attention score p99 / max 快速增长 | QK Norm 用来限制 score 生成幅度 |
| attention entropy collapse | QK Norm 用来防止 softmax 过尖 |
| 某些 head Q/K norm 长期 outlier | QK Norm 用来压制局部 head 尺度 |
| 长上下文训练中 score 随 position 增长 | QK Norm 用来稳定长位置 attention |
| 加 QK Norm 后主要变化是 score range 下降 | 更像数值约束 |


#### 什么时候 QK Norm 更像架构稳定性设计？
| 场景 | 判断 |
| --- | --- |
| 从模型设计一开始就采用 | 是 attention 几何的一部分 |
| 与 learned scale、初始化、LR recipe 配套 | 不是临时止血 |
| 训练全程低触发、低饱和 | 是结构性稳定机制 |
| 改善 attention entropy 同时不损害 eval | 更像合理架构选择 |
| 与长上下文、多模态或高分辨率输入 recipe 联动 | 更像模型设计 |


#### 使用 QK Norm 时建议记录
| 指标 | 用途 |
| --- | --- |
| Q norm / K norm before norm | 判断原始尺度是否失控 |
| learned QK scale | 判断模型是否通过 scale 补偿 |
| score p99 / max | 判断 score 是否被稳定控制 |
| entropy per head | 判断 attention 是否过尖或过平 |
| head diversity | 判断 QK Norm 是否压平 head 差异 |
| long-context eval | 判断是否损害远距离依赖 |


如果 QK Norm 使 attention entropy 恢复正常，但 learned scale 持续增大、score saturation rate 仍然高，则说明它可能只是在延迟问题，而没有完全解决 score 失控的原因。

---

### 5.9 LayerNorm / RMSNorm：不应简单视为“退烧药”
LayerNorm 和 RMSNorm 也处在 B 类与 C 类边界上。若它们是模型架构的一部分，例如 Transformer block 的标准组件，则不应被简单视为“治标不治本”。但如果是在训练已经不稳定后临时插入、额外包裹或强行约束某层 hidden state，它们就具有 B 类数值约束属性。

#### LayerNorm / RMSNorm 作为 B 类方法的场景
| 场景 | 说明 |
| --- | --- |
| 某层 hidden RMS 长期异常，临时增加 norm | 用归一化压制 activation scale |
| 在模块输出后额外加 norm 防 NaN | 主要是止血 |
| 用 norm 替代查找上游 outlier 来源 | 可能掩盖根因 |
| 插入 norm 后训练稳定但质量下降 | 说明改变了模型函数和优化路径 |


#### LayerNorm / RMSNorm 作为架构稳定组件的场景
| 场景 | 说明 |
| --- | --- |
| Pre-LN Transformer 默认结构 | 改善深层梯度流 |
| RMSNorm 作为 decoder-only LLM 标准 recipe | 控制 residual stream RMS |
| 与 residual scaling、初始化、LR 配套 | 属于结构性稳定设计 |
| 从训练开始就存在，而非异常后添加 | 不是临时止血 |


#### 推荐监控
| 指标 | 判断 |
| --- | --- |
| norm input RMS | 上游 residual 是否失控 |
| norm output RMS | norm 是否稳定输出尺度 |
| gamma / scale 参数 norm | 模型是否通过 gamma 补偿 |
| gamma max / p99 | 是否出现局部通道过大 |
| pre-norm vs post-norm activation | 判断 norm 是否掩盖输入异常 |
| layer-wise update ratio | norm 参数是否异常更新 |


常见误区：

| 误区 | 修正 |
| --- | --- |
| “加了 norm 就等于解决稳定性问题” | 只能说明输出尺度被控制，上游输入可能仍然失控 |
| “所有 norm 都是治标” | 架构内的 Pre-LN / RMSNorm 是稳定性设计 |
| “norm 越多越稳” | 过多 norm 会改变表达路径和优化几何 |
| “只看 norm 后输出” | 必须同时记录 norm 前输入 |


---

### 5.10 Weight norm / weight clipping / spectral norm：不要与 weight decay 混淆
Weight clipping、weight norm constraint、spectral norm 与 weight decay 有本质差异。

| 方法 | 作用方式 | 与 weight decay 的区别 |
| --- | --- | --- |
| Weight clipping | 直接把参数值截断到范围内 | hard constraint，非平滑，可能破坏优化方向 |
| Weight max-norm | 把参数向量投影到范数球 | 是投影约束，不是连续正则 |
| Spectral norm | 限制矩阵最大奇异值 | 约束 operator norm，而不是简单减小所有权重 |
| Weight decay | 在优化中惩罚或衰减权重 | 通常是软约束，影响长期参数尺度 |


#### 5.10.1 Weight clipping
Weight clipping 在 LLM 预训练中通常不作为默认方法，因为它会直接破坏参数空间结构。可用于：

+ 特殊约束模型；
+ 调试某个参数是否异常爆炸；
+ embedding row 或 lm_head row 局部失控时短期验证；
+ 避免某个极端 checkpoint 继续扩散。

推荐监控：

+ weight max / p99；
+ weight histogram；
+ clipping ratio；
+ row norm for embedding / lm_head；
+ post-clip update-to-weight ratio。

如果 weight clipping 长期触发，应优先检查 LR、weight decay、optimizer state、param group、embedding/lm_head 初始化和 tokenizer 分布。

#### 5.10.2 Weight max-norm
Max-norm 更适合用于 row-wise 参数，例如 embedding row 或 lm_head row。它限制的是每个 row 的范数，而不是每个元素。

适用场景：

| 场景 | 说明 |
| --- | --- |
| 特定 token embedding row norm 极端 | 可防止该 token 影响 logits |
| lm_head 某些 row 过大 | 可限制 vocab outlier |
| 新增 special token 训练不稳 | 可防止新 token row 过度更新 |


风险：

+ rare token 可能学习不足；
+ special token 表达能力受限；
+ tied embedding 情况下同时影响输入和输出表示。

#### 5.10.3 Spectral norm
Spectral norm 限制矩阵的最大奇异值：

[  
|W|_2  
]

它控制线性映射的最大放大能力，比普通 weight norm 更接近 operator-level 稳定约束。它可能用于需要强 Lipschitz 控制的模块，但在大规模 LLM 中要谨慎，因为：

+ 计算开销高；
+ power iteration 近似会引入额外复杂度；
+ 可能显著限制表达能力；
+ 与 tensor parallel / sharding 结合实现复杂。

因此，在 LLM 预训练中，spectral norm 更像特殊模块约束，而不是默认稳定性工具。

---

### 5.11 Optimizer update clipping / update norm constraint：比 gradient clipping 更接近真实风险
在 Adam 类优化器中，真正作用到参数上的不是原始 gradient，而是经过 m、v、epsilon、bias correction、weight decay 和 LR 处理后的 update：

[  
\Delta W = -\eta \cdot \frac{m}{\sqrt{v} + \epsilon}  
]

因此，gradient norm 正常不代表 update norm 正常。尤其在以下场景中，update-level constraint 比 gradient clipping 更有意义：

| 场景 | 原因 |
| --- | --- |
| Adam v 过小 | 小 v 会放大 effective update |
| resume 后 optimizer state 错误 | m/v 与参数不匹配 |
| 新增参数或部分参数 state 未初始化 | update 可能异常 |
| 参数范数很小 | 绝对 update 不大，但相对 update 过大 |
| 不同参数组 LR 配错 | 某组 update-to-weight ratio 异常 |
| weight decay contribution 异常 | decay 部分可能主导 update |


#### 5.11.1 Update norm clipping
约束：

[  
\Delta W \leftarrow \Delta W \cdot \min \left(1,\frac{\tau}{|\Delta W| + \epsilon}\right)  
]

推荐监控：

| 指标 | 说明 |
| --- | --- |
| update norm pre/post clip | 判断 update 失控程度 |
| update-to-weight ratio | 判断相对参数变化 |
| Adam m norm / v norm | 判断 update 放大来源 |
| effective LR | 判断 LR 与 v 的交互 |
| state finite ratio | 检查 optimizer state 是否损坏 |
| param-group update ratio | 检查 LR group 配置 |


#### 5.11.2 Update-to-weight ratio cap
约束：

[  
\frac{|\Delta W_l|}{|W_l|} \leq r  
]

它比单纯 update norm 更合理，因为不同层参数尺度不同。对大模型预训练尤其重要。

判断标准：

| 现象 | 可能原因 |
| --- | --- |
| 某层 U/W ratio 长期高 | 该层 LR、weight norm、optimizer state 或输入尺度异常 |
| U/W ratio 在 resume 后突增 | checkpoint / scheduler / optimizer state 问题 |
| embedding U/W ratio 高 | token 分布、embedding row、LR group 问题 |
| norm 参数 U/W ratio 高 | 是否应对 norm 参数使用不同 LR / WD |
| expert U/W ratio 高 | MoE routing load 或 expert optimizer state 问题 |


副作用：

+ 限制小范数层快速适应；
+ 对新初始化参数可能过强；
+ 如果 cap 太低，会造成训练慢或局部欠拟合；
+ 如果只记录 post-cap update，会掩盖真实 update 异常。

---

### 5.12 Loss clipping / reward clipping：在 RLHF 和 preference learning 中尤其危险
Loss clipping 和 reward clipping 属于强 B 类方法，因为它们直接改变训练信号。

#### 5.12.1 Loss clipping
Loss clipping 可以对 per-sample loss、per-token loss 或 batch loss 施加上界。它能降低异常样本、噪声 label、bad batch 对训练的影响，但也可能把真正困难样本的学习信号裁掉。

适用场景：

| 场景 | 使用建议 |
| --- | --- |
| 数据中存在明显噪声 label | 可短期使用，同时做数据清洗 |
| 某些样本 loss 极端且不可修复 | 可结合 sample quarantine |
| SFT 中模板错误导致异常 loss | 不应只 clip，应修模板 |
| pretraining 中 long-tail 难样本 | 不建议简单裁掉 |


推荐监控：

+ per-sample loss histogram；
+ top-k loss 样本 ID；
+ clipped sample ratio；
+ clipped token ratio；
+ clipped 样本的数据源、长度、语言、模板；
+ clip 前后 gradient norm；
+ valid loss 和 hard subset eval。

#### 5.12.2 Reward clipping
Reward clipping 在 RLHF / RL / preference learning 中常用于限制 reward outlier，但副作用更严重，因为 reward 是 policy update 的核心驱动信号。

风险包括：

| 风险 | 表现 |
| --- | --- |
| 改变优化目标 | policy 学到的是 clipped reward，不是原 reward |
| 掩盖 reward model bug | reward model 极端输出被截断后难以发现 |
| 加剧 reward hacking | policy 可能学会贴近 clipping 边界 |
| 压制高质量样本差异 | 好样本之间 reward 被压平 |
| 与 KL penalty 交互复杂 | reward 被 clip 后 KL 可能主导优化 |


推荐监控：

| 指标 | 用途 |
| --- | --- |
| raw reward mean/std/max | 判断原始 reward 是否异常 |
| clipped reward mean/std | 判断训练实际信号 |
| reward clip rate | 判断 clipping 强度 |
| policy KL | 判断 update 是否被 reward outlier 推动 |
| advantage std/max | 判断 policy gradient 是否失控 |
| win-rate / preference eval | 判断 clipping 是否损害偏好学习 |
| reward by prompt category | 判断是否某类 prompt 触发 outlier |


如果 reward clipping 长期高频触发，应优先检查 reward model calibration、prompt distribution、response length bias、KL coefficient、advantage normalization 和 rejection sampling 逻辑，而不是继续调 clipping threshold。

---

### 5.13 Gradient normalization / update normalization：不是修复根因，而是改变优化动力学
Gradient normalization 和 update normalization 通常把尺度信息移除，只保留方向或相对方向：

[  
g' = \frac{g}{|g| + \epsilon}  
]

或：

[  
\Delta W' = \frac{\Delta W}{|\Delta W| + \epsilon}  
]

它们可以降低尺度波动，但会丢失梯度范数中包含的重要信息。梯度范数本身反映了 batch 难度、loss 曲率、参数敏感度和优化器状态。如果完全归一化，模型无法区分“小错误”和“大错误”的更新强度。

适用场景：

| 场景 | 说明 |
| --- | --- |
| 实验性优化器 | 作为特定算法设计的一部分 |
| trust-region 风格训练 | 控制每步 update 尺度 |
| 极端噪声梯度环境 | 降低尺度噪声 |
| 调试 | 判断不稳定是否主要来自尺度而非方向 |


不适合：

+ 替代 LR 修复；
+ 替代 optimizer state 修复；
+ 替代数据清洗；
+ 在没有 direction cosine 和 update ratio 监控时长期使用。

推荐监控：

| 指标 | 判断 |
| --- | --- |
| raw grad norm | 原始尺度是否失控 |
| normalized grad norm | 归一化是否过强 |
| raw/update direction cosine | 方向是否保留 |
| loss decrease per update | 是否仍有效下降 |
| update-to-weight ratio | 实际步长是否合理 |
| layer-wise learning progress | 是否某些层学习停滞 |


---

### 5.14 B 类方法的选择流程
当训练出现异常但尚未定位根因时，可按以下流程选择 B 类保护机制。

#### Step 1：确定异常首先出现在哪里
| 首个异常对象 | 优先考虑的 B 类方法 |
| --- | --- |
| gradient norm | global grad norm clipping、layer-wise clipping |
| optimizer update | update clipping、update-to-weight cap |
| residual / hidden RMS | residual norm cap、activation clamp，短期使用 |
| MLP activation | MLP activation clipping、activation percentile clipping |
| logits | logit soft-capping，必要时 hard clipping |
| attention scores | score softcap、QK score scaling、QK Norm |
| softmax / CE NaN | stable softmax、fp32 softmax、logsumexp |
| reward / advantage | reward normalization、advantage clipping |
| MoE expert update | expert-level grad clipping、expert update cap |
| FP16 overflow | loss scaling、skip step、bf16/fp32 fallback |


#### Step 2：选择尽量靠近异常源头的约束
原则：

| 不推荐 | 更推荐 |
| --- | --- |
| logits 爆了就直接 clip logits | 先看 hidden RMS、lm_head norm、attention score |
| loss spike 就 clip loss | 先看 per-sample loss、label、mask、batch source |
| global grad 爆就只调 clip threshold | 同时查 layer-wise grad 和 update ratio |
| activation 爆就 clamp hidden | 找第一个异常层，区分 attention 和 MLP |
| update 爆就 clip gradient | 直接监控和约束 update |


越靠近异常源头，越不容易误伤其他正常信号。

#### Step 3：选择最小扰动形式
扰动强度从低到高大致为：

1. stable numerical implementation，例如 stable softmax、logsumexp；
2. smooth scaling，例如 temperature、softcap；
3. norm-based rescale，例如 gradient norm clipping；
4. ratio-based cap，例如 update-to-weight cap；
5. percentile clipping；
6. hard clamp；
7. per-value clipping；
8. skip step / drop batch。

一般优先选择平滑、连续、低副作用的方法，最后才使用 hard clamp 或 value clipping。

#### Step 4：设置触发日志，而不是只设置阈值
每个 B 类方法都应至少记录：

| 日志 | 用途 |
| --- | --- |
| trigger step | 什么时候触发 |
| trigger layer / head / expert | 哪里触发 |
| pre-constraint value | 约束前真实值 |
| post-constraint value | 约束后实际值 |
| trigger ratio | 触发比例 |
| source batch id | 是否与数据相关 |
| rank id | 是否与分布式相关 |
| pre/post direction cosine | 方向是否被改变 |
| eval impact | 是否影响质量 |


没有这些日志，clip / clamp 只会让训练看起来稳定，而不是让问题变得可解释。

---

### 5.15 判断 B 类方法是否掩盖真实问题
以下信号说明 B 类方法可能正在掩盖根因。

| 信号 | 具体含义 | 后续动作 |
| --- | --- | --- |
| clip rate 长期高 | 约束已经成为主训练机制 | 降 LR、查 optimizer state、查数据和 layer-wise 指标 |
| pre/post norm ratio 长期很小 | 原始信号远超阈值 | 找到触发最大贡献层或 batch |
| clamp ratio 持续上升 | upstream activation 仍在恶化 | hook 第一个异常层 |
| logits softcap saturation rate 上升 | 输出分布持续极端化 | 查 hidden RMS、lm_head norm、tokenizer |
| attention score clamp 集中在长序列 | 可能是 RoPE / position / mask 问题 | 按长度分桶 replay |
| clipping 集中在某 rank | distributed 或 rank data 问题 | 查 rank batch、checksum、all-reduce |
| clipping 集中在某 expert | MoE routing 或 expert state 问题 | 查 expert load、router logits、capacity |
| loss 稳定但 valid loss 变差 | 有效学习信号被压制 | 调低约束强度或修根因 |
| gradient cosine 低 | clipping 改变了方向 | 改用更局部或 update-level 约束 |
| 关闭 clipping 立即 NaN | clipping 不是充分修复 | 从最近稳定 checkpoint 做根因定位 |
| 阈值越调越低 | 训练越来越依赖约束 | 回查 LR、初始化、precision、data |
| 阈值越调越高 | 约束逐渐失效 | 改查根因，不要继续放宽 |


一个实用标准是：

**如果某个 B 类约束在训练初期短暂触发，随后触发率下降，它更像合理保险丝；如果触发率随 step 增加，或者集中在固定层、固定 head、固定 expert、固定 rank、固定数据源，它更像根因定位线索。**

---

### 5.16 典型工程案例
#### 案例 1：global grad clipping 让训练“不炸”，但 loss 始终不收敛
现象：

+ global grad norm 经常超过阈值；
+ clipping 后没有 NaN；
+ train loss 有下降但 valid loss 较差；
+ clip rate 在 warmup 后仍然很高。

可能原因：

+ LR 过大；
+ warmup 不足；
+ 某些 batch loss 极端；
+ optimizer state 或 param group 错；
+ global clipping 压制了正常层学习。

处理方式：

1. 记录 pre-clip grad norm，而不是只看 post-clip。
2. 分 layer 统计 grad norm contribution。
3. 检查 update-to-weight ratio。
4. 将 LR 降低 2–4 倍 replay。
5. 检查 top-loss batch 是否集中在某类数据。
6. 若降低 LR 后 clip rate 大幅下降，说明 clipping 原先掩盖了 LR/update 问题。

#### 案例 2：logit clipping 消除了 NaN，但生成质量变差
现象：

+ CE NaN 消失；
+ logits max 被限制在阈值附近；
+ output entropy 异常偏高或偏低；
+ generation 重复、校准变差或 rare token 表现异常。

可能原因：

+ hidden RMS 增大；
+ lm_head norm 增大；
+ tokenizer / special token 分布异常；
+ attention score 失控导致 final hidden 极端；
+ logit clipping 改变了 CE 梯度。

处理方式：

1. 记录 logits pre-clip 分布。
2. 检查 final hidden RMS。
3. 检查 lm_head row norm。
4. 查 top logit token 是否集中在 special token。
5. 将 hard clipping 改为 soft-capping 或修上游尺度。
6. 若 softcap saturation rate 仍高，继续查 hidden / attention / lm_head 根因。

#### 案例 3：attention score clamp 稳住训练，但长上下文能力下降
现象：

+ score clamp 后不再 NaN；
+ long-context eval 明显变差；
+ attention entropy 在长位置异常变平；
+ clamp 主要发生在长序列 batch。

可能原因：

+ RoPE scaling 不合理；
+ position id 错误；
+ long sequence packing mask 错误；
+ attention score clamp 破坏远距离 attention pattern。

处理方式：

1. 按 sequence length 分桶统计 score p99 和 entropy。
2. 检查 position id 是否连续、是否跨 packed sample 错误延续。
3. 检查 causal mask 和 packed boundary mask。
4. 对比短序列与长序列 replay。
5. 优先修 position / RoPE / mask，再考虑 QK Norm 或 score softcap。
6. 不建议长期依赖 hard clamp。

#### 案例 4：update-to-weight cap 高频触发，发现 optimizer state resume 错误
现象：

+ grad norm 正常；
+ update norm 和 U/W ratio 异常；
+ resume 后第一批 step 出现大量 update cap；
+ Adam v 部分参数接近 0 或 state step 不连续。

可能原因：

+ optimizer state 未正确加载；
+ ZeRO/FSDP shard state 错位；
+ scheduler step 与 optimizer step 不一致；
+ 新增参数 state 未初始化。

处理方式：

1. 检查 checkpoint 中 optimizer state shape、dtype、step。
2. 对比 resume 前后同 batch 的 update。
3. 打印 per-param-group LR 和 state step。
4. 回滚到正确 checkpoint 或重建 optimizer state。
5. update cap 只能短期防止损坏参数，不能替代 state 修复。

#### 案例 5：reward clipping 稳定 RLHF，但 policy 学到边界行为
现象：

+ reward clipping 后 PPO loss 稳定；
+ policy KL 有时异常；
+ reward 分布大量贴近 clipping 上界；
+ 人评或 preference eval 没有提升。

可能原因：

+ reward model calibration 差；
+ reward clipping 改变了优化目标；
+ policy 学会 exploit clipping boundary；
+ advantage normalization / KL coefficient 不合适。

处理方式：

1. 同时记录 raw reward 和 clipped reward。
2. 查看 reward by prompt category 和 response length。
3. 监控 clip rate 与 KL 的相关性。
4. 检查 reward model 是否对长度、格式、模板有偏置。
5. 调整 reward normalization、KL coefficient 或 reward model，而不是只改 clipping threshold。

---

### 5.17 B 类方法的使用原则
#### 原则 1：先记录 pre-constraint，再记录 post-constraint
只记录 post-clip/post-clamp 会掩盖真实异常。所有 B 类方法都应记录约束前后的值。

| 必须记录 | 示例 |
| --- | --- |
| pre value | pre-clip grad norm、pre-clamp activation max |
| post value | post-clip grad norm、post-softcap logits |
| trigger ratio | clip rate、clamp ratio、saturation rate |
| location | layer、head、expert、rank、batch |
| direction change | gradient/update cosine |
| quality impact | valid loss、eval、generation quality |


#### 原则 2：阈值应绑定统计分布，而不是拍脑袋设置
阈值可以来自：

+ warmup 稳定区间的 p95 / p99；
+ 小模型稳定 run 的对应统计；
+ 同规模模型历史 run；
+ layer-wise baseline；
+ batch length / token count normalized value；
+ update-to-weight ratio 的历史稳定范围。

不推荐只用单个固定阈值处理所有层、所有参数、所有训练阶段。不同层、不同模块、不同数据阶段的尺度可能不同。

#### 原则 3：区分“偶发保护”和“长期 recipe”
| 类型 | 使用方式 |
| --- | --- |
| 偶发保护 | 允许低频触发，重点记录异常来源 |
| 长期 recipe | 需要验证不会损害收敛、eval、校准和生成质量 |
| 调试工具 | 使用后应关闭或替换成根因修复 |
| 自动化安全机制 | 需要 trigger logs 和 alert policy |


#### 原则 4：优先使用平滑约束，谨慎使用 hard clamp
一般优先级：

[  
\text{stable implementation} > \text{smooth softcap} > \text{norm rescale} > \text{ratio cap} > \text{hard clamp} > \text{value clipping}  
]

例如：

+ logits 极端：优先 soft-capping，而不是 hard clipping；
+ attention score 极端：优先 QK scaling / softcap / QK Norm，而不是 hard clamp；
+ update 过大：优先 update-to-weight cap，而不是单纯 per-value gradient clipping；
+ softmax overflow：优先 stable softmax / fp32 softmax，而不是直接 clamp scores。

#### 原则 5：B 类方法必须反哺根因定位
每一次 clipping / clamping 都应产生定位线索：

| 触发模式 | 根因定位方向 |
| --- | --- |
| 按 batch 集中 | 数据、label、mask、packing |
| 按 layer 集中 | 初始化、架构、LR group、模块实现 |
| 按 head 集中 | QK norm、attention score、RoPE、mask |
| 按 expert 集中 | MoE routing、capacity、expert state |
| 按 rank 集中 | distributed consistency、rank data |
| 按 token 集中 | tokenizer、special token、template |
| 按 sequence length 集中 | long-context、position、packing |


如果 B 类方法没有产生任何可解释日志，它只能防崩，不能帮助训练系统变得更可靠。

---

### 5.18 小结
B 类方法的价值不在于“治本”，而在于：

1. **防止异常数值扩散**：避免一次 bad batch、overflow、局部 outlier 或 update spike 污染整个模型。
2. **提供工程安全边界**：让大规模训练在不可完全预测的数据和系统噪声下可恢复。
3. **暴露定位线索**：clip rate、saturation rate、clamp ratio、pre/post norm ratio、direction cosine 可以帮助定位真正根因。
4. **稳定训练 recipe**：在合理监控和低触发率下，某些约束可以成为长期 recipe 的一部分。

但 B 类方法的使用底线是：

**任何 norm / clip / clamp / rescale 都必须同时回答三个问题：约束了什么、触发了多少、改变了什么。**

如果一个约束长期高频触发、显著改变梯度方向、导致 valid loss 或 eval 变差，或者集中在特定 layer / head / expert / rank / batch，那么它不是最终解决方案，而是根因定位入口。

