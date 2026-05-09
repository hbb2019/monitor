## 三、分类标准：A / B / C
本章的作用是给全文后续方法列表提供一个统一的判别口径：当训练中出现 loss spike、gradient explosion、activation explosion、logits explosion、attention entropy collapse、NaN / Inf、MoE expert collapse 或训练整体发散时，应该判断当前采取的手段到底是在**修复根因**、**压制数值**，还是处于二者之间的**稳定性设计 / 优化约束 / 参数化调整**。本章沿用全文提出的 A / B / C 三类方法体系，并进一步细化其工程判别标准。

需要强调的是，A / B / C 的分类对象不是单个方法名，而是一个完整的“使用实例”：

[  
\text{分类对象} = \text{方法} + \text{触发原因} + \text{证据链} + \text{作用对象} + \text{验证结果}  
]

因此，同一个方法在不同上下文中可能属于不同类别。例如 QK Norm 可以是架构层面的稳定性设计，也可以是 attention score 爆炸后的数值约束；logit soft-capping 可以是模型 recipe 的一部分，也可以是 lm_head / hidden state 尺度异常时的临时保险丝；loss scaling 通常是混合精度数值范围修复，但不能用来掩盖模型本身已经发散的问题。

---

### 3.1 分类的基本原则
A / B / C 分类不应只看“用了什么技术”，而应看四个问题：

1. **异常是否有明确机制解释**  
例如是否已经定位到 attention mask 错误、position id 错误、LR resume 错误、Adam state 损坏、FP16 overflow、MoE router collapse、某数据源样本权重错误等。
2. **干预是否作用在异常产生链条的上游**  
如果修复的是导致异常的输入、代码、状态、参数化或优化器机制，通常更接近 A 类。  
如果只是在异常值已经产生后对 gradient、activation、logits、attention scores、update 进行裁剪或缩放，通常更接近 B 类。
3. **修复后是否能在较少额外硬约束下恢复稳定**  
如果修复后不再依赖新增 hard clip / hard clamp，且 loss、gradient、activation、update-to-weight ratio、rank-wise consistency 等指标回归正常，更接近 A 类。  
如果训练稳定主要来自长期高频触发 clip / clamp，则更接近 B 类。
4. **该方法是否本来就是模型 recipe、优化器定义或数值精度机制的一部分**  
LayerNorm、RMSNorm、Pre-LN、residual scaling、μP、loss scaling、optimizer update constraint 等方法不能简单视为“治标不治本”。它们是否属于 A / B / C，要看使用位置、使用动机、触发方式和验证结果。

---

### 3.2 A 类：找到“预兆爆炸”的根因并解决
**定义**：A 类方法不是简单压制异常数值，而是定位并消除导致 loss spike、gradient explosion、activation explosion、logits explosion、attention entropy collapse、NaN / Inf 或训练发散的源头机制。

A 类的关键不是“训练不炸了”，而是能够回答：

为什么之前会炸？  
哪个对象最早异常？  
异常从哪里传播到哪里？  
修复后是否不再依赖额外硬 clip / clamp？  
修复是否能在 replay、ablation、rank-wise 检查或小规模复现实验中被验证？

#### 3.2.1 A 类判断标准
| 判断维度 | A 类应满足的标准 | 工程验证方式 | 反例 |
| --- | --- | --- | --- |
| 根因明确性 | 能指出具体机制，而不是只说“梯度太大” | 定位到数据、mask、LR、optimizer state、precision、distributed、MoE routing 或代码实现问题 | 只看到 grad norm 大，然后加 clip |
| 时序证据 | 根因指标先于 loss spike / NaN 出现 | 查看异常 step 前 N 个 step 的指标趋势 | loss 已经 NaN 后才观察到 activation NaN |
| 空间定位 | 能定位到具体 batch、layer、head、expert、rank、param group 或 kernel | layer-wise / head-wise / rank-wise / expert-wise logging | 全局指标异常但无定位 |
| 可复现性 | 同 checkpoint + 同 batch / 同配置可复现异常 | replay bad batch；固定 seed；关闭随机增强 | 异常不可复现且无进一步定位 |
| 因果干预 | 修复该机制后异常消失或显著减弱 | ablation：修 mask、降 LR、恢复 optimizer state、切 bf16 等 | 只改变 clip 阈值后不炸 |
| 对硬约束依赖 | 修复后 clip rate、overflow rate、NaN retry rate 回归低水平 | 监控 clip rate、overflow count、skip step | clip rate 长期接近 100% |
| 质量恢复 | 训练稳定且 validation loss / downstream metric 不异常恶化 | 对比修复前后 loss 曲线、eval、校准指标 | loss 稳定但模型性能明显变差 |


#### 3.2.2 A 类常见根因域
A 类通常落在以下根因域中：

| 根因域 | 典型根因 | 对应异常 |
| --- | --- | --- |
| 数据 | 脏数据、错误 label、极长样本、样本权重错误、数据源比例突变 | loss spike、grad spike、batch-specific NaN |
| sequence / mask | padding 未 mask、causal mask 错、packed boundary 串样本、position id 错 | attention entropy 异常、logits 异常、loss 异常 |
| LR / schedule | LR 过大、warmup 不足、resume 后 scheduler step 错、param group LR 配错 | update norm 过大、loss 发散 |
| optimizer | Adam m/v 损坏、epsilon 不合适、weight decay 配错、bias correction 错 | grad 正常但 update 爆炸 |
| 初始化 / 参数尺度 | embedding / lm_head 尺度异常、residual branch 初始化不合理 | logits explosion、activation explosion |
| 架构 | Post-LN 深层不稳、attention score 过大、RoPE scaling 错、residual path 失衡 | deep layer residual RMS 上升、attention collapse |
| 精度 / kernel | FP16 overflow、softmax overflow、fused kernel 数值错误 | Inf / NaN、loss scale 频繁下降 |
| 分布式 | rank 间 batch 不一致、all-reduce 错、ZeRO / FSDP state 错、resume shard 错 | rank-wise divergence、checkpoint 后发散 |
| MoE | router logits 爆炸、expert load 不均、capacity drop 过高、aux loss 权重异常 | expert collapse、per-expert grad spike |
| 代码实现 | label shift 错、ignore_index 错、loss reduction 错、tokenizer/vocab 不一致 | loss 异常、无法 overfit tiny batch |


#### 3.2.3 A 类典型案例
| 案例 | 异常表现 | 根因证据 | A 类修复 |
| --- | --- | --- | --- |
| packed sequence mask 串样本 | loss 偶发 spike，某些 head attention entropy 极低 | replay 同一 batch 必现；可视化 mask 发现 token attend 到另一个样本 | 修 packed boundary mask；增加 mask 单元测试 |
| resume 后 LR 错误 | 从 checkpoint 恢复后几十 step 内 loss 发散 | scheduler step 与 global step 不一致；param group LR 比预期高 | 恢复 scheduler state；修 checkpoint metadata |
| Adam state 丢失 | grad norm 未明显异常，但 update-to-weight ratio 突然升高 | Adam m/v step count 重置；v 过小 | 正确加载 optimizer state；必要时回滚 checkpoint |
| FP16 attention softmax overflow | 某层 attention score max 极大后出现 NaN | bf16 / fp32 replay 不炸；FP16 fused kernel 出 Inf | 改 bf16 或 FP32 softmax；修 kernel / autocast |
| MoE expert collapse | 少数 expert grad norm 和 update norm 极大 | expert load 分布极不均；router entropy collapse | 调整 router z-loss、capacity factor、aux loss 权重或 router precision |


---

### 3.3 B 类：使用 norm / clip / clamp / rescale 进行数值约束
**定义**：B 类方法主要通过对 gradient、activation、hidden state、logits、attention scores、weight、optimizer update、loss、reward 等对象施加上界、归一化、重缩放或裁剪，防止异常数值继续放大或传播。

B 类方法可以是必要的训练保险丝，但通常不能单独说明异常为什么发生。其核心作用是：

[  
\text{限制幅度} \quad \text{而不是} \quad \text{解释机制}  
]

因此，B 类方法的正确使用方式不是“加了就结束”，而是要同时记录触发率、触发位置、触发前后的方向变化和质量影响。

#### 3.3.1 B 类判断标准
| 判断维度 | B 类典型特征 | 工程观察指标 |
| --- | --- | --- |
| 作用对象 | 已经产生的梯度、激活、logits、attention scores、update、loss、reward | grad norm、activation RMS、logit max、score p99、update norm |
| 作用方式 | clamp、clip、normalize、rescale、cap、threshold | clip threshold、saturation rate、clamp ratio |
| 机制解释 | 不直接解释异常源头 | 只能回答“哪里数值大”，不能回答“为什么大” |
| 触发方式 | 通常由阈值触发 | clip rate、overflow count、skip step |
| 对训练动力学影响 | 可能改变梯度方向、更新幅度、概率分布或优化目标 | gradient cosine、update cosine、valid loss、calibration |
| 典型角色 | 止血、保险丝、缓冲带、数值安全边界 | spike 后训练是否继续、NaN 是否减少 |
| 风险 | 掩盖数据 / 代码 / 优化器 / 分布式根因 | 长期高频触发、局部模块持续被 clip |


#### 3.3.2 B 类常见方法
| 方法 | 作用对象 | 为什么通常归为 B 类 | 主要风险 |
| --- | --- | --- | --- |
| global grad norm clipping | 全局梯度 | 限制梯度范数，通常不解释梯度为何异常 | 高频触发会改变有效学习率 |
| per-layer grad clipping | 层级梯度 | 限制局部层梯度 | 破坏层间梯度比例 |
| per-parameter value clipping | 单个梯度元素 | 对极端 outlier 硬截断 | 严重改变梯度方向 |
| activation clamp | hidden / MLP activation | 压制 activation outlier | 掩盖 residual / MLP / attention 上游问题 |
| logits clipping | 输出 logits | 防止 softmax / CE 数值极端 | 改变概率分布和校准 |
| attention score clamp | QK scores | 防止 attention softmax 饱和 | 破坏 attention head 选择性 |
| update norm clipping | optimizer update | 限制实际参数更新 | 掩盖 LR / optimizer state 问题 |
| update-to-weight cap | 相对更新量 | 防止小权重层被过大更新 | 可能让部分层学习不足 |
| loss clipping | per-sample loss 或总 loss | 限制异常样本影响 | 改变训练目标 |
| reward clipping | reward / advantage | 限制 RLHF / RL 中极端 reward | 掩盖 reward model 或 preference data 问题 |


#### 3.3.3 B 类方法的正确使用条件
B 类方法适合在以下场景中使用：

| 场景 | 使用 B 类方法的合理性 | 必须同步做的监控 |
| --- | --- | --- |
| 训练早期需要防止偶发极端梯度摧毁 checkpoint | 合理，作为保险丝 | grad clip rate、loss spike source |
| 大规模预训练中偶发坏 batch 不可完全避免 | 合理，但应结合 bad batch attribution | per-batch loss、sample source、length |
| FP16 / fused kernel 中偶发 overflow | 可临时 skip step / 降 loss scale | overflow count、loss scale trend |
| RLHF reward 分布重尾 | reward clipping / normalization 可用 | reward hist、KL、policy entropy |
| MoE 某些 expert 偶发 grad spike | expert-level clipping 可临时保护 | expert load、expert update ratio |
| 根因尚未定位但训练资源昂贵 | 可短期使用，避免全局崩溃 | 触发位置、频率、质量影响 |


B 类方法不适合被当作最终结论，特别是在以下情况下：

| 危险信号 | 含义 |
| --- | --- |
| clip rate 长期高于预期且没有下降趋势 | clip 已成为主要训练机制 |
| 某一层、某个 expert、某个 rank 长期被 clip | 局部根因未解决 |
| clip 后 loss 稳定但 validation loss 变差 | 数值稳定掩盖了优化质量下降 |
| gradient cosine / update cosine 大幅下降 | clip 改写了训练方向 |
| hard clamp 后 histogram 出现明显截断 | 分布真实形状被隐藏 |
| 关闭 clip 立刻 NaN | 仍存在未修复根因 |


---

### 3.4 C 类：混合型或需要结合场景判断的方法
**定义**：C 类方法不能简单归为 A 或 B。它们可能是架构稳定性设计、优化器机制、参数化策略、数值精度机制、正则化方法或 curriculum 策略；也可能在某些场景下被临时用作数值止血。

C 类的关键判断点是：

这个方法是在改变训练系统的合理参数化和信号传播方式，还是在异常已经发生后强行压制数值？

#### 3.4.1 C 类判断标准
| 判断维度 | 更像 A 类 / 结构性修复 | 更像 B 类 / 数值止血 |
| --- | --- | --- |
| 使用时机 | 训练 recipe 设计阶段就引入 | 训练炸了之后临时加 |
| 机制解释 | 有明确动力学或数值机制 | 只因某指标太大而压制 |
| 触发方式 | 持续作为模型/优化器定义的一部分 | 阈值触发或异常触发 |
| 是否高频截断 | 不依赖 hard threshold 高频触发 | 频繁 clip / clamp / saturation |
| 对指标影响 | residual RMS、update ratio、entropy 等回归合理区间 | 只把 max 值压下去 |
| 对质量影响 | loss 和 eval 同时改善或不恶化 | 训练稳定但质量下降 |
| 可迁移性 | 在不同 batch / seed / scale 下稳定有效 | 只在当前异常下有效 |
| 是否解释根因 | 能解释为什么原设置不稳 | 不能解释，只是避免崩溃 |


#### 3.4.2 典型 C 类方法的判别
| 方法 | 为什么属于 C 类 | 更像根因修复 / 稳定性设计的场景 | 更像数值止血的场景 | 推荐判别指标 |
| --- | --- | --- | --- | --- |
| LayerNorm | 既是 Transformer 架构组件，也有归一化作用 | 作为模型 block 标准设计，用于稳定层间信号传播 | 训练炸后临时在异常位置插 LN | layer-wise RMS、grad norm、valid loss |
| RMSNorm | 是 LLM 常用归一化组件，也限制 hidden RMS | 从 recipe 开始使用，配合初始化和 LR | 某层 activation 爆后临时替换以压 RMS | residual RMS、RMSNorm scale、update ratio |
| Pre-LN | 改变 residual block 的梯度流 | 深层 Transformer 默认使用，降低 warmup 敏感性 | Post-LN 炸了但未验证根因，仅换结构 | early grad distribution、warmup sensitivity |
| Post-LN | 可能有表达和训练动态差异，但深层更敏感 | 配合 DeepNorm / residual scaling / warmup 成熟使用 | 深层不稳仍依赖大量 clip 保持 | gradient by depth、clip rate |
| QK Norm | 同时改变 attention 几何和 score 范围 | Q/K norm drift 或 attention entropy collapse 被验证为根因 | attention score 大就直接加，未查 mask/RoPE/LR | Q/K norm、score p99、attention entropy |
| residual scaling | 控制 residual path 尺度，也可能只是压小输出 | 深层 residual stream 累积是主要机制 | 不定位模块，统一把 residual branch 缩小 | residual RMS by depth、block output/input ratio |
| logit soft-capping | 平滑限制 logits，也可作为模型 recipe | logits 长期增长导致校准/CE 不稳，且 hidden/lm_head 已排查 | lm_head 或 hidden RMS 异常时直接压 logits | logit max、entropy、saturation rate、calibration |
| weight decay | 正则化和优化几何调整，不是普通 clipping | weight norm drift、泛化问题或 AdamW recipe 配套 | 权重一大就盲目加大 decay | weight norm、train/valid gap、decay/update ratio |
| spectral norm | 限制 operator norm，有结构约束含义 | 明确需要 Lipschitz 或 attention/MLP operator 控制 | activation 爆了就对所有矩阵加谱约束 | spectral norm、activation RMS、compute overhead |
| μP / 参数化缩放 | scale-aware 参数化，不是简单 rescale | 小模型超参向大模型迁移不稳，按 μP 从头设计 | 中途用缩放掩盖 LR / init 问题 | width sweep、update scale、hyperparam transfer |
| loss scaling | 混合精度数值范围机制 | FP16 underflow/overflow 是明确问题 | 模型已发散却试图靠调 loss scale 稳住 | overflow count、loss scale、bf16/fp32 replay |
| optimizer update constraint | 可以是优化器定义，也可以是保险丝 | Adafactor / trust-ratio 类机制的一部分 | Adam update 爆了但不查 m/v/LR，只加 cap | update norm、U/W ratio、state health |
| warmup 增加 | 改变早期优化动力学 | early update 过大或 Post-LN 深层梯度问题被验证 | 数据/mask bug 导致 spike，却靠加 warmup | early update ratio、clip rate、loss slope |
| batch size / accumulation 调整 | 改变梯度噪声和 effective update | batch noise 或 token/update 不匹配是根因 | 不查 loss scaling / LR scaling，只增大 batch | tokens/update、grad noise、LR relation |
| sequence length curriculum | 控制训练难度和长程 attention 稳定性 | 长序列阶段导致真实稳定性瓶颈 | long-context mask / position bug 被 curriculum 掩盖 | length-bucket loss、position id、mask correctness |
| router z-loss | 约束 MoE router logits，也改变 routing objective | router logits 爆炸和 expert collapse 被验证 | expert 不均衡就盲目加大 z-loss | router entropy、expert load、z-loss scale |
| auxiliary load balancing loss | 改善 expert load，但改变目标 | expert overload / drop token 是根因 | 为了压平 expert 分布而牺牲 specialization | load CV、drop ratio、task loss、expert specialization |


---

### 3.5 A / B / C 的工程判别流程
当出现稳定性异常时，可以按以下流程分类当前干预手段。

#### Step 1：先判断是否已有明确根因
如果满足以下任意一组强证据，优先归为 A 类方向：

| 强证据 | 例子 |
| --- | --- |
| replay 同一 batch 稳定复现 | 某 packed batch 每次都触发 attention score NaN |
| 单个配置项修复后异常消失 | 修正 LR resume step 后 update ratio 回归 |
| rank-wise 定位明确 | rank 7 的 batch shape 与其他 rank 不一致 |
| layer/head/expert 定位明确 | 第 23 层第 5 个 head attention entropy collapse |
| precision 对照明确 | bf16 / fp32 不炸，fp16 炸 |
| tiny batch 单元实验失败 | label shift / ignore_index / mask toy case 不通过 |


此时应优先做根因修复，不应直接把“加 clip 后不炸”当作最终结论。

#### Step 2：如果没有根因，只有限幅动作，则暂归 B 类
如果当前干预主要满足以下特征，暂归 B 类：

| 特征 | 判断 |
| --- | --- |
| 只设置了阈值 | grad norm clip、activation clamp、logit cap |
| 无法解释异常来源 | 只知道某个 max / norm 过大 |
| 修复依赖持续触发 | clip rate 长期较高 |
| 关闭后立刻复现异常 | 说明根因未消除 |
| 质量指标不确定 | 只关注不 NaN，未检查 eval / calibration |


B 类并不意味着方法错误，而是说明它的角色主要是**保险丝**。使用 B 类方法后，仍需继续做 attribution。

#### Step 3：如果方法改变了训练 recipe 或参数化，进入 C 类判断
如果方法属于 LayerNorm、RMSNorm、Pre-LN、QK Norm、residual scaling、μP、warmup、loss scaling、router z-loss 等，应进一步判断：

| 问题 | 判别方向 |
| --- | --- |
| 是否从训练 recipe 开始就存在？ | 是，则更可能是结构性稳定设计 |
| 是否有明确机制解释？ | 是，则可能接近 A 类修复 |
| 是否依赖异常阈值触发？ | 是，则更可能接近 B 类 |
| 是否只压低 max 值但不改善分布？ | 是，则更偏 B 类 |
| 是否降低 clip rate 并改善 loss / eval？ | 是，则更偏 A / 稳定性设计 |
| 是否改变优化目标或概率校准？ | 是，需要单独评估副作用 |


#### Step 4：用“关闭实验”和“替代实验”验证分类
分类不是静态标签，应通过实验校正：

| 实验 | 目的 | 解释 |
| --- | --- | --- |
| 关闭新增 clip / clamp | 看是否仍稳定 | 若关闭后仍稳定，可能根因已修 |
| 降低 LR 对照 | 判断是否只是 LR 过大 | 若降 LR 后 clip 不再触发，LR 可能是根因 |
| bf16 / fp32 对照 | 判断是否为精度问题 | 若高精度稳定，precision 是关键 |
| bad batch replay | 判断是否数据 / mask 相关 | 若同 batch 必炸，查数据和 sequence |
| reset optimizer state | 判断是否 state 损坏 | 若 reset 后稳定，查 m/v/step |
| single-rank / fewer-rank | 判断是否分布式问题 | 单卡稳定多卡不稳，查同步和 shard |
| module ablation | 判断是否架构局部问题 | 替换某模块后稳定，查该模块实现 |


---

### 3.6 A / B / C 的边界案例
为了避免误判，下面列出几个常见边界案例。

#### 案例 1：global grad norm clipping 后训练不炸
| 现象 | 分类判断 |
| --- | --- |
| clip rate 偶发，loss / eval 正常 | B 类保险丝，合理 |
| clip rate 长期很高，关闭即炸 | B 类止血，根因未解决 |
| 后续发现 LR resume 错误，修复后 clip rate 下降 | 真正修复是 A 类，clip 只是过渡手段 |


#### 案例 2：加入 QK Norm 后 attention entropy 恢复
| 现象 | 分类判断 |
| --- | --- |
| 已排除 mask / RoPE / LR，确认 Q/K norm 随训练系统性膨胀 | QK Norm 更接近 C 类中的结构性修复 |
| 未定位根因，只因 score max 大而加 QK Norm | 更接近 B 类数值约束 |
| QK Norm 是模型从头设计的一部分 | C 类架构稳定性设计 |


#### 案例 3：logit soft-capping 后 CE 不再 NaN
| 现象 | 分类判断 |
| --- | --- |
| hidden RMS、lm_head norm 正常，仅 logits tail 长期过重 | C 类稳定性设计或 soft constraint |
| lm_head norm 异常增长但只靠 softcap 压住 | B 类止血，根因未修 |
| 修复 lm_head LR group 后 softcap saturation rate 很低 | A 类修复完成，softcap 退化为保险机制 |


#### 案例 4：loss scaling 解决 FP16 overflow
| 现象 | 分类判断 |
| --- | --- |
| fp32 / bf16 replay 正常，FP16 overflow，动态 loss scaling 后稳定 | C 类数值精度机制，接近根因修复 |
| 模型已经 activation explosion，却不断降低 loss scale | 不是根因修复，只是在处理后果 |
| overflow 由 fused kernel bug 导致，修 kernel 后恢复 | kernel 修复是 A 类，loss scaling 是辅助机制 |


#### 案例 5：router z-loss 改善 MoE expert collapse
| 现象 | 分类判断 |
| --- | --- |
| router logits 爆炸、expert load collapse 被证实 | C 类中的机制性稳定修复 |
| expert load 不均但未查 capacity、token distribution、router precision | 可能只是数值止血 |
| z-loss 过大导致 routing entropy 过高、expert specialization 消失 | 稳定性改善但目标被扭曲，需要重新调权重 |


---

### 3.7 分类后的工程动作
A / B / C 分类的最终目的不是贴标签，而是决定下一步工程动作。

| 分类 | 主要工程动作 | 必须监控 | 成功标准 | 风险 |
| --- | --- | --- | --- | --- |
| A 类 | 修数据、修代码、修 mask、修 LR、修 optimizer state、修 precision、修 distributed state、修 MoE routing | 根因指标、loss、grad、update、rank/expert consistency | 修复后异常不再复现，额外 hard clip 依赖下降 | 修复可能改变训练分布或需要回滚 checkpoint |
| B 类 | 设置 clip / clamp / cap / rescale，防止异常扩散 | clip rate、saturation rate、gradient/update cosine、eval loss | 低频触发，防止灾难性崩溃，不显著损害质量 | 掩盖根因，改变优化方向 |
| C 类 | 作为 recipe、参数化、归一化、精度或优化器机制进行系统调参 | 相关动力学指标和质量指标 | 稳定性提升且 clip rate 降低，质量不恶化 | 被误用为临时止血，或过度约束表达能力 |


工程上可以采用如下判定结论格式：

```plain
当前异常：第 N step 发生 loss spike，随后 global grad norm 和 logits max 升高。
最早异常对象：第 18 层 attention score p99 在 spike 前 20 step 持续上升。
候选根因：Q/K norm drift；已排除 mask、position id、bad batch、LR resume。
干预方法：引入 QK Norm，并降低 attention temperature。
分类判断：C 类，偏结构性稳定修复；不是单纯 B 类 clip。
验证标准：attention entropy 恢复，score p99 回落，clip rate 不升高，valid loss 不恶化。
```

---

### 3.8 本章小结
A / B / C 的核心区别可以压缩为一句话：

**A 类修复异常产生机制；B 类限制异常数值传播；C 类改变训练系统的参数化、归一化、精度、优化器或 recipe，需要结合使用动机和验证结果判断。**

更具体地说：

| 类别 | 关键问题 | 典型回答 |
| --- | --- | --- |
| A 类 | 为什么会炸？ | 因为 mask / data / LR / optimizer / precision / distributed / MoE / code 某处有明确根因，已修复 |
| B 类 | 怎么不让它继续炸？ | 对 gradient / activation / logits / attention scores / update / reward 加上界或重缩放 |
| C 类 | 这是治本、止血，还是 recipe？ | 取决于它是否有机制解释、是否从设计阶段引入、是否依赖阈值触发、是否改善整体训练动力学 |


因此，判断一个方法属于哪一类时，不应问“这个方法叫什么名字”，而应问：

1. **它作用在异常链条的上游还是下游？**
2. **它是否解释了异常产生机制？**
3. **它是否只是限制数值幅度？**
4. **它是否长期高频触发？**
5. **它是否改善了 loss、activation、gradient、optimizer update 和 eval，而不是只避免 NaN？**
6. **关闭或减弱该方法后，训练是否仍然稳定？**

只有同时回答这些问题，A / B / C 分类才具有工程意义。

