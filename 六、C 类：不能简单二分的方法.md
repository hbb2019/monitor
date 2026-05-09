C 类方法指的是：**不能仅根据方法名称判断其属于“根因修复”还是“数值止血”**。同一个方法在不同训练阶段、不同使用方式、不同触发条件下，可能分别扮演架构稳定性设计、优化器动力学设计、正则化、数值保护、临时止血、甚至掩盖真实问题的角色。

本章的核心任务不是重新列举 A 类或 B 类，而是给出一套工程判断标准：**当一个方法同时具有 norm / clip / rescale 的形式，又可能属于模型 recipe、参数化设计或优化器机制时，如何判断它到底是在解决根因，还是只是在压制症状。** 这与前文“大模型稳定性训练”的整体分类目标保持一致。

---

### 6.1 C 类方法的本质：同一机制，不同角色
C 类方法通常具有以下特征之一：

| 特征 | 说明 | 典型例子 |
| --- | --- | --- |
| 形式上像数值约束 | 对 activation、attention score、logits、update、router logits 等做缩放、归一化或上界控制 | QK Norm、logit soft-capping、update constraint |
| 机制上属于架构设计 | 从模型结构层面改变信号传播、梯度流或 residual path | LayerNorm、RMSNorm、Pre-LN、residual scaling、DeepNorm |
| 作用上可能是根因修复 | 如果根因确实是某个尺度路径、参数化或训练动力学不合理，则该方法是在修机制 | Post-LN 深层不稳时改 Pre-LN；attention score 系统性过大时引入 QK Norm |
| 作用上也可能是止血 | 如果未定位根因，只是因为训练炸了就加入约束，则该方法可能只是压制症状 | loss spike 后盲目加 logit clamp、activation clamp、增大 warmup |
| 长期作为 recipe 时难以归类 | 一些方法在现代 LLM 中已经是默认稳定性设计，不能简单视为“治标” | RMSNorm、Pre-LN、AdamW、warmup、bf16 loss scaling 策略 |


因此，C 类方法的判断不应问：

“这个方法是不是 norm / clip？”

而应问：

**它约束的对象是否就是失稳链路中的根因节点？它是否改变了训练动力学的正确结构？它是否在没有高频触发硬阈值的情况下恢复稳定？它是否保持或改善了模型质量？**

---

### 6.2 判断一个方法属于 A 倾向、B 倾向还是真正 C 类的标准
C 类方法需要从 **使用动机、作用位置、触发方式、监控结果、质量影响** 五个维度判断。

| 判断维度 | 更偏 A 类：根因修复 | 更偏 B 类：数值止血 | 典型 C 类：需要保留双重解释 |
| --- | --- | --- | --- |
| 使用动机 | 已定位失稳机制，例如 Post-LN 梯度异常、QK score 过大、router logits 爆炸 | 只观察到 loss spike / NaN / grad explosion，尚未定位机制 | 有合理机制假设，但仍需实验验证 |
| 作用位置 | 作用在异常链路的上游或根因处 | 作用在异常链路的下游输出处 | 既影响上游动力学，也限制下游数值 |
| 触发方式 | 作为固定架构、参数化或 optimizer recipe 使用 | 只有异常时才触发，或高频触发 hard threshold | 固定存在，但有 saturation / clipping rate 需要监控 |
| 对梯度方向的影响 | 改善梯度流，减少异常 update，不大幅扭曲方向 | 大量截断梯度或激活，改变优化方向 | 可能改变几何结构，但不一定是硬截断 |
| 对训练质量的影响 | loss 更平滑，valid loss / downstream 指标不下降或改善 | loss 稳定但收敛变慢、valid loss 变差、能力下降 | 稳定性改善，但需要重新调 LR、init、schedule |
| 可替代性 | 修复后不依赖额外硬 clip | 移除后训练立刻发散，但根因未解释 | 方法本身就是 recipe 的一部分，不能随意移除 |
| 诊断证据 | 指标链路闭合：异常源头消失 | 只看到异常结果被压小 | 有部分链路证据，但仍可能掩盖其他问题 |


工程上可以用下面的判定公式：

[  
\text{C 类有效性} =  
\text{机制解释}

+ \text{指标闭环}
+ \text{质量保持}
+ \text{高频硬触发}
+ \text{异常迁移}  
]

如果一个方法加入后只是让 loss spike 消失，但 residual RMS、attention score p99、update-to-weight ratio 或 router entropy 仍持续恶化，那么它更偏 B 类止血。  
如果加入后上游异常指标同步恢复，clip rate / saturation rate 低，且验证集指标不下降，则它更偏 A 类根因修复或稳定性 recipe。

---

### 6.3 C 类方法总表
| 方法名称 | 为什么不能简单归为 A 或 B | 在什么情况下更像根因修复 | 在什么情况下更像数值止血 | 如何判断它是否真的解决了根因 | 推荐使用方式 |
| --- | --- | --- | --- | --- | --- |
| LayerNorm | 它形式上是 normalization，但在 Transformer 中是信号传播和梯度流结构的一部分 | 原模型缺少合适归一化，导致 hidden state / residual stream 随深度漂移；加入后层间 RMS 和 gradient flow 恢复稳定 | 训练炸了以后在任意位置插 LN，只为了压 activation max | 观察 LN 前后 RMS、gamma norm、layer-wise grad norm；若上游残差尺度恢复、clip rate 降低、valid loss 不降，说明有效 | 作为架构设计使用；不建议训练中途随意插入，除非做明确 ablation |
| RMSNorm | 不做 mean centering，只控制 RMS；既是稳定性组件，也是参数化选择 | LLM recipe 中从一开始使用，配合 Pre-LN、合理 init、LR 后稳定训练 | 某层激活过大时临时插 RMSNorm，未查清 MLP / attention / residual source | 看 residual RMS 是否平稳、RMSNorm weight 是否异常放大、是否降低 update outlier | 适合作为 decoder-only LLM 默认组件之一；需要与 init、LR、residual scale 联合设计 |
| Pre-LN | 改变归一化位置，直接影响深层 Transformer 梯度传播 | Post-LN 深层模型 warmup 敏感、早期 grad 不稳；改 Pre-LN 后 early step 稳定 | 只因训练不稳就改 Pre-LN，但实际问题是 LR resume、mask 或数据 bug | 比较 Post-LN / Pre-LN 下 step0 到 warmup 期的 grad norm、update-to-weight ratio、loss spike frequency | 深层 LLM 通常优先采用；如果追求特定性能，需要与 residual scaling 和 schedule 联调 |
| Post-LN | 不是错误结构，但在深层训练中更依赖 warmup、init、residual scaling | 若目标架构经过验证，且配合 DeepNorm、充分 warmup、合适 init 后稳定 | 深层训练不稳仍强行保留 Post-LN，仅靠 grad clipping 压制 | 看 early gradients 是否集中在特定层，是否需要异常长 warmup 或高 clip rate 才能训练 | 可用于特定架构，但不应把高频 clip 当作正常成本 |
| QK Norm | 既限制 Q/K 几何，也改变 attention score 生成机制 | attention score p99 持续过大、attention entropy collapse、Q/K norm 随 step 增长；QK Norm 后 score 和 entropy 恢复 | 未确认是 QK 问题，只因 logits 或 loss spike 加 QK Norm | 监控 Q norm、K norm、QK score p99、attention entropy、max attention prob；若这些上游指标恢复，说明不是单纯止血 | 对长上下文、高 LR、多模态或 attention score 易失控场景有价值；应监控 learnable scale |
| attention temperature | 调整 attention softmax sharpness，既可能修 score scale，也可能只是调分布 | attention entropy 系统性过低或过高，且 QK score scale 与 head 行为匹配该判断 | loss spike 后随意增大 temperature，使 attention 变平但不解决 Q/K 或 mask 问题 | 看 entropy、max prob、head specialization、valid loss；若 entropy 恢复但能力下降，可能过度止血 | 与 QK Norm、RoPE scaling、mask 检查一起使用；不应单独作为万能稳定器 |
| residual scaling | 通过缩放 residual branch 控制深层 residual accumulation，既是架构设计也是数值约束 | residual stream RMS 随层数增长，block output/input ratio 偏大；缩放后层间 RMS 平稳 | 不定位哪个 branch 失控，直接整体缩小所有 residual，导致学习能力下降 | 监控每层 residual input/output RMS、attention/MLP branch contribution、update ratio | 应在模型设计阶段确定；中途修改需要重新调 LR 和初始化 |
| DeepNorm / depth-aware residual scaling | 通过深度相关缩放稳定极深 Transformer，属于结构性稳定机制 | 模型深度增加后 Post-LN 或 residual path 不稳定，普通 init/warmup 不足 | 浅层模型或非深度问题中盲目套用，掩盖数据/optimizer bug | 做 depth sweep，比较不同深度下 residual RMS、grad norm、loss spike；若深度扩展稳定则有效 | 用于极深模型时应按完整 recipe 使用，包括 residual scaling 和初始化 |
| μP / maximal update parametrization | 它不是简单缩放，而是让宽度扩展时 update 量级可迁移 | 小模型调参稳定，大模型宽度扩展后 LR / init 不可迁移；采用 μP 后 width sweep 下超参更稳定 | 没有完整实现 μP，只是局部改 scale，导致混合参数化 | 做 width sweep：不同宽度下 activation、logits、update-to-weight ratio 是否保持可比 | 项目初始阶段决定；不宜训练中途混用 standard parameterization 和 μP |
| weight decay | 既是正则化，也是参数尺度控制；不同于 hard weight clipping | weight norm 持续漂移、train-valid gap 扩大，且 AdamW decoupled WD 配置正确 | 参数爆了就盲目加大 WD，但根因是 LR、optimizer state 或 data spike | 观察 weight norm、update decomposition、valid loss；区分 gradient update 和 decay contribution | 使用 decoupled weight decay；明确 norm/bias/embedding/lm_head 是否 decay |
| spectral norm | 通过限制 operator norm 改变层的 Lipschitz 性质；既可能是约束，也可能是稳定性设计 | 某些线性层 operator norm 增长导致 activation 或 attention score 放大 | 作为普通 weight clipping 替代品使用，未验证 operator norm 是根因 | 监控 spectral norm、activation amplification ratio、质量指标 | 大规模 LLM 中需谨慎，计算和性能成本高；更适合作为特定模块约束 |
| Adam epsilon 调整 | epsilon 是数值稳定项，也改变 effective update，尤其在 v 很小时 | Adam v 过小导致 (m / (\sqrt{v}+\epsilon)) 异常大；调 epsilon 后 update-to-weight ratio 恢复 | 不查 Adam state，只因 loss spike 调大 epsilon，导致所有更新被压小 | 监控 v min/p1、effective update norm、update-to-weight ratio、loss convergence | 对稀疏梯度、低精度、embedding、MoE expert 要重点检查；不应孤立调 |
| Adam beta 调整 | beta 影响动量和二阶矩时间尺度，既影响稳定性也影响收敛 | loss/grad 噪声时间尺度与 beta2 不匹配，v 估计滞后导致 update spike | 看到 grad 大就调 beta，但实际是数据异常或 LR 错误 | 看 m/v 的响应速度、update spike 是否滞后于 grad spike、不同 beta 下 valid loss | 与 LR、batch size、warmup 联调；记录 optimizer state 统计 |
| optimizer update constraint | 限制的是实际 update，不是 raw gradient；可能是优化器设计的一部分 | 根因是 adaptive optimizer 产生过大 effective update，update clipping 直接作用于异常节点 | 未查 LR / state / epsilon，直接限制 update norm，掩盖 optimizer state bug | 监控 grad norm 与 update norm 分离情况；如果 grad 正常但 update 异常，update constraint 更合理 | 比单纯 grad clipping 更贴近 Adam 类优化器；必须监控 update clip rate |
| loss scaling | 用于混合精度动态范围管理，不等于模型层面的治标 | FP16 下出现 underflow/overflow，bf16/fp32 单步稳定；loss scaling 修复数值范围 | 模型本身 activation/logits 已失控，却试图靠调 loss scale 继续训练 | 比较 fp16/bf16/fp32 one-step；看 overflow count、loss scale 是否稳定、non-finite 首发位置 | FP16 训练必备；若连续 overflow，应回到模型/数据/optimizer 根因排查 |
| logit soft-capping | 平滑限制 logits 上界，既可能是 recipe，也可能是输出端止血 | logits 极端增长是长期稳定性瓶颈，softcap 后 entropy、CE、valid loss 稳定 | hidden RMS 或 lm_head norm 已失控，只在输出端压 logits | 监控 pre-cap logits、post-cap logits、saturation rate、entropy、calibration | 优先于 hard logit clipping；但 saturation rate 长期高说明上游仍有问题 |
| temperature scaling for logits | 改变输出分布 sharpness，影响 CE 梯度尺度 | logits 过尖/过平是模型 recipe 中预期控制对象 | 为掩盖 lm_head scale、hidden RMS 或 label 问题而调 temperature | 看 logits RMS、entropy、top-1 prob、valid loss 和 calibration | 用于明确目标时；不应替代 hidden/lm_head 尺度诊断 |
| gradient accumulation 调整 | 改变 effective batch、梯度噪声和 update 频率 | 原因是 microbatch 太小导致梯度噪声 spike；增加 accumulation 后 update 方差下降 | 实际是 loss reduction / no_sync / scaling bug，却通过 accumulation 暂时变稳 | 对比等价大 batch 与 accumulation；检查 loss scaling、grad averaging、tokens/update | 必须与 LR、warmup、loss reduction 一起校验 |
| batch size 调整 | 改变 gradient noise scale，也改变 LR scaling 条件 | 小 batch 噪声过大或大 batch LR scale 不当被识别为根因 | 数据/mask bug 在大 batch 下被平均掉，看似稳定 | 按 tokens/update 归一化比较 grad variance、update ratio、loss curve | 调 batch 必须重新审视 LR schedule；不要用大 batch 掩盖坏样本 |
| sequence length curriculum | 控制训练难度和 attention/position 范围，既是数据策略也是稳定机制 | 长序列阶段性引入导致 position/attention/activation 不稳；curriculum 平滑过渡 | 长序列 mask 或 RoPE bug 未修，只用 curriculum 延后暴露 | 分长度桶监控 loss、attention score、position id、activation RMS | 长上下文训练常用；每个长度阶段都要有独立稳定性验收 |
| warmup 增加 | 减小早期 update，可能解决初始化/梯度流问题，也可能掩盖 bug | early update-to-weight ratio 过大、Post-LN/深层模型对 LR 敏感 | resume step 错、LR 配错、data bug 导致 spike，却靠更长 warmup 压住 | 看 warmup 内外 update ratio、clip rate、loss spike 是否自然下降 | 与模型深度、batch size、optimizer state 联动；过长 warmup 会降低训练效率 |
| dropout | 主要是正则化，但也改变 activation/gradient 噪声 | 过拟合、co-adaptation 或 fine-tune 阶段不稳 | loss spike 时盲目加 dropout，导致优化噪声更大或收敛变慢 | 看 train-valid gap、activation variance、gradient variance、eval 指标 | 预训练中需按 recipe；SFT/RLHF 中要特别关注分布偏移 |
| stochastic depth / layer drop | 改变残差路径和有效深度，既是正则也影响稳定性 | 极深模型训练中作为结构正则化缓解过拟合或路径 co-adaptation | 训练不稳时随机丢层，掩盖某些层实现 bug | 监控被 drop 层、effective depth、layer-wise loss contribution | LLM 预训练中需谨慎；若使用，必须和深度、LR、residual scaling 联调 |
| router z-loss | 约束 MoE router logits，同时改变 routing objective | router logits 爆炸、router entropy collapse、expert load 极不均 | 所有 MoE 不稳都加大 z-loss，但实际问题是 capacity 或数据分布 | 监控 router logits、router entropy、expert load、drop rate、z-loss 占比 | 与 aux loss、capacity factor、router precision 联调；权重过大会削弱专家选择性 |
| auxiliary load balancing loss | 促进 expert load 均衡，但改变 MoE 训练目标 | expert overload、token drop、expert collapse 是主要根因 | 为了让 load 看起来均匀而过强约束，破坏 expert specialization | 看 load CV、drop tokens、expert grad/update、主 loss 与 aux loss 比例 | 权重应足够小且可监控；不要只追求 load 完全均匀 |
| capacity factor 调整 | 控制 expert 容量和 token drop，是 MoE 系统与优化共同参数 | token drop 高导致有效训练信号损失或 expert update 异常 | 增大 capacity 掩盖 router collapse，显存/通信成本暴涨 | 监控 capacity usage、drop ratio、expert load、throughput | 与 batch size、top-k routing、aux loss 一起调 |
| router precision / router fp32 | 改变 routing 数值稳定性，不只是普通精度选择 | router logits / softmax 在低精度下 overflow 或排序不稳定 | MoE 架构/负载问题未修，只把 router 提到 fp32 | 比较 bf16/fp32 router 下 route consistency、entropy、load | router 通常值得更高精度；但不能替代 routing objective 修复 |
| label smoothing | 改变目标分布，也会降低极端 logits 梯度 | 目标过尖导致 logits 过度自信，且任务允许平滑 | label 错误、tokenizer 错误或数据噪声被 smoothing 掩盖 | 看 calibration、entropy、valid loss、hard-label accuracy | 对预训练 LLM 要谨慎；更常用于特定监督任务 |
| normalization of rewards | RLHF/RL 中稳定 advantage/reward scale | reward scale 漂移导致 PPO/DPO 类更新不稳 | reward model bug、prompt 分布异常被归一化掩盖 | 监控 raw reward、normalized reward、KL、advantage、win-rate | 必须同时记录 raw 与 normalized 指标，避免隐藏 reward model 漂移 |


---

### 6.4 C 类方法的工程判定流程
当一个方法看起来既像稳定性设计又像数值止血时，建议按以下流程判断。

#### Step 1：明确它约束的对象
先写清楚该方法直接作用在哪个对象上：

| 方法 | 直接作用对象 |
| --- | --- |
| LayerNorm / RMSNorm | hidden state / residual stream |
| QK Norm | query/key 向量与 attention score scale |
| logit soft-capping | output logits |
| update clipping | optimizer update |
| loss scaling | FP16 梯度动态范围 |
| router z-loss | MoE router logits |
| aux load balancing | expert routing distribution |
| warmup | early-stage update magnitude |
| batch size / accumulation | gradient noise 与 effective update frequency |


如果直接作用对象位于异常链路的上游，它更可能是根因修复。  
如果直接作用对象位于异常链路的末端，它更可能是止血。

例如：  
attention entropy collapse 的上游通常是 Q/K norm、attention score scale、mask、position id。此时 QK Norm 或 RoPE 修复更接近根因。  
如果只是最终 logits 爆炸，而上游 hidden RMS 仍持续增大，logit soft-capping 更像输出端止血。

---

#### Step 2：检查是否有明确机制假设
每个 C 类方法上线前都应配一个可验证的机制假设。

| 方法 | 合格的机制假设 | 不合格的使用理由 |
| --- | --- | --- |
| Pre-LN | Post-LN 在当前深度和 LR 下 early gradient flow 不稳定 | “Pre-LN 通常更稳，所以试试” |
| QK Norm | Q/K norm 增长导致 attention score p99 过大和 entropy collapse | “attention 可能有问题” |
| logit soft-capping | logits tail 持续增大，CE 梯度受极端 logits 影响 | “loss spike 了，压一下 logits” |
| warmup 增加 | early update-to-weight ratio 超过稳定区间 | “训练炸了，warmup 加长一点” |
| router z-loss | router logits 增大导致 expert collapse | “MoE 不稳，z-loss 加大” |
| Adam epsilon 调整 | v 过小导致 effective update 异常 | “Adam 不稳，调 epsilon” |


机制假设必须能被指标验证，否则该方法应按 B 类止血处理，并持续监控是否掩盖问题。

---

#### Step 3：做最小 ablation
C 类方法不应只靠一次完整训练判断。推荐最小验证包括：

| 验证方式 | 目的 |
| --- | --- |
| same checkpoint + same batch replay | 判断是否直接消除触发异常的路径 |
| short-run ablation | 比较加入前后 100–1000 step 的稳定性指标 |
| layer-wise / head-wise hook | 判断异常是否从上游消失 |
| precision ablation | 区分数值范围问题和模型动力学问题 |
| LR sweep / warmup sweep | 判断是否只是降低 effective update 后变稳 |
| remove-protection test | 在安全短跑中移除或减弱该方法，看是否立即恢复异常 |
| saturation / trigger rate logging | 判断该方法是否长期高频介入 |


如果一个方法只有在极低 LR 或极高 clip threshold 下才稳定，它可能不是充分的根因修复。  
如果一个方法加入后无需额外硬 clip，且上游异常指标恢复，则更接近稳定性设计。

---

#### Step 4：监控它有没有改变训练目标或模型能力
C 类方法的副作用往往不是“训练继续不继续”，而是“训练学到的东西是否变了”。

| 方法 | 需要额外监控的质量风险 |
| --- | --- |
| logit soft-capping | calibration、token entropy、rare token probability、CE tail |
| attention temperature / QK Norm | long-context retrieval、head specialization、attention sparsity |
| residual scaling | 深层表达能力、收敛速度、梯度流 |
| warmup 增加 | token budget 浪费、早期欠更新 |
| weight decay | underfitting、embedding/lm_head norm 过小 |
| dropout | 预训练 loss 噪声、SFT 指令遵循下降 |
| aux load balancing loss | expert specialization 被削弱 |
| router z-loss | routing 过平、专家选择能力下降 |
| batch size 增大 | 泛化变化、sharpness、LR scaling 失配 |
| sequence curriculum | 长上下文能力是否真正学到，而非后期才暴露问题 |


一个方法如果稳定了 loss 但降低 valid loss 表现、下游能力、校准或长上下文指标，应视为“稳定性换性能”，不能直接判定为根因修复。

---

### 6.5 典型 C 类案例
#### 案例 1：QK Norm 是根因修复还是数值止血
**现象**：训练中 attention entropy 在若干层逐渐 collapse，部分 head 的 max attention probability 长期接近 1，随后 logits max 增大并出现 loss spike。

**错误处理**：直接对 attention scores 做 hard clamp。loss spike 减少，但 head 行为异常仍然存在，valid loss 变差。

**C 类处理**：引入 QK Norm 或调整 attention temperature。

**判断标准**：

| 指标 | 预期变化 |
| --- | --- |
| Q norm / K norm | 不再随 step 持续增大 |
| QK score p99 | 回到稳定范围 |
| attention entropy | 不再 collapse，也不过度均匀 |
| max attention prob | 不再长期饱和 |
| downstream / valid loss | 不下降或改善 |
| score clamp 触发率 | 显著下降或不再需要 |


如果这些指标同步恢复，QK Norm 更接近根因修复或架构稳定性设计。  
如果只是 clamp 后 loss 不炸，但 Q/K norm 继续上升，则仍是 B 类止血。

---

#### 案例 2：warmup 增加是有效修复还是掩盖 LR / resume bug
**现象**：训练前几千 step 频繁 loss spike，global grad norm 和 update-to-weight ratio 偏大。

**可能解释 1**：模型深度较大，初始化阶段需要更平滑的 LR ramp-up。  
**可能解释 2**：LR schedule 配错，或者 resume 后 scheduler step 错误。  
**可能解释 3**：batch size / accumulation 实际值与配置不一致，导致 effective LR 偏大。

**判断标准**：

| 检查项 | 如果是根因修复 | 如果是掩盖问题 |
| --- | --- | --- |
| scheduler step | 正确连续 | resume 后跳变或归零 |
| tokens/update | 与配置一致 | accumulation 或 global batch 计算错误 |
| update-to-weight ratio | warmup 后自然平稳 | 仍贴近危险阈值 |
| clip rate | warmup 增加后下降到低频 | 仍长期高频触发 |
| small LR replay | 与增加 warmup 行为一致 | 发现真实 LR 远高于预期 |


如果 warmup 增加只是补偿错误 LR，那么它是止血。  
如果 schedule、batch、optimizer state 均正确，而 early update 确实过大，则 warmup 调整更像训练动力学修复。

---

#### 案例 3：logit soft-capping 是 recipe 还是掩盖 hidden RMS 爆炸
**现象**：输出 logits max 持续增大，CE loss 偶发 spike。

**两种可能**：

| 可能根因 | 解释 |
| --- | --- |
| 输出端 logits tail 过重 | hidden RMS 基本稳定，但 lm_head 输出尾部过大 |
| 上游 residual stream 爆炸 | hidden RMS、MLP output、attention output 已经持续失控 |


**判断方式**：

| 指标 | 更支持 recipe | 更支持止血 |
| --- | --- | --- |
| pre-cap hidden RMS | 稳定 | 持续增大 |
| lm_head weight norm | 稳定或轻微增长 | 快速漂移 |
| softcap saturation rate | 低频触发 | 高频触发 |
| post-cap entropy | 合理 | 被强行压平 |
| valid loss / calibration | 不下降 | 下降或校准恶化 |


如果 softcap saturation rate 长期很高，应继续追查 hidden state、lm_head、LR group 或数据问题。

---

#### 案例 4：router z-loss 是 MoE 根因修复还是过度正则
**现象**：MoE 训练中 expert load 不均，部分 expert 梯度异常，router logits 变大，token drop 增加。

**合理使用**：如果 router logits 过大导致 routing entropy collapse，z-loss 可以作为 router 稳定机制。  
**不合理使用**：如果真实问题是 capacity factor 太小、数据分布高度偏斜或 expert parallel state 错误，只增大 z-loss 可能让 routing 看起来平滑，但专家学习质量下降。

**判断标准**：

| 指标 | 合理修复 | 过度止血 |
| --- | --- | --- |
| router logits max | 下降到稳定范围 | 被压得过小 |
| router entropy | 恢复合理 | 过度均匀 |
| expert load CV | 降低 | 降低但 expert specialization 消失 |
| token drop ratio | 降低 | 仍高，说明 capacity 问题未解 |
| aux/z-loss 占比 | 可控 | 辅助 loss 主导训练 |
| expert update norm | 稳定 | 某些 expert 仍异常或学习不足 |


---

### 6.6 C 类方法的监控清单
C 类方法上线后，至少应额外记录以下信息，避免把“稳定”误判为“修复”。

| 方法类型 | 必须监控的额外指标 | 危险信号 |
| --- | --- | --- |
| normalization 类 | pre-norm RMS、post-norm RMS、norm weight、layer-wise grad | norm weight 持续放大，说明模型在补偿归一化 |
| QK / attention 类 | Q/K norm、score p99、entropy、max prob、head diversity | entropy 过度均匀或长期 collapse |
| residual scaling 类 | branch output/input ratio、residual RMS by depth、update ratio | residual 过小导致学习停滞 |
| logit softcap 类 | pre-cap logits、post-cap logits、saturation rate、entropy | saturation rate 长期高 |
| optimizer constraint 类 | raw grad norm、effective update norm、U/W ratio、clip rate | update constraint 高频触发 |
| loss scaling 类 | loss scale、overflow count、first non-finite tensor | 连续 overflow 或 step skip |
| warmup / LR 类 | current LR、scheduler step、U/W ratio、clip rate | warmup 结束后立即 spike |
| batch / accumulation 类 | tokens/update、grad variance、loss reduction factor | effective batch 与配置不一致 |
| MoE balancing 类 | router logits、entropy、expert load、drop ratio、aux loss 占比 | load 均衡但专家能力下降 |


---

### 6.7 C 类方法的推荐实验模板
为了判断 C 类方法是否真的解决根因，可以使用以下短跑实验模板。

| 实验 | 配置 | 观察重点 | 结论判断 |
| --- | --- | --- | --- |
| Baseline replay | 原始 checkpoint + bad batch | 异常是否复现 | 确认问题可复现 |
| Method-on replay | 同 checkpoint + 同 batch + C 类方法 | 上游指标是否恢复 | 若只压下下游指标，则偏止血 |
| Lower-LR control | 不加方法，只降低 LR | 判断是否只是 update 太大 | 若降 LR 同样有效，需查 schedule/optimizer |
| Precision control | bf16/fp32 对比 | 判断是否数值范围问题 | 若高精度稳定，先修 precision |
| Remove hard clip | 保留 C 类方法，去掉额外 clip | 判断是否还依赖保险丝 | 若仍需高频 clip，根因未完全解决 |
| Short validation | 运行固定 token budget | 看 valid loss / downstream proxy | 稳定但质量下降不是充分修复 |
| Saturation logging | 记录 saturation / trigger rate | 判断方法是否长期介入 | 高频触发说明更像 B 类 |


---

### 6.8 C 类方法的使用原则
C 类方法的工程原则可以概括为以下几条。

#### 原则 1：先定义角色，再上线方法
同一个方法上线前必须明确它的角色：

| 角色 | 示例 | 验收标准 |
| --- | --- | --- |
| 架构 recipe | RMSNorm、Pre-LN、residual scaling | 从 step0 开始参与训练，指标长期稳定 |
| 根因修复 | QK Norm 修 attention score 失控 | 上游异常指标恢复 |
| 优化器控制 | update-to-weight cap | update 异常消失，grad 不被过度改写 |
| 数值范围修正 | loss scaling、stable softmax | overflow / non-finite 消失 |
| 临时保险丝 | grad clipping、activation clamp | 低频触发，不作为主要训练机制 |
| 正则化 | weight decay、dropout、aux load balancing | 泛化或 routing 改善，不牺牲主任务 |


如果角色不清楚，该方法应默认按 B 类风险处理。

---

#### 原则 2：监控 pre-constraint 与 post-constraint
对任何带约束的 C 类方法，都要同时记录约束前和约束后的数值。

| 方法 | pre 指标 | post 指标 |
| --- | --- | --- |
| logit soft-capping | raw logits max/RMS | capped logits max/RMS |
| attention score softcap | raw score p99/max | capped score p99/max |
| update clipping | raw update norm | clipped update norm |
| loss scaling | unscaled grad finite ratio | scaled/backward overflow |
| QK Norm | raw Q/K norm 或 normalized 前尺度 | normalized score scale |
| router z-loss | raw router logits | z-loss 后 router entropy/load |


只看 post-constraint 指标会严重低估问题。  
如果 raw 指标持续恶化，而 post 指标稳定，说明该方法正在掩盖上游异常。

---

#### 原则 3：监控触发率或饱和率
C 类方法是否变成 B 类止血，最直接的信号是触发率。

| 指标 | 含义 |
| --- | --- |
| grad clip rate | 梯度有多少 step 被裁剪 |
| update clip rate | optimizer update 有多少 step 被限制 |
| logit softcap saturation rate | 多少 logits 进入 softcap 非线性饱和区 |
| attention score cap rate | 多少 score 被限制 |
| loss scale overflow rate | 多少 step 出现 overflow |
| router z-loss 占比 | 辅助稳定 loss 是否主导训练 |
| expert drop ratio | capacity 是否仍不足 |
| norm gamma growth rate | 归一化层是否被模型反向补偿 |


经验上，**低频触发的保护机制更像保险丝，高频触发的保护机制更像新的训练动力学**。  
一旦高频触发，就必须重新评估它是否正在改变训练目标。

---

#### 原则 4：不要把“稳定”与“正确”混淆
一个方法让训练不 NaN，只说明它改善了数值连续性，不说明训练目标、数据语义和优化动力学正确。

下面这些现象都属于“稳定但不正确”：

| 现象 | 可能问题 |
| --- | --- |
| loss 不炸，但 valid loss 持续变差 | 过度 clipping / softcap / 正则化 |
| attention entropy 恢复，但所有 head 过度均匀 | temperature / QK Norm 过强 |
| expert load 均衡，但专家 specialization 消失 | aux loss 或 z-loss 过强 |
| logits 不极端，但 calibration 变差 | logit softcap 或 label smoothing 过强 |
| grad norm 稳定，但 update-to-weight ratio 仍异常 | optimizer state / epsilon / LR 未修 |
| activation RMS 稳定，但 norm gamma 持续放大 | normalization 在被模型补偿 |
| loss scale 稳定，但 fp32 replay 仍出现异常 | 不是单纯混合精度问题 |


---

### 6.9 C 类章节的最终分类结论
C 类方法不是“模糊地带”，而是大模型训练稳定性中最需要工程判断的一类方法。它们的正确分类取决于 **是否有明确机制、是否作用于异常链路的根因位置、是否低频触发、是否保持模型质量、是否能通过最小实验验证**。

可以用以下规则总结：

| 判断结果 | 归类倾向 |
| --- | --- |
| 有明确机制，修复后上游异常消失，不再依赖额外 hard clip | C → A 倾向，属于根因修复或稳定性 recipe |
| 没有机制解释，只压住 loss / NaN / max value，raw 指标继续恶化 | C → B 倾向，属于数值止血 |
| 方法从训练开始就是架构或优化器 recipe，且没有高频饱和/触发 | 保持 C 类，作为稳定性设计 |
| 方法引入后质量下降、entropy 过平、expert specialization 消失 | C 类方法使用过强，需回退或重新调参 |
| 方法只有在高频触发时训练才稳定 | 实际上已经成为 B 类保护，应继续追根因 |


最终可以把 C 类方法理解为：

**C 类方法不是简单的“治标”或“治本”，而是训练动力学中的结构性调节器。它们只有在作用位置、机制假设、监控指标和质量验证形成闭环时，才可以被视为稳定性设计或根因修复；否则，即使名称看起来像高级架构方法，也可能只是更隐蔽的数值止血。**

