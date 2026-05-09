本章讨论的 A 类方法，是指在训练中观察到 loss spike、gradient explosion、activation explosion、logits explosion、attention entropy collapse、NaN / Inf、optimizer update 异常、rank-wise divergence、MoE expert collapse 等“预兆爆炸”后，**定位并修复导致异常产生的具体机制**，而不是仅通过 clip / clamp / rescale 把异常数值压下去。它延续全文对 A/B/C 类方法的分类标准：A 类的关键不是“用了什么手段”，而是是否找到了可验证的根因，并且修复后训练不再依赖额外高频硬约束来维持稳定。

需要温和修正一点：A 类修复并不意味着训练中完全不能保留 gradient clipping、loss scale overflow skip、bad batch quarantine 等保险机制。工业级训练通常仍会保留这些安全阀。A 类的判断标准是：**训练稳定性不再主要依赖这些保险机制；clip / clamp 的触发率应回到低频、异常保护级别，而不是成为主训练动力学的一部分。**

---

## 4.1 A 类方法的判定标准
一个修复能否被归为 A 类，建议至少满足以下四个条件。

| 判定维度 | A 类要求 | 不满足时的风险 |
| --- | --- | --- |
| 机制明确 | 能说明异常从哪里产生，例如 mask 错误、LR resume 错误、Adam state 损坏、position id 错误、router collapse | 只是观察到“数值变大”，可能仍停留在症状层 |
| 可复现 | 能用固定 checkpoint、固定 batch、固定 seed 或固定 rank 复现异常 | 异常可能来自非确定性、日志误差或系统状态不一致 |
| 可干预 | 修改该根因后，异常显著减弱或消失 | 可能只是相关指标，不是因果源头 |
| 可验收 | 修复后 loss、grad norm、activation RMS、update-to-weight ratio、rank consistency 等回到正常区间，且不需要高频 hard clip | 可能只是把异常转移到其他模块或更晚 step |


因此，A 类修复的基本形式不是：

“grad norm 太大，所以加大 gradient clipping。”

而是：

“grad norm 变大是因为某个 packed sequence 的 boundary mask 错误，导致跨样本 attention；修复 mask 后，同一 bad batch replay 不再触发 activation spike 和 grad spike。”

---

## 4.2 A 类根因定位的最小闭环
在进入具体根因表之前，先给出一个适用于大多数训练稳定性问题的 A 类定位闭环。

### 4.2.1 保留现场
当出现异常时，应优先保留以下对象：

| 对象 | 具体内容 |
| --- | --- |
| checkpoint | 异常前最近稳定 checkpoint、异常 step checkpoint、异常后 checkpoint |
| batch 信息 | sample id、source、sequence length、token ids、attention mask、position ids、labels、loss mask、sample weights |
| 训练状态 | global step、LR、warmup / decay 状态、optimizer step、loss scale、gradient accumulation step |
| 分布式状态 | rank-wise loss、grad norm、overflow、batch shape、parameter checksum、optimizer shard metadata |
| 模型内部信号 | layer-wise residual RMS、attention score p99、attention entropy、MLP activation RMS、logits max、update-to-weight ratio |


如果只保留 mean loss 和 global grad norm，通常不足以支持 A 类定位。A 类定位至少需要能回答三个问题：**哪个 step 最先异常、哪个对象最先异常、哪个输入或状态触发异常。**

---

### 4.2.2 复现异常
优先做以下复现实验：

| 实验 | 操作 | 解释 |
| --- | --- | --- |
| 同 checkpoint + 同 batch replay | 固定 seed，从异常前 checkpoint 重放触发 batch | 若稳定复现，batch、mask、precision、模型状态更可疑 |
| 同 checkpoint + 不同 batch | 换正常 batch replay | 若不复现，数据 / sequence / sample-specific 问题更可疑 |
| 同 batch + 更早 checkpoint | 用更早稳定 checkpoint 跑同一 batch | 若也复现，batch / mask / data 更可疑；若不复现，模型状态或 optimizer state 更可疑 |
| 单卡 / 少卡 replay | 去掉 ZeRO/FSDP/多机通信因素 | 若单卡不复现，多半是分布式、sharding、rank 数据或同步问题 |
| bf16 / fp32 replay | 提高数值精度或关闭部分 autocast | 若高精度不复现，precision、softmax、fused kernel、loss scaling 更可疑 |
| 关闭 fused kernel replay | 替换 FlashAttention、fused CE、fused optimizer 等 | 若不复现，kernel 数值稳定性或实现 bug 更可疑 |


---

### 4.2.3 定位第一个异常对象
A 类定位的关键是寻找 **first bad tensor / first bad state**，而不是只看最后出现 NaN 的地方。

| 第一个异常对象 | 常见根因方向 |
| --- | --- |
| batch loss / per-sample loss 先异常 | 数据、label、sample weight、template、tokenizer |
| attention score 先异常 | Q/K norm、RoPE、position id、mask、attention temperature |
| residual RMS 先异常 | residual scaling、Post-LN 深层不稳、MLP/attention 输出尺度 |
| MLP activation 先异常 | activation function、SwiGLU gate、init、MLP scale |
| logits 先异常 | lm_head / embedding scale、hidden RMS、vocab/tokenizer 错配 |
| grad norm 先异常 | loss spike、backward bug、optimizer scale、某层局部异常 |
| update norm 先异常 | LR、Adam state、epsilon、weight decay、param group 配置 |
| rank-wise loss 先分叉 | 分布式数据、all-reduce、gradient accumulation、shard state |
| router load 先塌缩 | MoE router logits、capacity factor、aux loss、expert parallel |


---

### 4.2.4 修复后验收
A 类修复完成后，不能只看“没有 NaN”。建议至少检查：

| 验收项 | 合格信号 |
| --- | --- |
| bad batch replay | 原触发 batch 不再触发 spike、NaN、Inf |
| layer-wise 信号 | residual RMS、attention score p99、MLP activation RMS 不再出现局部阶跃 |
| optimizer update | update norm、update-to-weight ratio 回到历史正常分布 |
| clip / skip 触发率 | gradient clip、overflow skip、bad batch quarantine 触发率下降到低频保护级别 |
| rank consistency | rank-wise loss、grad norm、loss scale、param checksum 不再分叉 |
| 训练质量 | train loss / valid loss 曲线连续，修复没有造成明显欠拟合或退化 |
| 小规模复现 | tiny batch overfit、toy mask test、single-step reference test 通过 |


---

## 4.3 A 类根因与修复方法总表
下面按根因方向展开。所有条目均围绕同一逻辑：**识别预兆指标 → 做最小验证实验 → 修复产生异常的机制 → 验证修复后不再依赖高频硬约束。**

---

### 4.3.1 数据问题
| 根因类别 | 典型异常现象 | 预兆指标 | 如何验证是不是这个根因 | 根因修复方法 | 为什么这属于 A 类 | 修复后的验证方式 | 可能副作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 脏数据 / 损坏文本 | loss spike 由少数样本稳定触发；per-sequence loss 极高；某些 batch replay 必炸 | per-sample loss top-k、异常 Unicode 比例、乱码比例、非文本 token 比例、source id | 固定 checkpoint replay bad batch；移除 top-loss 样本后 replay；按 source 分桶比较 loss | 清洗损坏样本；过滤乱码、空样本、异常格式；为高风险 source 设置 quarantine | 修复异常输入分布，而不是压制由异常输入产生的梯度 | 原 bad batch 不再 spike；同 source loss 分布回归；clip rate 降低 | 过度清洗可能移除困难样本，降低鲁棒性 |
| 错误 label / 错误 target | loss 长期偏高或局部 spike；tiny batch 无法 overfit；模型学到错位目标 | per-token loss、label token 分布、ignore ratio、input-label alignment | 对单个 batch 手算 CE；打印 input / label shift；tiny batch overfit | 修正 label 构造、shift 方向、target mask、template 中 assistant/user 区间 | 修复训练目标本身 | tiny batch 可快速 overfit；per-token loss 不再集中在错误位置 | 修复后 loss 标尺可能变化，需要重建 baseline |
| 重复样本 / 高重复数据 | train loss 下降异常快，valid loss 恶化；某些 token logits 过度自信 | duplicate rate、n-gram repetition、source repetition、train/valid gap | 对高频样本去重后对比；按 source 统计重复比例 | 文档级 / paragraph 级去重；降低重复 source 权重；修 sampler | 修复数据分布偏差和过拟合源头 | valid loss 改善；logits entropy 不再异常下降 | 去重过强可能损失高质量模板数据 |
| 极长样本 / 长度分布突变 | 长序列 batch 触发 activation spike、attention score spike 或 OOM 前的 loss spike | seq length p95/p99/max、tokens per batch、attention mask density、position id range | 按长度分桶 replay；短序列 / 长序列分别训练数百 step；固定 batch size 改 token batch size | length curriculum；长度分桶；限制极端长度；修 token batch sampler | 修复输入难度和计算图尺度突变 | 长度分桶下 loss / grad 平滑；long-context 阶段不再突然 spike | 过度限制长度会影响长上下文能力 |
| 特殊 token 异常 | BOS/EOS/PAD/UNK 等 token loss 极端；lm_head 某些 row norm 异常；输出偏向特殊 token | special token ratio、special token per-token loss、embedding row norm、top logit token | 打印 tokenized sample；检查 chat template；对特殊 token loss 分桶 | 修 tokenizer 配置、special token id、chat template、loss mask | 修复输入/目标语义错误 | 特殊 token loss 和 row norm 回归正常 | 修改 template 会改变数据分布，需要重新评估 SFT/RLHF 指标 |
| 数据源分布突变 | 某个 step 后 loss、grad、activation 分布整体改变；source-specific loss 失衡 | source mix ratio、language ratio、domain ratio、batch source entropy | 对异常窗口统计 source mix；固定 source replay；恢复旧 sampler 对比 | 修 sampler、mixing weight、数据 shard 顺序、epoch boundary 逻辑 | 修复训练分布漂移源头 | source mix 与配置一致；loss 不再在 source 切换处 spike | 数据混合调整可能改变最终能力分布 |
| 样本权重错误 | 少数样本造成极大 loss / grad；loss 与 sample weight 强相关 | sample weight max、weighted loss p99、source weight、task weight | 将 sample weight 设为 1 replay；按 weight bucket 统计 grad norm | 修 sample weighting、task mixing coefficient、loss normalization | 修复优化目标尺度错误 | weighted / unweighted loss 比例正常；grad norm 降低 | 修复后任务间权重改变，可能影响多任务能力 |
| 数据 shard / dataloader 异常 | 某些 worker 或 rank 稳定产出异常 batch；多卡下复现，单卡不复现 | worker id、rank id、shard id、batch shape、sample id 重复率 | 固定 dataloader seed；单独跑异常 rank 的 shard；检查 shard metadata | 修 shard 切分、worker seed、drop_last、resume sampler state | 修复数据供应链状态 | rank-wise 数据分布一致；resume 前后 sample order 正常 | 修复 sampler 后训练可复现性基线需重置 |


---

### 4.3.2 Sequence packing / mask / position 问题
| 根因类别 | 典型异常现象 | 预兆指标 | 如何验证是不是这个根因 | 根因修复方法 | 为什么这属于 A 类 | 修复后的验证方式 | 可能副作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attention mask 错误 | 模型 attend 到 padding、未来 token 或非法区域；loss 异常低或异常高 | illegal attention ratio、mask density、attention entropy、attention score on masked positions | 构造 toy sequence 检查 attention matrix；可视化 mask；比较 masked / unmasked logits | 修 attention mask 形状、dtype、广播维度、加性 mask 符号 | 修复注意力语义错误 | toy mask test 通过；非法 attention score 为负无穷或不可达 | mask 修复后 loss 可能上升，因为原先存在信息泄漏 |
| padding 未 mask | pad token 参与 loss 或 attention；pad 多的 batch loss 异常 | pad ratio、pad token loss、label ignore ratio、attention to pad | 构造含大量 padding 的 batch；检查 labels 中 pad 是否为 ignore_index | 修 loss mask、padding mask、ignore_index、collator | 修复无效 token 参与训练的问题 | pad token loss 消失；不同 padding 比例 batch loss 一致 | loss denominator 改变后曲线不可直接与旧实验比较 |
| causal mask 错误 | 模型看到未来 token，loss 异常低；或过度 mask 导致 loss 异常高 | future attention ratio、train loss 低于合理范围、attention entropy | 对短序列手工检查每个位置可见 token；比较 teacher-forcing toy case | 修 causal mask 方向、upper/lower triangular、position offset | 修复自回归目标定义 | toy autoregressive test 通过；future attention 为 0 | 修复信息泄漏后短期 loss 可能明显升高 |
| packed sequence 串样本 | 不同样本之间互相 attention；packed batch 特有 loss spike 或异常低 loss | packed boundary attention、segment id、position reset、boundary token loss | packed vs unpacked 同样本对比；可视化 segment attention | 修 block diagonal mask、segment ids、position reset、loss mask boundary | 修复样本隔离机制 | packed / unpacked loss 接近；cross-boundary attention 消失 | 更严格 mask 可能降低 packing 吞吐 |
| position id 错误 | 长序列退化；RoPE 位置异常；某些位置 loss spike | position id min/max、重复 position、position gap、loss by position | 打印 position id；构造短/长序列 reference；禁用 packing 对比 | 修 position id 生成、padding offset、packed reset、sliding window offset | 修复位置编码输入 | position-wise loss 平滑；长短序列均稳定 | 修复后 long-context 指标需重新评估 |
| RoPE scaling / 长上下文扩展错误 | 短序列稳定，长序列 loss spike；attention score 随位置变极端 | attention score by distance、QK norm by position、long-seq loss | 短/中/长长度分桶；对比不同 RoPE scaling 配置 | 修 RoPE base、scaling ratio、interpolation/extrapolation 实现 | 修复位置几何和 attention score 源头 | 长度分桶下 attention entropy 正常；long context loss 稳定 | 改 RoPE scaling 会改变位置泛化能力 |
| sliding window / block attention mask 错误 | 局部 attention 模型在边界处 loss spike；窗口外 attention 泄漏或过度屏蔽 | window boundary loss、illegal attention ratio、block mask density | 构造跨 window toy case；检查每个 token 的可见范围 | 修 window offset、block mask、global token mask | 修复稀疏 attention 结构 | 边界位置 loss 不再异常 | 严格窗口可能降低远距离依赖能力 |


---

### 4.3.3 Learning rate / schedule 问题
| 根因类别 | 典型异常现象 | 预兆指标 | 如何验证是不是这个根因 | 根因修复方法 | 为什么这属于 A 类 | 修复后的验证方式 | 可能副作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learning rate 过大 | early training loss spike；grad norm 和 update norm 同时放大；clip rate 持续高 | LR、global update norm、update-to-weight ratio、clip rate | 从同 checkpoint 用 LR /2、/5、/10 replay；对比 warmup 延长 | 降低 base LR；按 token batch size 重新 scale；调 peak LR | 修复过大有效步长 | U/W ratio 回归；clip rate 降低；loss 曲线平滑 | 收敛速度可能下降 |
| warmup 不足 | warmup 前后或训练早期不稳；深层模型初期 spike | warmup step、early update norm、layer-wise grad norm | 延长 warmup；降低初始斜率；对比前几千 step | 增加 warmup tokens；降低 warmup slope；分阶段解冻或分阶段长度 curriculum | 修复早期优化器状态尚未稳定时的更新过大 | early steps 不再 spike；Adam v 逐渐建立 | warmup 过长会浪费 token budget |
| decay schedule 不合理 | decay 边界处 loss 抖动；LR plateau 太长导致后期不稳 | LR curve、loss slope、update ratio trend | 切换 cosine / linear / constant-with-decay 对比 | 修 schedule 类型、decay start、min LR、token-based step | 修复训练后期有效步长问题 | LR 变化处 loss 连续 | 可能改变最终收敛速度 |
| resume 后 LR 错误 | checkpoint 恢复后立刻 spike 或 loss 曲线断崖 | global step、scheduler step、param-group LR、optimizer step | 对比 resume 前后第一步 LR 和 update；加载前后打印 scheduler state | 正确恢复 global step、scheduler state、optimizer step | 修复状态不连续 | resume 前后单步 update 连续；loss 无断崖 | 需要回滚错误 resume 之后的训练段 |
| 参数组 LR 配错 | embedding、lm_head、norm 或 MoE router 层异常更新 | param-group LR、per-group update ratio、layer clip rate | 打印所有 param group；临时统一 LR 对比 | 修 param group mapping；为 embedding/lm_head/router 设置合理 LR | 修复局部有效学习率错误 | 各参数组 U/W ratio 回到合理区间 | 改 LR group 后需重新调最终 recipe |
| batch size / LR scale 不匹配 | 改 global batch 后训练不稳；effective update 变化过大 | tokens/update、samples/update、LR/token ratio | 保持 LR 不变 / 重新 scale 对比；模拟旧 batch size | 按实际 token batch 重新调 LR、warmup、gradient accumulation | 修复有效 batch 与 LR 的动力学错配 | 不同 batch 配置下 update ratio 接近 | 可能影响梯度噪声和泛化 |


---

### 4.3.4 Optimizer 问题
| 根因类别 | 典型异常现象 | 预兆指标 | 如何验证是不是这个根因 | 根因修复方法 | 为什么这属于 A 类 | 修复后的验证方式 | 可能副作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Adam beta 设置不合适 | 梯度噪声大，m/v 跟踪滞后；loss spike 后恢复慢 | Adam m norm、v norm、effective update、grad variance | 改 beta2；缩短/延长统计窗口对比；观察 effective update | 调 beta1/beta2；按 batch size 和任务阶段重新设定 | 修复优化器统计时间尺度 | m/v 分布稳定；spike 后不再产生过大 update | beta 调整会改变收敛和泛化 |
| Adam epsilon 不合适 | v 很小的参数 update 异常大；稀疏参数或 norm 层不稳 | v min、effective update max、U/W ratio by param | 调大 epsilon；检查小 v 参数集合；重放异常 step | 调整 epsilon；对特定参数组设置更稳健 epsilon | 修复 denominator 过小导致的有效步长异常 | 小 v 参数 update 不再极端 | epsilon 过大可能削弱 Adam 自适应性 |
| bias correction / step count 错误 | resume 或初期 update 明显异常；optimizer step 与 global step 不一致 | optimizer step、bias correction factor、m/v scale | 打印 step count；与连续训练 reference 对比 | 正确恢复 optimizer step；修 bias correction 实现 | 修复 Adam update 公式状态 | resume 前后 update 一致 | 需丢弃错误状态训练结果 |
| weight decay 配错 | norm、bias、embedding 或 lm_head 权重异常漂移；valid loss 恶化 | decay contribution norm、param norm、row norm | 分别关闭 / 开启特定参数 decay；打印 param group | 使用 decoupled weight decay；排除 norm/bias 或按 recipe 设置 embedding/lm_head | 修复正则项施加对象错误 | param norm trend 正常；valid loss 恢复 | weight decay 改变泛化，需要重新调 |
| optimizer state 损坏 / 丢失 | grad 正常但 update 爆；resume 后立刻异常 | state finite ratio、m/v shape、state checksum、optimizer step | 从 full-state checkpoint 对比；reset optimizer state 对比 | 修 checkpoint save/load、state dict mapping、ZeRO/FSDP shard restore | 修复实际 update 源头 | 同 checkpoint 连续训练与 resume 单步一致 | 有时只能回滚到更早 checkpoint |
| update-to-weight ratio 过大 | 某层参数相对更新过大；训练未必 NaN 但质量退化 | U/W ratio mean/p95/max、layer-wise update norm | 降 LR 或修 optimizer state 后 replay；按层统计异常参数组 | 修 LR、epsilon、state、param group；必要时调整初始化尺度 | 修复有效更新尺度 | U/W ratio 回到历史正常范围；loss 平滑 | 若只降低全局 LR，可能使其他层欠学习 |
| optimizer 与 gradient accumulation 缩放不匹配 | effective update 比预期大/小；更改 accumulation 后不稳定 | loss reduction scale、grad norm per microbatch、optimizer step frequency | 手算一个 accumulation cycle；比较 true large batch | 修 loss division、accumulation step、optimizer step 调用频率 | 修复更新频率和梯度尺度 | accumulation 与等价大 batch update 接近 | 修复后吞吐或显存策略可能变化 |


---

### 4.3.5 初始化与参数尺度问题
| 根因类别 | 典型异常现象 | 预兆指标 | 如何验证是不是这个根因 | 根因修复方法 | 为什么这属于 A 类 | 修复后的验证方式 | 可能副作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 初始化方差不合理 | step 0 forward 已出现 activation RMS 层间漂移；第一步 grad 异常 | step0 layer RMS、param std、fan-in/fan-out、grad norm | 只跑 step0 forward/backward；与 reference init 对比 | 修 fan-in/fan-out、std、truncated normal、projection init | 修复信号传播初始条件 | step0 各层 RMS 平稳；初期 grad 不爆 | 初始化过小会降低早期学习速度 |
| embedding 尺度异常 | logits 或 hidden RMS 从输入端开始异常；特殊 token row norm 大 | embedding RMS、row norm p99、input hidden RMS | 缩放 embedding replay；检查 tokenizer/vocab resize 后 init | 修 embedding init、resize 后新 token 初始化、embedding scale | 修复输入表示尺度 | input hidden RMS 正常；logits 不再早期极端 | 新 token 学习可能变慢 |
| lm_head 尺度异常 | logits max 极大；softmax entropy 过低；CE 易 spike | lm_head row norm、logits RMS/max、top-1 prob | 缩放 lm_head；冻结 lm_head 对比；检查 tied weight | 修 lm_head init、tie weight、output scaling、param group LR | 修复输出层尺度源头 | logits 分布正常；entropy 不再塌缩 | 可能影响早期 loss 下降速度 |
| residual branch 初始化不合理 | 深层 residual stream 随 depth 累积放大；越深越不稳 | residual input/output RMS、block output/input ratio、depth trend | 缩放 residual branch；减少层数对比；同宽浅模型对比 | 调整 residual branch init、residual scaling、block output scale | 修复深层信号累积机制 | layer-wise RMS 随 depth 平稳 | 缩放过强可能限制表达能力 |
| norm 参数初始化异常 | LayerNorm/RMSNorm 输出尺度异常；gamma 快速放大 | norm gamma norm、norm input/output RMS | 将 gamma 恢复默认值 replay；检查加载权重 | 修 norm gamma/bias 初始化或 checkpoint mapping | 修复归一化层尺度 | norm output RMS 稳定 | 可能改变已有 pretrained checkpoint 行为 |
| 权重加载 / shape mapping 错误 | 某些层参数 norm 与相邻层差异极大；加载后第一步即异常 | param norm by layer、state dict missing/unexpected keys、checksum | 与 reference checkpoint 对比；随机初始化该层对比 | 修 state dict mapping、tensor parallel merge/split、transpose 逻辑 | 修复参数语义错误 | 加载后 layer norm/param norm 与 reference 一致 | 可能需要重新生成 checkpoint |


---

### 4.3.6 架构问题
| 根因类别 | 典型异常现象 | 预兆指标 | 如何验证是不是这个根因 | 根因修复方法 | 为什么这属于 A 类 | 修复后的验证方式 | 可能副作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Post-LN 深层不稳定 | 深层模型早期梯度不稳；需要极长 warmup 或高频 clip | layer-wise grad norm、early update ratio、residual RMS | 与 Pre-LN / DeepNorm / 更长 warmup 对比；减少深度对比 | 改 Pre-LN、引入 residual scaling 或 DeepNorm 类设计、重新调 warmup | 修复归一化位置导致的梯度流问题 | early training 不再依赖高频 clip；深层 grad 平稳 | 架构变化可能影响最终上限，需要重新调 recipe |
| attention logits 过大 | attention entropy collapse；某些 head 近似 one-hot；logits spike | Q/K norm、QK score p99、attention entropy、max attention prob | 对比 QK Norm、temperature、RoPE 修复；hook head-wise score | 修 Q/K 初始化、QK Norm、attention scale、temperature、RoPE 参数 | 修复 attention score 生成机制 | score p99 与 entropy 回归；head 不再异常塌缩 | attention 变钝可能影响长距离检索 |
| RoPE / position 架构配置不合理 | 长上下文训练不稳；高位置 token loss spike | score by distance、loss by position、position id、QK norm by position | 不同上下文长度分桶；替换 RoPE scaling 配置 | 修 RoPE base、scaling、插值策略、position offset | 修复位置编码与训练长度不匹配 | 长序列 loss 平滑；attention entropy 随距离合理 | 可能牺牲短上下文或长上下文某一端性能 |
| residual path 尺度失衡 | attention 或 MLP residual contribution 明显大于主流 | residual delta/input ratio、attention output RMS、MLP output RMS | 缩放 attention/MLP residual branch；冻结某 branch replay | 调整 residual branch scale、init、gate、normalization placement | 修复残差路径贡献失衡 | delta/input ratio 回归；深层 RMS 不再累积 | 可能改变模块贡献比例 |
| MLP 输出尺度异常 | MLP down_proj 输出 outlier；SwiGLU gate 产生极端激活 | gate RMS/max、up_proj RMS、down_proj RMS、activation kurtosis | 替换 activation；缩放 gate/up/down init；hook 第一个异常层 | 修 activation 实现、SwiGLU gate scale、MLP init、bias 设置 | 修复 FFN 激活源头 | MLP output RMS 与 attention output 同量级 | 可能降低模型非线性表达 |
| activation function 实现错误 | GELU/SwiGLU/SILU 输出分布异常；grad 在激活层后爆 | activation input/output histogram、nonfinite ratio | 用 reference implementation 对比单层输出 | 修 activation 公式、dtype、fused kernel、inplace 操作 | 修复计算图语义错误 | 单层 reference test 通过；histogram 正常 | 替换 kernel 可能降低吞吐 |
| normalization placement / residual order 错误 | 与预期架构 loss 曲线不一致；梯度流异常 | block input/output RMS、norm input RMS、grad by depth | 对照 reference block；单层 forward equivalence test | 修 Pre-LN/Post-LN 顺序、residual add 顺序、dropout 位置 | 修复架构实现错误 | block 输出与 reference 接近；训练曲线恢复 | 修复后旧 checkpoint 可能不可直接兼容 |


---

### 4.3.7 混合精度与数值精度问题
| 根因类别 | 典型异常现象 | 预兆指标 | 如何验证是不是这个根因 | 根因修复方法 | 为什么这属于 A 类 | 修复后的验证方式 | 可能副作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 overflow | loss scale 频繁下降；Inf/NaN 出现在 attention、MLP 或 CE | overflow count、loss scale、nonfinite tensor location、activation max | bf16/fp32 replay；关闭 autocast；定位 first nonfinite op | 改 bf16；关键 op 用 fp32；动态 loss scaling；降低初期 LR | 修复数值范围不足 | overflow 频率下降；loss scale 稳定 | 显存和吞吐可能下降 |
| bf16 / fp32 混用错误 | 某些 op 输出 dtype 不符合预期；grad 或 optimizer state 异常 | dtype trace、master weight dtype、optimizer state dtype | 打印关键 tensor dtype；用 reference precision 跑单步 | 修 autocast scope、master weight、optimizer state dtype、cast 位置 | 修复 dtype 语义和精度路径 | dtype trace 符合预期；single-step 与 reference 接近 | 更多 fp32 会增加显存 |
| loss scaling 不合理 | loss scale 过大频繁 overflow，或过小导致 underflow 风险 | loss scale trend、skipped steps、grad finite ratio | 固定 loss scale 对比；动态 scaler 参数对比 | 调 scaler growth/backoff、initial scale；同步 rank overflow | 修复混合精度梯度范围 | skipped step 下降；grad finite | scaler 过保守会降低有效训练效率 |
| softmax overflow / underflow | attention softmax 或 CE 出 NaN；score 大时崩溃 | pre-softmax max、logsumexp finite、attention prob finite | 替换 stable softmax；fp32 softmax replay；构造极端 score toy test | 使用减 max、logsumexp、fp32 accumulation、稳定 CE kernel | 修复 exp/log 数值实现 | toy extreme case 稳定；attention prob finite | 稳定 kernel 可能更慢 |
| exp / log / sqrt / divide 不稳定 | NaN 出现在自定义 loss、regularizer、router loss 或 norm 计算 | denominator min、log input min、sqrt input min、finite ratio | 给 denominator 加断言；对比 reference 实现 | 加 epsilon、clamp 合法域、稳定公式重写 | 修复数学域错误 | first nonfinite 消失；单元测试覆盖边界输入 | epsilon 过大会改变数值行为 |
| fused kernel 数值问题 | 只在开启 fused op 时出现异常；关闭后稳定 | kernel on/off 对比、dtype trace、first nonfinite op | 替换 unfused reference；逐 kernel bisect | 修 kernel、换版本、关闭高风险 fused path、增加 fp32 accumulation | 修复实现级数值误差 | fused/unfused 单步输出误差可控 | 关闭 fused kernel 会降低吞吐 |


---

### 4.3.8 分布式训练问题
| 根因类别 | 典型异常现象 | 预兆指标 | 如何验证是不是这个根因 | 根因修复方法 | 为什么这属于 A 类 | 修复后的验证方式 | 可能副作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 梯度同步错误 | 单卡稳定，多卡不稳；rank-wise grad norm 分叉 | pre/post all-reduce grad norm、rank grad norm、param checksum | 单卡 / 多卡对比；关闭 bucket/fused all-reduce；检查 no_sync | 修 DDP all-reduce、bucket、no_sync、gradient hook | 修复跨 rank 更新不一致 | rank-wise grad 和参数 checksum 一致 | 可能降低通信效率 |
| gradient accumulation 错误 | effective batch 与预期不符；accumulation 改变后 loss 大幅变化 | microbatch loss scale、accum step、optimizer step frequency | 手算 accumulation cycle；与真实大 batch 对比 | 修 loss division、accumulation boundary、optimizer step 调用 | 修复有效梯度尺度 | accumulation update 与等价大 batch 接近 | 可能改变吞吐与显存 |
| global batch size / token batch 计算错误 | LR scaling 错；tokens/update 与配置不一致 | tokens/update、samples/update、world size、grad accumulation | 打印每 step token count；对比配置表 | 修 batch size 公式、drop_last、packing token accounting | 修复有效 batch-LR 关系 | tokens/update 稳定；LR scale 合理 | 改 batch 后需重新调 schedule |
| rank 间数据不一致 | 某个 rank loss 长期偏高；overflow 只在少数 rank 出现 | rank loss、rank batch length、source mix by rank、overflow by rank | 单独 replay 异常 rank batch；检查 sampler shard | 修 distributed sampler、seed、shard assignment、data resume state | 修复分布式数据分布错误 | rank-wise loss 分布接近 | 修 sampler 可能改变数据顺序 |
| ZeRO / FSDP optimizer state 错误 | resume 后某些 shard update 异常；局部参数 NaN | shard finite ratio、optimizer shard checksum、state shape | full-state checkpoint round-trip；禁 ZeRO/FSDP 小规模 replay | 修 shard save/load、reshard mapping、state dict key、offload dtype | 修复 optimizer/parameter shard 状态 | full-state 与 sharded-state 单步一致 | checkpoint 成本可能增加 |
| checkpoint resume 状态不一致 | resume 后 loss 断崖、LR 错、loss scale 错、optimizer step 错 | scheduler state、global step、optimizer step、scaler state、rng state | 连续训练 vs resume 后第一步对比 | 完整保存/恢复 model、optimizer、scheduler、scaler、sampler、rng | 修复训练状态连续性 | resume 前后一到数步 loss/update 连续 | 需要增强 checkpoint 元数据 |
| pipeline / tensor parallel mapping 错误 | 某些层或分片输出异常；TP/PP 模式下才炸 | shard param norm、activation checksum、stage-wise loss | 小模型关闭 TP/PP 对比；检查 split/merge/transpose | 修 TP shard 维度、PP stage mapping、通信顺序 | 修复并行计算图语义 | 并行与非并行 reference 单步接近 | 修复后可能影响 checkpoint 兼容 |


---

### 4.3.9 MoE 问题
| 根因类别 | 典型异常现象 | 预兆指标 | 如何验证是不是这个根因 | 根因修复方法 | 为什么这属于 A 类 | 修复后的验证方式 | 可能副作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 专家负载不均 | 少数 expert token load 极高；某些 expert grad/update 爆 | expert load CV、tokens/expert、expert grad norm、drop ratio | 打印 routing top-k；调 aux loss / capacity 对比；dense ablation | 调 load balancing loss、capacity factor、router temperature、routing init | 修复 routing 分配机制 | expert load 更均匀；expert grad 不再极端 | 过强均衡可能损害专家专化 |
| router logits 爆炸 | router entropy collapse；top expert 占比过高；expert collapse | router logits max/RMS、router entropy、top expert share | 缩放 router logits；增加 z-loss；fp32 router 对比 | router z-loss、router init、router precision、temperature | 修复 gating score 源头 | router entropy 稳定；top expert share 回落 | z-loss 过大可能削弱路由表达 |
| capacity factor 设置不合理 | dropped tokens 高；loss 与 drop ratio 强相关 | capacity usage、dropped tokens ratio、overflow tokens | 增大 capacity factor；按长度/source 分析 drop | 调 capacity factor、expert parallelism、token dispatch policy | 修复 token 被丢弃或过载的机制 | drop ratio 降低；loss 不再随 drop spike | capacity 增大提高显存和通信 |
| aux loss 权重不合理 | aux loss 主导训练或无法平衡负载；main loss 退化 | aux/main loss ratio、expert load、router entropy | 扫描 aux weight；观察 main loss 与 load balance | 调 aux loss 权重、归一化方式、调度策略 | 修复辅助目标对主目标的干扰 | main loss 稳定且 load 合理 | 权重过小负载不均，过大损害任务学习 |
| 某些 expert 梯度异常 | 个别 expert grad norm 或 update ratio 长期极端 | per-expert grad norm、U/W ratio、expert load、expert state finite | 冻结/重置异常 expert；检查该 expert 数据流和 optimizer state | 修 expert init、optimizer state、dispatch、LR group | 修复局部 expert 状态 | 异常 expert 与其他 expert 指标接近 | 重置 expert 可能造成短期能力波动 |
| expert parallel / all-to-all 错误 | MoE 多机下不稳，dense 或单机稳定；token dispatch 错位 | dispatch checksum、tokens sent/received、expert id mapping | 禁用 expert parallel 对比；toy routing case 检查 all-to-all | 修 all-to-all mapping、expert id、token combine order | 修复 MoE 通信语义 | toy dispatch/merge 完全一致；rank-wise load 正常 | 通信实现修复可能影响吞吐 |
| shared expert / routed expert 尺度失衡 | shared expert 输出压过 routed expert，或反之；residual RMS 异常 | shared/routed output norm、expert contribution ratio | 缩放 shared expert；分别冻结 shared/routed 对比 | 调 shared expert scale、router gate、residual merge 方式 | 修复 MoE 分支贡献失衡 | contribution ratio 稳定；loss 不再 spike | 改变专家分工和最终能力 |


---

### 4.3.10 代码实现 bug
| 根因类别 | 典型异常现象 | 预兆指标 | 如何验证是不是这个根因 | 根因修复方法 | 为什么这属于 A 类 | 修复后的验证方式 | 可能副作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| label shift 错误 | loss 无法下降或异常低；模型预测错位 token | input-label alignment、per-position loss、tiny batch overfit | 打印 input/label 对齐；手算短序列 CE | 修 shift 方向、slice 范围、BOS/EOS 处理 | 修复训练目标语义 | tiny batch 可 overfit；per-position loss 合理 | 修复后 loss 标尺变化 |
| loss reduction 错误 | 改 batch size 或 seq length 后 grad scale 异常 | loss denominator、valid token count、grad norm vs length | 手算 loss mean/sum；固定 token 数对比 | 按 valid token 正确归一化；修 microbatch reduction | 修复梯度尺度 | 不同 padding/length 下 grad scale 稳定 | 旧实验曲线不可直接比较 |
| tokenizer 与模型 vocab 不一致 | embedding 越界、特殊 token 错、某些 token logits 异常 | vocab size、token id range、special token id、embedding row norm | 检查 tokenizer hash；打印 token ids；对齐 vocab 文件 | resize embedding；同步 tokenizer/config；修 special ids | 修复输入 id 到参数的映射 | token id 无越界；special token loss 正常 | 新增 token 参数需训练适应 |
| ignore_index 配错 | pad、prompt 或 masked token 参与 loss；SFT loss 异常 | ignore ratio、masked token loss、pad token loss | 打印 labels 中 ignore_index；构造 prompt-only batch | 修 ignore_index、loss mask、collator | 修复哪些 token 应参与优化的问题 | pad/prompt masked 区域 loss 为 0 | 有效 token 数变化影响 LR / loss scale 感知 |
| position embedding / RoPE 实现错 | 某些位置 loss spike；长序列不稳 | position id、cos/sin cache shape、position offset | 与 reference RoPE 单步输出对比；短序列 toy test | 修 cos/sin cache、offset、interleaving、head dim 维度 | 修复位置计算图 | reference equivalence test 通过 | 旧 checkpoint 位置行为可能变化 |
| dropout / eval mode 错误 | train/valid 行为异常；resume 后随机性不一致 | model.training、dropout mask、rng state、eval loss variance | 固定 seed 检查输出；切换 train/eval 对比 | 修 mode 切换、rng restore、dropout placement | 修复训练状态机 | train/eval 输出符合预期；valid 稳定 | 修复后正则强度变化 |
| causal mask dtype / additive mask 符号错误 | softmax 后 masked position 仍有概率；或所有位置被 mask | mask dtype、mask value、attention prob on masked positions | 构造极短序列检查 attention probability | 修 bool/additive mask 转换、负无穷值、广播维度 | 修复 attention 语义 | masked prob 为 0；无 all-masked NaN | 某些 kernel 对 mask dtype 有性能差异 |
| 梯度检查点 / recomputation bug | 开启 checkpointing 才炸；forward 正常 backward 异常 | checkpoint on/off 对比、activation recompute dtype、grad mismatch | 关闭 gradient checkpointing；与 non-checkpoint reference 对比 | 修 recompute function、rng、autocast scope、inplace op | 修复 backward 计算图 | checkpoint on/off 单步 grad 接近 | 关闭或修正后显存/吞吐变化 |
| 自定义 loss / reward / regularizer bug | 主 loss 正常但 auxiliary loss 触发 NaN 或 update 异常 | aux loss value、aux/main ratio、nonfinite op | 单独计算各 loss；关闭 auxiliary term 对比 | 修 loss 公式、归一化、合法域、权重 | 修复额外目标的数值或语义错误 | aux loss 稳定且不主导更新 | 可能改变训练目标权重 |


---

## 4.4 A 类定位中的典型案例
### 案例 1：loss spike 来自 packed sequence 串样本
**现象**：训练到某个 step 后出现固定 batch loss spike，global grad norm 随后升高。加大 gradient clipping 后 NaN 消失，但 valid loss 变差。

**预兆指标**：

| 指标 | 异常 |
| --- | --- |
| per-sequence loss | 某些 packed sample 边界附近 token loss 极高 |
| attention mask density | packed batch 的 mask density 与预期不一致 |
| illegal attention ratio | token 可以 attend 到上一个样本结尾 |
| packed vs unpacked loss | 同样本 unpacked 后 loss 正常 |


**A 类修复**：

1. 构造两个短样本拼接的 toy batch。
2. 打印 block diagonal attention mask。
3. 发现 segment boundary 处没有完全屏蔽。
4. 修复 packed boundary mask 和 position id reset。
5. replay 原 bad batch，loss spike 和 grad spike 消失。

**为什么不是 B 类**：  
加 gradient clipping 只能限制 spike 后的梯度幅度，但模型仍然看到错误上下文。修复 mask 后，异常输入语义被消除。

---

### 案例 2：resume 后 update-to-weight ratio 异常
**现象**：从 checkpoint resume 后第一百个 step 内 loss 断崖式上升。global grad norm 不算极端，但 update norm 和 update-to-weight ratio 明显大于 resume 前。

**预兆指标**：

| 指标 | 异常 |
| --- | --- |
| scheduler step | 恢复成 0 或偏移 |
| Adam step | 与 global step 不一致 |
| LR | 比预期大 |
| Adam v | 部分参数组缺失或被重新初始化 |
| U/W ratio | embedding 和 lm_head 层异常大 |


**A 类修复**：

1. 对比连续训练和 resume 后第一步的 LR、Adam m/v、optimizer step。
2. 发现 scheduler state 和部分 optimizer shard 未正确恢复。
3. 修 checkpoint loader，恢复 optimizer、scheduler、scaler、sampler、rng。
4. replay resume step，update norm 与连续训练对齐。

**为什么不是 B 类**：  
update clipping 可以暂时限制参数变化，但错误的 optimizer state 会持续产生错误 update。修复 checkpoint state 才是根因治理。

---

### 案例 3：attention entropy collapse 来自 position id 错误
**现象**：短序列训练稳定，长序列 batch 经常出现 loss spike。某些 attention head 的 entropy 接近 0，score p99 极高。

**预兆指标**：

| 指标 | 异常 |
| --- | --- |
| position id range | packed 后部分样本 position id 未 reset 或出现跳变 |
| QK score by position | 高位置 token score 异常放大 |
| attention entropy | 部分 head 在长序列处 collapse |
| loss by position | 靠近高 position 区域 loss spike |


**A 类修复**：

1. 按 sequence length 分桶 replay。
2. 打印 position id 和 RoPE offset。
3. 修 packed sequence 内的 position id reset 或 RoPE offset。
4. 长序列 replay 后 attention score 和 entropy 恢复。

**为什么不是 B 类**：  
attention score clamp 可以降低 score max，但不能修复位置编码语义错误。位置 id 修复后，attention score 不再从源头异常。

---

### 案例 4：MoE expert collapse 来自 router logits 失控
**现象**：MoE 模型训练中某些 expert load 极高，其他 expert 几乎空闲。局部 expert grad norm 和 update ratio 爆炸，主 loss 随 drop token ratio 波动。

**预兆指标**：

| 指标 | 异常 |
| --- | --- |
| expert load CV | 持续升高 |
| router entropy | 下降到极低 |
| top expert share | 少数 expert 占据多数 token |
| dropped tokens | capacity overflow 上升 |
| expert U/W ratio | overloaded expert 更新异常 |


**A 类修复**：

1. 打印 router logits、top-k routing 和 expert load。
2. 发现 router logits 尺度过大，top expert 被过度选择。
3. 调整 router init / temperature / z-loss / aux loss 权重 / capacity factor。
4. replay 后 expert load、router entropy、drop ratio 稳定。

**为什么不是 B 类**：  
只 clip overloaded expert 的 gradient 会降低局部爆炸，但 routing collapse 仍然存在。修复 router 动力学才是根因治理。

---

## 4.5 A 类修复的优先级
当多个异常同时出现时，不应按“哪个数最大”排序，而应按“哪个最早、最局部、最可复现、最能解释后续异常”排序。

建议优先级如下：

| 优先级 | 根因方向 | 原因 |
| --- | --- | --- |
| 1 | 数据、label、mask、packing、position id、tokenizer、loss reduction | 这些是训练语义正确性的基础；错误时 clip 无法修复目标 |
| 2 | resume、LR、scheduler、optimizer state、param group | 这些直接决定 effective update，错误时会持续污染参数 |
| 3 | precision、softmax、loss scaling、kernel | 这些可能直接产生 NaN/Inf，需要尽快隔离 |
| 4 | 分布式同步、ZeRO/FSDP、rank 数据一致性 | 多卡问题会让异常不可复现，必须先恢复一致性 |
| 5 | 初始化、参数尺度、架构残差路径、attention/MLP 尺度 | 这些通常影响长期稳定性和可扩展性 |
| 6 | MoE routing、expert load、aux/z-loss | 稀疏模型中 routing 本身就是训练动力学核心 |
| 7 | 长期 recipe 调整 | 如 warmup、curriculum、depth scaling、normalization placement，需要与质量指标一起评估 |


一个实用规则是：

**凡是会改变训练目标语义或输入语义的问题，优先作为 A 类根因修复；凡是只改变数值幅度的问题，先判断是否有上游语义或状态错误。**

---

## 4.6 A 类修复后的验收清单
A 类修复完成后，应同时进行局部验收和全局验收。

### 4.6.1 局部验收
| 验收项 | 操作 |
| --- | --- |
| 原 bad batch replay | 修复前触发异常的 batch，修复后应不再触发同类异常 |
| toy case 单元测试 | mask、position id、loss shift、packing、router dispatch 等必须可手工验证 |
| single-step reference | 与 unfused / fp32 / non-sharded reference 做单步输出和梯度对比 |
| first bad tensor 复查 | 修复后原 first bad tensor 不再异常 |
| 参数组检查 | LR、weight decay、dtype、optimizer state、requires_grad 与预期一致 |


### 4.6.2 全局验收
| 验收项 | 合格标准 |
| --- | --- |
| loss 曲线 | 不再出现相同模式 spike；恢复与历史正常 run 接近的斜率 |
| activation 分布 | residual RMS、MLP activation、attention output 不再出现持续上升或层间阶跃 |
| attention 分布 | score p99、entropy、max prob 回到合理范围 |
| optimizer update | update norm 和 U/W ratio 稳定，不再由少数层主导 |
| clip / overflow | clip rate、overflow skip、bad batch quarantine 触发率下降 |
| rank consistency | rank-wise loss、grad norm、batch shape、param checksum 一致 |
| MoE 指标 | expert load、drop ratio、router entropy、expert update 稳定 |
| 质量指标 | train loss、valid loss、下游 eval 不因修复明显退化 |


---

## 4.7 A 类修复中的常见误判
| 误判 | 为什么危险 | 更可靠的判断方式 |
| --- | --- | --- |
| “NaN 出现在 loss，所以 loss 是根因” | loss 往往只是最后一个暴露非有限值的地方 | 找 first nonfinite tensor，追溯到 attention、MLP、logits 或 optimizer update |
| “grad norm 大，所以根因是 gradient explosion” | gradient explosion 通常是下游症状 | 看 activation / logits / per-sample loss 是否先异常 |
| “clip 后不炸，所以问题解决了” | clip 可能只是让异常不再扩散 | 检查 clip rate、valid loss、update direction cosine、上游指标是否仍恶化 |
| “某个 batch loss 高，所以数据坏” | 难样本也可能产生高 loss | 做 replay、去掉样本、检查 label/mask/source，而不是直接删除 |
| “单卡稳定，多卡不稳定，所以模型结构没问题” | 多卡问题可能来自数据、同步、shard state，也可能放大已有数值问题 | 单卡/多卡、fp32/bf16、sharded/full-state 多维对比 |
| “warmup 增加后稳定，所以根因是 warmup” | warmup 可能掩盖 mask、state 或 LR 配置错误 | 检查 early update、optimizer state、mask 和 batch 数据 |
| “某层 activation 大，所以该层有 bug” | 该层可能只是接收了上游异常输入 | 比较该层 input 是否已异常，定位第一个 output/input ratio 异常的模块 |
| “MoE 某 expert 梯度大，所以 clip expert grad” | overloaded expert 的梯度大可能来自 routing collapse | 先看 router entropy、expert load、drop ratio、capacity |


---

## 4.8 本章小结
A 类方法的核心是 **因果归因和机制修复**。在大模型训练稳定性工程中，真正高质量的 A 类修复通常具备以下特征：

1. 能指出异常产生的具体链路，例如：  
`packed boundary mask 错 → attention 跨样本 → logits 异常 → loss spike → grad norm 爆炸`。
2. 能通过最小实验复现，例如：  
`同 checkpoint + 同 bad batch replay`、`packed vs unpacked 对比`、`fp32 vs fp16 对比`、`single-rank vs multi-rank 对比`。
3. 能修复源头机制，例如：  
修 mask、修 label shift、修 scheduler state、修 Adam state、修 RoPE position id、修 optimizer shard restore、修 router logits 动力学。
4. 修复后不再依赖高频硬约束，例如：  
gradient clipping 仍可保留，但 clip rate 应显著降低；loss scale skip 仍可存在，但不应连续触发；bad batch quarantine 仍可作为保护，但不应成为常规训练路径。

可以将 A 类方法概括为一句工程原则：

**不要只问“哪个值太大”，要问“哪个机制让它第一次变大”。A 类修复的目标，是让异常不再从源头产生，而不是让异常产生后被压住。**

