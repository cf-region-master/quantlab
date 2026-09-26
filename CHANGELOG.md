# Changelog

## feature/debug-upgrade-innovation（2026-09-27）

全面代码审查后的修复与升级。基线 33 项测试 → **53 项全部通过**（新增 21 项；
含 2 项此前在干净克隆上必然失败、现已有 skip 守卫的环境测试）。

### 修复（正确性 / 一致性）

| # | 问题 | 修复 |
|---|---|---|
| 1 | `universe.index_member_mask` 在 pandas 3.0 下 `col \|= sel` 触发 "output array is read-only"（CoW 只读视图），**新环境流水线直接崩溃** | 改为独立 bool 数组装配后一次性建 DataFrame（`universe.py`） |
| 2 | `factors/model.py`（linear/tree 信号）对每个分量写死 `["winsorize","zscore"]` 且不做股票池限定——同一信号换 model_type 因子口径漂移 | 与 `signal_panel` weighted 路径同源：按各因子自己的 `preprocess_spec` 处理 + `_apply_pool` 限定池（`model.py::fit_walk_forward`） |
| 3 | `diagnose_signal` 分层净值传 `pool_mask=None`：池约束信号的分层统计跑在全样本上 | 新增 `_signal_pool_mask`：分层域 = 各分量因子所属池的逐日交集（`store.py`） |
| 4 | `refresh_factor` 非原子：先覆盖值 parquet 再算诊断/提交，中途失败留下「新值文件+旧元数据」且旧值不可恢复 | 先写 `.tmp`，DB 提交成功后原子替换（`_write_values_tmp/_promote_values_tmp`） |
| 5 | `delete_factors` 在 commit 前 unlink 值文件：提交失败则 DB 行在、文件已丢 | 收集路径，commit 成功后再删 |
| 6 | `check_data_sufficiency` 循环内 `if not problems` —— 前面因子出问题后，后续正常因子不进 `checked` | 按因子名判定该因子自身是否有问题 |
| 7 | 质量报告 `missing_close_adj_cells` 统计「未 ffill 的段内因子拼出的内存调整价」，与加载派生语义（ffill 因子）不符（1883 vs 601 格），**报告与数据不一致** | 按 `raw 缺失 \|\| ffill 后因子缺失` 统计；移除误导性的内存调整价面板（`clean.py`） |
| 8 | `minute_expr._rolling` 在 w==1 时走恒等分支，`Ref(x,1)/Delta(x,1)` 错误返回 x（引擎语义：Ref/Delta 先于 w==1 快捷分支，是真 shift） | Ref/Delta 提前分流（`minute_expr.py`，移植自分钟层，测试捕获） |
| 9 | `signals.html`、`/signals/new` 硬编码 h=20：AlphaGen 因子常只有 h=5，页面显示为空 | 取每个因子实际存在的最大 horizon 诊断，随附期数展示 |
| 10 | `job.html` 同一 div 两个 `style` 属性（第二个被浏览器忽略） | 合并 |
| 11 | `pool.metrics(exact=...)` 参数被实现完全忽略（误导性接口） | 移除参数并对齐 3 处调用方，docstring 说明精确重算走平台诊断链路 |
| 12 | `checks.py` 注释里粘着死代码文本（不存在的 `fdates` 变量） | 清理并补准确注释 |
| 13 | `model.py` 死代码（未用的 `train_end`、弃用的 `seg` 列表）；`refit_points` 为空时静默返回全 NaN 面板 | 删除死代码；空拟合点显式报错并说明原因 |

### 修复（健壮性 / 工程）

| # | 问题 | 修复 |
|---|---|---|
| 14 | `tests/test_repro.py` 依赖本地数据，干净克隆必失败（本次实测 2 个 FAILED） | 数据缺失时 `pytest.skip`（抓数后自动恢复） |
| 15 | `lab/alphagen/tracking.py` 顶层 `import psutil` 但 requirements 无 psutil，且模块无任何引用（死模块，接线即 ImportError） | 删除；`gymnasium` 显式进 requirements |
| 16 | `lab/alphagen/data.py` 的 5 分钟 `build_cache` 死路径（日频流程不使用，BARS_PER_DAY=48 与日频版混淆） | 删除 |
| 17 | `@app.on_event("startup")` 已被 FastAPI 弃用 | 迁移到 lifespan |
| 18 | `GpTaskCreate` pydantic 模型定义后从未使用 | 删除 |
| 19 | `market()` 单例与池缓存永不失效：服务运行期间重跑流水线后页面用旧面板 | `store.reset_caches()` + `POST /api/admin/cache/reset` |
| 20 | `fundamentals._wide` 每次调用对百万行长表重新 pivot（批量中性化反复触发） | 按字段 lru_cache（`clear_cache` 一并清理） |
| 21 | 陈旧文案：api/main.py docstring（复现包已下线）、未知引擎报错仍说 GPU 版未接入、`PROJECT1_CHECKLIST.md` 测试数/页面引用过期、`configs/data.yaml` 与 PIT 主流程矛盾 | 全部更新为现状（data.yaml 注明快照模式与 Tushare PIT 模式两种口径） |

### 升级创新

1. **成本印花税分项**（卖出单边）：`cost.stamp_duty_sell`（默认 0.05%，2023-08-28 起 A 股口径；
   时间变动未建模、已声明）。逐笔账本分项 `commission/stamp_duty`，成本门禁按同口径逐笔重算，
   metrics 输出 `commission_total/stamp_duty_total`。实测 base_v3：总成本 278,651 = 佣金 238,968 + 印花税 39,683。
2. **分钟级数据层**（自研，已在真实 918MB 沪深300 五分钟宽表上验证）：
   `quantlab/data/minute.py`（两遍扫描 + float32 多字段面板）、`minute_expr.py`
   （alphagen5m RPN 表达式解释器，逐算子对齐引擎语义）、向量化快速诊断
   （`ic_table_fast/quantile_table_fast/run_diagnostics_fast`）、`configs/factors_5m.yaml`
   三只 5 分钟因子卡。9 个合成数据测试（不依赖真实数据）。真实数据上：1666 日×48bar×300 股、
   三因子诊断与 AlphaGen 候选交叉验证（8/10 与引擎 IC 差 <0.005）。
3. **沙箱 AST 静态防护**：用户代码 exec 前先过语法树检查——拒绝双下划线属性链、
   按名拦截 eval/exec/getattr/open/type 等危险内建、import 双重校验（与运行时 safe_import 叠加）。
   10 个专项测试（`tests/test_user_code.py`）。
4. **GitHub Actions CI**：`.github/workflows/tests.yml`——精简依赖（无 torch）跑全量测试，
   重依赖测试由 importorskip 守卫自动跳过。

### 与主线的关系

在 main@1bcbcc4 之上，不改变任何既有 API/页面契约；所有新配置键都有默认值，向后兼容。


## feature/debug-upgrade-innovation · 第二批（2026-09-27）：异步治理 + AlphaGen 划分修复

### 修复（用户实测痛点）

1. **AlphaGen「自己划分」必 FAILED —— 根因修复**：`segment_split` 对结束类边界
   （train_end/valid_end/test_end）误用 searchsorted-left，周末/节假日的边界被映射到
   下一个交易日、与下一段起点撞位，顺序校验误判拒绝。按自然月末/年末划分的合理输入
   全部失败。现改为：开始类边界取 ≥ 日期的首个交易日，结束类边界取 ≤ 日期的最后一个
   交易日。回归测试 3 项（周末边界/相邻交易日/部分边界）。
2. **表单防呆**：lab 页三个日期默认值原为同一起点，直接提交「自己划分」会 purge 后空段
   而 FAILED。现默认按 60%/80% 分割点预填；后端在创建任务前干跑 segment_split，
   非法边界返回 400 + 人话提示（不再入队后失败）。
3. **轮询接口 NaN 500**：训练曲线早期轮次常含 NaN（entropy 等），Starlette JSONResponse
   禁 NaN 导致轮询接口 500。响应统一做非有限值清洗。

### 升级：异步治理（根治「前端卡」）

| 改动 | 说明 | 实测 |
|---|---|---|
| **引擎进程隔离** | GP/AlphaGen 从 uvicorn 同进程线程改为独立子解释器（`python -m quantlab.lab.task_runner`）。numpy/torch 不再占 Web 进程 GIL；Windows 下不依赖 multiprocessing spawn 的 `__main__` 重导；服务重启不影响已派发任务 | 4 万步 GPU 挖矿运行中，全站页面最大延迟 **119ms**（多数 <40ms） |
| 信号/回测任务化 | 创建信号（walk-forward 拟合）与回测（逐日模拟）移出 HTTP 请求路径：POST 立即 303 到 `/jobs/{id}` 进度页，完成自动跳结果实体页（复用 BatchJob 轮询） | 创建信号 303 即回，job SUCCESS → /signals/{id} |
| SQLite WAL | journal_mode=WAL + busy_timeout=5000 + synchronous=NORMAL：挖矿高频写进度不再顶掉页面读 | — |
| 进度写节流 | 子进程 report() ≥2 秒才落盘一次，减少锁竞争 | — |
| 轮询瘦身 | `/api/lab/tasks/{id}` 支持 `log_since` 增量日志（不再全量回传）、`curve_max` 服务端降采样；前端只追加新日志、ETA 按进度速率外推 | — |
| 取消跨进程 | 取消旗标经文件传递（`data/store/task_cancel/{id}`），子进程在进度回调里检测并停止引擎 | — |

### 其他

- `AlphaGenParams.device`（auto/cpu/cuda）：本机 RTX 4060 + torch cu130 实测 GPU 训练
  （device=cuda 时 summary.device=cuda）；无 CUDA 自动回落 CPU。
- `norm_pool_id` 兼容内置池字符串键（`index_csi300` → 数字 id），不再 `int()` 崩溃。
- 任务创建前预检三段边界（两引擎共用）；陈旧 PENDING 任务在启动时清理。
- 测试 53 → **57 项全部通过**（新增：边界语义 3、池键兼容 1）。


## feature/debug-upgrade-innovation · 第三批（2026-09-27）：因子组合升级（"怎么更好地组合因子"）

### 新功能（四件套）

1. **组合前体检：因子相关性矩阵 + 去冗余建议**（`factors/combine.py::pooled_corr/redundancy_prune`）
   - `GET /api/factors/correlation?ids=…`：池化相关矩阵 + 按 |RankIC| 降序的贪心去冗余
     （|ρ|≥0.8 的低分因子被建议剔除）；signals/new 页选中因子后自动渲染热力图与建议。
   - 价值：5 个高相关因子等权相加 = 把单一风险放大 5 倍 —— 组合变差的头号原因。

2. **滚动 ICIR 加权 `ic_weight_rolling`**（walk-forward）
   - w(t) ∝ mean(IC_f[·<t-h]) / std(IC_f[·<t-h])；IC(t) 的标签用到 t+h 价格，
     因此权重先把 IC 序列 shift(h) —— **严格无前视**（专项测试断言：改动未来标签
     不影响历史权重）。
   - 同时如实标注旧 `ic_weight` 的全样本口径存在轻微样本内偏误（只作基线）。

3. **IC 均值-方差凸组合 `ic_meanvar`**（Grinold-Kahn 式，walk-forward）
   - w(t) ∝ (Σ(t)+λI)⁻¹μ(t)：μ=滚动因子 IC 均值、Σ=因子间 IC 协方差（同样 shift(h)
     纪律）；λ 相对化，闭式解毫秒级；Σ 不可逆/样本不足时退化 ICIR 权重。

4. **组合增益报告**（`combination_report` + 信号详情页 + `GET /api/signals/{id}/combination-report`）
   - 组合信号 vs 各分量单因子同口径 RankIC 对照、分量相关性摘要（min/max/平均|ρ|）、
     相对最强单因子的提升比例；符号翻转（组合翻正）时如实提示"增益比例不适用"。

### 实测（真实数据，mom20/rev5/lowvol20 三因子）

- `ic_meanvar` 组合 RankIC(5d) = **+0.0202**，而最强单因子为 **-0.0213**（动量）——
  分散化把负 IC 单因子的组合翻正，相关性摘要显示 mom20 与 rev5 相关 -0.51；
- `ic_weight_rolling` 组合 RankIC(5d) = +0.0192，同样为正；
- 相关矩阵给出明确的去冗余依据（本例无 ≥0.8 冗余对）。

### 新增测试

`tests/test_combine.py` 6 项：池化相关恒等、贪心去冗余剔除克隆、**无前视性质**
（截断数据重算，历史权重逐位一致）、Σ 对角闭式解方向、增益符号翻转提示、常数
因子 IC 记缺失。全量 **63 项测试通过**。


## feature/debug-upgrade-innovation · 第四批（2026-09-27）：IC 衰减/半衰期、行业中性、波动率目标、正交化合成、对照实验自动化

1. **Newey-West HAC 显著性修正**（重叠标签纪律）：`newey_west_tstat`（Bartlett 核，
   lag=h-1）；ic_summary 并列 t_stat / t_stat_nw；组合报告重算组合 IC 的 NW t；
   因子详情页与信号详情页展示。测试：重叠标签构造下 NW 收缩虚高 t、IID 时与普通 t 一致。
2. **IC 衰减曲线与半衰期**：`factors/decay.py` + `/api/factors/{id}/decay` + 因子详情页
   折线图；实测 mom20 最优 h=2、半衰期 10 个交易日（调仓频率量化依据，与 E1 对照互证）。
3. **组合层行业中性**：回测可选开关，申万 PIT 逐日标签 + 单行业上限
   ceil(top_n/10)；「未知」组同样受限；毛/净同口径。
4. **组合层波动率目标**（只降不升）：vol_target/vol_window，缩放系数
   min(1, vt/realized)；升仓需融资而引擎未建模融资（文档化取舍）；
   高波动市场实测波动显著压低。
5. **Schmidt 正交化合成 `ortho_ic_weight_rolling`**：按分量顺序保留原始信息、
   后续因子只留增量残差；对正交残差做滚动 ICIR —— 权重衡量增量信息贡献。
6. **组合方式对照实验自动化**：`scripts/compare_combinations.py` + `lab/comparison.py`
   —— 同因子集只改组合模型，输出 RankIC/普通 t/NW t 对照表并落盘 JSON。

### 对照实验实测（mom20/rev5/lowvol20，h=5，601 交易日）

| 模型 | RankIC | t(普通) | t(NW) |
|---|---|---|---|
| equal_weight | +0.0255 | 2.67 | 1.55 |
| ic_weight（全样本权重） | **-0.0208** | -2.30 | -1.28 |
| ic_weight_rolling | +0.0192 | 1.96 | 1.09 |
| ic_meanvar | +0.0202 | 2.10 | 1.17 |
| ortho_ic_weight_rolling | +0.0245 | 2.58 | 1.46 |

发现：ic_weight 全样本权重被动量的负 IC 主导把组合做成负——**全样本加权的
样本内偏误的活例证**（本仓库已如实标注并推荐 walk-forward 变体）。

### 测试
70 → **71 项全部通过**（新增 NW 3 项、行业中性 2 项、波动率目标 1 项、
组合器 7 项中的部分已在第三批计入）。


## feature/debug-upgrade-innovation · 第五批（2026-09-27 夜间续）：批评估增强 + 组合报告/增量 IC 导出

1. **批评估增强**：批量重算结果新增前后 RankIC(20d) 对照表（重算是否改善一目了然）。
2. **对照实验 CSV 导出**：`/experiments/{filename}.csv`（含净值层对照列），
   文件名白名单防目录穿越；实验历史卡片加导出链接。
3. **组合增益报告 CSV 导出**：`/api/signals/{id}/combination-report.csv`
   （combined/component/correlation/gain 分区呈现）+ 详情页导出入口。
4. **因子增量 IC API**：`/api/factors/incremental-ic?ids=…` —— Schmidt 正交化
   分解，回答"这个因子在已有因子之外还提供多少新信息"；signals/new 相关性
   区块联动展示（实测：低波动 +0.0143，动量的信息大部分被前者解释 −0.0165）。
5. **README 快速开始/§2.5 润色**：组合对照实验 --backtest/--rebalances 用法。
