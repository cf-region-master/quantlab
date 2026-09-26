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
