# CF2026 Project 1 要求逐项对照清单

> 依据：`CF2026_Project1.pdf`（19 页）。每项要求 → 实现位置 → 验证产物/证据。状态：✅ 已实现并验证。

## 一、总览与目标（P2 页）

| 要求 | 实现 | 证据 |
|---|---|---|
| 最小但**完整**：从原始数据到策略净值各模块连接 | `quantlab/pipeline.py` 一键端到端 | run v1 用时 9.3s 产出全链路产物 |
| **正确**：数据、因子、收益与成本计算符合定义 | 计算口径集中实现 + 8 项回测检查 + 20 项测试 | `tests/` 20/20 PASS；backtest checks 全 PASS |
| **可复现**：相同数据与配置复现主要结果 | manifest（数据/代码/配置 sha256）+ 确定性引擎 | 两次独立运行输出字节级一致（实测） |
| **可扩展**：后续 Project 直接建立在框架上 | 因子注册表/算子插件/引擎协议/策略模型可插拔 | README §6；`docs` 引擎协议说明 |

## 二、数据模块（P5–P7 页）

| 要求 | 实现 | 证据 |
|---|---|---|
| 研究范围：市场、资产、样本期、日频数据来源 | `configs/data.yaml`（A股·CSI300 子集 50 只·2022-01-04~2025-12-31·akshare/sina 日频） | 配置文件 + 报告 §1 |
| 数据快照：原始与清洗分别保存，记录下载时间/版本/校验值 | `data/raw/`（CSV）与 `data/clean/`（parquet）分离；`raw/manifest.json` 含 fetched_at、akshare 版本、**逐文件 sha256**；`clean_manifest.json` 同理 | manifest 实文件；`tests/test_repro.py::test_raw_snapshot_manifest_integrity` 逐文件校验通过 |
| 复现入口：数据获取说明、导入脚本与配置；可分享小样本 | `scripts/fetch_data.py`（幂等/断点续跑/--force）+ README §1；**50 只×4 年快照随仓库分发**（约 5MB CSV） | 脚本实跑 50/50 成功；快照在仓库内 |
| 日期×资产唯一标识、字段含义一致 | 清洗按 (date, code) 去重；宽表统一 date×code | `quantlab/data/clean.py`；质量报告 duplicate 字段 |
| 行情与单位：开/收/量，注明频率与单位 | `configs/data.yaml` units（价格=元、量=股、额=元、换手=比例）；sina 口径在 manifest.units | 报告附录A 数据字典 |
| 价格口径：原始价与复权信息；**乘法累计因子约定** | 原始价与 adj_factor 分别保存；`P_adj=P_raw×a(t)/a(τ)`，τ=样本首日；因子阶梯前填充（沿用公告值） | `clean.py`；`tests/test_data_clean.py::test_adjustment_formula` |
| 资产范围：名单、筛选条件、覆盖区间与样本数量 | constituents.csv + 抽样规则（代码升序步长，确定性）+ 每股票 first/last date、覆盖率 | quality_report.per_stock |
| **质量报告**：重复键/日期顺序/缺失值/非法价格成交量/覆盖率/清洗前后数量 | `quality_report.json`：`duplicate_dates_dropped_total`、`raw_monotonic`、`missing_close_*`、`illegal_total`（4 类规则）、`coverage_mean/min`、`rows_raw_total→rows_in_window_total`；未上市 523 格与上市后停牌 1360 格分开统计 | `tests/test_repro.py::test_clean_quality_report_fields`；报告复现信息（manifest/API） |
| 异常处理说明原因；**缺失值不统一填零** | policy 声明 + 实现：非法置 NaN 计数不删行；下游显式跳过；估值用最后有效价并记事件 | `clean.py._check_illegal`；`tests/test_data_clean.py` |

## 三、因子模块（P8–P9 页）

| 要求 | 实现 | 证据 |
|---|---|---|
| **三个逻辑有区别的因子**：20日动量、5日反转、20日低波动 | `configs/factors.yaml` 三因子卡 + `factors/operators.py` 三算子 | run v1 因子诊断输出 |
| 因子卡：名称/假设/公式/字段/窗口/预期方向/缺失处理/失效情形 | YAML 因子卡全字段 | 报告 §2.1；Web `/factors/{id}` |
| 统一输出 `date, asset, factor, value`；配置切换因子与参数；公式与回测核心分离 | 宽表 date×code 存储；factor key+params 全配置化；回测只接收信号面板 | `pipeline.build_signal_panel` |
| 统一方向："分数越高，预期收益越高"；低波动取负号并说明 | 因子卡 direction 字段；rev5/lowvol20 公式内负号 | `tests/test_factors.py::test_reversal_sign/test_lowvol_direction` |
| 因子预处理：截面排名、中性化、标准化、去极值 | `preprocess.py`：winsorize/zscore/rank_pct/**neutralize（分组中性化接口）** | `tests/test_factors.py::test_winsorize_and_zscore` |

## 四、因子诊断（P12–P13 页）

| 要求 | 实现 | 证据 |
|---|---|---|
| IC = Pearson；RankIC = Spearman 并列平均秩；固定持有期 h | `diagnostics.py`：逐日截面 pearson + rank(average) 后 pearson；h∈{5,20} | `tests/test_factors.py::test_ic_recovers_sorted_signal`（同序因子 IC→1） |
| 报告覆盖率、分布、有效资产数、IC 序列/均值/标准差；RankIC 同样报告 | `ic_table`（ic/rank_ic/n_valid/coverage 逐日）+ `ic_summary`（mean/std/icir/t/win_rate/n_obs） | 报告 §2.2 与附录C；Web 因子详情 IC 时序图 |
| 事先设最低样本要求；**常数或样本不足记缺失，不改 0** | `min_cross_section_samples: 10` 配置；常数序列→NaN | `tests/test_factors.py::test_ic_constant_factor_is_nan` |
| 分组收益：K 组、组内等权、Spread=R_K−R_1；检查单调性 | `quantile_table`：K=5、rank 定序（并列按代码，确定性）、组内等权均值、spread、Spearman 单调性 | 报告 §2.2；Web 分层柱状图 |
| 并列值处理与样本不足规则；**标签缺失报原组/有效人数，不重新分组** | n_formed（按因子分组）与 n_valid（组内标签有效）分列 | 附录C 首个形成日分组人数；`tests/test_factors.py::test_quantile_groups_counts_and_spread` |
| 重叠标签不能复利为策略净值；分组诊断与策略净值分别展示 | 分层为描述性日均收益不复利；策略净值独立回测产出 | 报告 §2.2 注 + §2.3 |

## 五、回测模块（P10–P11、P14–P15 页）

| 要求 | 实现 | 证据 |
|---|---|---|
| 输入：标准化行情、因子/信号、策略参数与成本参数 | `run_backtest(signal, market, bt_cfg)`；bt_cfg 含区间/调仓/持仓/费率 | `tests/test_backtest.py` |
| 配置：样本区间、调仓频率、持仓数量、费率、缺失/停牌处理 | `configs/backtest.yaml`：weekly/monthly/every_h、top_n、c_b/c_s/slippage、suspend/missing_price/delisted 处理策略 | 配置文件 + 报告 §2.3 |
| 输出：日收益、净值、持仓记录、换手与累计成本；**参数改变后可重新运行** | BacktestResult：nav/daily_return/trades/holdings_history/turnover/cost_series/events；改 yaml 即重跑 | run v1 backtests/*.json |
| **成本 = c_b·B_t + c_s·S_t，按实际成交金额扣费** | 引擎按成交金额计费，逐笔与账本可核对 | `tests/test_backtest.py::test_cost_math_exact` |
| 检查-输入与配置：与实验说明一致 | 结果内嵌 config 回显（config_echo/`config` 字段） | backtests/*.json.config |
| 检查-收益与净值：净值变化与日收益一致，累计收益可独立核对 | checks.nav_vs_daily_return（1e-8 重算）+ cumulative_return_consistent | checks 全 PASS；测试断言 |
| 检查-交易成本：按规则计算，**未成交部分不扣成交费** | checks.cost_ledger（账本=逐笔；每笔成交日价格有效且非停牌） | `tests/test_backtest.py::test_suspended_no_trade_no_fee` |
| 检查-异常与缺失：有明确处理与记录 | events 留痕（sell_deferred/buy_skipped/nav_fallback），checks.events_logged | events 字段 |
| 检查-重复运行：同数据同配置结果在容差内一致 | 输出哈希（nav+trades sha256）+ 复现测试 | `tests/test_repro.py::test_pipeline_rerun_deterministic`；两次全量运行字节级一致（实测） |
| 绩效：累计/年化收益 `(V_N/V_0)^{A/N}−1`、年化波动 `√A·std`、年化 Sharpe `√A·mean(e)/std(e)`（扣费后为主） | `metrics.py` 严格按公式；rf 与 A 来自配置 | `tests/test_backtest.py::test_metrics_formulas` |
| 最大回撤 `1−V_t/H_t`；换手 `(B_t+S_t)/V⁻_t`；TotalCost=ΣCost（货币单位） | 引擎+metrics 实现；成本以名义货币报告 | `test_metrics_formulas` |
| 持仓数量、最大单资产权重、交易次数；**毛收益 vs 净收益对比（同一路径归因）** | metrics.portfolio；毛净值=同成交序列费率置零重跑，checks.gross_net_same_trades 断言成交序列一致 | 报告 §2.3 表；策略详情页三线图 |

## 六、对照实验与报告（P16 页）

| 要求 | 实现 | 证据 |
|---|---|---|
| 先写问题和预期，再只改变一个主要因素 | `configs/experiment.yaml`：question/expectation/keep_fixed；E1 只改调仓频率，E2 只改因子 | 报告 §3 |
| 对照：如周度 vs 月度调仓；或不同因子 | **两个对照都做了**（E1 周度 vs 月度：换手 112.7→54.9、成本减半；E2 三因子对照） | run v1 experiments/*.json |
| 报告：数据与假设/因子与回测/对照与解释/复现信息 | `quantlab/report.py` 五章 + 附录 A–D | `reports/研究报告.md`（~10KB） |
| 保留不支持预期的结果；未检验称描述性 | mom20 的 RankIC 为负、候选 test IC 为负等均如实呈现并声明描述性 | 报告 §2.2/§3 解释段 |
| 研究报告 6–10 页 PDF | fpdf2 中文排版（微软雅黑）+ 4 张图 | `reports/研究报告.pdf` **6 页**（pypdf 实测） |

## 七、自主拓展（P17 页，20 分）

| 方向 | 可选工作 | 本项目对应 | 验收证据 |
|---|---|---|---|
| 数据工程 | 增量更新、缓存、动态资产池 | fetch_data 幂等续跑 + manifest 去重；clean parquet 缓存；未上市/停牌区分（动态有效集） | 重复导入不重复；质量报告分列 |
| 因子研究工具 | **因子注册、批量诊断、中性化** | 因子库（SQLite 注册+hash 幂等去重+来源标签）；三因子批量诊断一次跑出；neutralize 分组中性化接口 | Web `/factors`；新增 GP 因子不改回测核心（注册即诊断） |
| 回测真实性 | 公司行动、部分成交、流动性限制 | 复权因子处理公司行动；停牌不可成交顺延留痕；缺价冻结估值并记事件 | tests + events 留痕 |
| 组合与风险 | 波动率目标、权重限制、风险平价 | 等权/IC 加权模型；最大单资产权重监控；MDD/Calmer；三线净值对照 | 策略库 metrics |

> 按课程口径"一个扎实扩展可取得全部拓展分"——本项目做了 4 个方向的扎实扩展（因子研究工具+自动挖掘为最重投入）。

## 八、提交材料（P19 页）

| 要求 | 实现 | 证据 |
|---|---|---|
| 可运行代码：依赖、README、入口、配置 | `requirements.txt`、`README.md`（快速开始/架构图/职责接口取舍）、`scripts/*` 三入口、`configs/*.yaml` | 全部在仓库 |
| 架构图与简要文字说明职责、接口与取舍 | README §2 架构图（mermaid）+ 职责表 + §7 取舍对照表 | README |
| 数据与复现包：可分享快照或小样本、字典、文件清单、校验值与获取说明 | data/raw 随仓库（50 只小样本）+ manifest sha256 + 报告附录A 数据字典 + fetch 脚本 | manifest.json（102 文件校验值） |
| 研究报告：设计、结果、对照、局限与下一步 | 报告 §1–§5 | 报告.md / 报告.pdf |

## 九、验证状态汇总（截至 2026-09-27）

- pytest：**78 项全部通过**（含 skip 守卫的环境测试；覆盖分钟数据层/沙箱/组合器/分段防泄漏/基准指标）
- 端到端流水线 run base_v3：回测基本检查**全部 PASS**
- 可复现：两次独立运行全部结果文件 **sha256 一致**；/runs 页 run 间 diff
- Web 平台：30 端点巡检全部 200；GP + AlphaGen 双引擎任务 202→SUCCESS→候选采纳闭环
- 研究报告：md + **6 页 PDF**（中文字体嵌入，4 张图表）+ 组合实验 JSON 12 组落盘
- 分析增强（9-27 批次）：Newey-West 修正 t / 功效分析(MDE) / 组合增益瀑布分解 /
  跨模型稳健性聚合徽章 / IC 衰减半衰期与调仓频率建议 / 基准相对指标
  （超额·Beta·Alpha·TE·IR·日胜率）/ 月度收益热力图 / 对照实验跨实验聚合总览
