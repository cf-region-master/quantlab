"""SQLite 存储层（课程版）：因子库 / 策略库 / 算法实验室的元数据。

取舍说明：设计文档的生产版使用 PostgreSQL+ArcticDB+MinIO+Celery；
课程复现版以 SQLite+Parquet 等价实现同样的实体与边界（因子库为唯一事实源、
候选经采纳进入因子库、任务异步化），保证零外部依赖可运行。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "quantlab.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False},
                       pool_pre_ping=True)


from sqlalchemy import event as _sa_event  # noqa: E402


@_sa_event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):
    """WAL + busy_timeout：挖掘线程/进程高频写进度时，页面读不再被写锁顶掉。

    历史 bug：默认 journal 模式下训练线程每轮 commit 会拿写锁，
    同一时刻的页面读请求报 database is locked / 明显变慢（前端"卡"的一部分）。
    """
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


def now() -> datetime:
    return datetime.now(timezone.utc)


class Factor(Base):
    __tablename__ = "factors"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    hash_code = Column(String, unique=True, index=True)      # 内容哈希，幂等去重
    source = Column(String, default="builtin")               # builtin | gp | user
    expression = Column(Text, default="")                    # 中缀公式/表达式
    hypothesis = Column(Text, default="")
    direction = Column(Text, default="")
    missing_policy = Column(Text, default="")
    failure_modes = Column(Text, default="")                 # 分号分隔
    operator = Column(String, default="")                    # builtin 算子名 / expression / user
    params = Column(JSON, default=dict)
    preprocess = Column(JSON, default=list)
    values_path = Column(String, default="")                 # 因子值 parquet
    algo_task_id = Column(String, nullable=True)             # GP 产出溯源
    notes = Column(Text, default="")
    # ---- 计算范围：因子值覆盖的股票池与区间（扩建/更新时延长区间）----
    pool_id = Column(Integer, ForeignKey("stock_pools.id", ondelete="SET NULL"), nullable=True)
    value_start = Column(DateTime, nullable=True)            # 因子值首个交易日
    value_end = Column(DateTime, nullable=True)              # 因子值最后交易日
    universe_note = Column(Text, default="")                 # 计算时所用股票池的可读说明
    code = Column(Text, default="")                          # source=user 时的因子源代码
    # 隐藏因子：由挖掘任务自动产生的内部因子，默认不在因子库列表里露出（列表页可切换显示）。
    # 目的是让"挖矿产出的因子池"能作为一个整体被策略引用，而不必把几十条中间产物塞进因子库主视图。
    hidden = Column(Boolean, default=False)
    created_at = Column(DateTime, default=now)

    @property
    def failure_modes_list(self) -> list[str]:
        """列存的是分号分隔字符串，模板必须用本属性而不是直接遍历 failure_modes。

        （直接 `{% for x in f.failure_modes %}` 会逐字符迭代，
        在页面渲染成"一行一个汉字"。）
        """
        return [x for x in (self.failure_modes or "").split(";") if x]

    @property
    def preprocess_spec(self) -> dict:
        """归一化后的预处理参数（可编辑、可复现）。

        `preprocess` 列历史上存过 `["winsorize","zscore"]` 这种步骤名列表，
        也存 spec dict。一律经 `normalize_spec` 收敛成完整参数字典，
        这样下游（诊断 / 分层回测 / 信号合成 / 页面口径表）读到的一定是全字段。
        """
        from ..factors.preprocess import normalize_spec

        return normalize_spec(self.preprocess)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        for k in ("created_at", "value_start", "value_end"):
            d[k] = str(d[k]) if d[k] else None
        d["failure_modes"] = self.failure_modes_list
        d["preprocess_spec"] = self.preprocess_spec
        return d


class StockPool(Base):
    """股票池：静态代码集合，或动态 PIT 指数成分。"""
    __tablename__ = "stock_pools"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    kind = Column(String, nullable=False, default="all")     # all | index | static
    indices = Column(JSON, default=list)                     # kind=index: ["csi300","csi500"]
    codes = Column(JSON, default=list)                       # kind=static: ["600519", ...]
    created_at = Column(DateTime, default=now)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        d["created_at"] = str(self.created_at)
        return d

    def spec(self) -> dict:
        return {"kind": self.kind, "indices": list(self.indices or []),
                "codes": list(self.codes or [])}


class Signal(Base):
    """信号（策略）：纯权重输出的定义，可像因子一样做 IC/RankIC/分层诊断。

    回测不属于信号：一个信号可以按不同区间/股票池/权重/成本回测多次，见 BacktestRun。
    """
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    kind = Column(String, default="composite")               # single | composite | model
    components = Column(JSON, default=list)                  # [{"factor_id":1,"weight":0.5}]
    model_type = Column(String, default="equal_weight")      # equal_weight|ic_weight|linear|tree
    model_params = Column(JSON, default=dict)                # linear/tree: refit_days, train_window, purge
    preprocess = Column(JSON, default=list)
    start_date = Column(DateTime, nullable=False)            # 策略有效区间 [a, b]
    end_date = Column(DateTime, nullable=False)
    diagnostics_json = Column(JSON, default=dict)            # 与因子库同类的 IC/分层诊断
    # 若该策略由挖掘任务自动生成（AlphaGen 因子池 → 线性组合），记录来源任务
    algo_task_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=now)

    @property
    def preprocess_spec(self) -> dict:
        """归一化后的预处理参数（与 Factor.preprocess_spec 同一套 schema）。"""
        from ..factors.preprocess import normalize_spec

        return normalize_spec(self.preprocess)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        for k in ("created_at", "start_date", "end_date"):
            d[k] = str(d[k]) if d[k] else None
        return d


class BacktestRun(Base):
    """对某个信号的一次配置化模拟成交（区间/股票池/权重/成本都可不同）。"""
    __tablename__ = "backtest_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, ForeignKey("signals.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    name = Column(String, default="")
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    pool_id = Column(Integer, ForeignKey("stock_pools.id", ondelete="SET NULL"), nullable=True)
    pool_note = Column(Text, default="")
    top_n = Column(Integer, default=10)
    weighting = Column(String, default="equal_weight")
    rebalance_freq = Column(String, default="weekly")
    initial_cash = Column(Float, default=1_000_000.0)
    cost_json = Column(JSON, default=dict)
    metrics_json = Column(JSON, default=dict)
    checks_json = Column(JSON, default=dict)
    result_path = Column(String, default="")
    created_at = Column(DateTime, default=now)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        for k in ("created_at", "start_date", "end_date"):
            d[k] = str(d[k]) if d[k] else None
        return d


class FactorMetric(Base):
    __tablename__ = "factor_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    factor_id = Column(Integer, index=True, nullable=False)
    horizon = Column(Integer, nullable=False)                # 持有期 h
    ic_mean = Column(Float)
    ic_std = Column(Float)
    rank_ic_mean = Column(Float)
    rank_ic_std = Column(Float)
    icir = Column(Float)
    t_stat = Column(Float)
    coverage_mean = Column(Float)
    n_valid_mean = Column(Float)
    spread_mean = Column(Float)
    monotonicity = Column(Float)
    summary_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=now)

    def summary(self):
        return {"horizon": self.horizon, "ic_mean": self.ic_mean, "ic_std": self.ic_std,
                "rank_ic_mean": self.rank_ic_mean, "rank_ic_std": self.rank_ic_std,
                "icir": self.icir, "t_stat": self.t_stat, "coverage_mean": self.coverage_mean,
                "n_valid_mean": self.n_valid_mean, "spread_mean": self.spread_mean,
                "monotonicity": self.monotonicity}


class AlgoTask(Base):
    __tablename__ = "algo_tasks"
    id = Column(String, primary_key=True)                    # uuid
    engine = Column(String, default="gp_daily")
    objective = Column(String, default="factor_mining")
    status = Column(String, default="PENDING")               # PENDING/RUNNING/SUCCESS/FAILED/CANCELLED
    params = Column(JSON, default=dict)
    progress = Column(Float, default=0.0)
    stage = Column(String, default="")
    message = Column(Text, default="")
    metrics_json = Column(JSON, default=dict)                # 训练曲线等
    log = Column(Text, default="")
    created_at = Column(DateTime, default=now)
    finished_at = Column(DateTime, nullable=True)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        for k in ("created_at", "finished_at"):
            d[k] = str(d[k]) if d[k] else None
        return d


class AlgoCandidate(Base):
    __tablename__ = "algo_candidates"
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, index=True, nullable=False)
    expression = Column(Text, nullable=False)                # 中缀表达式（展示）
    expr_json = Column(JSON, default=None)                   # 表达式树（采纳时直接求值）
    train_ic = Column(Float)
    valid_ic = Column(Float)
    test_ic = Column(Float)
    size = Column(Integer)                                   # token 数
    adopted = Column(Boolean, default=False)
    factor_id = Column(Integer, nullable=True)               # 采纳后的因子 id
    created_at = Column(DateTime, default=now)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        d["created_at"] = str(self.created_at)
        return d


class BatchJob(Base):
    """批量后台任务（重算/删除多因子）：把分钟级 CPU 活从 HTTP 请求里挪出去。

    请求只负责建任务并 303 到进度页，页面轮询 /api/jobs/{id}；完成后跳结果页。
    """
    __tablename__ = "batch_jobs"
    id = Column(String, primary_key=True)                    # uuid8
    kind = Column(String, nullable=False)                    # refresh | delete
    status = Column(String, default="PENDING")               # PENDING/RUNNING/SUCCESS/FAILED
    total = Column(Integer, default=0)
    done = Column(Integer, default=0)
    stage = Column(String, default="")
    message = Column(Text, default="")
    params = Column(JSON, default=dict)
    result_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=now)
    finished_at = Column(DateTime, nullable=True)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        for k in ("created_at", "finished_at"):
            d[k] = str(d[k]) if d[k] else None
        d["progress"] = (self.done / self.total) if self.total else 0.0
        return d


# 轻量迁移表：新增列时写在这里。`create_all` 只建缺失的【表】，不会给已存在的表加列，
# 所以已经跑过一轮的库必须靠 ALTER TABLE 补列，否则老库读新代码会报 no such column。
_MIGRATIONS: dict[str, dict[str, str]] = {
    "factors": {"hidden": "BOOLEAN DEFAULT 0"},
    "signals": {"algo_task_id": "VARCHAR"},
}


def _migrate() -> list[str]:
    """按需补列（幂等）。返回本次实际执行的 DDL，便于启动日志说明。"""
    from sqlalchemy import text

    done: list[str] = []
    with engine.begin() as conn:
        for table, cols in _MIGRATIONS.items():
            have = {r[1] for r in conn.execute(text(f"PRAGMA table_info({table})"))}
            if not have:            # 表还不存在，create_all 会建全
                continue
            for col, ddl in cols.items():
                if col not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))
                    done.append(f"{table}.{col}")
    return done


def init_db() -> None:
    Base.metadata.create_all(engine)
    applied = _migrate()
    if applied:
        print(f"[db] 已补列: {', '.join(applied)}")


def get_session():
    return SessionLocal()
