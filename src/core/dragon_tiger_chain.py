# -*- coding: utf-8 -*-
"""
===================================
A股监链 - 龙虎榜量化监控流程
===================================

监控流程（Top-Down）：
  Step 1  政策扫描    → 找今日热点政策/催化剂
  Step 2  板块热度    → 结合政策 + 资金流锁定热门板块
  Step 3  热门个股    → 龙虎榜 + 涨停板 + 热门板块个股
  Step 4  资金分类    → 区分机构(主力/超大单/大单)流入 vs 散户流入
  Step 5  基本面过滤  → 审计机构质量检查
  Step 6  综合报告    → AI 生成结构化操作建议

适用时间：11:00 / 13:00 / 收盘后（非仅盯早盘涨停）
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from data_provider.lhb_fetcher import LHBFetcher, CapitalFlowFetcher, AuditorFilter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────

@dataclass
class PolicySignal:
    """政策/催化剂信号"""
    topic: str                          # 政策主题
    summary: str                        # 简要描述
    affected_sectors: List[str]         # 受影响板块
    confidence: float = 0.5             # 置信度 0-1
    source: str = ""


@dataclass
class SectorHeat:
    """板块热度综合评分"""
    name: str
    policy_score: float = 0.0           # 政策驱动分（0-40）
    flow_score: float = 0.0             # 资金流入分（0-40）
    lhb_score: float = 0.0             # 龙虎榜出现分（0-20）
    total_score: float = 0.0            # 综合分
    net_inflow: float = 0.0             # 资金净流入（亿元）
    change_pct: float = 0.0             # 板块今日涨跌幅
    top_stocks: List[str] = field(default_factory=list)


@dataclass
class StockCandidate:
    """候选股票"""
    code: str
    name: str
    source: str                         # "lhb" / "limit_up" / "strong" / "sector"
    sector: str = ""
    change_pct: float = 0.0

    # 龙虎榜信息
    on_lhb: bool = False
    institutional_net: float = 0.0      # 机构席位净买（万元）
    retail_net: float = 0.0             # 游资席位净买（万元）
    institutional_seats: int = 0
    retail_seats: int = 0
    lhb_net_buy: float = 0.0           # 龙虎榜总净买（万元）

    # 资金流
    main_net_inflow: float = 0.0        # 主力净流入（万元）
    retail_net_inflow: float = 0.0      # 散户净流入（万元）
    flow_dominance: str = "无数据"
    main_net_pct: float = 0.0

    # 基本面
    auditor: str = "未知"
    auditor_quality: str = "unknown"
    auditor_pass: bool = True

    # 连板信息
    consecutive_days: int = 0

    # 综合评分
    total_score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    alert_tags: List[str] = field(default_factory=list)
    risk_tags: List[str] = field(default_factory=list)


@dataclass
class DragonTigerReport:
    """监链最终报告"""
    date: str
    scan_time: str

    policy_signals: List[PolicySignal] = field(default_factory=list)
    hot_sectors: List[SectorHeat] = field(default_factory=list)
    candidates: List[StockCandidate] = field(default_factory=list)

    # 危险监控（散户>主力，单独列出供参考/回避）
    # 这些股票评分被扣至负数，不进入 candidates 主列表，但在报告中专门警示
    danger_watch: List[StockCandidate] = field(default_factory=list)

    # 统计
    total_candidates: int = 0
    institutional_dominant: int = 0     # 机构主导流入的股票数
    retail_dominant: int = 0            # 散户主导流入的股票数
    limit_up_count: int = 0
    lhb_count: int = 0
    danger_count: int = 0               # 危险信号个股数

    ai_report: str = ""
    errors: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────
# 主链类
# ─────────────────────────────────────────────────

class DragonTigerChain:
    """
    A股监链主流程

    使用方式：
        chain = DragonTigerChain(search_service=svc, data_manager=dm, analyzer=ai)
        report = chain.run()
        print(report.ai_report)
    """

    # 评分权重
    SCORE_LHB_INSTITUTIONAL = 30        # 机构席位净买（满分）
    SCORE_LHB_PRESENCE = 15             # 仅出现在龙虎榜
    SCORE_MAIN_FLOW_INFLOW = 25         # 主力资金净流入
    SCORE_RETAIL_DIVERGENCE = 10        # 主力进/散户出（最强形态）
    SCORE_LIMIT_UP = 10                 # 涨停板加分
    SCORE_CONSECUTIVE = 10              # 连板加分

    # 危险信号扣分（散户 > 主力）
    # 逻辑：散户净流入 > 主力净流入 → 资金结构恶化，主力可能在减仓出货
    PENALTY_RETAIL_GT_MAIN_MILD = -15   # 散户>主力（同向流入，散户略多）
    PENALTY_RETAIL_GT_MAIN_STRONG = -25 # 散户远超主力（>2倍）
    PENALTY_MAIN_OUT_RETAIL_IN = -35    # 最危险：主力净流出 + 散户净流入

    # 散户主导流出的股票仍可关注（看机构动作）
    MIN_SCORE_THRESHOLD = 20

    def __init__(
        self,
        search_service=None,
        data_manager=None,
        analyzer=None,
        max_stocks: int = 30,           # 最多处理候选股数（控制速度）
        enable_auditor_check: bool = True,
        enable_flow_detail: bool = True,
        trade_date: Optional[str] = None,
    ):
        self.search_service = search_service
        self.data_manager = data_manager
        self.analyzer = analyzer
        self.max_stocks = max_stocks
        self.enable_auditor_check = enable_auditor_check
        self.enable_flow_detail = enable_flow_detail
        self.trade_date = trade_date

        self.lhb_fetcher = LHBFetcher()
        self.flow_fetcher = CapitalFlowFetcher()
        self.auditor_filter = AuditorFilter()

    # ── 公开入口 ──────────────────────────────────

    @staticmethod
    def _sf(v: Any, default: float = 0.0) -> float:
        """从 DataFrame row 安全提取 float，兼容 Series/NaN/None。"""
        if isinstance(v, pd.Series):
            v = v.iloc[0] if not v.empty else default
        try:
            f = float(v)
            return default if f != f else f  # f != f 捕获 NaN
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _ss(v: Any, default: str = "") -> str:
        """从 DataFrame row 安全提取 str，兼容 Series。"""
        if isinstance(v, pd.Series):
            v = v.iloc[0] if not v.empty else default
        if v is None:
            return default
        s = str(v).strip()
        return default if s in ("nan", "None", "NaN") else s

    def run(self) -> DragonTigerReport:
        """
        执行完整监链，返回报告对象。
        """
        now = datetime.now()
        date_str = self.trade_date or now.strftime("%Y%m%d")
        scan_time = now.strftime("%H:%M")

        logger.info("=" * 60)
        logger.info("[监链] 开始 A 股监链扫描，日期: %s  时间: %s", date_str, scan_time)
        logger.info("=" * 60)

        report = DragonTigerReport(date=date_str, scan_time=scan_time)

        # Step 1: 政策扫描
        report.policy_signals = self._step1_scan_policies()

        # Step 2: 板块热度
        report.hot_sectors = self._step2_find_hot_sectors(report.policy_signals)

        # Step 3: 候选个股
        candidates = self._step3_find_hot_stocks(report.hot_sectors, date_str)

        # Step 4: 资金流分类
        candidates = self._step4_classify_flow(candidates, date_str)

        # Step 5: 基本面过滤
        if self.enable_auditor_check:
            candidates = self._step5_filter_fundamentals(candidates)

        # Step 6: 评分排序（危险股被分离进 _danger_buffer）
        self._danger_buffer: List[StockCandidate] = []
        candidates = self._step6_score_and_rank(candidates)

        # 统计
        report.candidates = candidates
        report.danger_watch = self._danger_buffer
        report.total_candidates = len(candidates)
        report.danger_count = len(report.danger_watch)
        report.institutional_dominant = sum(
            1 for c in candidates if "机构主导流入" in c.flow_dominance
        )
        report.retail_dominant = sum(
            1 for c in candidates if "散户主导流入" in c.flow_dominance
        )
        report.limit_up_count = sum(1 for c in candidates if c.source in ("limit_up", "consecutive"))
        report.lhb_count = sum(1 for c in candidates if c.on_lhb)

        # Step 7: AI 报告
        report.ai_report = self._step7_generate_report(report)

        logger.info("[监链] 扫描完成，候选 %d 只，危险监控 %d 只，机构主导 %d 只",
                    report.total_candidates, report.danger_count, report.institutional_dominant)

        return report

    # ── Step 1: 政策扫描 ─────────────────────────

    def _step1_scan_policies(self) -> List[PolicySignal]:
        """扫描今日热点政策与催化剂"""
        logger.info("[监链 Step1] 政策扫描...")
        signals: List[PolicySignal] = []

        if not self.search_service:
            logger.warning("[监链 Step1] 搜索服务未配置，跳过政策扫描")
            return signals

        queries = [
            "今日A股政策利好 板块",
            "产业政策 补贴 新能源 科技 医药",
            "国家队增持 ETF 资金",
            "今日涨停原因 题材 热点",
        ]

        news_by_sector: Dict[str, List[str]] = {}

        for q in queries:
            try:
                resp = self.search_service.search_stock_news(
                    stock_code="policy",
                    stock_name="政策",
                    max_results=3,
                    focus_keywords=q.split(),
                )
                if resp and resp.results:
                    for item in resp.results:
                        title = getattr(item, "title", "") or ""
                        snippet = getattr(item, "snippet", "") or ""
                        text = title + " " + snippet

                        # 简单关键词匹配，找板块
                        sectors = self._extract_sectors_from_text(text)
                        topic = title[:40] if title else q
                        summary = snippet[:80] if snippet else ""

                        sig = PolicySignal(
                            topic=topic,
                            summary=summary,
                            affected_sectors=sectors,
                            confidence=0.6 if sectors else 0.3,
                            source=getattr(item, "url", ""),
                        )
                        signals.append(sig)

                        for s in sectors:
                            news_by_sector.setdefault(s, []).append(topic)

            except Exception as e:
                logger.debug("[监链 Step1] 搜索失败: %s", e)

        # 去重合并同主题
        seen_topics = set()
        unique_signals = []
        for sig in signals:
            key = sig.topic[:20]
            if key not in seen_topics:
                seen_topics.add(key)
                unique_signals.append(sig)

        logger.info("[监链 Step1] 政策信号 %d 条，涉及板块: %s",
                    len(unique_signals),
                    list({s for sig in unique_signals for s in sig.affected_sectors})[:8])
        return unique_signals

    def _extract_sectors_from_text(self, text: str) -> List[str]:
        """从文本中提取板块关键词"""
        sector_keywords = {
            "新能源": ["新能源", "光伏", "储能", "电动车", "锂电"],
            "人工智能": ["人工智能", "AI", "大模型", "算力", "芯片"],
            "半导体": ["半导体", "芯片", "集成电路", "晶圆"],
            "医药": ["医药", "创新药", "生物医药", "医疗器械"],
            "军工": ["军工", "国防", "航空航天", "武器"],
            "消费": ["消费", "零售", "食品饮料", "白酒"],
            "房地产": ["房地产", "地产", "房企", "楼市"],
            "银行": ["银行", "金融", "保险", "券商"],
            "科技": ["科技", "互联网", "软件", "云计算"],
            "农业": ["农业", "粮食", "种植", "养殖"],
            "有色金属": ["有色", "铜", "铝", "黄金", "稀土"],
            "化工": ["化工", "材料", "石化"],
            "电力": ["电力", "核电", "水电", "电网"],
            "交通": ["交通", "高铁", "航空", "港口"],
        }

        found = []
        for sector, keywords in sector_keywords.items():
            if any(kw in text for kw in keywords):
                found.append(sector)

        return found[:4]  # 最多返回4个板块

    # ── Step 2: 板块热度 ─────────────────────────

    def _step2_find_hot_sectors(self, policy_signals: List[PolicySignal]) -> List[SectorHeat]:
        """结合政策信号 + 资金流排名，锁定热门板块"""
        logger.info("[监链 Step2] 板块热度分析...")

        sector_map: Dict[str, SectorHeat] = {}

        # 2a. 政策驱动分
        for sig in policy_signals:
            for sector in sig.affected_sectors:
                if sector not in sector_map:
                    sector_map[sector] = SectorHeat(name=sector)
                sector_map[sector].policy_score = min(
                    40.0,
                    sector_map[sector].policy_score + sig.confidence * 20
                )

        # 2b. 资金流排名
        flow_data = self.flow_fetcher.get_sector_flow_ranking(top_n=15)
        for i, s in enumerate(flow_data.get("inflow_sectors", [])):
            name = s["name"]
            if name not in sector_map:
                sector_map[name] = SectorHeat(name=name)
            # 排名越靠前分越高
            flow_pts = max(0, 40 - i * 3)
            sector_map[name].flow_score = flow_pts
            sector_map[name].net_inflow = s.get("net_inflow", 0) / 1e8  # 转亿
            sector_map[name].change_pct = s.get("change_pct", 0)

        # 2c. 大盘板块涨幅榜（akshare）
        try:
            import akshare as ak
            sector_df = ak.stock_board_concept_name_em()
            if sector_df is not None and not sector_df.empty:
                chg_col = next((c for c in sector_df.columns if "涨跌幅" in str(c)), None)
                name_col = next((c for c in sector_df.columns if "板块名称" in str(c) or "名称" in str(c)), None)
                if chg_col and name_col:
                    sector_df[chg_col] = pd.to_numeric(sector_df[chg_col], errors="coerce")
                    top10 = sector_df.nlargest(10, chg_col)
                    for _, row in top10.iterrows():
                        n = str(row[name_col])
                        if n not in sector_map:
                            sector_map[n] = SectorHeat(name=n)
                        sector_map[n].lhb_score = min(20, sector_map[n].lhb_score + 10)
                        sector_map[n].change_pct = float(row[chg_col])
        except Exception as e:
            logger.debug("[监链 Step2] 板块涨幅获取失败: %s", e)

        # 计算综合分并排序
        for sh in sector_map.values():
            sh.total_score = sh.policy_score + sh.flow_score + sh.lhb_score

        hot = sorted(sector_map.values(), key=lambda x: x.total_score, reverse=True)[:10]

        logger.info("[监链 Step2] 热门板块 TOP5: %s",
                    [(s.name, f"{s.total_score:.0f}分") for s in hot[:5]])
        return hot

    # ── Step 3: 候选个股 ─────────────────────────

    def _step3_find_hot_stocks(
        self,
        hot_sectors: List[SectorHeat],
        date_str: str,
    ) -> List[StockCandidate]:
        """
        从三个来源收集候选个股：
        1. 当日龙虎榜
        2. 涨停板个股池（含连板）
        3. 强势股池
        """
        logger.info("[监链 Step3] 收集候选个股...")
        candidates: Dict[str, StockCandidate] = {}

        # 3a. 龙虎榜
        lhb_df = self.lhb_fetcher.get_daily_lhb_list(trade_date=date_str, lookback_days=1)
        for _, row in lhb_df.iterrows():
            code = self._ss(row.get("code", "")).zfill(6)
            name = self._ss(row.get("name", ""))
            if not code or code == "000000":
                continue

            c = StockCandidate(
                code=code, name=name, source="lhb",
                change_pct=self._sf(row.get("change_pct", 0)),
                on_lhb=True,
                lhb_net_buy=self._sf(row.get("lhb_net_buy", 0)) / 1e4,
            )
            candidates[code] = c

        logger.info("[监链 Step3] 龙虎榜 %d 只", len(candidates))

        # 3b. 涨停板个股池
        zt_df = self.lhb_fetcher.get_limit_up_pool(trade_date=date_str)
        for _, row in zt_df.iterrows():
            code = self._ss(row.get("code", "")).zfill(6)
            name = self._ss(row.get("name", ""))
            if not code:
                continue
            if code in candidates:
                candidates[code].source = "lhb_and_zt"
            else:
                c = StockCandidate(
                    code=code, name=name, source="limit_up",
                    change_pct=self._sf(row.get("change_pct", 0)),
                    sector=self._ss(row.get("sector", "")),
                )
                candidates[code] = c

        # 3c. 连板池
        consec_df = self.lhb_fetcher.get_consecutive_limit_up_pool(trade_date=date_str)
        for _, row in consec_df.iterrows():
            code = self._ss(row.get("code", "")).zfill(6)
            name = self._ss(row.get("name", ""))
            if not code:
                continue
            days = int(self._sf(row.get("consecutive_days", 2), 2.0))
            if code in candidates:
                candidates[code].consecutive_days = days
                candidates[code].source = "consecutive"
            else:
                c = StockCandidate(
                    code=code, name=name, source="consecutive",
                    change_pct=self._sf(row.get("change_pct", 0)),
                    sector=self._ss(row.get("sector", "")),
                    consecutive_days=days,
                )
                candidates[code] = c

        # 3d. 强势股池（补充）
        strong_df = self.lhb_fetcher.get_strong_stocks_pool(trade_date=date_str)
        for _, row in strong_df.iterrows():
            code = self._ss(row.get("code", "")).zfill(6)
            if code in candidates:
                continue
            name = self._ss(row.get("name", ""))
            c = StockCandidate(
                code=code, name=name, source="strong",
                change_pct=self._sf(row.get("change_pct", 0)),
                sector=self._ss(row.get("sector", "")),
            )
            candidates[code] = c

        total = list(candidates.values())

        # 限制处理数量（从龙虎榜/涨停优先）
        priority_order = ["consecutive", "lhb_and_zt", "lhb", "limit_up", "strong", "sector"]
        total.sort(key=lambda c: priority_order.index(c.source) if c.source in priority_order else 99)
        total = total[:self.max_stocks]

        logger.info("[监链 Step3] 候选个股 %d 只（上限 %d）", len(total), self.max_stocks)
        return total

    # ── Step 4: 资金流分类 ────────────────────────

    def _step4_classify_flow(
        self,
        candidates: List[StockCandidate],
        date_str: str,
    ) -> List[StockCandidate]:
        """
        为每只候选股获取：
        1. 龙虎榜席位明细（机构 vs 游资）
        2. 个股资金流细分（超大单/大单 vs 中小单）
        """
        if not self.enable_flow_detail:
            return candidates

        logger.info("[监链 Step4] 资金流分类（%d 只）...", len(candidates))

        for i, c in enumerate(candidates):
            # 龙虎榜席位详情
            if c.on_lhb:
                try:
                    seat_data = self.lhb_fetcher.get_stock_lhb_seat_detail(c.code, trade_date=date_str)
                    c.institutional_net = seat_data["institutional_net"]
                    c.retail_net = seat_data["retail_net"]
                    c.institutional_seats = seat_data["institutional_seats"]
                    c.retail_seats = seat_data["retail_seats"]
                except Exception as e:
                    logger.debug("[监链 Step4] %s 席位详情失败: %s", c.code, e)

            # 个股资金流细分
            try:
                flow = self.flow_fetcher.get_stock_flow_detail(c.code)
                c.main_net_inflow = flow["main_net_inflow"]
                c.retail_net_inflow = flow["retail_net_inflow"]
                c.flow_dominance = flow["flow_dominance"]
                c.main_net_pct = flow["main_net_pct"]
            except Exception as e:
                logger.debug("[监链 Step4] %s 资金流失败: %s", c.code, e)

            # 适当限速
            if (i + 1) % 5 == 0:
                time.sleep(0.5)

        logger.info("[监链 Step4] 资金流分类完成")
        return candidates

    # ── Step 5: 基本面过滤 ────────────────────────

    def _step5_filter_fundamentals(
        self, candidates: List[StockCandidate]
    ) -> List[StockCandidate]:
        """审计机构质量检查，标记高风险股"""
        logger.info("[监链 Step5] 基本面过滤（审计质量）...")

        for c in candidates:
            try:
                audit = self.auditor_filter.check_auditor(c.code)
                c.auditor = audit["auditor"]
                c.auditor_quality = audit["quality"]
                c.auditor_pass = audit["pass_filter"]
                if not c.auditor_pass:
                    c.risk_tags.append(f"审计风险: {audit['note']}")
            except Exception as e:
                logger.debug("[监链 Step5] %s 审计查询失败: %s", c.code, e)

        failed = sum(1 for c in candidates if not c.auditor_pass)
        logger.info("[监链 Step5] 审计过滤完成，%d 只标记风险", failed)
        return candidates

    # ── Step 6: 评分排序 ──────────────────────────

    def _step6_score_and_rank(self, candidates: List[StockCandidate]) -> List[StockCandidate]:
        """综合评分并排序"""
        logger.info("[监链 Step6] 综合评分排序...")

        for c in candidates:
            score = 0.0
            breakdown: Dict[str, float] = {}

            # --- 龙虎榜机构席位净买 ---
            if c.on_lhb and c.institutional_net > 0:
                pts = min(self.SCORE_LHB_INSTITUTIONAL, c.institutional_net / 500)
                score += pts
                breakdown["机构席位净买"] = pts
                c.alert_tags.append(f"机构席位净买 {c.institutional_net:.0f}万")
            elif c.on_lhb:
                score += self.SCORE_LHB_PRESENCE
                breakdown["龙虎榜上榜"] = self.SCORE_LHB_PRESENCE

            # --- 主力资金净流入 ---
            if c.main_net_inflow > 0:
                pts = min(self.SCORE_MAIN_FLOW_INFLOW, c.main_net_inflow / 1000)
                score += pts
                breakdown["主力净流入"] = pts
                c.alert_tags.append(f"主力净流 +{c.main_net_inflow:.0f}万")
            elif c.main_net_inflow < 0:
                c.risk_tags.append(f"主力净流出 {c.main_net_inflow:.0f}万")

            # --- 主力进散户出（最强共振形态）---
            if c.main_net_inflow > 0 and c.retail_net_inflow < 0:
                score += self.SCORE_RETAIL_DIVERGENCE
                breakdown["主力进散户出"] = self.SCORE_RETAIL_DIVERGENCE
                c.alert_tags.append("主力进/散户出（强势形态）")

            # --- 散户净流入 > 主力净流入（危险信号）---
            # 情形1（最危险）：主力净流出，散户净流入 → 经典出货接盘
            elif c.main_net_inflow < 0 and c.retail_net_inflow > 0:
                penalty = self.PENALTY_MAIN_OUT_RETAIL_IN
                score += penalty
                breakdown["主力出货散户接盘"] = penalty
                ratio = c.retail_net_inflow / max(abs(c.main_net_inflow), 1)
                c.risk_tags.append(
                    f"⚠️【危险】主力净流出{abs(c.main_net_inflow):.0f}万，"
                    f"散户接盘{c.retail_net_inflow:.0f}万（散户/主力={ratio:.1f}x）"
                )

            # 情形2（较危险）：两者均为净流入，但散户 > 主力 2倍以上
            elif (c.main_net_inflow > 0 and c.retail_net_inflow > 0
                  and c.retail_net_inflow > c.main_net_inflow * 2):
                penalty = self.PENALTY_RETAIL_GT_MAIN_STRONG
                score += penalty
                breakdown["散户远超主力"] = penalty
                ratio = c.retail_net_inflow / max(c.main_net_inflow, 1)
                c.risk_tags.append(
                    f"⚠️【较危险】散户净流入{c.retail_net_inflow:.0f}万 是主力的{ratio:.1f}倍，"
                    f"资金结构偏散户化"
                )

            # 情形3（温和风险）：两者均为净流入，散户略多于主力
            elif (c.main_net_inflow > 0 and c.retail_net_inflow > 0
                  and c.retail_net_inflow > c.main_net_inflow):
                penalty = self.PENALTY_RETAIL_GT_MAIN_MILD
                score += penalty
                breakdown["散户略多于主力"] = penalty
                c.risk_tags.append(
                    f"注意：散户净流入({c.retail_net_inflow:.0f}万) > "
                    f"主力净流入({c.main_net_inflow:.0f}万)，追高需谨慎"
                )

            # --- 涨停 ---
            if c.source in ("limit_up", "lhb_and_zt", "consecutive"):
                score += self.SCORE_LIMIT_UP
                breakdown["涨停"] = self.SCORE_LIMIT_UP

            # --- 连板加分 ---
            if c.consecutive_days >= 2:
                pts = min(self.SCORE_CONSECUTIVE, c.consecutive_days * 3)
                score += pts
                breakdown[f"{c.consecutive_days}连板"] = pts
                c.alert_tags.append(f"{c.consecutive_days}连板")

            # --- 审计风险扣分 ---
            if not c.auditor_pass:
                score -= 20
                breakdown["审计风险扣分"] = -20

            c.total_score = round(score, 1)
            c.score_breakdown = breakdown

        # 分类：正常候选 vs 危险监控（散户>主力导致负分）
        # 危险股单独保存进 danger_watch，不混入主候选列表
        normal = [c for c in candidates if c.total_score >= self.MIN_SCORE_THRESHOLD]
        danger = [c for c in candidates if c.total_score < self.MIN_SCORE_THRESHOLD]

        normal.sort(key=lambda x: x.total_score, reverse=True)
        # 危险股按危险程度排序（分越低越危险，放最前面）
        danger.sort(key=lambda x: x.total_score)

        # 把危险股挂在 _danger_buffer 供 run() 取用
        self._danger_buffer = danger

        logger.info("[监链 Step6] 评分完成：达标 %d 只，危险监控 %d 只", len(normal), len(danger))
        return normal

    # ── Step 7: AI 报告 ───────────────────────────

    def _step7_generate_report(self, report: DragonTigerReport) -> str:
        """使用 AI 生成结构化报告，无 AI 则使用模板"""
        if self.analyzer and hasattr(self.analyzer, "is_available") and self.analyzer.is_available():
            prompt = self._build_report_prompt(report)
            try:
                text = self.analyzer.generate_text(prompt, max_tokens=3000, temperature=0.6)
                if text:
                    return text
            except Exception as e:
                logger.warning("[监链 Step7] AI 报告生成失败: %s", e)

        return self._build_template_report(report)

    def _build_report_prompt(self, report: DragonTigerReport) -> str:
        """构建 AI 报告 Prompt"""
        # 政策摘要
        policy_text = "\n".join(
            f"- {p.topic}（涉及板块: {', '.join(p.affected_sectors) or '待确认'}）"
            for p in report.policy_signals[:5]
        ) or "暂无明确政策信号"

        # 热门板块
        sector_text = "\n".join(
            f"- {s.name}：综合热度 {s.total_score:.0f}分"
            f"（政策{s.policy_score:.0f} + 资金{s.flow_score:.0f} + 涨幅{s.lhb_score:.0f}）"
            f"，资金净流入 {s.net_inflow:.2f}亿，今日涨幅 {s.change_pct:.1f}%"
            for s in report.hot_sectors[:5]
        ) or "暂无热门板块"

        # 候选股票（TOP 15）
        top_cands = report.candidates[:15]
        stock_rows = []
        for c in top_cands:
            tags = " | ".join(c.alert_tags[:3])
            risks = " | ".join(c.risk_tags[:2])
            flow_str = f"主力{c.main_net_inflow:+.0f}万/散户{c.retail_net_inflow:+.0f}万"
            lhb_str = f"机构席位净买{c.institutional_net:+.0f}万" if c.on_lhb else "未上龙虎榜"
            auditor_str = f"审计:{c.auditor_quality}" if c.auditor != "未知" else ""
            row = (
                f"{c.code} {c.name} | 评分:{c.total_score:.0f} | {flow_str} | "
                f"{lhb_str} | {c.flow_dominance} | {auditor_str}"
            )
            if tags:
                row += f" | 信号:{tags}"
            if risks:
                row += f" | ⚠️ {risks}"
            stock_rows.append(row)

        stock_text = "\n".join(stock_rows) or "无达标候选股"

        # 危险监控股（散户>主力）
        danger_rows = []
        for c in report.danger_watch[:10]:
            ratio = (c.retail_net_inflow / max(abs(c.main_net_inflow), 1)) if c.retail_net_inflow > 0 else 0
            level = "🔴最危险" if c.main_net_inflow < 0 else ("🟠较危险" if ratio >= 2 else "🟡温和风险")
            danger_rows.append(
                f"{level} {c.code} {c.name} | 主力{c.main_net_inflow:+.0f}万 / 散户{c.retail_net_inflow:+.0f}万"
                f"（散户/主力={ratio:.1f}x）| 评分:{c.total_score:.0f}"
            )
        danger_text = "\n".join(danger_rows) or "无危险信号个股"

        return f"""你是一位专业的A股量化分析师。请根据以下监链数据生成今日操作建议报告。

【扫描时间】{report.date} {report.scan_time}

【Step1 政策信号】
{policy_text}

【Step2 热门板块 TOP5】
{sector_text}

【Step3-6 候选个股（按评分排序，已过滤危险股）】
{stock_text}

【危险监控个股（散户净流入>主力，出货接盘风险）】
{danger_text}

【统计】
- 候选总数: {report.total_candidates} 只
- 危险监控: {report.danger_count} 只（散户>主力，不在主推荐列表）
- 机构主导流入: {report.institutional_dominant} 只
- 散户主导流入: {report.retail_dominant} 只
- 今日上龙虎榜: {report.lhb_count} 只
- 涨停/连板: {report.limit_up_count} 只

---

请按以下格式生成报告（纯 Markdown，不要代码块）：

## {report.date} A股监链报告（{report.scan_time}）

### 一、今日政策催化
（总结今日政策/事件驱动，1-2段）

### 二、热门板块解析
（分析TOP3板块的逻辑，资金流向，注意上午盘/下午盘差异）

### 三、重点关注个股（机构主导 TOP5）
（对机构净流入最多的股票逐一分析，包括上榜原因、资金性质、操作建议）

### 四、散户 vs 主力资金结构警示
（重点分析以下危险形态，按严重程度排序：
  🔴 最危险：主力净流出 + 散户净流入（出货接盘）
  🟠 较危险：散户净流入是主力2倍以上（散户FOMO）
  🟡 温和风险：散户净流入略高于主力（追高需慎）
列出具体股票代码、散户/主力比例、操作建议）

### 五、今日操作策略
（结合时间段给出具体操作策略：早盘已过→关注尾盘/隔日，下午盘→可介入点位）

### 六、风险提示
（审计风险个股、过热板块、止损建议）

---

请直接输出报告，不要额外说明。建议仅供参考，不构成投资建议。
"""

    def _build_template_report(self, report: DragonTigerReport) -> str:
        """无 AI 时的模板报告"""
        lines = [
            f"## {report.date} A股监链报告（{report.scan_time}）",
            "",
            "### 一、政策信号",
        ]

        if report.policy_signals:
            for p in report.policy_signals[:5]:
                lines.append(f"- **{p.topic}**：涉及板块 {', '.join(p.affected_sectors) or '未识别'}")
        else:
            lines.append("- 暂无明确政策信号（建议手动关注财经新闻）")

        lines += ["", "### 二、热门板块 TOP5"]
        for s in report.hot_sectors[:5]:
            lines.append(
                f"- **{s.name}**：热度 {s.total_score:.0f}分"
                f" | 净流入 {s.net_inflow:.2f}亿"
                f" | 涨幅 {s.change_pct:.1f}%"
            )

        lines += ["", f"### 三、候选个股（共 {report.total_candidates} 只）"]
        lines.append("")
        lines.append("| 代码 | 名称 | 评分 | 主力流入(万) | 散户流入(万) | 特征 | 审计 |")
        lines.append("|------|------|------|------------|------------|------|------|")

        for c in report.candidates[:20]:
            tags = "、".join(c.alert_tags[:2]) or c.source
            risks = " ⚠️" + c.risk_tags[0] if c.risk_tags else ""
            lines.append(
                f"| {c.code} | {c.name} | {c.total_score:.0f} "
                f"| {c.main_net_inflow:+.0f} | {c.retail_net_inflow:+.0f} "
                f"| {tags}{risks} | {c.auditor_quality} |"
            )

        # 按危险等级分类散户>主力的个股
        danger_critical = [
            c for c in report.candidates
            if c.main_net_inflow < 0 and c.retail_net_inflow > 0
        ]
        danger_strong = [
            c for c in report.candidates
            if c.main_net_inflow > 0 and c.retail_net_inflow > c.main_net_inflow * 2
        ]
        danger_mild = [
            c for c in report.candidates
            if c.main_net_inflow > 0 < c.retail_net_inflow
            and c.main_net_inflow < c.retail_net_inflow <= c.main_net_inflow * 2
        ]

        lines += [
            "",
            "### 四、散户 vs 主力资金结构警示",
        ]

        if danger_critical:
            lines.append("#### 🔴 最危险：主力净流出 + 散户净流入（出货接盘形态）")
            for c in danger_critical:
                ratio = c.retail_net_inflow / max(abs(c.main_net_inflow), 1)
                lines.append(
                    f"- **{c.code} {c.name}**：主力流出 {abs(c.main_net_inflow):.0f}万，"
                    f"散户接盘 {c.retail_net_inflow:.0f}万（{ratio:.1f}x）→ 建议回避"
                )
        if danger_strong:
            lines.append("#### 🟠 较危险：散户净流入是主力2倍以上")
            for c in danger_strong:
                ratio = c.retail_net_inflow / max(c.main_net_inflow, 1)
                lines.append(
                    f"- **{c.code} {c.name}**：散户{c.retail_net_inflow:.0f}万 vs 主力{c.main_net_inflow:.0f}万"
                    f"（{ratio:.1f}x）→ 谨慎，散户FOMO情绪主导"
                )
        if danger_mild:
            lines.append("#### 🟡 温和风险：散户略高于主力")
            for c in danger_mild:
                lines.append(
                    f"- **{c.code} {c.name}**：散户{c.retail_net_inflow:.0f}万 > 主力{c.main_net_inflow:.0f}万"
                    f" → 关注但不追高"
                )
        if not (danger_critical or danger_strong or danger_mild):
            lines.append("- 本次候选股中暂无明显散户主导风险（资金结构健康）")

        lines += [
            "",
            "### 五、机构 vs 散户汇总",
            f"- 机构主导流入个股：**{report.institutional_dominant}** 只",
            f"- 散户主导流入个股：**{report.retail_dominant}** 只",
            f"- 今日上龙虎榜：**{report.lhb_count}** 只（机构席位净买为强信号）",
            f"- 🔴主力出货散户接盘：**{len(danger_critical)}** 只",
            f"- 🟠散户远超主力：**{len(danger_strong)}** 只",
            "",
            "### 六、操作建议",
            f"> 扫描时间 {report.scan_time}，注意区分盘中/收盘数据。",
            "> ✅ 机构净买 + 主力净流入 同时满足的个股，优先关注。",
            "> ✅ 主力净流入 + 散户净流出（主力进散户出），最强介入形态。",
            "> ❌ 主力净流出 + 散户净流入，坚决回避，出货接盘典型信号。",
            "> ⚠️ 散户净流入是主力2倍以上，高位谨慎，追高风险大。",
            "",
            "> ⚠️ 以上数据仅供参考，不构成投资建议。市场有风险，投资需谨慎。",
        ]

        return "\n".join(lines)

    def format_report_text(self, report: DragonTigerReport) -> str:
        """将报告格式化为控制台输出文本"""
        return report.ai_report

    @staticmethod
    def build_compact_push(report: "DragonTigerReport", top_n: int = 8) -> str:
        """
        生成适合微信单条推送的精简版报告（约 2-4 KB）。
        包含：热门板块、重点推荐股、危险回避股、操作提示。
        """
        lines = [
            f"## 监链速报 {report.date} {report.scan_time}",
            "",
        ]

        # 热门板块（前3）
        if report.hot_sectors:
            lines.append("**热门板块**")
            for s in report.hot_sectors[:3]:
                lines.append(
                    f"- {s.name}：净流入 {s.net_inflow:.1f}亿 | 涨幅 {s.change_pct:.1f}%"
                )
            lines.append("")

        # 重点推荐（机构主导，评分靠前）
        top = [c for c in report.candidates if c.main_net_inflow > 0][:top_n]
        if top:
            lines.append(f"**重点关注（共 {len(report.candidates)} 只，取前 {len(top)}）**")
            for c in top:
                flag = "🏛机构" if c.institutional_net > 0 else "💰主力"
                lines.append(
                    f"- **{c.code} {c.name}** {flag} | 评分{c.total_score:.0f}"
                    f" | 主力{c.main_net_inflow:+.0f}万 | 散户{c.retail_net_inflow:+.0f}万"
                )
            lines.append("")

        # 危险回避
        danger = [
            c for c in report.candidates
            if c.main_net_inflow < 0 and c.retail_net_inflow > 0
        ]
        if danger:
            lines.append("**⚠️ 危险回避（主力出货 + 散户接盘）**")
            for c in danger[:5]:
                lines.append(f"- ❌ {c.code} {c.name}：主力{c.main_net_inflow:+.0f}万 / 散户{c.retail_net_inflow:+.0f}万")
            lines.append("")

        lines += [
            "**操作原则**",
            "✅ 机构净买 + 主力净流入 → 优先关注",
            "✅ 主力进 + 散户出 → 最强介入",
            "❌ 主力出 + 散户进 → 坚决回避",
            "",
            "> 以上仅供参考，不构成投资建议。",
        ]

        return "\n".join(lines)


def run_dragon_tiger_chain(
    search_service=None,
    data_manager=None,
    analyzer=None,
    trade_date: Optional[str] = None,
    max_stocks: int = 30,
    enable_auditor_check: bool = True,
) -> DragonTigerReport:
    """
    便捷入口：运行完整监链。
    """
    chain = DragonTigerChain(
        search_service=search_service,
        data_manager=data_manager,
        analyzer=analyzer,
        trade_date=trade_date,
        max_stocks=max_stocks,
        enable_auditor_check=enable_auditor_check,
    )
    return chain.run()
