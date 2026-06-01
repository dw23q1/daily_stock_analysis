# -*- coding: utf-8 -*-
"""
===================================
龙虎榜 & 资金流 数据获取器
===================================

功能：
1. 获取当日龙虎榜上榜个股列表
2. 按席位名称区分机构 vs 游资/散户
3. 获取个股资金流明细（超大单/大单=主力机构，中小单=散户）
4. 获取涨停板个股池

数据来源：akshare（东方财富接口）
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# 机构席位关键词（东方财富龙虎榜）
_INSTITUTION_SEAT_KEYWORDS = ["机构专用", "机构"]

# 游资/营业部关键词（非机构）
_HOT_MONEY_SEAT_PATTERNS: List[str] = []  # 非机构默认归为游资

# 中国知名审计事务所（基本面过滤白名单）
# 四大 + 国内头部事务所
REPUTABLE_AUDITORS: List[str] = [
    # 四大国际
    "普华永道", "毕马威", "德勤", "安永",
    "PWC", "KPMG", "Deloitte", "Ernst",
    # 国内头部（中注协百强前20）
    "天健", "大华", "立信", "容诚",
    "信永中和", "致同", "中审众环", "广发",
    "中汇", "中喜", "中证天通", "众华",
    "大信", "正中珠江", "亚太", "中联",
    "国富浩华", "中准", "瑞华",  # 注：瑞华已暂停，仅历史参考
]

# 明确存在风险的小所（参考历史违规记录）
RISKY_AUDITORS: List[str] = [
    "新天域", "华兴", "中和正信", "方正",
]


def _import_akshare():
    try:
        import akshare as ak
        return ak
    except ImportError:
        logger.error("akshare 未安装，龙虎榜功能不可用")
        return None


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _date_str(d: datetime) -> str:
    return d.strftime("%Y%m%d")


class LHBFetcher:
    """龙虎榜 & 涨停板 数据获取器"""

    def get_daily_lhb_list(
        self,
        trade_date: Optional[str] = None,
        lookback_days: int = 1,
    ) -> pd.DataFrame:
        """
        获取龙虎榜上榜个股列表。

        Args:
            trade_date: 交易日 YYYYMMDD，默认今日
            lookback_days: 往回查询天数（处理非交易日情况）

        Returns:
            DataFrame，列包含：代码, 名称, 上榜日期, 涨跌幅, 龙虎榜净买额,
                               龙虎榜买入额, 龙虎榜卖出额, 换手率
        """
        ak = _import_akshare()
        if ak is None:
            return pd.DataFrame()

        end_date = trade_date or _today_str()
        # 往前推 lookback_days 找到有数据的交易日
        try:
            end_dt = datetime.strptime(end_date, "%Y%m%d")
        except ValueError:
            end_dt = datetime.now()

        start_dt = end_dt - timedelta(days=lookback_days + 3)
        start_date = _date_str(start_dt)

        try:
            df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                logger.warning("[LHB] 龙虎榜数据为空，日期范围: %s ~ %s", start_date, end_date)
                return pd.DataFrame()

            logger.info("[LHB] 获取龙虎榜 %d 条记录，日期范围: %s ~ %s", len(df), start_date, end_date)

            # 统一列名
            col_map = {}
            for c in df.columns:
                cs = str(c)
                if "代码" in cs:
                    col_map[c] = "code"
                elif "名称" in cs:
                    col_map[c] = "name"
                elif "上榜日期" in cs or "日期" in cs:
                    col_map[c] = "date"
                elif "涨跌幅" in cs:
                    col_map[c] = "change_pct"
                elif "龙虎榜净买额" in cs or "净买额" in cs:
                    col_map[c] = "lhb_net_buy"
                elif "龙虎榜买入额" in cs:
                    col_map[c] = "lhb_buy"
                elif "龙虎榜卖出额" in cs:
                    col_map[c] = "lhb_sell"
                elif "换手率" in cs:
                    col_map[c] = "turnover"
                elif "收盘价" in cs:
                    col_map[c] = "close"
            df = df.rename(columns=col_map)

            # 过滤只保留最新 lookback_days 天
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                cutoff = end_dt - timedelta(days=lookback_days)
                df = df[df["date"] >= cutoff]

            return df.reset_index(drop=True)

        except Exception as e:
            logger.warning("[LHB] 获取龙虎榜失败: %s", e)
            return pd.DataFrame()

    def get_stock_lhb_seat_detail(
        self,
        stock_code: str,
        trade_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        获取单只股票的龙虎榜席位明细，区分机构 vs 游资。

        Returns:
            {
                "is_on_lhb": bool,
                "institutional_buy": float,    # 机构买入总额（万元）
                "institutional_sell": float,
                "institutional_net": float,
                "retail_buy": float,           # 游资买入总额（万元）
                "retail_sell": float,
                "retail_net": float,
                "institutional_seats": int,
                "retail_seats": int,
                "seats": [{"name": str, "type": str, "buy": float, "sell": float, "net": float}],
            }
        """
        result = {
            "is_on_lhb": False,
            "institutional_buy": 0.0,
            "institutional_sell": 0.0,
            "institutional_net": 0.0,
            "retail_buy": 0.0,
            "retail_sell": 0.0,
            "retail_net": 0.0,
            "institutional_seats": 0,
            "retail_seats": 0,
            "seats": [],
        }

        ak = _import_akshare()
        if ak is None:
            return result

        date_str = trade_date or _today_str()

        try:
            df = ak.stock_lhb_stock_detail_date_em(symbol=stock_code, date=date_str)
            if df is None or df.empty:
                return result

            result["is_on_lhb"] = True

            for _, row in df.iterrows():
                seat_name = str(row.get("席位名称", row.get("营业部名称", ""))).strip()
                buy_val = self._safe_float(row.get("买入金额", row.get("买入额", 0))) or 0.0
                sell_val = self._safe_float(row.get("卖出金额", row.get("卖出额", 0))) or 0.0
                net_val = self._safe_float(row.get("净额", buy_val - sell_val)) or (buy_val - sell_val)

                is_inst = any(kw in seat_name for kw in _INSTITUTION_SEAT_KEYWORDS)
                seat_type = "机构" if is_inst else "游资"

                result["seats"].append({
                    "name": seat_name,
                    "type": seat_type,
                    "buy": buy_val / 1e4,   # 转万元
                    "sell": sell_val / 1e4,
                    "net": net_val / 1e4,
                })

                if is_inst:
                    result["institutional_buy"] += buy_val / 1e4
                    result["institutional_sell"] += sell_val / 1e4
                    result["institutional_net"] += net_val / 1e4
                    result["institutional_seats"] += 1
                else:
                    result["retail_buy"] += buy_val / 1e4
                    result["retail_sell"] += sell_val / 1e4
                    result["retail_net"] += net_val / 1e4
                    result["retail_seats"] += 1

        except Exception as e:
            logger.debug("[LHB] %s 席位明细获取失败: %s", stock_code, e)

        return result

    def get_limit_up_pool(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取涨停板个股池（今日涨停股票）。

        Returns:
            DataFrame，含：代码, 名称, 涨跌幅, 最终涨停时间, 涨停统计, 板块, 连板数
        """
        ak = _import_akshare()
        if ak is None:
            return pd.DataFrame()

        date_str = trade_date or _today_str()

        for fn_name in ["stock_zt_pool_em", "stock_zt_pool_previous_em"]:
            try:
                fn = getattr(ak, fn_name)
                df = fn(date=date_str)
                if df is not None and not df.empty:
                    logger.info("[LHB] 涨停板池 %d 只，来源: %s", len(df), fn_name)
                    df = self._normalize_zt_columns(df)
                    return df
            except Exception as e:
                logger.debug("[LHB] %s 失败: %s", fn_name, e)

        return pd.DataFrame()

    def get_strong_stocks_pool(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """获取强势股票池（近期多次涨停或趋势强势）"""
        ak = _import_akshare()
        if ak is None:
            return pd.DataFrame()

        date_str = trade_date or _today_str()

        try:
            df = ak.stock_zt_pool_strong_em(date=date_str)
            if df is not None and not df.empty:
                logger.info("[LHB] 强势股池 %d 只", len(df))
                return self._normalize_zt_columns(df)
        except Exception as e:
            logger.debug("[LHB] 强势股池获取失败: %s", e)

        return pd.DataFrame()

    def get_consecutive_limit_up_pool(self, trade_date: Optional[str] = None) -> pd.DataFrame:
        """获取连板股票池"""
        ak = _import_akshare()
        if ak is None:
            return pd.DataFrame()

        date_str = trade_date or _today_str()

        try:
            df = ak.stock_zt_pool_zbgc_em(date=date_str)
            if df is not None and not df.empty:
                logger.info("[LHB] 连板股池 %d 只", len(df))
                return self._normalize_zt_columns(df)
        except Exception as e:
            logger.debug("[LHB] 连板股池获取失败: %s", e)

        return pd.DataFrame()

    def _normalize_zt_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一涨停板列名"""
        col_map = {}
        for c in df.columns:
            cs = str(c)
            if "代码" in cs:
                col_map[c] = "code"
            elif "名称" in cs:
                col_map[c] = "name"
            elif "涨跌幅" in cs:
                col_map[c] = "change_pct"
            elif "连板" in cs or "连续" in cs:
                col_map[c] = "consecutive_days"
            elif "板块" in cs or "所属" in cs:
                col_map[c] = "sector"
            elif "涨停时间" in cs or "最终" in cs:
                col_map[c] = "limit_up_time"
        return df.rename(columns=col_map)

    @staticmethod
    def _safe_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(str(v).replace(",", "").replace("%", "").strip())
        except (ValueError, TypeError):
            return None


class CapitalFlowFetcher:
    """
    个股 & 板块资金流向获取器

    资金分类逻辑（按单笔金额）：
    - 超大单(>1000万) + 大单(200-1000万) = 主力/机构资金
    - 中单(20-200万) + 小单(<20万)        = 散户/中小资金
    """

    def get_stock_flow_detail(
        self,
        stock_code: str,
        market: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        获取个股资金流明细（主力/散户区分）。

        Args:
            stock_code: 股票代码（纯数字，如 600519）
            market: sh / sz / None（自动推断）

        Returns:
            {
                "main_net_inflow": float,     # 主力净流入（超大单+大单），万元
                "retail_net_inflow": float,   # 散户净流入（中单+小单），万元
                "super_large_net": float,     # 超大单净流入，万元
                "large_net": float,           # 大单净流入，万元
                "medium_net": float,          # 中单净流入，万元
                "small_net": float,           # 小单净流入，万元
                "main_net_pct": float,        # 主力净流入占比（%）
                "flow_dominance": str,        # "机构主导" / "散户主导" / "中性"
                "date": str,
            }
        """
        result = {
            "main_net_inflow": 0.0,
            "retail_net_inflow": 0.0,
            "super_large_net": 0.0,
            "large_net": 0.0,
            "medium_net": 0.0,
            "small_net": 0.0,
            "main_net_pct": 0.0,
            "flow_dominance": "无数据",
            "date": "",
        }

        ak = _import_akshare()
        if ak is None:
            return result

        if market is None:
            market = "sh" if stock_code.startswith("6") else "sz"

        try:
            df = ak.stock_individual_fund_flow(stock=stock_code, market=market)
            if df is None or df.empty:
                return result

            latest = df.iloc[-1]
            result["date"] = str(latest.get("日期", ""))

            # 超大单净流入
            super_large = self._pick(latest, ["超大单净流入-净额", "超大单净额"])
            # 大单净流入
            large = self._pick(latest, ["大单净流入-净额", "大单净额"])
            # 中单净流入
            medium = self._pick(latest, ["中单净流入-净额", "中单净额"])
            # 小单净流入
            small = self._pick(latest, ["小单净流入-净额", "小单净额"])

            result["super_large_net"] = (super_large or 0.0) / 1e4
            result["large_net"] = (large or 0.0) / 1e4
            result["medium_net"] = (medium or 0.0) / 1e4
            result["small_net"] = (small or 0.0) / 1e4

            result["main_net_inflow"] = result["super_large_net"] + result["large_net"]
            result["retail_net_inflow"] = result["medium_net"] + result["small_net"]

            total_abs = abs(result["main_net_inflow"]) + abs(result["retail_net_inflow"])
            if total_abs > 0:
                result["main_net_pct"] = result["main_net_inflow"] / total_abs * 100

            # 流向判断
            main = result["main_net_inflow"]
            retail = result["retail_net_inflow"]
            if main > 0 and main > abs(retail) * 0.5:
                result["flow_dominance"] = "机构主导流入"
            elif retail > 0 and retail > abs(main) * 0.5:
                result["flow_dominance"] = "散户主导流入"
            elif main < 0 and abs(main) > abs(retail) * 0.5:
                result["flow_dominance"] = "机构主导流出"
            elif retail < 0 and abs(retail) > abs(main) * 0.5:
                result["flow_dominance"] = "散户主导流出"
            else:
                result["flow_dominance"] = "资金中性"

        except Exception as e:
            logger.debug("[资金流] %s 获取失败: %s", stock_code, e)

        return result

    def get_sector_flow_ranking(self, top_n: int = 10) -> Dict[str, Any]:
        """
        获取板块资金流排名。

        Returns:
            {
                "inflow_sectors": [{"name": str, "net_inflow": float, "change_pct": float}],
                "outflow_sectors": [{"name": str, "net_inflow": float, "change_pct": float}],
            }
        """
        result: Dict[str, Any] = {"inflow_sectors": [], "outflow_sectors": []}
        ak = _import_akshare()
        if ak is None:
            return result

        for fn_name, indicator in [
            ("stock_sector_fund_flow_rank", "今日"),
            ("stock_board_industry_fund_flow_rank", "今日"),
        ]:
            try:
                fn = getattr(ak, fn_name)
                df = fn(indicator=indicator)
                if df is None or df.empty:
                    continue

                name_col = next((c for c in df.columns if any(k in str(c) for k in ("板块", "行业", "名称"))), None)
                flow_col = next((c for c in df.columns if any(k in str(c) for k in ("净流入", "主力净流入", "净额"))), None)
                chg_col = next((c for c in df.columns if "涨跌幅" in str(c)), None)

                if name_col and flow_col:
                    work = df[[name_col, flow_col] + ([chg_col] if chg_col else [])].copy()
                    work[flow_col] = pd.to_numeric(work[flow_col], errors="coerce")
                    work = work.dropna(subset=[flow_col])

                    top_df = work.nlargest(top_n, flow_col)
                    bot_df = work.nsmallest(top_n, flow_col)

                    def row_to_dict(r):
                        d = {"name": str(r[name_col]), "net_inflow": float(r[flow_col])}
                        if chg_col:
                            try:
                                d["change_pct"] = float(r[chg_col])
                            except Exception:
                                d["change_pct"] = 0.0
                        return d

                    result["inflow_sectors"] = [row_to_dict(r) for _, r in top_df.iterrows()]
                    result["outflow_sectors"] = [row_to_dict(r) for _, r in bot_df.iterrows()]
                    logger.info("[资金流] 板块流向获取成功，来源: %s", fn_name)
                    return result

            except Exception as e:
                logger.debug("[资金流] %s 失败: %s", fn_name, e)

        return result

    @staticmethod
    def _pick(row: pd.Series, keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in row.index:
                v = row[k]
                try:
                    return float(str(v).replace(",", ""))
                except (ValueError, TypeError):
                    pass
        return None


class AuditorFilter:
    """
    基本面审计质量过滤器

    过滤逻辑：
    - 审计机构属于知名事务所 → 通过
    - 审计机构为小所或列入风险名单 → 不通过
    - 无法获取审计信息 → 保守通过（不因缺数据误杀）
    """

    def check_auditor(self, stock_code: str) -> Dict[str, Any]:
        """
        查询股票审计机构并给出质量评级。

        Returns:
            {
                "auditor": str,
                "quality": "top4" | "domestic_top" | "acceptable" | "small" | "risky" | "unknown",
                "pass_filter": bool,
                "note": str,
            }
        """
        result = {
            "auditor": "未知",
            "quality": "unknown",
            "pass_filter": True,  # 默认通过（避免误杀）
            "note": "无审计数据",
        }

        ak = _import_akshare()
        if ak is None:
            return result

        for fn_name, kwargs in [
            ("stock_financial_abstract_ths", {"symbol": stock_code, "indicator": "按年度"}),
            ("stock_financial_abstract_ths", {"symbol": stock_code}),
        ]:
            try:
                fn = getattr(ak, fn_name)
                df = fn(**kwargs)
                if df is None or df.empty:
                    continue

                # 找审计相关列
                audit_col = next(
                    (c for c in df.columns if any(k in str(c) for k in ("审计", "会计师", "事务所", "审计机构"))),
                    None,
                )
                if audit_col is not None:
                    val = str(df[audit_col].iloc[0]).strip()
                    if val and val not in ("nan", "None", "-"):
                        result["auditor"] = val
                        result["quality"], result["pass_filter"], result["note"] = self._classify(val)
                        return result
            except Exception:
                pass

        return result

    def _classify(self, auditor_name: str) -> Tuple[str, bool, str]:
        """返回 (quality, pass_filter, note)"""
        name = auditor_name.lower()

        # 四大
        big4 = ["普华永道", "毕马威", "德勤", "安永", "pwc", "kpmg", "deloitte", "ernst"]
        if any(kw.lower() in name for kw in big4):
            return "top4", True, "四大审计，质量最高"

        # 国内头部
        top_domestic = ["天健", "大华", "立信", "容诚", "信永中和", "致同", "中审众环", "中汇", "大信", "正中珠江"]
        if any(kw in auditor_name for kw in top_domestic):
            return "domestic_top", True, "国内头部事务所"

        # 其他知名所
        if any(kw in auditor_name for kw in REPUTABLE_AUDITORS):
            return "acceptable", True, "知名事务所"

        # 高风险小所
        if any(kw in auditor_name for kw in RISKY_AUDITORS):
            return "risky", False, f"高风险事务所: {auditor_name}"

        # 无法识别 → 保守通过
        return "small", True, f"未知/小型事务所: {auditor_name}（建议人工核查）"
