# -*- coding: utf-8 -*-
"""
PushPlus 发送提醒服务

职责：
1. 通过 PushPlus API 发送 PushPlus 消息
"""
import logging
import time
from typing import Optional
from datetime import datetime
import requests

from src.config import Config
from src.formatters import chunk_content_by_max_bytes


logger = logging.getLogger(__name__)


class PushplusSender:
    
    def __init__(self, config: Config):
        """
        初始化 PushPlus 配置

        Args:
            config: 配置对象
        """
        self._pushplus_token = getattr(config, 'pushplus_token', None)
        self._pushplus_topic = getattr(config, 'pushplus_topic', None)
        self._pushplus_max_bytes = getattr(config, 'pushplus_max_bytes', 20000)
        
    def send_to_pushplus(self, content: str, title: Optional[str] = None) -> bool:
        """
        推送消息到 PushPlus

        PushPlus API 格式：
        POST http://www.pushplus.plus/send
        {
            "token": "用户令牌",
            "title": "消息标题",
            "content": "消息内容",
            "template": "html/txt/json/markdown"
        }

        PushPlus 特点：
        - 国内推送服务，免费额度充足
        - 支持微信公众号推送
        - 支持多种消息格式

        Args:
            content: 消息内容（Markdown 格式）
            title: 消息标题（可选）

        Returns:
            是否发送成功
        """
        if not self._pushplus_token:
            logger.warning("PushPlus Token 未配置，跳过推送")
            return False

        api_url = "http://www.pushplus.plus/send"

        if title is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
            title = f"📈 股票分析报告 - {date_str}"

        try:
            # Convert Markdown→HTML once, then size-check the HTML (not raw Markdown).
            # HTML is 2-3x larger than Markdown, so checking Markdown size would
            # skip chunking even when the delivered payload exceeds the limit.
            html_content = self._markdown_to_html(content)
            template = "html" if html_content != content else "markdown"

            html_bytes = len(html_content.encode('utf-8'))
            if html_bytes > self._pushplus_max_bytes:
                logger.info(
                    "PushPlus HTML 内容超长(%s字节)，将分批发送",
                    html_bytes,
                )
                return self._send_pushplus_chunked_html(
                    api_url,
                    html_content,
                    title,
                    template,
                    self._pushplus_max_bytes,
                )

            return self._send_pushplus_payload(api_url, html_content, title, template)
        except Exception as e:
            logger.error(f"发送 PushPlus 消息失败: {e}")
            return False

    def _markdown_to_html(self, md: str) -> str:
        """Convert Markdown to HTML for reliable PushPlus rendering."""
        try:
            import markdown2
            return markdown2.markdown(
                md,
                extras=["tables", "fenced-code-blocks", "strike", "break-on-newline"],
            )
        except Exception:
            return md

    def _send_pushplus_message(self, api_url: str, content: str, title: str) -> bool:
        """Legacy entry point kept for _send_pushplus_chunked compatibility."""
        html_content = self._markdown_to_html(content)
        template = "html" if html_content != content else "markdown"
        return self._send_pushplus_payload(api_url, html_content, title, template)

    def _send_pushplus_payload(self, api_url: str, content: str, title: str, template: str) -> bool:
        payload = {
            "token": self._pushplus_token,
            "title": title,
            "content": content,
            "template": template,
        }

        if self._pushplus_topic:
            payload["topic"] = self._pushplus_topic

        response = requests.post(api_url, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            if result.get('code') == 200:
                logger.info("PushPlus 消息发送成功")
                return True

            error_msg = result.get('msg', '未知错误')
            logger.error(f"PushPlus 返回错误: {error_msg}")
            return False

        logger.error(f"PushPlus 请求失败: HTTP {response.status_code}")
        return False

    def _send_pushplus_chunked(self, api_url: str, content: str, title: str, max_bytes: int) -> bool:
        """Legacy Markdown-chunked path (kept for external callers)."""
        html_content = self._markdown_to_html(content)
        template = "html" if html_content != content else "markdown"
        return self._send_pushplus_chunked_html(api_url, html_content, title, template, max_bytes)

    def _send_pushplus_chunked_html(
        self, api_url: str, html: str, title: str, template: str, max_bytes: int
    ) -> bool:
        """分批发送已转换为 HTML 的 PushPlus 消息。"""
        budget = max(1000, max_bytes - 1500)
        chunks = chunk_content_by_max_bytes(html, budget, add_page_marker=True)
        total_chunks = len(chunks)
        success_count = 0

        logger.info(f"PushPlus 分批发送：共 {total_chunks} 批")

        for i, chunk in enumerate(chunks):
            chunk_title = f"{title} ({i+1}/{total_chunks})" if total_chunks > 1 else title
            if self._send_pushplus_payload(api_url, chunk, chunk_title, template):
                success_count += 1
                logger.info(f"PushPlus 第 {i+1}/{total_chunks} 批发送成功")
            else:
                logger.error(f"PushPlus 第 {i+1}/{total_chunks} 批发送失败")

            if i < total_chunks - 1:
                time.sleep(1)

        return success_count == total_chunks
