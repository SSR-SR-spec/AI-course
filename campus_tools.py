"""
校园常用工具 — 为 LLM 提供获取校园日常信息的工具。
"""

from __future__ import annotations

import time
from typing import Any

from app.tools import ToolDef, get_tool_registry


def _get_current_time_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
    }


def _calculate_gpa_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "courses": {
                "type": "array",
                "description": "课程列表，每门课包含课程名、学分和成绩",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "课程名称"},
                        "credit": {"type": "number", "description": "学分"},
                        "score": {"type": "number", "description": "百分制成绩"},
                    },
                    "required": ["name", "credit", "score"],
                },
            },
        },
        "required": ["courses"],
    }


def _get_campus_guide_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": '咨询主题，如"图书馆""食堂""校车""宿舍"等',
            },
        },
        "required": ["topic"],
    }


async def _get_current_time() -> str:
    """获取当前日期和时间。"""
    now = time.localtime()
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    wd = weekdays[now.tm_wday]
    return (
        f"当前时间：{now.tm_year}年{now.tm_mon}月{now.tm_mday}日 "
        f"星期{wd} {now.tm_hour:02d}:{now.tm_min:02d}"
    )


def _score_to_gpa_point(score: float) -> float:
    """百分制成绩转绩点（五分制标准）。"""
    if score >= 90:
        return 4.0
    if score >= 85:
        return 3.7
    if score >= 82:
        return 3.3
    if score >= 78:
        return 3.0
    if score >= 75:
        return 2.7
    if score >= 72:
        return 2.3
    if score >= 68:
        return 2.0
    if score >= 64:
        return 1.5
    if score >= 60:
        return 1.0
    return 0.0


async def _calculate_gpa(courses: list[dict[str, Any]]) -> str:
    """根据课程成绩计算 GPA。"""
    if not courses:
        return "未提供课程数据。"
    total_points = 0.0
    total_credits = 0.0
    details = []
    for c in courses:
        name = c.get("name", "未知课程")
        credit = float(c.get("credit", 0))
        score = float(c.get("score", 0))
        point = _score_to_gpa_point(score)
        total_points += point * credit
        total_credits += credit
        details.append(
            f"  {name}：{score}分 → {point}绩点（{credit}学分）"
        )
    if total_credits == 0:
        return "学分总和为0，无法计算GPA。"
    gpa = total_points / total_credits
    lines = [
        f"GPA 计算结果（五分制）：",
        *details,
        f"",
        f"总学分：{total_credits}",
        f"总绩点：{total_points:.2f}",
        f"平均绩点（GPA）：{gpa:.2f}",
    ]
    return "\n".join(lines)


async def _get_campus_guide(topic: str) -> str:
    """查询校园生活指南（模拟数据）。"""
    guide = {
        "图书馆": (
            "湖北经济学院图书馆相关信息：\n"
            "- 开放时间：周一至周日 8:00-22:00\n"
            '- 座位预约：通过微信公众号"湖北经济学院图书馆"预约\n'
            "- 借阅限额：本科生可借10册，借期30天\n"
            "- 位置：校园中心区域，行政楼旁边"
        ),
        "食堂": (
            "湖北经济学院食堂信息：\n"
            "- 一粟堂：位于教学区附近，主要供应学生套餐\n"
            "- 三清园：风味食堂，有各地美食窗口\n"
            "- 五味轩：教工食堂，学生也可就餐\n"
            "- 七品居：特色餐厅，适合聚餐\n"
            "- 九华厅：自助餐厅\n"
            "- 营业时间：早餐 6:30-9:00，午餐 11:00-13:00，晚餐 17:00-19:30"
        ),
        "校车": (
            "湖北经济学院校车信息：\n"
            "- 路线：学校 ↔ 光谷广场\n"
            "- 发车时间：工作日 7:00-18:00 每小时一班\n"
            "- 票价：5元/人（刷校园卡）\n"
            "- 上车点：校门口广场"
        ),
        "宿舍": (
            "湖北经济学院宿舍信息：\n"
            "- 宿舍类型：四人间（大部分），上床下桌\n"
            "- 设施：空调、热水器、独立卫生间\n"
            "- 网络：校园网全覆盖，需办理开户\n"
            "- 水电费：每月每人有免费额度，超出部分自理"
        ),
    }
    result = guide.get(topic)
    if result:
        return result
    return (
        f"暂无「{topic}」的详细信息。建议访问校园官网或咨询学生事务中心。"
        f"可查询的主题：{', '.join(guide.keys())}"
    )


def register_campus_tools() -> None:
    """注册所有校园生活相关工具。"""
    registry = get_tool_registry()
    registry.register(
        ToolDef(
            name="get_current_time",
            description="获取当前的日期和时间。",
            parameters=_get_current_time_schema(),
            function=_get_current_time,
        )
    )
    registry.register(
        ToolDef(
            name="calculate_gpa",
            description="根据课程成绩（百分制）计算平均绩点（五分制GPA）。",
            parameters=_calculate_gpa_schema(),
            function=_calculate_gpa,
        )
    )
    registry.register(
        ToolDef(
            name="get_campus_guide",
            description="查询校园生活指南，如图书馆、食堂、校车、宿舍等信息。",
            parameters=_get_campus_guide_schema(),
            function=_get_campus_guide,
        )
    )
