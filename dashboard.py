"""
YiCeNet Streamlit Dashboard — three panels.

Usage:
  streamlit run dashboard.py --server.port 8501

Views:
  1. Performance — reward curves, token cost, latency, termination types
  2. Hexagram Heatmap — 64 hexagrams × time, usage frequency
  3. Tai Chi Compass — overall health + Bagua radar

Data source: metrics.db (SQLite, shared with inference + training)
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ──
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "data", "metrics.db"))
REFRESH_INTERVAL = 10  # seconds between auto-refresh

HEXAGRAM_NAMES = [
    "乾","坤","屯","蒙","需","讼","师","比",
    "小畜","履","泰","否","同人","大有","谦","豫",
    "随","蛊","临","观","噬嗑","贲","剥","复",
    "无妄","大畜","颐","大过","坎","离","咸","恒",
    "遯","大壮","晋","明夷","家人","睽","蹇","解",
    "损","益","夬","姤","萃","升","困","井",
    "革","鼎","震","艮","渐","归妹","丰","旅",
    "巽","兑","涣","节","中孚","小过","既济","未济",
]

TRIGRAM_NAMES = ["乾 (天)", "兑 (泽)", "离 (火)", "震 (雷)",
                 "巽 (风)", "坎 (水)", "艮 (山)", "坤 (地)"]


def init_db():
    """Create tables if not exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            hexagram_id INTEGER,
            candidate_values TEXT,
            action_id INTEGER,
            reward REAL,
            terminal_type TEXT DEFAULT 'active',
            latency_ms REAL,
            token_cost REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT,
            avg_reward REAL,
            win_rate REAL,
            episodes INTEGER,
            duration_sec REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hexagram_usage (
            date TEXT,
            hexagram_id INTEGER,
            count INTEGER DEFAULT 0,
            avg_q_value REAL,
            PRIMARY KEY (date, hexagram_id)
        )
    """)
    conn.commit()
    conn.close()


@st.cache_data(ttl=REFRESH_INTERVAL)
def load_trajectories(hours: int = 24):
    """Load recent trajectories."""
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    df = pd.read_sql_query(
        "SELECT * FROM trajectories WHERE created_at >= ? ORDER BY created_at",
        conn, params=(cutoff,)
    )
    conn.close()
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


@st.cache_data(ttl=REFRESH_INTERVAL)
def load_evaluations(hours: int = 168):
    """Load evaluation history."""
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    df = pd.read_sql_query(
        "SELECT * FROM evaluations WHERE created_at >= ? ORDER BY created_at",
        conn, params=(cutoff,)
    )
    conn.close()
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


@st.cache_data(ttl=REFRESH_INTERVAL)
def load_hexagram_usage(days: int = 7):
    """Load hexagram usage stats."""
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT * FROM hexagram_usage WHERE date >= ? ORDER BY date",
        conn, params=(cutoff,)
    )
    conn.close()
    return df


def panel_performance():
    """Panel 1: Performance metrics."""
    st.header("📈 模型效能 (Performance)")

    col1, col2, col3, col4 = st.columns(4)
    df = load_trajectories(hours=24)

    if df.empty:
        st.info("尚无轨迹数据。等待推理服务写入...")
        return

    # Summary metrics
    avg_reward = df["reward"].mean()
    total_trajs = len(df)
    success_rate = (df["terminal_type"] == "success").mean() * 100
    abandon_rate = (df["terminal_type"] == "abandoned").mean() * 100

    col1.metric("平均奖励", f"{avg_reward:.2f}")
    col2.metric("总轨迹数", total_trajs)
    col3.metric("成功率", f"{success_rate:.0f}%")
    col4.metric("抛弃率", f"{abandon_rate:.0f}%")

    # Reward over time
    st.subheader("奖励趋势 (Reward)")
    df_sorted = df.sort_values("created_at")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted["created_at"],
        y=df_sorted["reward"].rolling(window=20, min_periods=1).mean(),
        mode="lines", name="Reward (MA20)",
        line=dict(color="#00bcd4"),
    ))
    fig.update_layout(
        xaxis_title="时间", yaxis_title="Reward",
        height=300, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Termination type distribution
    st.subheader("终止类型分布")
    type_counts = df["terminal_type"].value_counts()
    colors = {"success": "#4caf50", "abandoned": "#f44336",
              "timeout": "#ff9800", "active": "#9e9e9e"}
    fig2 = go.Figure(data=[
        go.Bar(x=type_counts.index, y=type_counts.values,
               marker_color=[colors.get(t, "#ccc") for t in type_counts.index])
    ])
    fig2.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    # Evaluation history
    st.subheader("版本评估历史 (v1 → v5)")
    df_eval = load_evaluations(hours=720)  # 30 days
    if not df_eval.empty:
        df_agg = df_eval.groupby("version", as_index=False).agg(
            avg_reward=("avg_reward", "mean"),
            episodes=("episodes", "max"),
        )
        # Sort versions numerically (v1, v2, v5…)
        def _ver_key(v):
            try: return int(v.lstrip("v"))
            except: return 0
        df_agg["_sort"] = df_agg["version"].apply(_ver_key)
        df_agg = df_agg.sort_values("_sort").drop(columns="_sort")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_agg["version"], y=df_agg["avg_reward"],
            mode="lines+markers", name="Avg Reward",
            marker=dict(size=10, color="#ff9800"),
            line=dict(color="#ff9800"),
        ))
        fig3.add_trace(go.Bar(
            x=df_agg["version"], y=df_agg["episodes"],
            name="Episodes", yaxis="y2",
            marker_color="rgba(0, 188, 212, 0.5)",
        ))
        fig3.update_layout(
            xaxis_title="版本", yaxis_title="Avg Reward",
            yaxis2=dict(title="Episodes", overlaying="y", side="right"),
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(x=0, y=1),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.caption("尚无评估记录")


def panel_hexagram_heatmap():
    """Panel 2: 64 hexagram usage heatmap."""
    st.header("☯ 卦象热力图 (Hexagram Usage)")

    df_usage = load_hexagram_usage(days=7)

    if df_usage.empty:
        st.info("尚无卦象统计。等待推理服务写入...")
        return

    # Pivot: date × hexagram_id → count
    pivot = df_usage.pivot_table(
        values="count", index="hexagram_id", columns="date",
        fill_value=0
    )

    # Label hexagrams
    pivot.index = [f"{i} {HEXAGRAM_NAMES[i]}" if i < 64 else f"{i}"
                   for i in pivot.index]

    fig = px.imshow(
        pivot,
        color_continuous_scale="Viridis",
        aspect="auto",
        labels=dict(x="日期", y="卦象", color="使用次数"),
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Top hexagrams
    st.subheader("使用最频繁的卦象")
    total_usage = df_usage.groupby("hexagram_id")["count"].sum().sort_values(ascending=False)
    top5 = total_usage.head(5)
    for h_id, cnt in top5.items():
        name = HEXAGRAM_NAMES[h_id] if h_id < 64 else "???"
        st.write(f"  #{h_id+1:2d} {name} — {int(cnt)} 次使用")


def panel_tai_chi():
    """Panel 3: Tai Chi compass / Bagua radar."""
    st.header("🐉 太极八卦效能罗盘 (Tai Chi Compass)")

    df = load_trajectories(hours=24)

    if df.empty:
        st.info("尚无轨迹数据。")
        return

    # Compute overall yang/yin ratio
    # Yang hexagrams = more solid lines (≥3 yang lines)
    total = len(df)
    if total > 0 and "hexagram_id" in df.columns:
        yang_count = df["hexagram_id"].apply(
            lambda h: bin(h).count("1") >= 3
        ).sum()
        yang_pct = yang_count / total * 100
        yin_pct = 100 - yang_pct
    else:
        yang_pct = yin_pct = 50

    # Success rate
    success_rate = (df["terminal_type"] == "success").mean() * 100 if "terminal_type" in df.columns else 50

    # Tai Chi display
    col1, col2 = st.columns([1, 1])

    with col1:
        # Simple Tai Chi visual
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode="markers+text",
            marker=dict(size=80, color="#f0f0f0",
                        line=dict(color="#333", width=2)),
            text=[f"{success_rate:.0f}%"],
            textfont=dict(size=20, color="#333"),
            textposition="middle center",
        ))
        # Yang dot
        fig.add_trace(go.Scatter(
            x=[0], y=[0.25],
            mode="markers",
            marker=dict(size=25, color="#d32f2f"),
        ))
        # Yin dot
        fig.add_trace(go.Scatter(
            x=[0], y=[-0.25],
            mode="markers",
            marker=dict(size=25, color="#1976d2"),
        ))
        fig.update_layout(
            title="综合效能",
            xaxis=dict(visible=False, range=[-1, 1]),
            yaxis=dict(visible=False, range=[-1, 1]),
            height=300, margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bagua radar chart
        st.subheader("八卦原型效能")
        categories = TRIGRAM_NAMES
        # Assign each hexagram to a trigram (upper trigram)
        if not df.empty and "hexagram_id" in df.columns:
            trigram_scores = [0.0] * 8
            trigram_counts = [0] * 8
            for _, row in df.iterrows():
                h = row["hexagram_id"]
                # Upper trigram = top 3 bits
                upper = (h >> 3) & 0b111
                if upper < 8:
                    trigram_scores[upper] += max(0, row.get("reward", 0))
                    trigram_counts[upper] += 1
            for i in range(8):
                if trigram_counts[i] > 0:
                    trigram_scores[i] /= trigram_counts[i]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=trigram_scores if not df.empty else [0]*8,
            theta=categories,
            fill="toself",
            line=dict(color="#00bcd4"),
            name="效能",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[-2, 2]),
            ),
            height=300, margin=dict(l=40, r=40, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Yang/Yin meter
    st.subheader(f"阴阳比例 — 阳 {yang_pct:.0f}% / 阴 {yin_pct:.0f}%")
    st.progress(yang_pct / 100)
    st.caption("阳 = 刚性/探索型卦象（≥3阳爻），阴 = 柔顺/保守型卦象（<3阳爻）")


def main():
    st.set_page_config(
        page_title="YiCeNet 仪表盘",
        page_icon="☯",
        layout="wide",
    )

    st.title("☯ YiCeNet 易策网络仪表盘")
    st.caption(f"数据源: {DB_PATH} | 自动刷新: 每{REFRESH_INTERVAL}秒")

    init_db()

    tab1, tab2, tab3 = st.tabs(["📈 效能", "☯ 卦象", "🐉 罗盘"])

    with tab1:
        panel_performance()
    with tab2:
        panel_hexagram_heatmap()
    with tab3:
        panel_tai_chi()

    # Auto-refresh
    time.sleep(0.1)
    st.rerun()


if __name__ == "__main__":
    main()
