"""pages/ai_prediction.py — Tab dự báo nhiệt độ AI (LSTM T+1)."""

import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.styles import inject_scifi_styles


def render_ai_page(predict_fn, selected_province: str, lat: float, lon: float):
    """
    Render tab AI Prediction.

    predict_fn: callable(lat, lon) → dict với keys:
        predicted_temp, predict_time, coords, history_temps, history_times
    """
    inject_scifi_styles()

    st.markdown("<div class='scifi-title'>🤖 AI WEATHER PREDICTION</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='scifi-subtitle'>"
        "LSTM Neural Network · Hourly Forecast · 34 Provinces of Vietnam"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center;color:#8b949e;margin-bottom:24px;'>"
        f"Địa điểm: <b style='color:#00bfff;'>{selected_province}</b> "
        f"({lat}°N, {lon}°E)"
        f"</p>",
        unsafe_allow_html=True,
    )

    col_btn = st.columns([1, 2, 1])[1]
    with col_btn:
        predict_btn = st.button(
            "🚀 KÍCH HOẠT MÔ HÌNH DỰ ĐOÁN",
            type="primary",
            use_container_width=True,
        )

    if not predict_btn:
        return

    with st.spinner("⚡ Đang khởi tạo Neural Network · Phân tích Pattern..."):
        try:
            pred = predict_fn(lat, lon)
        except FileNotFoundError as e:
            st.error(f"⚠️ Không tìm thấy file mô hình: {e}")
            st.info(
                "Đảm bảo 3 file sau trong thư mục `models/`:\n"
                "- `lstm_weather_model_temp.pt`\n"
                "- `scaler_temp.pkl`\n"
                "- `feature_cols_temp.pkl`"
            )
            return
        except ValueError as e:
            st.error(f"⚠️ Lỗi dữ liệu: {e}")
            return
        except Exception as e:
            st.error(f"❌ Lỗi không mong đợi: {e}")
            with st.expander("Xem traceback"):
                st.code(traceback.format_exc())
            return

    predicted_temp = pred["predicted_temp"]
    predict_time   = pred["predict_time"]
    history_temps  = pred["history_temps"]
    history_times  = pred["history_times"]
    actual_lat, actual_lon = pred["coords"]

    # ── Màu sắc theo nhiệt độ ─────────────────────────────────────────────────
    if predicted_temp < 20:
        temp_color = "#00bfff"
        glow_color = "rgba(0,191,255,0.6)"
    elif predicted_temp < 30:
        temp_color = "#ffd700"
        glow_color = "rgba(255,215,0,0.6)"
    else:
        temp_color = "#ff4500"
        glow_color = "rgba(255,69,0,0.6)"

    st.markdown(
        f"""
        <div class="scifi-container">
            <div class="scifi-province">📍 {selected_province}</div>
            <div class="scifi-time">
                ⏱ Dự báo cho: <b>{predict_time}</b>
                &nbsp;|&nbsp; Lưới: ({actual_lat}, {actual_lon})
            </div>
            <div class="scifi-temp" style="
                color:{temp_color};
                text-shadow:0 0 8px {temp_color},0 0 24px {temp_color},0 0 60px {glow_color};
            ">
                {predicted_temp:.1f}<span class="scifi-unit">°C</span>
            </div>
            <div class="scifi-label">Nhiệt độ dự báo T+1</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Biểu đồ dark-theme ────────────────────────────────────────────────────
    connect_times = [history_times[-1], predict_time]
    connect_temps = [history_temps[-1], predicted_temp]

    _BG   = "#0d1117"
    _GRID = "rgba(255,255,255,0.08)"
    _FONT = "#c9d1d9"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_times, y=history_temps,
        mode="lines+markers", name="Nhiệt độ thực tế (24h qua)",
        line=dict(color="#00bfff", width=2.5),
        marker=dict(size=5, color="#00bfff"),
        hovertemplate="<b>%{x}</b><br>%{y:.2f} °C<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=connect_times, y=connect_temps,
        mode="lines", name="Kết nối dự đoán",
        line=dict(color="#ff69b4", width=2, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[predict_time], y=[predicted_temp],
        mode="markers+text",
        name=f"🤖 Dự đoán AI: {predicted_temp:.1f}°C",
        marker=dict(size=22, symbol="star", color="#ff69b4",
                    line=dict(color="#ff1493", width=2)),
        text=[f" {predicted_temp:.1f}°C"],
        textposition="top right",
        textfont=dict(color="#ff69b4", size=14),
        hovertemplate=(
            f"<b>Dự báo AI</b><br>{predict_time}<br>"
            f"<b>{predicted_temp:.2f} °C</b><extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(
            text=(
                f"<span style='color:#00bfff'>"
                f"24 Giờ Qua + Dự Đoán Giờ Tiếp Theo · {selected_province}"
                f"</span>"
            ),
            font=dict(size=16), x=0.5, xanchor="center",
        ),
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_FONT, size=12),
        xaxis=dict(
            title="Thời Gian", showgrid=True, gridcolor=_GRID, gridwidth=1,
            tickangle=-35, color=_FONT, linecolor="rgba(255,255,255,0.2)",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Nhiệt Độ (°C)", showgrid=True, gridcolor=_GRID, gridwidth=1,
            color=_FONT, linecolor="rgba(255,255,255,0.2)",
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0.5)", bordercolor="#00bfff", borderwidth=1,
            font=dict(color=_FONT),
        ),
        margin=dict(t=70, b=90, l=60, r=30),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Chi tiết dữ liệu 24 giờ qua"):
        detail_df = pd.DataFrame({
            "Thời gian":    history_times,
            "Nhiệt độ (°C)": [f"{t:.2f}" for t in history_temps],
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
