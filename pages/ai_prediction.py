"""pages/ai_prediction.py — Tab dự báo nhiệt độ AI (Seq2Seq LSTM, 168 giờ)."""

import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.styles import inject_scifi_styles

_BG, _GRID, _FONT = "#0d1117", "rgba(255,255,255,0.08)", "#c9d1d9"


def _run_prediction(predict_fn, lat, lon):
    """Gọi mô hình, bắt lỗi và hiển thị thông báo thân thiện. Trả dict hoặc None."""
    with st.spinner("⚡ Đang chạy Encoder-Decoder · Suy luận 168 giờ tới..."):
        try:
            return predict_fn(lat, lon)
        except FileNotFoundError as e:
            st.error(f"⚠️ Không tìm thấy file mô hình: {e}")
        except ValueError as e:
            st.error(f"⚠️ Lỗi dữ liệu: {e}")
        except Exception as e:
            st.error(f"❌ Lỗi không mong đợi: {e}")
            with st.expander("Xem traceback"):
                st.code(traceback.format_exc())
    return None


def _render_metrics(forecast):
    """3 khối chỉ số: trung bình / cao nhất / thấp nhất tuần tới."""
    s = pd.Series(forecast)
    c1, c2, c3 = st.columns(3)
    c1.metric("🌡️ Nhiệt độ TB tuần tới", f"{s.mean():.1f} °C")
    c2.metric("🔥 Cao nhất dự báo",        f"{s.max():.1f} °C")
    c3.metric("❄️ Thấp nhất dự báo",       f"{s.min():.1f} °C")


def _build_chart(pred, province):
    """Đồ thị nối liền 72h thực tế (xanh) + 168h dự báo (hồng nét đứt)."""
    h_t, h_v = pred["history_times"], pred["history_temps"]
    f_t, f_v = pred["forecast_times"], pred["forecast_temps"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h_t, y=h_v, mode="lines", name="Thực tế (72h qua)",
        line=dict(color="#00bfff", width=2.5),
        hovertemplate="<b>%{x}</b><br>%{y:.2f} °C<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[h_t[-1]] + f_t, y=[h_v[-1]] + f_v, mode="lines",
        name="🤖 Dự báo AI (168h tới)",
        line=dict(color="#ff69b4", width=2.5, dash="dash"),
        hovertemplate="<b>%{x}</b><br>%{y:.2f} °C<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<span style='color:#00bfff'>Sóng Nhiệt Độ 7 Ngày · {province}</span>",
            font=dict(size=16), x=0.5, xanchor="center",
        ),
        paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_FONT, size=12),
        xaxis=dict(title="Thời Gian", showgrid=True, gridcolor=_GRID,
                   tickangle=-35, color=_FONT, tickfont=dict(size=10)),
        yaxis=dict(title="Nhiệt Độ (°C)", showgrid=True, gridcolor=_GRID, color=_FONT),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="#00bfff", borderwidth=1, font=dict(color=_FONT)),
        margin=dict(t=70, b=90, l=60, r=30), hovermode="x unified",
    )
    return fig


def _render_table(pred):
    """Bảng 168 giờ + nút tải CSV."""
    df = pd.DataFrame({
        "Thời gian":     pred["forecast_times"],
        "Nhiệt độ (°C)": [round(t, 2) for t in pred["forecast_temps"]],
    })
    with st.expander("📊 Chi tiết dự báo 168 giờ"):
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Tải CSV", df.to_csv(index=False).encode("utf-8-sig"),
            file_name="du_bao_168h.csv", mime="text/csv",
        )


def render_ai_page(predict_fn, selected_province=None, lat=None, lon=None):
    """Render tab AI Prediction (dự báo 7 ngày). lat/lon do sidebar truyền vào."""
    if lat is None or lon is None:
        st.warning("Hãy chọn Tỉnh/Thành và điểm lưới ở thanh bên.")
        return
    inject_scifi_styles()

    st.markdown("<div class='scifi-title'>🤖 AI WEATHER PREDICTION</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='scifi-subtitle'>"
        "Seq2Seq LSTM Encoder-Decoder · Dự báo 168 giờ · Lưới đất liền Việt Nam"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center;color:#8b949e;margin-bottom:24px;'>"
        f"Địa điểm: <b style='color:#00bfff;'>{selected_province}</b> "
        f"({lat}°N, {lon}°E)</p>",
        unsafe_allow_html=True,
    )

    with st.columns([1, 2, 1])[1]:
        if not st.button("🚀 KÍCH HOẠT DỰ BÁO 7 NGÀY", type="primary", width="stretch"):
            return

    pred = _run_prediction(predict_fn, lat, lon)
    if pred is None:
        return

    a_lat, a_lon = pred["coords"]
    st.caption(f"Lưới thực tế dùng để dự báo: ({a_lat}, {a_lon})")
    _render_metrics(pred["forecast_temps"])
    st.plotly_chart(_build_chart(pred, selected_province), width="stretch")
    _render_table(pred)
