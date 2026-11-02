import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Battery RUL Predictor", page_icon="🔋", layout="centered")

PALETTE = {
    "light": {"bg": "#fffaf1", "fg": "#1c1c1c", "accent": "#344f88", "accent2": "#66914a", "gold": "#b5803f"},
    "dark": {"bg": "#1c1c1c", "fg": "#fffaf1", "accent": "#87c0c8", "accent2": "#9cc7b8", "gold": "#d1b06f"},
}
theme = "light" if st.query_params.get("theme") == "light" else "dark"
c = PALETTE[theme]

st.markdown(
    f"""
    <style>
    :root {{
        --bg: {c["bg"]}; --fg: {c["fg"]}; --accent: {c["accent"]}; --accent-2: {c["accent2"]}; --gold: {c["gold"]};
        --line: color-mix(in srgb, var(--fg) 18%, transparent);
        --field: color-mix(in srgb, var(--fg) 5%, var(--bg));
    }}
    .stApp {{ background: var(--bg); color: var(--fg); font-family: Helvetica, Arial, sans-serif; }}
    header[data-testid="stHeader"], div[data-testid="InputInstructions"] {{ display: none !important; }}
    .stApp p, .stApp label, .stApp li, .stApp summary, .stApp span {{ color: var(--fg); }}
    h1, h2, h3 {{ font-family: Helvetica, Arial, sans-serif !important; color: var(--fg) !important; font-weight: 400 !important; }}
    .eyebrow {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }}

    [data-testid="stForm"], [data-testid="stExpander"] details {{ border: 1px solid var(--line) !important; border-radius: 12px; background: transparent; }}
    [data-baseweb="input"], [data-baseweb="input"] > div, [data-testid="stNumberInput"] input, [data-testid="stNumberInput"] button {{
        background: var(--field) !important; color: var(--fg) !important; border-color: var(--line) !important;
    }}
    [data-baseweb="input"]:focus-within {{ border-color: var(--accent) !important; }}
    [data-testid="stNumberInputContainer"] {{ border: 1px solid var(--line) !important; }}
    [data-testid="stNumberInputContainer"]:focus-within {{ border-color: var(--accent) !important; }}
    [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{ display: none !important; }}
    [data-testid="stNumberInput"] input::placeholder {{ color: var(--fg); opacity: .4; }}

    .stButton button, [data-testid="stFormSubmitButton"] button {{
        background: transparent !important; color: var(--accent) !important; border: 1px solid var(--accent) !important; border-radius: 999px;
    }}
    .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{ color: var(--gold) !important; border-color: var(--gold) !important; }}
    .stButton button p, [data-testid="stFormSubmitButton"] button p {{ color: inherit !important; }}

    [data-testid="stAlert"] > div {{ background: color-mix(in srgb, var(--gold) 14%, var(--bg)) !important; border: 1px solid var(--gold); border-radius: 12px; }}
    [data-testid="stAlert"] p {{ color: var(--fg) !important; }}

    .result-wrap {{ padding: 1.5rem 0 1rem; }}
    .result {{ padding: 1.25rem 1.5rem; border: 1px solid var(--accent); border-radius: 12px; text-align: center; }}
    .result .value {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 2.4rem; color: var(--accent); }}
    code {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; color: var(--gold) !important; background: var(--field) !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Live demo · Gradient Boosting</div>', unsafe_allow_html=True)
st.title("Battery RUL Predictor")
st.markdown("Input the battery feature parameters below to estimate the Remaining Useful Life (RUL).")

FIELDS = [
    ("Discharge Time (s)", "discharge_time", "e.g., 6026.76"),
    ("Decrement 3.6-3.4V (s)", "decrement_v", "e.g., 1220.00"),
    ("Max. Voltage Discharge (V)", "max_v_discharge", "e.g., 4.25"),
    ("Min. Voltage Charge (V)", "min_v_charge", "e.g., 3.21"),
    ("Time at 4.15V (s)", "time_415v", "e.g., 5505.99"),
    ("Time Constant Current (s)", "time_cc", "e.g., 6161.00"),
    ("Charging Time (s)", "charging_time", "e.g., 10500.33"),
]

TRAIN_RANGE = {
    "discharge_time": (8.69, 958320.37),
    "decrement_v": (31.16, 406703.77),
    "max_v_discharge": (3.043, 4.363),
    "min_v_charge": (3.022, 4.3),
    "time_415v": (14.05, 245101.12),
    "time_cc": (5.98, 880728.1),
    "charging_time": (5.98, 880728.1),
}

with st.form("prediction_form"):
    cols = st.columns(2)
    values = {
        key: cols[i % 2].number_input(label, min_value=0.0, value=None, placeholder=hint, format="%.10g")
        for i, (label, key, hint) in enumerate(FIELDS)
    }
    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    submit_button = mid.form_submit_button("Predict RUL", use_container_width=True)

if submit_button:
    if any(v is None for v in values.values()):
        st.error("All feature parameters must be filled out.")
    else:
        try:
            with st.spinner("Asking the model…"):
                res = requests.post(f"{API_URL}/predict", json=values, timeout=60)
            res.raise_for_status()
            rul = res.json()["predicted_rul"]
            st.markdown(
                f'<div class="result-wrap"><div class="result"><div class="eyebrow">Predicted Remaining Useful Life</div>'
                f'<div class="value">{int(round(rul))} cycles</div></div></div>',
                unsafe_allow_html=True,
            )
            outside = [
                f"{label} ({TRAIN_RANGE[key][0]:g}–{TRAIN_RANGE[key][1]:g})"
                for label, key, _ in FIELDS
                if not TRAIN_RANGE[key][0] <= values[key] <= TRAIN_RANGE[key][1]
            ]
            if outside:
                st.warning(
                    "Some inputs are outside the range the model was trained on: "
                    + "; ".join(outside)
                    + ". Treat this prediction with caution."
                )
        except requests.RequestException:
            st.error("The prediction API is not responding. It may be waking up, so please try again in a minute.")


def styled(chart):
    return chart.configure(background="transparent").configure_view(stroke=None).configure_axis(
        labelColor=c["fg"], titleColor=c["fg"], gridColor=c["fg"], gridOpacity=0.12, domainColor=c["fg"], domainOpacity=0.3,
        labelFont="Helvetica", titleFont="Helvetica", titleFontWeight="normal",
    )


with st.expander("View Model Insights"):
    st.markdown("The contribution of each feature to the Gradient Boosting model's prediction decision:")
    importance = pd.DataFrame({
        "Feature": [f[0] for f in FIELDS],
        "Importance": [0.3907, 0.0742, 0.0147, 0.0083, 0.3132, 0.1927, 0.0061],
    })
    bars = alt.Chart(importance).mark_bar(color=c["accent"], cornerRadiusEnd=3).encode(
        x=alt.X("Importance:Q", title="Importance"),
        y=alt.Y("Feature:N", sort="-x", title=None),
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")],
    ).properties(height=260)
    st.altair_chart(styled(bars), use_container_width=True, theme=None)

    st.markdown("Predicted versus actual Remaining Useful Life on the test set (n = 3,006):")
    scatter = pd.read_csv("scatter.csv")
    points = alt.Chart(scatter).mark_circle(size=14, opacity=0.45, color=c["accent2"]).encode(
        x=alt.X("actual:Q", title="Actual RUL (cycles)"),
        y=alt.Y("predicted:Q", title="Predicted RUL (cycles)"),
        tooltip=["actual", "predicted"],
    )
    ideal = alt.Chart(pd.DataFrame({"actual": [0, 1130], "predicted": [0, 1130]})).mark_line(
        color=c["gold"], strokeDash=[4, 4]
    ).encode(x="actual:Q", y="predicted:Q")
    st.altair_chart(styled((points + ideal).properties(height=380).interactive()), use_container_width=True, theme=None)

    st.markdown(
        """
        * **$R^2$ Score ($\\text{RUL} \\le 200$):** `0.7706`
        * **$R^2$ Score ($\\text{RUL} > 200$):** `0.9956`
        """
    )
