from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Klasterisasi Segmentasi Limit Finansial",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "Avg_Credit_Limit",
    "Total_Credit_Cards",
    "Total_visits_bank",
    "Total_visits_online",
    "Total_calls_made",
]
ID_COLUMNS = ["Sl_No", "Customer Key"]

# Palet warna Biru, Putih, Hitam
PALETTE = {
    "black": "#090D16",
    "charcoal": "#111827",
    "slate": "#475569",
    "blue": "#2563EB",
    "blue_dark": "#1E3A8A",
    "sky": "#0EA5E9",
    "white": "#FFFFFF",
    "bg": "#F4F7FB",
    "border": "#E2E8F0",
}

CLUSTER_COLORS = [
    "#2563EB",  # Royal Blue
    "#090D16",  # Obsidian Black
    "#0EA5E9",  # Sky Blue
    "#1E3A8A",  # Deep Navy
    "#64748B",  # Steel Slate
    "#38BDF8",  # Bright Cyan
    "#334155",  # Dark Charcoal
    "#60A5FA",  # Soft Blue
]


def inject_custom_css() -> None:
    """Memuat gaya visual kustom dari file CSS eksternal (styles.css)."""
    css_path = Path(__file__).resolve().parent / "styles.css"
    if css_path.is_file():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>\n{css_content}\n</style>", unsafe_allow_html=True)


def render_section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def style_axes(ax: plt.Axes) -> None:
    """Menyeragamkan gaya grafik matplotlib dengan tema putih, hitam, dan biru."""
    ax.set_facecolor(PALETTE["white"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors=PALETTE["black"], labelsize=9.5)
    ax.xaxis.label.set_color(PALETTE["black"])
    ax.yaxis.label.set_color(PALETTE["black"])
    ax.title.set_color(PALETTE["black"])
    ax.grid(True, linestyle="--", linewidth=0.7, color="#E2E8F0", alpha=0.9)
    ax.set_axisbelow(True)


def plot_elbow_chart(
    k_values: list[int], inertias: list[float], selected_k: int
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=130)
    fig.patch.set_facecolor(PALETTE["white"])
    style_axes(ax)

    ax.plot(
        k_values,
        inertias,
        color=PALETTE["blue"],
        linewidth=2.6,
        marker="o",
        markersize=7.5,
        markerfacecolor=PALETTE["white"],
        markeredgecolor=PALETTE["blue"],
        markeredgewidth=2.2,
        zorder=3,
    )
    ax.fill_between(
        k_values, inertias, min(inertias), color=PALETTE["blue"], alpha=0.07, zorder=2
    )

    if selected_k in k_values:
        idx = k_values.index(selected_k)
        sel_y = inertias[idx]
        ax.axvline(
            selected_k,
            color=PALETTE["black"],
            linestyle="--",
            linewidth=1.3,
            alpha=0.75,
            zorder=2,
        )
        ax.scatter(
            [selected_k],
            [sel_y],
            s=150,
            color=PALETTE["black"],
            edgecolors=PALETTE["white"],
            linewidths=2,
            zorder=4,
            label=f"K aktif = {selected_k}",
        )
        ax.annotate(
            f"K = {selected_k}\nInertia: {sel_y:,.1f}",
            xy=(selected_k, sel_y),
            xytext=(12, 14),
            textcoords="offset points",
            fontsize=8.8,
            fontweight="bold",
            color=PALETTE["white"],
            bbox=dict(
                boxstyle="round,pad=0.38",
                facecolor=PALETTE["black"],
                edgecolor=PALETTE["blue"],
                lw=1.2,
            ),
            arrowprops=dict(arrowstyle="->", color=PALETTE["black"], lw=1.1),
            zorder=5,
        )

    ax.set_xlabel("Jumlah Cluster (K)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("Inertia", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_title(
        "Elbow Method — Evaluasi Nilai Inertia per Jumlah Cluster",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(k_values)
    ax.legend(
        frameon=True,
        facecolor=PALETTE["white"],
        edgecolor=PALETTE["border"],
        loc="upper right",
    )
    fig.tight_layout()
    return fig


def plot_pca_clusters(
    pca_points: np.ndarray,
    cluster_labels: np.ndarray,
    pca_centers: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=130)
    fig.patch.set_facecolor(PALETTE["white"])
    style_axes(ax)

    plot_data = pd.DataFrame(
        {
            "Komponen 1": pca_points[:, 0],
            "Komponen 2": pca_points[:, 1],
            "Cluster": cluster_labels,
        }
    )

    for cluster_id in sorted(plot_data["Cluster"].unique()):
        subset = plot_data[plot_data["Cluster"] == cluster_id]
        color = CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]
        ax.scatter(
            subset["Komponen 1"],
            subset["Komponen 2"],
            s=52,
            color=color,
            edgecolors=PALETTE["white"],
            linewidths=0.8,
            alpha=0.86,
            label=f"Cluster {cluster_id}",
            zorder=3,
        )

    ax.scatter(
        pca_centers[:, 0],
        pca_centers[:, 1],
        s=165,
        marker="D",
        color=PALETTE["white"],
        edgecolors=PALETTE["black"],
        linewidths=2.2,
        label="Centroid",
        zorder=5,
    )
    for idx, (cx, cy) in enumerate(pca_centers):
        ax.text(
            cx,
            cy,
            f"C{idx}",
            fontsize=7.5,
            fontweight="bold",
            color=PALETTE["blue_dark"],
            ha="center",
            va="center",
            zorder=6,
        )

    ax.set_xlabel("Komponen Utama 1", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("Komponen Utama 2", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_title(
        "Visualisasi Cluster menggunakan PCA (2 Dimensi)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.legend(
        frameon=True,
        facecolor=PALETTE["white"],
        edgecolor=PALETTE["border"],
        ncol=2,
        fontsize=8.8,
        loc="best",
    )
    fig.tight_layout()
    return fig


def plot_cluster_distribution(counts: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
    fig.patch.set_facecolor(PALETTE["white"])
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, color="#E2E8F0", alpha=0.9)

    labels = [f"Cluster {idx}" for idx in counts.index]
    values = counts.values
    total = values.sum()
    colors = [CLUSTER_COLORS[int(idx) % len(CLUSTER_COLORS)] for idx in counts.index]

    bars = ax.barh(
        labels,
        values,
        color=colors,
        height=0.55,
        edgecolor=PALETTE["white"],
        linewidth=1,
    )
    ax.invert_yaxis()

    max_val = max(values) if len(values) else 1
    ax.set_xlim(0, max_val * 1.24)
    for bar, val in zip(bars, values):
        pct = (val / total) * 100 if total else 0
        ax.text(
            val + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,} ({pct:.1f}%)",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["black"],
        )

    ax.set_xlabel("Jumlah Pelanggan", fontsize=9.5, fontweight="bold", labelpad=6)
    ax.set_title(
        "Distribusi Pelanggan per Cluster", fontsize=11, fontweight="bold", pad=10
    )
    fig.tight_layout()
    return fig


def plot_cluster_profiles(cluster_means: pd.DataFrame) -> plt.Figure:
    """Membandingkan profil relatif fitur antar cluster (normalisasi min-max 0-100%)."""
    span = (cluster_means.max() - cluster_means.min()).replace(0, 1)
    normalized = ((cluster_means - cluster_means.min()) / span) * 100

    fig, ax = plt.subplots(figsize=(9.2, 4.3), dpi=130)
    fig.patch.set_facecolor(PALETTE["white"])
    style_axes(ax)
    ax.grid(axis="x", visible=False)

    n_clusters = len(normalized.index)
    n_features = len(normalized.columns)
    x = np.arange(n_features)
    bar_width = min(0.75 / max(n_clusters, 1), 0.24)

    for i, cluster_id in enumerate(normalized.index):
        offset = (i - (n_clusters - 1) / 2) * bar_width
        color = CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]
        ax.bar(
            x + offset,
            normalized.loc[cluster_id].values,
            width=bar_width,
            color=color,
            edgecolor=PALETTE["white"],
            linewidth=0.8,
            label=f"Cluster {cluster_id}",
            zorder=3,
        )

    short_labels = [
        "Avg Credit\nLimit",
        "Total Credit\nCards",
        "Visits\nBank",
        "Visits\nOnline",
        "Calls\nMade",
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9, fontweight="600")
    ax.set_ylabel("Intensitas Relatif (%)", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_title(
        "Perbandingan Profil Karakteristik Antar Cluster (Skala Relatif 0–100%)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.legend(
        frameon=True,
        facecolor=PALETTE["white"],
        edgecolor=PALETTE["border"],
        ncol=min(n_clusters, 4),
        fontsize=8.5,
        loc="upper right",
    )
    fig.tight_layout()
    return fig


inject_custom_css()

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">💳 Machine Learning • K-Means Clustering</div>
        <h1 class="hero-title">Klasterisasi Segmentasi Limit Finansial</h1>
        <p class="hero-subtitle">
            Dashboard analitik untuk mengelompokkan pelanggan kartu kredit menggunakan metode
            <strong>K-Means Clustering</strong> berdasarkan limit kredit, jumlah kartu, dan pola interaksi pelanggan.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_default_data():
    return pd.read_csv("Credit Card Customer Data.csv")


def validate_data(data):
    missing_columns = [col for col in FEATURES if col not in data.columns]
    if missing_columns:
        st.error(
            "Kolom berikut tidak ditemukan pada dataset: " + ", ".join(missing_columns)
        )
        st.stop()

    clean = data.copy()
    for col in FEATURES:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    invalid_count = int(clean[FEATURES].isna().sum().sum())
    if invalid_count:
        st.warning(
            f"Ditemukan {invalid_count} nilai kosong atau bukan angka pada kolom fitur. "
            "Baris yang memiliki nilai tersebut akan dihapus untuk proses clustering."
        )
        clean = clean.dropna(subset=FEATURES)

    if len(clean) < 2:
        st.error("Dataset harus memiliki minimal 2 baris data valid.")
        st.stop()

    return clean


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-tag">Control Panel</div>
            <div class="sidebar-brand-title">Segmentasi Limit Finansial</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.header("Pengaturan")
    uploaded_file = st.file_uploader("Unggah dataset CSV", type=["csv"])
    st.caption(
        "Jika tidak mengunggah file, aplikasi menggunakan dataset bawaan proyek."
    )
    st.divider()
    st.subheader("Parameter Model")
    n_clusters = st.slider(
        "Pilih jumlah cluster (K)", min_value=2, max_value=8, value=3
    )
    random_seed = st.number_input(
        "Random state", min_value=0, max_value=9999, value=42, step=1
    )
    st.markdown(
        """
        <div class="sidebar-feature-box">
            <strong>Fitur Clustering (5):</strong><br/>
            • Avg_Credit_Limit<br/>
            • Total_Credit_Cards<br/>
            • Total_visits_bank<br/>
            • Total_visits_online<br/>
            • Total_calls_made
        </div>
        """,
        unsafe_allow_html=True,
    )

try:
    raw_data = (
        pd.read_csv(uploaded_file) if uploaded_file is not None else load_default_data()
    )
except FileNotFoundError:
    st.error(
        "File dataset tidak ditemukan. Letakkan 'Credit Card Customer Data.csv' satu folder dengan app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Dataset tidak dapat dibaca: {error}")
    st.stop()

data = validate_data(raw_data)
valid_features = data[FEATURES].copy()

if len(valid_features) < n_clusters:
    st.error("Jumlah cluster tidak boleh lebih besar dari jumlah baris data valid.")
    st.stop()

# Standardisasi agar fitur dengan skala besar tidak mendominasi perhitungan jarak.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(valid_features)

# Elbow Method: menghitung inertia untuk beberapa nilai K.
max_k = min(8, len(valid_features) - 1)
k_values = list(range(1, max_k + 1))
inertias = []
for k in k_values:
    model = KMeans(n_clusters=k, random_state=int(random_seed), n_init=10)
    model.fit(scaled_features)
    inertias.append(model.inertia_)

# Model K-Means utama.
kmeans = KMeans(n_clusters=n_clusters, random_state=int(random_seed), n_init=10)
cluster_labels = kmeans.fit_predict(scaled_features)

result = data.copy()
result["Cluster"] = cluster_labels
centers_original = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=FEATURES,
)
centers_original.index = [f"Cluster {i}" for i in range(n_clusters)]

# Metrik evaluasi internal.
silhouette = None
if len(set(cluster_labels)) > 1 and len(set(cluster_labels)) < len(valid_features):
    silhouette = silhouette_score(scaled_features, cluster_labels)

# Ringkasan.
render_section_title("Ringkasan Dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah data awal", f"{len(raw_data):,}")
c2.metric("Data diproses", f"{len(result):,}")
c3.metric("Jumlah fitur", str(len(FEATURES)))
c4.metric("Jumlah cluster", f"K = {n_clusters}")

st.markdown(
    """
    <div class="info-strip">
        <strong>Catatan Identitas:</strong> Kolom <code>Sl_No</code> dan <code>Customer Key</code>
        diperlakukan sebagai identitas pelanggan, sehingga tidak digunakan sebagai fitur clustering.
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Data", "📉 Elbow Method", "🎯 Hasil Clustering", "🔍 Interpretasi Cluster"]
)

with tab1:
    render_section_title("Pratinjau Dataset")
    st.dataframe(raw_data.head(20), width="stretch")
    render_section_title("Statistik Deskriptif Fitur")
    st.dataframe(valid_features.describe().round(2), width="stretch")

with tab2:
    render_section_title("Elbow Method")
    st.write(
        "Grafik ini membantu melihat perubahan nilai inertia pada beberapa pilihan K. "
        "Titik siku dapat menjadi salah satu pertimbangan dalam menentukan jumlah cluster."
    )
    elbow_fig = plot_elbow_chart(k_values, inertias, n_clusters)
    st.pyplot(elbow_fig, width="stretch")
    plt.close(elbow_fig)
    st.info(
        f"Saat ini aplikasi menggunakan K = {n_clusters}. "
        "Silakan bandingkan grafik Elbow dengan kebutuhan analisis sebelum menetapkan K final."
    )

with tab3:
    render_section_title("Visualisasi Hasil Clustering")
    st.write(
        "PCA merangkum lima fitur menjadi dua komponen untuk membantu visualisasi. "
        "PCA hanya digunakan untuk tampilan, sedangkan K-Means dilatih menggunakan lima fitur yang telah distandardisasi."
    )
    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(scaled_features)
    pca_centers = pca.transform(kmeans.cluster_centers_)

    col_plot, col_eval = st.columns([2.2, 1], gap="large")
    with col_plot:
        pca_fig = plot_pca_clusters(pca_points, cluster_labels, pca_centers)
        st.pyplot(pca_fig, width="stretch")
        plt.close(pca_fig)

    with col_eval:
        if silhouette is not None:
            st.metric("Silhouette Score", f"{silhouette:.3f}")
            st.caption(
                "Silhouette Score mendekati 1 umumnya menunjukkan pemisahan cluster yang lebih baik; "
                "nilai mendekati 0 menunjukkan cluster saling berdekatan, dan nilai negatif dapat "
                "mengindikasikan beberapa data mungkin ditempatkan pada cluster yang kurang sesuai."
            )
        counts = result["Cluster"].value_counts().sort_index()
        count_table = counts.rename_axis("Cluster").reset_index(name="Jumlah Pelanggan")
        st.markdown("**Ringkasan Jumlah Pelanggan**")
        st.dataframe(count_table, width="stretch", hide_index=True)

    render_section_title("Distribusi Pelanggan per Cluster")
    dist_fig = plot_cluster_distribution(counts)
    st.pyplot(dist_fig, width="stretch")
    plt.close(dist_fig)

    render_section_title("Data dengan Label Cluster")
    st.dataframe(result, width="stretch")

    csv_bytes = result.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Download hasil clustering (CSV)",
        data=csv_bytes,
        file_name="hasil_klasterisasi_limit_finansial.csv",
        mime="text/csv",
    )

with tab4:
    cluster_means = result.groupby("Cluster")[FEATURES].mean().round(2)

    render_section_title("Profil Visual Antar Cluster")
    profile_fig = plot_cluster_profiles(cluster_means)
    st.pyplot(profile_fig, width="stretch")
    plt.close(profile_fig)

    render_section_title("Karakteristik Rata-rata Setiap Cluster")
    st.dataframe(cluster_means, width="stretch")

    render_section_title("Pusat Cluster (Centroid)")
    st.write(
        "Centroid menunjukkan nilai pusat/rata-rata karakteristik cluster dalam skala asli "
        "setelah dikembalikan dari proses standardisasi."
    )
    st.dataframe(centers_original.round(2), width="stretch")

    render_section_title("Panduan Membaca Hasil")
    st.write(
        "- **Avg_Credit_Limit**: semakin tinggi, rata-rata limit kredit pelanggan pada cluster tersebut semakin besar.\n"
        "- **Total_Credit_Cards**: menunjukkan rata-rata jumlah kartu kredit.\n"
        "- **Total_visits_bank** dan **Total_visits_online**: menunjukkan rata-rata frekuensi kunjungan ke bank dan layanan online.\n"
        "- **Total_calls_made**: menunjukkan rata-rata jumlah panggilan yang dilakukan pelanggan."
    )
    st.warning(
        "Nama segmen seperti 'limit rendah', 'limit menengah', atau 'limit tinggi' "
        "belum ditetapkan otomatis. Tentukan label berdasarkan perbandingan nilai centroid "
        "dan karakteristik cluster yang benar-benar muncul dari hasil analisis."
    )

st.divider()
st.markdown(
    '<div class="app-footer">Proyek Pembelajaran — <strong>K-Means Clustering</strong> | Klasterisasi Segmentasi Limit Finansial</div>',
    unsafe_allow_html=True,
)
