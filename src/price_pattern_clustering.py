from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = (
    PROJECT_DIR
    / "output"
    / "price_trend_features_output"
    / "全シート_項目別_都市別_線形回帰トレンド分析.csv"
)
OUTPUT_DIR = PROJECT_DIR / "output" / "price_pattern_clustering_output"
CLUSTER_N = 4


FEATURE_COLS = [
    "1月価格_円",
    "1か月あたりの変化額_円",
    "決定係数",
]


def make_cluster_name(row, price_median):
    slope = row["平均_1か月あたりの変化額_円"]
    price = row["平均_1月価格_円"]
    r2 = row["平均_決定係数"]

    if slope < -1:
        base = "価格低下型"
    elif slope >= 10:
        base = "急上昇型"
    elif slope >= 3:
        base = "上昇型"
    else:
        base = "横ばい・緩やか変動型"

    if price >= price_median:
        price_label = "高価格帯"
    else:
        price_label = "低価格帯"

    if r2 >= 0.7:
        r2_label = "直線性高"
    else:
        r2_label = "変動あり"

    return f"{base}_{price_label}_{r2_label}"


def evaluate_cluster_count(x_scaled):
    k_list = list(range(2, 9))
    rows = []

    for k in k_list:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )
        labels = model.fit_predict(x_scaled)
        rows.append({
            "クラスター数": k,
            "SSE_inertia": model.inertia_,
            "シルエット係数": silhouette_score(x_scaled, labels),
        })

    return pd.DataFrame(rows)


def plot_pca_scatter(cluster_df):
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(9, 6))

    for cluster_id in sorted(cluster_df["クラスター"].unique()):
        df = cluster_df[cluster_df["クラスター"] == cluster_id]
        label = df["クラスター名"].iloc[0]

        plt.scatter(
            df["主成分1"],
            df["主成分2"],
            label=f"クラスター{cluster_id}: {label}",
            alpha=0.7,
        )

    plt.xlabel("主成分1")
    plt.ylabel("主成分2")
    plt.title("都市×品目の価格変動パターンのクラスタリング")
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "クラスタリング結果_PCA散布図.png",
        dpi=200,
    )
    plt.close()


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            "クラスタリング用CSVが見つかりません。先に "
            "price_trend_regression.py を実行してください: "
            f"{INPUT_CSV}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trend_city = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    cluster_df = trend_city.dropna(subset=FEATURE_COLS).copy()

    x = cluster_df[FEATURE_COLS].copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    cluster_count_result = evaluate_cluster_count(x_scaled)
    cluster_count_result.to_csv(
        OUTPUT_DIR / "クラスター数検討結果.csv",
        index=False,
        encoding="utf-8-sig",
    )

    kmeans = KMeans(
        n_clusters=CLUSTER_N,
        random_state=42,
        n_init=20,
    )
    cluster_df["クラスター"] = kmeans.fit_predict(x_scaled)

    cluster_summary_raw = (
        cluster_df
        .groupby("クラスター")
        .agg(
            件数=("都市", "count"),
            品目数=("分析項目", "nunique"),
            都市数=("都市", "nunique"),
            平均_1月価格_円=("1月価格_円", "mean"),
            平均_12月価格_円=("12月価格_円", "mean"),
            平均_1月から12月の変化額_円=("1月から12月の変化額_円", "mean"),
            平均_1か月あたりの変化額_円=("1か月あたりの変化額_円", "mean"),
            平均_年間換算の変化額_円=("年間換算の変化額_円", "mean"),
            平均_決定係数=("決定係数", "mean"),
        )
        .reset_index()
    )

    price_median = cluster_summary_raw["平均_1月価格_円"].median()
    cluster_summary_raw["クラスター名"] = cluster_summary_raw.apply(
        make_cluster_name,
        axis=1,
        price_median=price_median,
    )

    cluster_name_map = dict(
        zip(
            cluster_summary_raw["クラスター"],
            cluster_summary_raw["クラスター名"],
        )
    )
    cluster_df["クラスター名"] = cluster_df["クラスター"].map(cluster_name_map)

    cluster_summary = (
        cluster_df
        .groupby(["クラスター", "クラスター名"])
        .agg(
            件数=("都市", "count"),
            品目数=("分析項目", "nunique"),
            都市数=("都市", "nunique"),
            平均_1月価格_円=("1月価格_円", "mean"),
            平均_12月価格_円=("12月価格_円", "mean"),
            平均_1月から12月の変化額_円=("1月から12月の変化額_円", "mean"),
            平均_1か月あたりの変化額_円=("1か月あたりの変化額_円", "mean"),
            平均_年間換算の変化額_円=("年間換算の変化額_円", "mean"),
            平均_決定係数=("決定係数", "mean"),
        )
        .reset_index()
        .round(2)
    )
    cluster_summary.to_csv(
        OUTPUT_DIR / "クラスター別_特徴まとめ.csv",
        index=False,
        encoding="utf-8-sig",
    )

    item_cluster_count = (
        cluster_df
        .groupby(["クラスター", "クラスター名", "分析項目"])
        .size()
        .reset_index(name="件数")
        .sort_values(["クラスター", "件数"], ascending=[True, False])
    )
    item_cluster_count.to_csv(
        OUTPUT_DIR / "クラスター別_品目構成.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(x_scaled)
    cluster_df["主成分1"] = pca_result[:, 0]
    cluster_df["主成分2"] = pca_result[:, 1]

    cluster_df.to_csv(
        OUTPUT_DIR / "都市別品目別_クラスタリング結果_PCA付き.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_pca_scatter(cluster_df)

    print("\nクラスター数の検討結果")
    print(cluster_count_result.round(3))
    print("\nクラスター別の特徴")
    print(cluster_summary)
    print(f"\n出力フォルダ: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
