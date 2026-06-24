import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


PROJECT_DIR = Path(__file__).resolve().parent.parent
EXCEL_PATH = PROJECT_DIR / "b00113.xlsx"
OUTPUT_DIR = PROJECT_DIR / "output" / "price_trend_features_output"
OUTPUT_CSV = OUTPUT_DIR / "全シート_項目別_都市別_線形回帰トレンド分析.csv"


def clean_city_name(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    return re.sub(r"[ァ-ヴー]+$", "", s)


def make_item_type(row):
    sheet = str(row["シート名"])

    if sheet.startswith("2121"):
        return "2121_すし"

    if sheet.startswith("2123"):
        return "2123_すし"

    return sheet


def read_sheet_to_long(raw_df, sheet_name):
    item_code = raw_df.iat[6, 8]
    item_name = raw_df.iat[7, 8]
    brand = raw_df.iat[8, 8]
    unit = raw_df.iat[9, 8]

    month_cols = list(range(11, 23))
    data = raw_df.iloc[16:, [7, 8] + month_cols].copy()
    data.columns = ["地域コード", "都市"] + list(range(1, 13))
    data = data.dropna(subset=["地域コード", "都市"])

    long_df = data.melt(
        id_vars=["地域コード", "都市"],
        value_vars=list(range(1, 13)),
        var_name="月",
        value_name="価格_円",
    )

    long_df["都市"] = long_df["都市"].map(clean_city_name)
    long_df["月"] = long_df["月"].astype(int)

    price = long_df["価格_円"].astype(str).str.strip()
    price = price.replace({
        "...": np.nan,
        "…": np.nan,
        "-": np.nan,
        "－": np.nan,
        "x": np.nan,
        "X": np.nan,
        "": np.nan,
        "nan": np.nan,
        "None": np.nan,
    })

    long_df["価格_円"] = pd.to_numeric(price, errors="coerce")

    long_df["シート名"] = sheet_name
    long_df["品目コード"] = str(item_code).strip()
    long_df["品目名"] = str(item_name).strip()
    long_df["銘柄"] = str(brand).strip()
    long_df["単位"] = str(unit).strip()

    return long_df[
        [
            "シート名",
            "品目コード",
            "品目名",
            "銘柄",
            "単位",
            "地域コード",
            "都市",
            "月",
            "価格_円",
        ]
    ]


def calc_trend(group):
    group = group.dropna(subset=["価格_円"]).sort_values("月").copy()

    if len(group) < 6:
        return pd.Series({
            "データ数": len(group),
            "1月価格_円": np.nan,
            "12月価格_円": np.nan,
            "1月から12月の変化額_円": np.nan,
            "1か月あたりの変化額_円": np.nan,
            "年間換算の変化額_円": np.nan,
            "決定係数": np.nan,
        })

    x = group["月"].values.reshape(-1, 1)
    y = group["価格_円"].values

    model = LinearRegression()
    model.fit(x, y)
    pred = model.predict(x)

    jan = group.loc[group["月"] == 1, "価格_円"]
    dec = group.loc[group["月"] == 12, "価格_円"]

    jan_price = jan.iloc[0] if len(jan) > 0 else np.nan
    dec_price = dec.iloc[0] if len(dec) > 0 else np.nan

    if pd.notna(jan_price) and pd.notna(dec_price):
        jan_to_dec_change = dec_price - jan_price
    else:
        jan_to_dec_change = np.nan

    return pd.Series({
        "データ数": len(group),
        "1月価格_円": jan_price,
        "12月価格_円": dec_price,
        "1月から12月の変化額_円": jan_to_dec_change,
        "1か月あたりの変化額_円": model.coef_[0],
        "年間換算の変化額_円": model.coef_[0] * 11,
        "決定係数": r2_score(y, pred),
    })


def main():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excelファイルが見つかりません: {EXCEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sheets = pd.read_excel(
        EXCEL_PATH,
        sheet_name=None,
        header=None,
        dtype=object,
    )

    monthly_long = pd.concat(
        [
            read_sheet_to_long(raw_df, sheet_name)
            for sheet_name, raw_df in sheets.items()
        ],
        ignore_index=True,
    )
    monthly_long["分析項目"] = monthly_long.apply(make_item_type, axis=1)

    trend_city = (
        monthly_long
        .groupby(["分析項目", "都市"])
        [["月", "価格_円"]]
        .apply(calc_trend)
        .reset_index()
        .round(2)
    )

    trend_city.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("項目別・都市別トレンド分析")
    print(trend_city)
    print(f"\n出力ファイル: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
