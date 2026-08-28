"""usedsientaL.csv（カーセンサー生スクレイピングデータ）を分析用に整形する。

入力: sampledata/scraped/usedsientaL.csv  （Octoparse の生出力・5513行）
出力: sampledata/processed/usedsienta_clean.parquet

生データは数値列に改行とタブが混ざり、区切り文字に NBSP(\xa0) と全角スペース(　)
が使われている。ここではそれらを剥がして型を付け、タイトル列を
「車名 / グレード / 装備の羅列」の3層に分解する。

装備の羅列（表記ゆれが激しい部分）はここでは正規化せず、テキストと語のリストの
まま残す。そこを LLM で特徴量化するのが本プロジェクトの主題なので、
手作りルールで潰さないでおく。

実行:
    .venv/bin/python scripts/clean_sienta.py
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "sampledata" / "scraped" / "usedsientaL.csv"
DST = ROOT / "sampledata" / "processed" / "usedsienta_clean.parquet"

# スクレイピング実施時点。車齢・車検残の基準にする（2026年3月取得）
SCRAPED_AT = pd.Timestamp("2026-03-01")

# 末尾が2文字以下でも切れていない語。これ以外の短い末尾は途中切れとみなす
SHORT_TAIL_OK = {"CD", "TV", "BT", "ナビ", "ETC", "AW", "PS", "MT", "AT", "HV"}


def strip_noise(s: pd.Series) -> pd.Series:
    """改行・タブ・NBSP を除去して前後の空白を落とす。"""
    return (
        s.astype(str)
        .str.replace(" ", " ", regex=False)
        .str.replace(r"[\n\r\t]", "", regex=True)
        .str.strip()
    )


def parse_man_yen(s: pd.Series) -> pd.Series:
    """「225.9万円」→ 225.9。「応談」「---万円」は欠損にする。"""
    v = strip_noise(s)
    return pd.to_numeric(v.str.extract(r"^([\d.]+)万円$")[0], errors="coerce")


def parse_km(s: pd.Series) -> pd.Series:
    """「2.2万km」→ 22000、「10km」→ 10。先頭の「交換車」（メーター交換）は落とす。"""
    v = strip_noise(s).str.replace("^交換車", "", regex=True)
    man = pd.to_numeric(v.str.extract(r"^([\d.]+)万km$")[0], errors="coerce") * 10000
    plain = pd.to_numeric(v.str.extract(r"^([\d.]+)km$")[0], errors="coerce")
    return man.fillna(plain)


def parse_shaken(v: pd.Series) -> pd.DataFrame:
    """車検列はカテゴリ（車検整備付など）と満了年月が混在しているので2列に割る。"""
    ym = v.str.extract(r"^(\d{4})\(.\d{2}\)年(\d{2})月$")
    expiry = pd.to_datetime(
        ym[0].str.cat(ym[1], sep="-", na_rep=""), format="%Y-%m", errors="coerce"
    )
    kubun = v.where(expiry.isna(), "期限指定")
    remain = (
        (expiry.dt.year - SCRAPED_AT.year) * 12 + (expiry.dt.month - SCRAPED_AT.month)
    ).astype("Float64")
    return pd.DataFrame({"車検区分": kubun, "車検満了": expiry, "車検残月数": remain})


def split_title(raw: pd.Series) -> pd.DataFrame:
    """タイトルを 車名 / グレード / 装備 の3層に割る。

    生の区切りは NBSP。装備の羅列は全角スペース区切りだが、途中切れの行では
    半角スペースになっているため、両方を区切りとして扱う。
    """
    parts = raw.astype(str).str.split(" ")
    shamei = parts.map(lambda p: p[0].strip())
    grade_raw = parts.map(lambda p: p[1].strip() if len(p) > 1 else "")
    equip_raw = parts.map(lambda p: p[2].strip() if len(p) > 2 else "")

    # グレード: 「ハイブリッド 1.5 G クエロ」→ HV=True, 排気量=1.5, グレード名="G クエロ"
    hybrid = grade_raw.str.contains("ハイブリッド")
    disp = pd.to_numeric(grade_raw.str.extract(r"(\d\.\d)")[0], errors="coerce")
    grade_name = (
        grade_raw.str.replace("ハイブリッド", "", regex=False)
        .str.replace(r"\d\.\d", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 装備: 全角/半角スペースどちらでも区切って語のリストにする
    equip_norm = equip_raw.str.replace("　", " ", regex=False).str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    equip_list = equip_norm.map(lambda s: [w for w in s.split(" ") if w])

    # 途中切れ判定: 末尾の語が2文字以下で、既知の短縮語でないもの
    tail = equip_list.map(lambda ws: ws[-1] if ws else "")
    truncated = (tail.str.len() <= 2) & (tail != "") & (~tail.isin(SHORT_TAIL_OK))

    return pd.DataFrame(
        {
            "車名": shamei,
            "グレード_原文": grade_raw,
            "ハイブリッド": hybrid,
            "排気量": disp,
            "グレード名": grade_name.replace("", pd.NA),
            "装備テキスト": equip_norm.replace("", pd.NA),
            "装備リスト": equip_list,
            "装備数": equip_list.map(len),
            "装備記載あり": equip_list.map(len) > 0,
            "タイトル切れ疑い": truncated,
            "タイトル文字数": raw.astype(str).str.len(),
        }
    )


def normalize_color(s: pd.Series) -> pd.DataFrame:
    """「黒Ｍ」「赤ＭII」→ 基本色「黒」「赤」とメタリック有無に分ける。"""
    v = strip_noise(s)
    metallic = v.str.contains("Ｍ")
    base = v.str.replace(r"[ＭＰI]+$", "", regex=True).str.strip().replace("", pd.NA)
    return pd.DataFrame({"色": v, "色_基本": base, "色_メタリック": metallic})


def main() -> None:
    raw = pd.read_csv(SRC, encoding="utf-8-sig")
    out = pd.DataFrame(index=raw.index)

    # 物件ID: 詳細URL の AU######## 部分。URLは34%欠損なので ID も欠損しうる
    out["物件ID"] = raw["cassettemain_subimg_URL"].str.extract(r"/detail/([A-Z0-9]+)/")[0]

    # --- 目的変数 ---
    out["車両本体価格_万円"] = parse_man_yen(raw["車両本体価格"])
    # 注意: 支払総額は車両本体価格＋諸費用。目的変数のリークになるので特徴量に使わない
    out["支払総額_万円"] = parse_man_yen(raw["支払総額"])

    # --- 数値・日付 ---
    out["年式"] = pd.to_numeric(
        strip_noise(raw["年式"]).str.extract(r"^(\d{4})")[0], errors="coerce"
    ).astype("Int64")
    out["車齢"] = (SCRAPED_AT.year - out["年式"]).astype("Int64")
    out["走行距離_km"] = parse_km(raw["走行距離"])
    out = out.join(parse_shaken(strip_noise(raw["車検"])))

    # --- フラグ ---
    out["修復歴あり"] = strip_noise(raw["修復歴"]).eq("あり")
    out["保証付"] = strip_noise(raw["保証"]).eq("保証付")

    # 200万km など明らかに桁がおかしい出品があるので落とさずに印だけ付ける
    out["走行距離_異常"] = out["走行距離_km"] > 300000

    # --- カテゴリ ---
    out = out.join(normalize_color(raw["色"]))
    out["都道府県"] = strip_noise(raw["都道府県"])
    out["店舗"] = strip_noise(raw["店舗"])

    # --- タイトル分解 ---
    out["タイトル"] = raw["タイトル"].astype(str).str.replace(" ", " ", regex=False)
    out = out.join(split_title(raw["タイトル"]))

    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DST, index=False)

    print(f"{SRC.name} {len(raw)}行 → {DST} {out.shape[0]}行 {out.shape[1]}列")
    print("\n--- 欠損率（上位） ---")
    na = out.isna().mean().sort_values(ascending=False)
    print(na[na > 0].round(4).to_string() or "（欠損なし）")
    print("\n--- 主要列の要約 ---")
    print(out[["車両本体価格_万円", "年式", "走行距離_km", "車検残月数", "装備数"]]
          .describe().round(2).to_string())
    print("\n--- フラグの比率 ---")
    for c in ["ハイブリッド", "修復歴あり", "保証付", "装備記載あり", "タイトル切れ疑い",
              "色_メタリック", "走行距離_異常"]:
        print(f"  {c}: {out[c].mean():.3f}")
    print("\n--- グレード名 上位10 ---")
    print(out["グレード名"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
