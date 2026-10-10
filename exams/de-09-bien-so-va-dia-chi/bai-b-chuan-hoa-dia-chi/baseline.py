from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

PHONE = r"(?:\+84|\b0)\d[\d. ]{7,12}\d"
HOUSE = re.compile(r"\d+[a-z]?(?:/\d+[a-z]?)*")


def simplify(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-z0-9/]+", " ", s.lower()).strip()


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Baseline Bài B: không học từ train; truy hồi TF-IDF n-gram ký tự địa chỉ -> cặp (đường, xã mới) dựng từ danh mục.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    gaz = args.data / "gazetteer"
    require_data(gaz / "xa_moi.csv")
    require_data(test_dir / "test.csv", args.split)
    xa, tinh, duong = (pd.read_csv(gaz / f"{n}.csv") for n in ("xa_moi", "tinh_moi", "duong"))
    xa_cu, sap_nhap = pd.read_csv(gaz / "xa_cu.csv"), pd.read_csv(gaz / "sap_nhap.csv")
    test = pd.read_csv(test_dir / "test.csv", dtype=str, keep_default_na=False)

    tinh_name = dict(zip(tinh["ma_tinh_moi"], tinh["ten_tinh_moi"]))
    xa_name = dict(zip(xa["ma_xa_moi"], xa["ten_xa_moi"]))
    xa_tinh = dict(zip(xa["ma_xa_moi"], xa["ma_tinh_moi"]))
    # tên các xã, huyện, tỉnh cũ được sáp nhập vào mỗi xã mới, nối thành một chuỗi
    merged = sap_nhap.merge(xa_cu, on="ma_xa_cu")
    merged["t"] = merged["ten_xa_cu"] + " " + merged["ten_huyen_cu"] + " " + merged["ten_tinh_cu"]
    olds = merged.groupby("ma_xa_moi")["t"].apply(" ".join)

    # mỗi dòng của duong.csv (một cặp đường - xã mới) thành một "tài liệu"
    docs = []
    for ma_xa, ten_duong in zip(duong["ma_xa_moi"], duong["ten_duong"]):
        street = simplify(ten_duong)
        rest = simplify(f"{xa_name[ma_xa]} {tinh_name[xa_tinh[ma_xa]]} {olds.get(ma_xa, '')}")
        docs.append(f"{street} {street} {rest}")
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)
    mat = vec.fit_transform(docs)

    text = test["dia_chi"].str.replace(PHONE, " ", regex=True)
    query = vec.transform(text.map(simplify))
    best = np.concatenate([(query[i:i + 500] @ mat.T).toarray().argmax(axis=1) for i in range(0, query.shape[0], 500)])

    def house(t: str) -> str:
        return next((tok.upper() for tok in simplify(t).split() if HOUSE.fullmatch(tok)), "")

    ma_xa_best = duong["ma_xa_moi"].to_numpy()[best]
    out = pd.DataFrame({
        "id": test["id"],
        "so_nha": text.map(house),
        "duong": duong["ten_duong"].to_numpy()[best],
        "xa_moi": [xa_name[m] for m in ma_xa_best],
        "tinh_moi": [tinh_name[xa_tinh[m]] for m in ma_xa_best],
    })
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
