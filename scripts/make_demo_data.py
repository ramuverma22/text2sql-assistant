"""Generate SMALL, FAKE CSVs in the exact Olist format.

Only for trying the app / running tests before you download the real dataset.
The values are synthetic, so benchmark results on this data are not meaningful.

Usage:
    python scripts/make_demo_data.py --out data/demo_csv
    python scripts/build_db.py --csv-dir data/demo_csv --db data/demo.db
"""
import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    "beleza_saude": "health_beauty", "cama_mesa_banho": "bed_bath_table",
    "esporte_lazer": "sports_leisure", "informatica_acessorios": "computers_accessories",
    "moveis_decoracao": "furniture_decor", "utilidades_domesticas": "housewares",
    "relogios_presentes": "watches_gifts", "telefonia": "telephony", "automotivo": "auto",
    "brinquedos": "toys", "cool_stuff": "cool_stuff", "perfumaria": "perfumery",
    "bebes": "baby", "eletronicos": "electronics", "papelaria": "stationery",
    "pet_shop": "pet_shop", "ferramentas_jardim": "garden_tools",
    "fashion_bolsas_e_acessorios": "fashion_bags_accessories",
}
STATES = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO", "PE", "CE", "PA", "MT", "MA"]
STATE_P = np.array([42, 13, 12, 5.5, 5, 3.7, 3.4, 2.2, 2, 2, 1.7, 1.3, 1, 0.9, 0.8])
CITIES = {"SP": ["sao paulo", "campinas", "guarulhos"], "RJ": ["rio de janeiro", "niteroi"],
          "MG": ["belo horizonte", "uberlandia"], "RS": ["porto alegre"], "PR": ["curitiba", "londrina"]}


def uid(rng):
    return uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63))).hex


def ts(t):
    return None if pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def main(out: Path, n_orders: int, seed: int):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    p_state = STATE_P / STATE_P.sum()

    def city(state):
        return rng.choice(CITIES.get(state, [f"cidade_{state.lower()}"]))

    sellers = []
    for _ in range(120):
        s = rng.choice(STATES, p=p_state)
        sellers.append({"seller_id": uid(rng), "seller_zip_code_prefix": f"{rng.integers(1000, 99999):05d}",
                        "seller_city": city(s), "seller_state": s})
    sellers = pd.DataFrame(sellers)

    cats = list(CATEGORIES)
    products = []
    for _ in range(400):
        products.append({
            "product_id": uid(rng),
            "product_category_name": rng.choice(cats) if rng.random() > 0.02 else None,
            "product_name_lenght": int(rng.integers(10, 70)),
            "product_description_lenght": int(rng.integers(50, 3000)),
            "product_photos_qty": int(rng.integers(1, 8)),
            "product_weight_g": float(rng.integers(50, 20000)),
            "product_length_cm": float(rng.integers(10, 100)),
            "product_height_cm": float(rng.integers(2, 80)),
            "product_width_cm": float(rng.integers(8, 80)),
        })
    products = pd.DataFrame(products)
    base_price = {p: float(np.round(rng.lognormal(4.2, 0.8), 2)) for p in products.product_id}

    # Real people; ~8% place more than one order.
    people = [uid(rng) for _ in range(int(n_orders * 0.93))]
    start, end = pd.Timestamp("2016-09-04"), pd.Timestamp("2018-10-17")
    span = (end - start).total_seconds()
    # Growth over time: sample purchase times skewed towards later dates.
    frac = np.sqrt(rng.random(n_orders))
    purchase = sorted(start + pd.to_timedelta(frac * span, unit="s"))

    customers, orders, items, payments, reviews = [], [], [], [], []
    for i in range(n_orders):
        person = people[i] if i < len(people) else rng.choice(people)
        state = rng.choice(STATES, p=p_state)
        cid, oid = uid(rng), uid(rng)
        customers.append({"customer_id": cid, "customer_unique_id": person,
                          "customer_zip_code_prefix": f"{rng.integers(1000, 99999):05d}",
                          "customer_city": city(state), "customer_state": state})
        p = purchase[i]
        status = rng.choice(["delivered", "shipped", "canceled", "unavailable", "invoiced", "processing"],
                            p=[0.95, 0.02, 0.01, 0.007, 0.007, 0.006])
        if p > pd.Timestamp("2018-09-01") and status == "delivered" and rng.random() < 0.5:
            status = "shipped"
        approved = p + pd.Timedelta(hours=float(rng.exponential(10)))
        carrier = approved + pd.Timedelta(days=float(rng.exponential(3))) if status in ("delivered", "shipped") else None
        est = (p + pd.Timedelta(days=int(rng.integers(15, 35)))).normalize()
        delivered = None
        if status == "delivered":
            transit = rng.gamma(3, 4) * (1.4 if state not in ("SP", "PR", "MG") else 1.0)
            delivered = carrier + pd.Timedelta(days=float(transit))
        orders.append({"order_id": oid, "customer_id": cid, "order_status": status,
                       "order_purchase_timestamp": ts(p), "order_approved_at": ts(approved),
                       "order_delivered_carrier_date": ts(carrier), "order_delivered_customer_date": ts(delivered),
                       "order_estimated_delivery_date": ts(est)})
        total = 0.0
        n_items = rng.choice([1, 2, 3, 4], p=[0.88, 0.08, 0.03, 0.01])
        seller = sellers.seller_id.iloc[rng.integers(len(sellers))]
        for k in range(1, n_items + 1):
            prod = products.product_id.iloc[rng.integers(len(products))]
            price = base_price[prod]
            freight = float(np.round(rng.uniform(7, 40), 2))
            total += price + freight
            items.append({"order_id": oid, "order_item_id": k, "product_id": prod, "seller_id": seller,
                          "shipping_limit_date": ts(p + pd.Timedelta(days=6)), "price": price, "freight_value": freight})
        ptype = rng.choice(["credit_card", "boleto", "voucher", "debit_card"], p=[0.74, 0.19, 0.055, 0.015])
        if ptype == "voucher" and rng.random() < 0.5:
            v = float(np.round(total * 0.3, 2))
            payments.append({"order_id": oid, "payment_sequential": 1, "payment_type": "voucher",
                             "payment_installments": 1, "payment_value": v})
            payments.append({"order_id": oid, "payment_sequential": 2, "payment_type": "credit_card",
                             "payment_installments": int(rng.integers(1, 10)), "payment_value": float(np.round(total - v, 2))})
        else:
            payments.append({"order_id": oid, "payment_sequential": 1, "payment_type": ptype,
                             "payment_installments": int(rng.integers(1, 11)) if ptype == "credit_card" else 1,
                             "payment_value": float(np.round(total, 2))})
        late = delivered is not None and delivered.normalize() > est
        score = int(rng.choice([1, 2, 3, 4, 5], p=[0.45, 0.15, 0.15, 0.13, 0.12] if late else [0.07, 0.03, 0.08, 0.2, 0.62]))
        if status == "canceled":
            score = int(rng.choice([1, 2, 3]))
        created = (delivered or p + pd.Timedelta(days=20)).normalize() + pd.Timedelta(days=1)
        reviews.append({"review_id": uid(rng), "order_id": oid, "review_score": score,
                        "review_comment_title": None,
                        "review_comment_message": rng.choice(["recebi bem antes do prazo", "produto otimo", None, None, "nao recebi"]),
                        "review_creation_date": ts(created), "review_answer_timestamp": ts(created + pd.Timedelta(days=2))})

    pd.DataFrame(customers).to_csv(out / "olist_customers_dataset.csv", index=False)
    pd.DataFrame(orders).to_csv(out / "olist_orders_dataset.csv", index=False)
    pd.DataFrame(items).to_csv(out / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame(payments).to_csv(out / "olist_order_payments_dataset.csv", index=False)
    pd.DataFrame(reviews).to_csv(out / "olist_order_reviews_dataset.csv", index=False)
    products.to_csv(out / "olist_products_dataset.csv", index=False)
    sellers.to_csv(out / "olist_sellers_dataset.csv", index=False)
    pd.DataFrame({"product_category_name": list(CATEGORIES),
                  "product_category_name_english": list(CATEGORIES.values())}).to_csv(
        out / "product_category_name_translation.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote fake Olist-format CSVs for {n_orders} orders to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/demo_csv", type=Path)
    p.add_argument("--orders", default=4000, type=int)
    p.add_argument("--seed", default=7, type=int)
    a = p.parse_args()
    main(a.out, a.orders, a.seed)
