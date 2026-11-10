"""Single source of truth for the Olist database: tables, columns, keys and
human-readable documentation.

`scripts/build_db.py` uses this to create the SQLite tables, and
`text2sql/schema.py` uses it to build the schema context given to the LLM.
"""

# Each column: (name, sqlite_type, description)
TABLES = {
    "orders": {
        "csv": "olist_orders_dataset.csv",
        "description": "One row per order. Holds order status and the full timeline "
        "of the order (purchase, approval, hand-off to carrier, delivery, estimated delivery).",
        "keywords": "order orders purchase status delivered delivery late delay on time "
        "estimated shipping timeline month year date when canceled",
        "columns": [
            ("order_id", "TEXT", "Unique order id"),
            ("customer_id", "TEXT", "Customer for this order (FK customers.customer_id). Each order has its own customer_id."),
            ("order_status", "TEXT", "Order status, e.g. delivered, shipped, canceled"),
            ("order_purchase_timestamp", "TEXT", "When the order was placed, 'YYYY-MM-DD HH:MM:SS'"),
            ("order_approved_at", "TEXT", "When payment was approved"),
            ("order_delivered_carrier_date", "TEXT", "When the order was handed to the logistics carrier"),
            ("order_delivered_customer_date", "TEXT", "When the customer actually received the order (NULL if not delivered)"),
            ("order_estimated_delivery_date", "TEXT", "Delivery date promised to the customer at purchase"),
        ],
        "primary_key": ["order_id"],
        "foreign_keys": [("customer_id", "customers", "customer_id")],
    },
    "customers": {
        "csv": "olist_customers_dataset.csv",
        "description": "Customer location. customer_id is per-order; customer_unique_id "
        "identifies the real person across orders (use it for repeat customers / retention).",
        "keywords": "customer customers buyer buyers state city location region repeat "
        "unique people users retention cohort",
        "columns": [
            ("customer_id", "TEXT", "Key used by orders.customer_id (one per order)"),
            ("customer_unique_id", "TEXT", "Real person id; same person can have many customer_id values"),
            ("customer_zip_code_prefix", "TEXT", "First 5 digits of zip code"),
            ("customer_city", "TEXT", "City name, lowercase"),
            ("customer_state", "TEXT", "Brazilian state, 2-letter code e.g. SP, RJ, MG"),
        ],
        "primary_key": ["customer_id"],
        "foreign_keys": [],
    },
    "order_items": {
        "csv": "olist_order_items_dataset.csv",
        "description": "One row per item in an order. Use price for revenue / sales and "
        "freight_value for shipping cost. Links orders to products and sellers.",
        "keywords": "item items revenue sales gmv price freight shipping cost sold "
        "quantity units basket product seller sell",
        "columns": [
            ("order_id", "TEXT", "FK orders.order_id"),
            ("order_item_id", "INTEGER", "Item number within the order (1, 2, 3...)"),
            ("product_id", "TEXT", "FK products.product_id"),
            ("seller_id", "TEXT", "FK sellers.seller_id"),
            ("shipping_limit_date", "TEXT", "Deadline for seller to hand item to carrier"),
            ("price", "REAL", "Item price in BRL (excludes freight)"),
            ("freight_value", "REAL", "Freight (shipping) cost charged for this item in BRL"),
        ],
        "primary_key": ["order_id", "order_item_id"],
        "foreign_keys": [
            ("order_id", "orders", "order_id"),
            ("product_id", "products", "product_id"),
            ("seller_id", "sellers", "seller_id"),
        ],
    },
    "payments": {
        "csv": "olist_order_payments_dataset.csv",
        "description": "Payments for orders. An order can be paid with several payment "
        "methods (one row each). payment_value is what the customer paid.",
        "keywords": "payment payments paid pay method type credit card boleto voucher "
        "debit installments amount spend value",
        "columns": [
            ("order_id", "TEXT", "FK orders.order_id"),
            ("payment_sequential", "INTEGER", "Sequence number when an order uses several payments"),
            ("payment_type", "TEXT", "Method: credit_card, boleto, voucher, debit_card, not_defined"),
            ("payment_installments", "INTEGER", "Number of installments chosen"),
            ("payment_value", "REAL", "Amount paid in BRL"),
        ],
        "primary_key": ["order_id", "payment_sequential"],
        "foreign_keys": [("order_id", "orders", "order_id")],
    },
    "reviews": {
        "csv": "olist_order_reviews_dataset.csv",
        "description": "Customer reviews of orders, score 1 to 5. review_id is not unique "
        "and a few orders have several reviews.",
        "keywords": "review reviews rating ratings score stars satisfaction feedback "
        "comment complaint happy unhappy",
        "columns": [
            ("review_id", "TEXT", "Review id (not unique)"),
            ("order_id", "TEXT", "FK orders.order_id"),
            ("review_score", "INTEGER", "Rating from 1 (worst) to 5 (best)"),
            ("review_comment_title", "TEXT", "Optional title (Portuguese)"),
            ("review_comment_message", "TEXT", "Optional comment text (Portuguese)"),
            ("review_creation_date", "TEXT", "When the review survey was sent"),
            ("review_answer_timestamp", "TEXT", "When the customer answered"),
        ],
        "primary_key": [],
        "foreign_keys": [("order_id", "orders", "order_id")],
    },
    "products": {
        "csv": "olist_products_dataset.csv",
        "description": "Product catalogue. Category names are in Portuguese; join "
        "category_translation to get English names.",
        "keywords": "product products category categories catalogue weight size dimensions "
        "photos description",
        "columns": [
            ("product_id", "TEXT", "Unique product id"),
            ("product_category_name", "TEXT", "Category in Portuguese (FK category_translation)"),
            ("product_name_length", "INTEGER", "Characters in product name"),
            ("product_description_length", "INTEGER", "Characters in product description"),
            ("product_photos_qty", "INTEGER", "Number of product photos"),
            ("product_weight_g", "REAL", "Weight in grams"),
            ("product_length_cm", "REAL", "Length in cm"),
            ("product_height_cm", "REAL", "Height in cm"),
            ("product_width_cm", "REAL", "Width in cm"),
        ],
        "primary_key": ["product_id"],
        "foreign_keys": [("product_category_name", "category_translation", "product_category_name")],
    },
    "sellers": {
        "csv": "olist_sellers_dataset.csv",
        "description": "Sellers (merchants) on the marketplace and their location.",
        "keywords": "seller sellers merchant merchants vendor shop store state city location",
        "columns": [
            ("seller_id", "TEXT", "Unique seller id"),
            ("seller_zip_code_prefix", "TEXT", "First 5 digits of zip code"),
            ("seller_city", "TEXT", "City name, lowercase"),
            ("seller_state", "TEXT", "Brazilian state, 2-letter code"),
        ],
        "primary_key": ["seller_id"],
        "foreign_keys": [],
    },
    "category_translation": {
        "csv": "product_category_name_translation.csv",
        "description": "Maps Portuguese product category names to English names.",
        "keywords": "category categories english translation name",
        "columns": [
            ("product_category_name", "TEXT", "Category in Portuguese"),
            ("product_category_name_english", "TEXT", "Category in English, e.g. health_beauty"),
        ],
        "primary_key": ["product_category_name"],
        "foreign_keys": [],
    },
}

# Original Olist CSV headers that we rename (the dataset misspells "length").
COLUMN_RENAMES = {
    "products": {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    }
}

# Business definitions the model should follow. Kept short on purpose.
GLOSSARY = [
    "Revenue / sales = SUM(order_items.price) unless the question says otherwise. "
    "GMV = SUM(price + freight_value).",
    "Use order_purchase_timestamp for 'when an order happened'. Month: strftime('%Y-%m', ...).",
    "Late delivery: order_delivered_customer_date > order_estimated_delivery_date "
    "(compare with date(); only delivered orders have a delivery date).",
    "Delivery time in days: julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp).",
    "Repeat customers / number of real customers: use customers.customer_unique_id, not customer_id.",
    "English category names: join products -> category_translation.",
    "Data covers Sept 2016 to Oct 2018; 2016 and late 2018 are very sparse.",
]
