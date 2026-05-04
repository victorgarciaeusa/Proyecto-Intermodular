CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE staging.sales_raw (
    load_id        SERIAL,
    load_date      TIMESTAMP DEFAULT NOW(),
    source         VARCHAR(100),
    sku            VARCHAR(50),
    sale_date      DATE,
    quantity       NUMERIC,
    price          NUMERIC,
    promotion_flag BOOLEAN,
    store_id       VARCHAR(50),
    raw_json       JSONB
);

CREATE TABLE analytics.sales_curated (
    sku            VARCHAR(50)  NOT NULL,
    sale_date      DATE         NOT NULL,
    quantity       NUMERIC      NOT NULL,
    price          NUMERIC,
    promotion_flag BOOLEAN      DEFAULT FALSE,
    store_id       VARCHAR(50),
    PRIMARY KEY (sku, sale_date)
);

CREATE TABLE analytics.features (
    sku                VARCHAR(50)  NOT NULL,
    sale_date          DATE         NOT NULL,
    quantity           NUMERIC      NOT NULL,
    quantity_lag_1     NUMERIC,
    quantity_lag_7     NUMERIC,
    ma_7               NUMERIC,
    std_7              NUMERIC,
    day_of_week        INTEGER,
    month              INTEGER,
    is_holiday         BOOLEAN      DEFAULT FALSE,
    season             VARCHAR(10),
    promotion_flag     BOOLEAN      DEFAULT FALSE,
    PRIMARY KEY (sku, sale_date)
);

CREATE TABLE analytics.predictions (
    prediction_id   SERIAL,
    sku             VARCHAR(50)  NOT NULL,
    prediction_date DATE         NOT NULL,
    horizon_week    INTEGER      NOT NULL,
    predicted_qty   NUMERIC      NOT NULL,
    ci_lower        NUMERIC,
    ci_upper        NUMERIC,
    model_name      VARCHAR(50),
    run_timestamp   TIMESTAMP    DEFAULT NOW(),
    PRIMARY KEY (sku, prediction_date, horizon_week, model_name)
);