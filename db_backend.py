from sqlalchemy import create_engine, text
import pandas as pd


def get_engine(db_url: str):
    return create_engine(db_url, pool_pre_ping=True)

def ensure_schema(engine):
    with engine.begin() as con:
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS recent_sales (
            id TEXT PRIMARY KEY,
            player_slug TEXT NOT NULL,
            rarity TEXT,
            card_slug TEXT,
            in_season_eligible BOOLEAN,
            price_eur DOUBLE PRECISION,
            buyer_slug TEXT,
            seller_slug TEXT,
            date TIMESTAMPTZ
        );
        """))
        con.execute(text("CREATE INDEX IF NOT EXISTS ix_recent_sales_player_date ON recent_sales (player_slug, date DESC);"))

        # make sure no null sellers remain, then enforce not null
        try:
            con.execute(text("DELETE FROM recent_sales WHERE seller_slug IS NULL"))
            con.execute(text("ALTER TABLE recent_sales ALTER COLUMN seller_slug SET NOT NULL"))
        except Exception:
            # column may already be not null or delete not needed
            pass


def max_date_for_player(engine, slug: str):
    with engine.begin() as con:
        row = con.execute(text("SELECT MAX(date) FROM recent_sales WHERE player_slug = :slug"), {"slug": slug}).fetchone()
        return row[0] if row else None

def upsert_sales(engine, rows: list[dict]):
    # keep only rows that have a seller
    rows = [r for r in rows if r.get("seller_slug")]
    if not rows:
        return
    with engine.begin() as con:
        con.execute(text("""
        INSERT INTO recent_sales (
            id, player_slug, rarity, card_slug, in_season_eligible,
            price_eur, buyer_slug, seller_slug, date
        )
        VALUES (
            UNNEST(:id), UNNEST(:player_slug), UNNEST(:rarity), UNNEST(:card_slug), UNNEST(:in_season_eligible),
            UNNEST(:price_eur), UNNEST(:buyer_slug), UNNEST(:seller_slug), UNNEST(:date)
        )
        ON CONFLICT (id) DO NOTHING;
        """), {
            "id":        [r.get("id") for r in rows],
            "player_slug":[r.get("player_slug") for r in rows],
            "rarity":    [r.get("rarity") for r in rows],
            "card_slug": [r.get("card_slug") for r in rows],
            "in_season_eligible":[r.get("in_season_eligible") for r in rows],
            "price_eur": [r.get("price_eur") for r in rows],
            "buyer_slug":[r.get("buyer_slug") for r in rows],
            "seller_slug":[r.get("seller_slug") for r in rows],
            "date":      [pd.to_datetime(r.get("date"), utc=True) for r in rows],
        })

def load_sales_for_owned(engine, owned_slugs: list[str], per_player_limit: int) -> pd.DataFrame:
    if not owned_slugs:
        return pd.DataFrame(columns=["player_slug","rarity","card_slug","in_season_eligible","price_eur","buyer_slug","seller_slug","date","id"])
    placeholders = ",".join([f":s{i}" for i in range(len(owned_slugs))])
    params = {f"s{i}": s for i, s in enumerate(owned_slugs)}
    sql = f"""
    WITH ranked AS (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY player_slug ORDER BY date DESC) AS rn
      FROM recent_sales
      WHERE player_slug IN ({placeholders})
    )
    SELECT id, player_slug, rarity, card_slug, in_season_eligible,
           price_eur, buyer_slug, seller_slug, date
    FROM ranked
    WHERE rn <= :limit
    """
    params["limit"] = per_player_limit
    with engine.begin() as con:
        df = pd.read_sql(text(sql), con, params=params)
    return df