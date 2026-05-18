#!/usr/bin/env python3
"""
ingest_nba_api.py
─────────────────
Pulls NBA game logs from nba_api into the local SQLite database.

Modes:
  Default (--incremental): Only fetches games since the last ingested date.
    Fast (~10–30s). Used in daily CI runs.
  --full: Re-ingests all historical seasons. Slow (~5–10 min). Use for
    initial setup or database repair.

Usage:
    python ingest_nba_api.py              # incremental (CI default)
    python ingest_nba_api.py --full       # full historical backfill
"""

import sys
import time
import datetime
from nba_api.stats.endpoints import leaguegamelog
from sqlalchemy import text
from database import SessionLocal
import schema
from sqlalchemy.orm import Session


def get_current_season() -> str:
    """Auto-detect the current NBA season string (e.g. '2025-26')."""
    today = datetime.date.today()
    year = today.year if today.month >= 10 else today.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def get_last_ingested_date() -> datetime.date | None:
    """Returns the most recent game_date in the database, or None if empty."""
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT MAX(game_date) as md FROM games")).fetchone()
        if result and result[0]:
            return datetime.date.fromisoformat(str(result[0])[:10])
    except Exception:
        pass
    finally:
        db.close()
    return None


def fetch_season_type_for_date(target_date: datetime.date) -> list[str]:
    """Return the appropriate season type(s) to query based on the date."""
    month = target_date.month
    # Playoffs: April through June
    if 4 <= month <= 6:
        return ["Playoffs", "Regular Season"]  # try both during transition
    return ["Regular Season"]


def fetch_and_store(seasons: list[str], season_types: list[str], since_date: datetime.date | None = None):
    """
    Pull game logs for the given seasons/types and store new records.
    If since_date is set, skips records older than that date (incremental mode).
    """
    for season in seasons:
        for season_type in season_types:
            label = f"{season} ({season_type})"
            print(f"\n📥 Fetching: {label}" + (f" [since {since_date}]" if since_date else " [FULL]"))
            try:
                logs = leaguegamelog.LeagueGameLog(
                    season=season,
                    player_or_team_abbreviation='P',
                    season_type_all_star=season_type
                )
                df = logs.get_data_frames()[0]
                if df.empty:
                    print(f"   No data returned for {label}. Skipping.")
                    time.sleep(1)
                    continue

                # Filter to only new games in incremental mode
                if since_date:
                    df['GAME_DATE'] = df['GAME_DATE'].astype(str)
                    df = df[df['GAME_DATE'] > since_date.isoformat()]
                    if df.empty:
                        print(f"   ✅ No new games since {since_date} for {label}.")
                        continue
                    print(f"   📊 {len(df)} new game log rows to ingest")
                else:
                    print(f"   📊 {len(df)} total game log rows")

                db: Session = SessionLocal()

                # Upsert players
                unique_players = df[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
                for _, row in unique_players.iterrows():
                    player_id = int(row['PLAYER_ID'])
                    if not db.query(schema.Player).filter(schema.Player.id == player_id).first():
                        db.add(schema.Player(id=player_id, full_name=row['PLAYER_NAME'], is_active=True))

                # Upsert games
                unique_games = df[['GAME_ID', 'GAME_DATE']].drop_duplicates()
                for _, row in unique_games.iterrows():
                    game_id = str(row['GAME_ID'])
                    if not db.query(schema.Game).filter(schema.Game.id == game_id).first():
                        g_date = datetime.date.fromisoformat(str(row['GAME_DATE'])[:10])
                        db.add(schema.Game(id=game_id, game_date=g_date))

                db.commit()

                # Upsert player game logs
                new_count = 0
                updated_count = 0
                for _, row in df.iterrows():
                    player_id = int(row['PLAYER_ID'])
                    game_id = str(row['GAME_ID'])

                    existing = db.query(schema.PlayerGameLog).filter_by(
                        player_id=player_id, game_id=game_id
                    ).first()

                    def safe_int(val):
                        try:
                            return int(val) if val is not None and str(val) != 'nan' else None
                        except Exception:
                            return None

                    def safe_float(val):
                        try:
                            return float(val) if val is not None and str(val) != 'nan' else None
                        except Exception:
                            return None

                    if not existing:
                        log = schema.PlayerGameLog(
                            player_id=player_id,
                            game_id=game_id,
                            minutes=safe_float(row.get('MIN', 0)) or 0.0,
                            points=safe_int(row.get('PTS', 0)) or 0,
                            rebounds=safe_int(row.get('REB', 0)) or 0,
                            assists=safe_int(row.get('AST', 0)) or 0,
                            threes_made=safe_int(row.get('FG3M', 0)) or 0,
                            field_goals_attempted=safe_int(row.get('FGA')),
                            free_throws_attempted=safe_int(row.get('FTA')),
                            turnovers=safe_int(row.get('TOV')),
                            plus_minus=safe_float(row.get('PLUS_MINUS')),
                            fg_pct=safe_float(row.get('FG_PCT')),
                            ft_pct=safe_float(row.get('FT_PCT')),
                            matchup=str(row['MATCHUP']) if row.get('MATCHUP') else None,
                            team_abbr=str(row['TEAM_ABBREVIATION']) if row.get('TEAM_ABBREVIATION') else None,
                        )
                        db.add(log)
                        new_count += 1
                    else:
                        # Backfill expanded fields if missing
                        if existing.field_goals_attempted is None:
                            existing.field_goals_attempted = safe_int(row.get('FGA'))
                            existing.free_throws_attempted = safe_int(row.get('FTA'))
                            existing.turnovers = safe_int(row.get('TOV'))
                            existing.plus_minus = safe_float(row.get('PLUS_MINUS'))
                            existing.fg_pct = safe_float(row.get('FG_PCT'))
                            existing.ft_pct = safe_float(row.get('FT_PCT'))
                            existing.matchup = str(row['MATCHUP']) if row.get('MATCHUP') else None
                            existing.team_abbr = str(row['TEAM_ABBREVIATION']) if row.get('TEAM_ABBREVIATION') else None
                            updated_count += 1

                db.commit()
                db.close()
                print(f"   ✅ ETL done: {new_count} new, {updated_count} backfilled")

                time.sleep(1)  # Rate limit courtesy pause

            except Exception as e:
                print(f"   ⚠️  Error for {label}: {e}")


def run_incremental():
    """Incremental mode: only fetch games newer than what's already in the DB."""
    print("=" * 60)
    print("  📈 NBA API INCREMENTAL INGESTION")
    print("=" * 60)

    last_date = get_last_ingested_date()
    current_season = get_current_season()
    today = datetime.date.today()

    if last_date is None:
        print("  ⚠️  No existing data found. Running full ingestion instead.")
        run_full()
        return

    days_stale = (today - last_date).days
    print(f"\n  📅 Current season: {current_season}")
    print(f"  🗓️  DB last updated: {last_date} ({days_stale} day(s) ago)")

    if days_stale == 0:
        print("  ✅ DB is already up to date. No ingestion needed.")
        return

    # Fetch only from the current season since last_date
    season_types = fetch_season_type_for_date(today)
    print(f"  🔍 Fetching season type(s): {season_types}")

    # Use last_date - 1 day to catch any late-reporting games
    since = last_date - datetime.timedelta(days=1)
    fetch_and_store(
        seasons=[current_season],
        season_types=season_types,
        since_date=since
    )

    new_last_date = get_last_ingested_date()
    print(f"\n  ✅ Incremental ingestion complete. DB now through: {new_last_date}")


def run_full():
    """Full mode: re-ingest all historical seasons."""
    print("=" * 60)
    print("  📦 NBA API FULL HISTORICAL INGESTION")
    print("=" * 60)

    today = datetime.date.today()
    # Build dynamic season list: 3 seasons ending with current
    current_season = get_current_season()
    season_year = int(current_season.split("-")[0])
    seasons = [
        f"{season_year - 2}-{str(season_year - 1)[-2:]}",
        f"{season_year - 1}-{str(season_year)[-2:]}",
        current_season,
    ]
    season_types = ["Regular Season", "Playoffs"]
    print(f"\n  Seasons: {seasons}")
    fetch_and_store(seasons=seasons, season_types=season_types, since_date=None)
    print(f"\n  ✅ Full ingestion complete.")


if __name__ == "__main__":
    if "--full" in sys.argv:
        run_full()
    else:
        run_incremental()
