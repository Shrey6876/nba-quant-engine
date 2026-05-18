from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Date
from sqlalchemy.orm import relationship
from database import Base
import datetime

class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True, index=True) # NBA API Player ID
    full_name = Column(String, index=True)
    is_active = Column(Boolean, default=True)

class Game(Base):
    __tablename__ = "games"
    id = Column(String, primary_key=True, index=True) # NBA API Game ID
    game_date = Column(Date, index=True)
    home_team_id = Column(Integer)
    away_team_id = Column(Integer)

class PlayerGameLog(Base):
    __tablename__ = "player_game_logs"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), index=True)
    game_id = Column(String, ForeignKey("games.id"), index=True)
    
    minutes = Column(Float)
    points = Column(Integer)
    rebounds = Column(Integer)
    assists = Column(Integer)
    threes_made = Column(Integer)
    usage_rate = Column(Float, nullable=True) # Advanced stat
    
    # Phase 1B: Extended box score fields for USG% calculation
    field_goals_attempted = Column(Integer, nullable=True)
    free_throws_attempted = Column(Integer, nullable=True)
    turnovers = Column(Integer, nullable=True)
    plus_minus = Column(Float, nullable=True)
    fg_pct = Column(Float, nullable=True)
    ft_pct = Column(Float, nullable=True)
    
    # Matchup context
    matchup = Column(String, nullable=True)       # e.g. "BOS vs. NYK" or "BOS @ NYK"
    team_abbr = Column(String, nullable=True)      # Player's team abbreviation
    
    player = relationship("Player")
    game = relationship("Game")

class LivePropLine(Base):
    """
    Stores the live lines retrieved from an odds API (e.g. The Odds API)
    """
    __tablename__ = "live_prop_lines"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True, index=True)
    player_name = Column(String, index=True)       # Full name from the odds API
    game_date = Column(Date, index=True)
    stat_type = Column(String) # 'player_points', 'player_rebounds', 'player_assists', etc.
    line = Column(Float)
    
    over_odds = Column(Integer) # American odds
    under_odds = Column(Integer)
    
    implied_prob_over = Column(Float)
    implied_prob_under = Column(Float)
    
    # No-vig (fair) probabilities
    fair_prob_over = Column(Float, nullable=True)
    fair_prob_under = Column(Float, nullable=True)
    
    book_name = Column(String, nullable=True)      # e.g. 'draftkings', 'fanduel'
    event_id = Column(String, nullable=True)
    home_team = Column(String, nullable=True)
    away_team = Column(String, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
class ModelProjection(Base):
    """
    Stores our Monte Carlo simulation outputs.
    """
    __tablename__ = "model_projections"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), index=True)
    game_date = Column(Date, index=True)
    stat_type = Column(String)
    
    predicted_mean = Column(Float)
    predicted_variance = Column(Float)
    
    # After simulating against the current line
    simulated_prob_over = Column(Float)
    simulated_prob_under = Column(Float)
    
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class MarketInefficiency(Base):
    """
    Flags the +EV bets identified by the system.
    """
    __tablename__ = "market_inefficiencies"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), index=True)
    game_date = Column(Date, index=True)
    stat_type = Column(String)
    line = Column(Float)
    
    sportsbook_implied_prob = Column(Float)
    model_prob = Column(Float)
    
    edge_percentage = Column(Float) # Expected Value edge
    expected_value = Column(Float)
    
    flagged_for_bet = Column(Boolean, default=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class BetLog(Base):
    """
    Tracks every prediction vs actual result for CLV and P&L analysis.
    """
    __tablename__ = "bet_log"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True, index=True)
    player_name = Column(String, index=True)
    game_date = Column(Date, index=True)
    stat_type = Column(String)                     # 'PTS', 'REB', 'AST', 'PRA'
    
    # Model output
    model_projection = Column(Float)
    model_prob_over = Column(Float)                # Monte Carlo % over line
    model_prob_under = Column(Float)
    
    # Market data at time of prediction
    opening_line = Column(Float)
    opening_over_odds = Column(Integer)            # American odds
    opening_under_odds = Column(Integer)
    book_name = Column(String, nullable=True)
    
    # Closing data (filled post-game)
    closing_line = Column(Float, nullable=True)
    closing_over_odds = Column(Integer, nullable=True)
    closing_under_odds = Column(Integer, nullable=True)
    
    # Result (filled post-game)
    actual_result = Column(Float, nullable=True)
    
    # Decision
    bet_direction = Column(String)                 # 'OVER' / 'UNDER' / 'NO_BET'
    expected_value = Column(Float)                 # EV%
    kelly_fraction = Column(Float)                 # Kelly % of bankroll
    wager_amount = Column(Float, nullable=True)    # $ amount
    
    # Validation (filled post-game)
    clv = Column(Float, nullable=True)             # Closing line value
    result = Column(String, nullable=True)         # 'WIN' / 'LOSS' / 'PUSH'
    pnl = Column(Float, nullable=True)             # Profit/loss in units
    
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
