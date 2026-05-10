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
    
    player = relationship("Player")
    game = relationship("Game")

class LivePropLine(Base):
    """
    Stores the live lines retrieved from an odds API (e.g. The Odds API)
    """
    __tablename__ = "live_prop_lines"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), index=True)
    game_date = Column(Date, index=True)
    stat_type = Column(String) # 'points', 'rebounds', 'assists', 'pts+rebs+asts'
    line = Column(Float)
    
    over_odds = Column(Integer) # American odds
    under_odds = Column(Integer)
    
    implied_prob_over = Column(Float)
    implied_prob_under = Column(Float)
    
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
