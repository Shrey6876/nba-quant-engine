import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

load_dotenv()

# We'll use SQLite for local prototyping if Postgres is not set up yet
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")

engine = create_engine(
    DATABASE_URL, 
    # Only for SQLite
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
