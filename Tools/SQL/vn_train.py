
import sys
import os
import plotly
import json
import pandas as pd

# Navigate up two levels to the project root and add it to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import plotly.io as pio
import logging
from typing import List, Tuple
from langchain_core.tools import tool
from helpers.database import get_sqlite_connection
from helpers.config import DB_PATH_RAW, VANNA_MODEL_NAME, VANNA_API_KEY
from vanna.remote import VannaDefault
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Vanna Model
vn = VannaDefault(model=VANNA_MODEL_NAME, api_key=VANNA_API_KEY)


vn.train(documentation=
         """
The segments avaialble in the Segments table are:
         Energy, Industry & Natural Resources
        Public Services
        Private Sector
        Engagement Portfolio (GM Level)
        Financial Services
        National Security & Private Affairs
        High Potential Portfolio
        """)



                  


# 1) Schema / documentation training
vn.train(documentation="""
Table: Sectors
- SectorID (INT, PK): unique identifier for each sector.
- SectorName (TEXT): high-level category name (e.g. 'Energy, Industry & Natural Resources').

Table: Segments
- SegmentID (INT, PK): unique identifier for each segment.
- SectorID (INT, FK → Sectors.SectorID): the parent sector.
- SegmentName (TEXT): descriptive segment name (e.g. 'Public Services').

Table: Clients
- ClientID (INT, PK): unique identifier for each client.
- SegmentID (INT, FK → Segments.SegmentID): which segment this client belongs to.
- ClientName (TEXT): official client name (e.g. 'Health Ministry').

Table: Products
- ProductID (INT, PK): unique identifier for each product or service.
- Category (TEXT): high-level product category (e.g. 'Cloud', 'Voice').
- ProductName (TEXT): descriptive name (if any).

Table: Metrics
- MetricID (INT, PK): unique identifier for a metrics record.
- Date (TEXT/DATE): date of the metric (YYYY-MM-DD).
- SectorID (INT, FK → Sectors.SectorID)
- SegmentID (INT, FK → Segments.SegmentID)
- ClientID (INT, FK → Clients.ClientID)
- ProductID (INT, FK → Products.ProductID)
- Revenue (DECIMAL): revenue amount.
- Billing (DECIMAL): billing amount.
- Services (TEXT): service type (e.g. 'Internet', 'Data').
""")

# 2) Question→SQL training examples

vn.train(
    question="What are the clients available under the Public Services segment?",
    sql="""
    SELECT c.ClientName
    FROM Clients AS c
    JOIN Segments AS s
      ON c.SegmentID = s.SegmentID
    WHERE s.SegmentName = 'Public Services'
    """
)

vn.train(
    question="Which products are offered in the Private Sector segment?",
    sql="""
    SELECT DISTINCT p.ProductName
    FROM Products AS p
    JOIN Metrics AS m
      ON p.ProductID = m.ProductID
    JOIN Segments AS s
      ON m.SegmentID = s.SegmentID
    WHERE s.SegmentName = 'Private Sector'
    """
)

vn.train(
    question="How many clients does the Financial Services segment have?",
    sql="""
    SELECT COUNT(*) AS num_clients
    FROM Clients AS c
    JOIN Segments AS s
      ON c.SegmentID = s.SegmentID
    WHERE s.SegmentName = 'Financial Services'
    """
)

vn.train(
    question="List each segment and its total revenue for the year 2024.",
    sql="""
    SELECT
      s.SegmentName,
      SUM(m.Revenue) AS total_revenue
    FROM Metrics AS m
    JOIN Segments AS s
      ON m.SegmentID = s.SegmentID
    WHERE strftime('%Y', m.Date) = '2024'
    GROUP BY s.SegmentName
    """
)

vn.train(
    question="What is the total billing for Cloud services across all clients in 2023?",
    sql="""
    SELECT SUM(m.Billing) AS total_billing
    FROM Metrics AS m
    WHERE m.Services = 'Cloud'
      AND strftime('%Y', m.Date) = '2023'
    """
)

vn.train(
    question="Which client had the highest revenue for Data services on May 20, 2024?",
    sql="""
    SELECT c.ClientName
    FROM Metrics AS m
    JOIN Clients AS c
      ON m.ClientID = c.ClientID
    WHERE m.Services = 'Data'
      AND m.Date = '2024-05-20'
    ORDER BY m.Revenue DESC
    LIMIT 1
    """
)

vn.train(
    question="What is the average revenue per service type?",
    sql="""
    SELECT
      m.Services,
      AVG(m.Revenue) AS avg_revenue
    FROM Metrics AS m
    GROUP BY m.Services
    """
)

vn.train(
    question="Calculate the percentage change in total revenue for each segment between 2023 and 2024.",
    sql="""
    SELECT
      s.SegmentName,
      ((SUM(CASE WHEN strftime('%Y', m.Date)='2024' THEN m.Revenue ELSE 0 END)
        - SUM(CASE WHEN strftime('%Y', m.Date)='2023' THEN m.Revenue ELSE 0 END))
       / NULLIF(SUM(CASE WHEN strftime('%Y', m.Date)='2023' THEN m.Revenue ELSE 0 END),0)
      ) * 100 AS pct_change
    FROM Metrics AS m
    JOIN Segments AS s
      ON m.SegmentID = s.SegmentID
    GROUP BY s.SegmentName
    """
)

vn.train(
    question="List the top 3 segments by total revenue in 2024.",
    sql="""
    SELECT
      s.SegmentName,
      SUM(m.Revenue) AS total_revenue
    FROM Metrics AS m
    JOIN Segments AS s
      ON m.SegmentID = s.SegmentID
    WHERE strftime('%Y', m.Date)='2024'
    GROUP BY s.SegmentName
    ORDER BY total_revenue DESC
    LIMIT 3
    """
)

vn.train(
    question="Show monthly revenue trend for Aramco in 2023.",
    sql="""
    SELECT
      substr(m.Date,1,7) AS month,
      SUM(m.Revenue) AS revenue
    FROM Metrics AS m
    JOIN Clients AS c
      ON m.ClientID = c.ClientID
    WHERE c.ClientName = 'Aramco'
      AND substr(m.Date,1,4) = '2023'
    GROUP BY month
    ORDER BY month
    """
)

vn.train(
    question="Which segments had any billing over 50,000,000 in Q2 2024?",
    sql="""
    SELECT DISTINCT s.SegmentName
    FROM Metrics AS m
    JOIN Segments AS s
      ON m.SegmentID = s.SegmentID
    WHERE m.Billing > 50000000
      AND m.Date BETWEEN '2024-04-01' AND '2024-06-30'
    """
)

vn.train(
    question="For each sector, how many unique clients were served in 2024?",
    sql="""
    SELECT
      sec.SectorName,
      COUNT(DISTINCT m.ClientID) AS num_clients
    FROM Metrics AS m
    JOIN Sectors AS sec
      ON m.SectorID = sec.SectorID
    WHERE strftime('%Y', m.Date) = '2024'
    GROUP BY sec.SectorName
    """
)

vn.train(
    question="What is the average billing-to-revenue ratio across all services?",
    sql="""
    SELECT
      AVG(m.Billing * 1.0 / NULLIF(m.Revenue,0)) AS avg_ratio
    FROM Metrics AS m
    """
)

vn.train(
    question="Find all clients in the High Potential Portfolio segment.",
    sql="""
    SELECT c.ClientName
    FROM Clients AS c
    JOIN Segments AS s
      ON c.SegmentID = s.SegmentID
    WHERE s.SegmentName = 'High Potential Portfolio'
    """
)
