from sqlalchemy import create_engine
from kedro.framework.context import KedroContext
import pandas as pd

connection_string = 'mysql://root:123456789@localhost:3306/Projekt_ASI'


engine = create_engine(connection_string)

query = "SELECT * FROM news_data"
news_data = pd.read_sql(query, engine)

print(news_data.head())
