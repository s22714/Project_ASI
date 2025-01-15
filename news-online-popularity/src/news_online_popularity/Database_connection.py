from sqlalchemy import create_engine
from kedro.framework.context import KedroContext
import pandas as pd
import sqlalchemy

connection_string = 'mysql://root:qwerty@localhost:3306/asi_project'

engine = sqlalchemy.create_engine(connection_string)

query = "SELECT * FROM newspop"
news_data = pd.read_sql(query, engine)

print(news_data.head())
