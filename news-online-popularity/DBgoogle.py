import os
import pandas as pd
from google.cloud.sql.connector import Connector, IPTypes
import pymysql
import yaml
import sqlalchemy


def connect_with_connector() -> sqlalchemy.engine.base.Engine:

    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

    connector = Connector(ip_type)

    with open('news-online-popularity\\conf\\local\\credentials.yml', 'r') as file:
        conn_str_service = yaml.safe_load(file)

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            conn_str_service['gclouddb_project'],
            "pymysql",
            user=conn_str_service['gclouddb_login'],
            password=conn_str_service['gcloud_password'],
            db=conn_str_service['gcloud_dbname'],
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        # ...
    )
    return pool

def SelectAll():
    pool = connect_with_connector()

    with pool.connect() as db_conn:
        result = db_conn.execute(sqlalchemy.text("SELECT * from newspop"))

        db_conn.commit()

        data = result.fetchall()
        columns = result.keys()
        df = pd.DataFrame(data, columns=columns)
        return df
