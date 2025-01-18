import os
import pandas as pd
from google.cloud.sql.connector import Connector, IPTypes
import pymysql

import sqlalchemy


def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of MySQL.

    Uses the Cloud SQL Python Connector package.
    """
    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.

    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

    connector = Connector(ip_type)

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            "balmy-component-447620-f8:europe-central2:asiproject4345",
            "pymysql",
            user="admin",
            password="ze0d`z]RA:.D\y3u",
            db="asi_project",
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        # ...
    )
    return pool


#pool = connect_with_connector()
#
#with pool.connect() as db_conn:
#    result = db_conn.execute(sqlalchemy.text("SELECT * from newspop"))
#
#    db_conn.commit()
#
#    data = result.fetchall()
#    columns = result.keys()
#    df = pd.DataFrame(data, columns=columns)
#    print(df.head())
