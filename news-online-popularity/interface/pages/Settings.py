import pandas as pd
import streamlit as st
import sqlalchemy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression
import yaml
import os
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
import kedro.framework.project

st.set_page_config(page_title="Settings")

if st.button("Run kedro"):
    with st.spinner("Running pipeline"):
        project_path = Path.cwd() / "news-online-popularity"
        bootstrap_project(project_path)
        with KedroSession.create(project_path=project_path) as session:
            session.run(pipeline_name="ASI")
    st.success("Run finished")
    st.rerun()

with open('news-online-popularity\\conf\\base\\parameters.yml', 'r') as file:
    param_service = yaml.safe_load(file)

with open('news-online-popularity\\conf\\local\\credentials.yml', 'r') as file:
    conn_str_service = yaml.safe_load(file)


filenames = next(os.walk('news-online-popularity\\data\\06_models\\'))
mindex = filenames[1].index(param_service['model_name'])
model_name = st.radio(f"Model name ( currently {param_service['model_name']} )",filenames[1],index=mindex)


if model_name:
    version_names = next(os.walk(f'news-online-popularity\\data\\06_models\\{model_name}'))
    vindex = version_names[1].index(param_service['model_version'])
    version_name = st.radio(f"Model version ( currently {param_service['model_version']} )",version_names[1],index=vindex)


test_size = param_service['test_size']
rand_state = param_service['random_state']

connection_string = conn_str_service['my_mysql_creds']['con']

newsize = st.number_input(label='Model test frame size',min_value=.0,value=test_size,max_value=0.9)
newrand = st.number_input(label='Random state',min_value=1,value=rand_state)

newwandbprojectname = st.text_input(label='W and B project name', value=conn_str_service['wandbprojectname'])
newwandbapikey = st.text_input(label='W and B api key', type="password", value=conn_str_service['wandbapikey'])



dboptions=['google cloud','local']

dbchoice = st.radio(label='Database type',options=dboptions, index=dboptions.index(param_service['db_type']))
if dbchoice == "local":
    newconnstr = st.text_input(label='Connection string',value=connection_string)
else:
    gcloud_projectnew = st.text_input(label='gcloud project', value=conn_str_service['gclouddb_project'])
    gclouddb_loginnew = st.text_input(label='gcloud login', value=conn_str_service['gclouddb_login'])
    gclouddb_passwordnew = st.text_input(label='gcloud password', type="password", value=conn_str_service['gcloud_password'])
    gcloud_dbnamenew = st.text_input(label='gcloud database', value=conn_str_service['gcloud_dbname'])


if st.button('Save'):
    with open('news-online-popularity\\conf\\base\\parameters.yml', 'w') as file:
        param_service['test_size'] = newsize
        param_service['random_state'] = newrand
        param_service['model_name'] = model_name
        param_service['model_version'] = version_name
        param_service['db_type'] = dbchoice
        yaml.safe_dump(param_service,file)

    with open('news-online-popularity\\conf\\local\\credentials.yml', 'w') as file:
        if dbchoice == "local":
            conn_str_service['my_mysql_creds']['con'] =  newconnstr
        else:
            conn_str_service['gclouddb_project'] = gcloud_projectnew
            conn_str_service['gclouddb_login'] = gclouddb_loginnew
            conn_str_service['gcloud_dbname'] = gcloud_dbnamenew
            conn_str_service['gcloud_password'] = gclouddb_passwordnew
        conn_str_service['wandbprojectname'] = newwandbprojectname
        conn_str_service['wandbapikey'] = newwandbapikey
        yaml.safe_dump(conn_str_service,file)

    st.rerun()