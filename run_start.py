# Databricks notebook source
# Databricks notebook source
# Databricks notebook source
import mlflow, os
# proforma, lists modules, maybe sets dictionary of available models as taskValue? 
experiment_id=dbutils.jobs.taskValues.get("ml-head","experiment_id")
# model_name=dbutils.widgets.get("model_name")
model_name = dbutils.jobs.taskValues.get("ml-head", "run_name")

with mlflow.start_run(experiment_id=experiment_id,run_name=f"{model_name}-") as run:
   run_id=run.info.run_id
   mlflow.log_param('run_id',run_id)
   mlflow.log_param('model_name',model_name)

os.environ['MLFLOW_RUN_ID']=run_id
dbutils.jobs.taskValues.set("run_id",run_id)
# dbutils.jobs.taskValues.set("experiment_id",experiment_id)
