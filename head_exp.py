# Databricks notebook source
# %sh pwd

# COMMAND ----------

# Databricks notebook source
import pyspark

dbutils.widgets.text('experiment_id',defaultValue='')
experiment_id=dbutils.widgets.get('experiment_id')
dbutils.jobs.taskValues.set('experiment_id',experiment_id)

dbutils.widgets.text("model_name", defaultValue="test")
model_name = dbutils.widgets.get("model_name")
dbutils.jobs.taskValues.set("model_name", model_name)
