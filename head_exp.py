# Databricks notebook source
# %sh pwd

# COMMAND ----------

# Databricks notebook source
import pyspark

dbutils.widgets.text('experiment_id',defaultValue='')
experiment_id=dbutils.widgets.get('experiment_id')
dbutils.jobs.taskValues.set('experiment_id',experiment_id)

dbutils.widgets.text("run_name", defaultValue="test")
model_name = dbutils.widgets.get("run_name")
dbutils.jobs.taskValues.set("run_name", model_name)




dbutils.jobs.taskValues.set("alpha", 3)
dbutils.jobs.taskValues.set("alpha_f", None)
dbutils.jobs.taskValues.set("lambda", "np.exp(2 + 0.4*(x_mat[:,0] + x_mat[:,1]))")
dbutils.jobs.taskValues.set("n", 100)
dbutils.jobs.taskValues.set("x_vars", 2)
dbutils.jobs.taskValues.set("cens_ind", False)
dbutils.jobs.taskValues.set("cens_scale", 60)
