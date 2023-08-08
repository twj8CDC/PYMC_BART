# Databricks notebook source
# %sh pwd

# COMMAND ----------

# Databricks notebook source
import pyspark

dbutils.widgets.text('experiment_id',defaultValue='')
experiment_id=dbutils.widgets.get('experiment_id')

dbutils.widgets.text("run_name", defaultValue="test")
model_name = dbutils.widgets.get("run_name")

dbutils.widgets.text("alpha", defaultValue="3")
ALPHA = int(dbutils.widgets.get("alpha"))

dbutils.widgets.text("alpha_f", defaultValue="None")
ALPHA_F = dbutils.widgets.get("alpha_f")
if ALPHA_F == "None":
    ALPHA_F = None

dbutils.widgets.text("lambda", defaultValue="np.exp(2 + 0.4*(x_mat[:,0] + x_mat[:,1]))")
LAMBDA = dbutils.widgets.get('lambda')

dbutils.widgets.text("n", defaultValue="100")
N = int(dbutils.widgets.get("n"))

dbutils.widgets.text("x_vars", defaultValue="2")
X_VARS = int(dbutils.widgets.get("x_vars"))

dbutils.widgets.text("cens_ind", defaultValue="False")
CENS_IND = dbutils.widgets.get("cens_ind")
if CENS_IND == "False":
    CENS_IND = False
else: 
    CENS_IND = True

dbutils.widgets.text("cens_scale", defaultValue="60")
CENS_SCALE = int(dbutils.widgets.get("cens_scale"))

dbutils.jobs.taskValues.set('experiment_id',experiment_id)
dbutils.jobs.taskValues.set("run_name", model_name)
dbutils.jobs.taskValues.set("alpha", ALPHA)
dbutils.jobs.taskValues.set("alpha_f", ALPHA_F)
dbutils.jobs.taskValues.set("lambda", LAMBDA)
dbutils.jobs.taskValues.set("n", N)
dbutils.jobs.taskValues.set("x_vars", X_VARS)
dbutils.jobs.taskValues.set("cens_ind", CENS_IND)
dbutils.jobs.taskValues.set("cens_scale", CENS_SCALE)
