# Databricks notebook source
# MAGIC %pip install mlflow

# COMMAND ----------

import mlflow as ml

# COMMAND ----------

exp = ml.get_experiment(502851330942627)
exp

# COMMAND ----------

runs = ml.search_runs([502851330942627])
runs = runs[runs["status"] == "FINISHED"]

# COMMAND ----------

runs

# COMMAND ----------

exp = ml.set_experiment(experiment_id = 502851330942627)

# COMMAND ----------

    d=ml.artifacts.load_dict("dbfs:/databricks/mlflow-tracking/3486404220763471/6040689ec5254e28a18ed869683fe204/artifacts/RSP_all_pdp_sample_summary.json")
    d

# COMMAND ----------

# gets the list of artifacts for each run
client = ml.tracking.MlflowClient()
client.list_artifacts(run_id = "57d3e4c1524b4a368ea2c395aba2ef54").path


# COMMAND ----------

runs[["run_id", "artifact_uri"]]

# COMMAND ----------

import pandas as pd

# COMMAND ----------


for idx,r in runs.iterrows():
    # print(r)
    rid = r["run_id"]
    uri = r["artifact_uri"]
    # print(rid,uri)
    af_list = [x.path for x in client.list_artifacts(run_id = rid) if "json" in x.path]
    for af in af_list:
        if "all_pdp_sample_summary" in af:
            print(uri + af)
            d = ml.artifacts.load_dict(uri + "/" + af)
            q= pd.DataFrame(d).T
            
    # print(af_list)
    if idx >3:
        break

    # d

# COMMAND ----------

q["var"] = q.index
q["cond"] = "MUS"
q["run"] = "RUNName"
q.reset_index(drop=TRUE)

# COMMAND ----------

try:
    print(x)
except:
    print("yes")

# COMMAND ----------

spark
