# Databricks notebook source
# MAGIC %pip install mlflow

# COMMAND ----------

import mlflow as ml
import pandas as pd

# COMMAND ----------

def get_exp_id_by_name(name):
    all_exp = ml.search_experiments()
    exp = [e for e in all_exp if name in e.name]
    return exp
exp = get_exp_id_by_name("pcc_04")

# COMMAND ----------

def get_runs_by_run_num(exp, num=None, finished = True):
    if type(exp) == list:
        if len(exp) > 1:
            print("Select a single experiment")
            return None
        else:
            print("here")
            exp = exp[0]
    runs = ml.search_runs([exp.experiment_id])
    if finished:
        runs = runs[runs["status"] == "FINISHED"]
    if num is not None:
        runs = runs[runs["tags.mlflow.runName"].str.contains(str(num)+"$")]
    return runs.reset_index()

runs1 = get_runs_by_run_num(exp, 1)

# COMMAND ----------

runs1

# COMMAND ----------

# exp = ml.set_experiment(experiment_id = 502851330942627)

# COMMAND ----------

def get_artifacts(runs):
    client = ml.tracking.MlflowClient()
    out = {}
    for idx,r in runs.iterrows():
        # print(r)
        rid = r["run_id"]
        uri = r["artifact_uri"]
        # select only json artifacts
        af_list = [x.path for x in client.list_artifacts(run_id = rid) if "json" in x.path]
        # instantiate dict
        if idx == 0:
            for af in af_list:
                d = ml.artifacts.load_dict(uri + "/" + af)
                af_nocode = af[4:-5]
                code = af[:3]
                out[af_nocode] = {str(code): d}
        # extend dict
        else:
            for af in af_list:
                d = ml.artifacts.load_dict(uri + "/" + af)
                af_nocode = af[4:-5]
                code = af[:3]
                out[af_nocode][code]=d
    return out

artf = get_artifacts(runs1)

# COMMAND ----------

[print(k) for k in artf.keys()]

# COMMAND ----------

pd.DataFrame(artf["tst_sv_ci_bs"]).T
pd.DataFrame(artf["trn_sv_ci_bs"]).T
pd.DataFrame(artf["all_cph_cindex"]).T
pd.DataFrame(artf["trn_cph_cindex"]).T
#
pd.DataFrame(artf["tst_sv_calib"]).T


# COMMAND ----------

# arrange into pandas 
