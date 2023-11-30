# Databricks notebook source
# MAGIC %pip install scikit-survival pymc pymc_experimental matplotlib colorcet pymc_bart lifelines mlflow

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.window as W
from pyspark.storagelevel import StorageLevel


import sksurv as sks
from sksurv import nonparametric
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import importlib
from pathlib import Path

from surv_bart_pkg import surv_bart as bmb
from surv_bart_pkg import utillities as ut

import mlflow as ml
import lifelines as ll
import time

importlib.reload(bmb)
# Set Seed
np_seed = int(np.ceil(time.time()))

# COMMAND ----------

# MAGIC %md
# MAGIC # Globals

# COMMAND ----------

# GLOBALS
CODE = "NVS"
RUN_NAME = f"{CODE}_run_01"
EXP_ID = 502851330942627
TIME_SCALE = 91.5
BALANCE = True
SAMPLE_TRN = 10_000
SAMPLE_TST = 20_000

TREES = 40
SPLIT_RULES =  [
    "pmb.ContinuousSplitRule()", # time
    "pmb.OneHotSplitRule", # ccsr_ind_p2
    "pmb.OneHotSplitRule",  # ccsr_ind_p1
    "pmb.ContinuousSplitRule()", # p1_sum
    "pmb.ContinuousSplitRule()", # p2_sum
    "pmb.ContinuousSplitRule()", #p3_sum
    "pmb.OneHotSplitRule", # covid_icd_lab
    "pmb.ContinuousSplitRule()", #init_date_ym
    "pmb.OneHotSplitRule()", # pat_type
    "pmb.OneHotSplitRule()", # ms_drg
    "pmb.OneHotSplitRule()", #std_payor
    "pmb.ContinuousSplitRule()", #los
    "pmb.ContinuousSplitRule()", # age
    "pmb.OneHotSplitRule()", #race
    "pmb.OneHotSplitRule()", #hispanic_ind
    "pmb.OneHotSplitRule()" # i_o_ind
    ]
DRAWS = 100
TUNE = 200
CORES = 8
CHAINS = 8




# COMMAND ----------

dbutils.widgets.text("seed", defaultValue = str(np_seed))
np_seed1 = dbutils.widgets.get("seed")
if np_seed1 != "na":
    np_seed = int(np_seed)

dbutils.widgets.text("code", defaultValue=str(CODE))
CODE1 = dbutils.widgets.get("code")
if CODE1 != "na":
    CODE = CODE1

dbutils.widgets.text("run_name", defaultValue=str(RUN_NAME))
RUN_NAME1 = dbutils.widgets.get("run_name")
if RUN_NAME1 != "na":
    RUN_NAME = RUN_NAME1

dbutils.widgets.text("exp_id", defaultValue=str(EXP_ID))
EXP_ID1 = dbutils.widgets.get("exp_id")
if EXP_ID1 != "na":
    EXP_ID = int(EXP_ID1)

dbutils.widgets.text("time_scale", defaultValue=str(TIME_SCALE))
TIME_SCALE1 = dbutils.widgets.get("time_scale")
if TIME_SCALE1 != "na":
    TIME_SCALE = float(TIME_SCALE1)

dbutils.widgets.text("balance", defaultValue=str(BALANCE))
BALANCE1 = dbutils.widgets.get("balance")
if BALANCE1 != "na":
    BALANCE = bool(BALANCE1)

dbutils.widgets.text("sample_trn", defaultValue=str(SAMPLE_TRN))
SAMPLE_TRN1 = dbutils.widgets.get("sample_trn")
if SAMPLE_TRN1 != "na":
    SAMPLE_TRN = int(SAMPLE_TRN1)

dbutils.widgets.text("sample_tst", defaultValue=str(SAMPLE_TST))
SAMPLE_TST1 = dbutils.widgets.get("sample_tst")
if SAMPLE_TST1 != "na":
    SAMPLE_TST = int(SAMPLE_TST1)

dbutils.widgets.text("trees", defaultValue=str(TREES))
TREES1 = dbutils.widgets.get("trees")
if TREES1 != "na":
    TREES = int(TREES)

dbutils.widgets.text("split_rules", defaultValue=str(SPLIT_RULES))
SPLIT_RULES1 = dbutils.widgets.get("split_rules")
if SPLIT_RULES1 != "na":
    SPLIT_RULES =  eval(SPLIT_RULES1)

dbutils.widgets.text("draws", defaultValue=str(DRAWS))
DRAWS1 = dbutils.widgets.get("draws")
if DRAWS1 != "na":
    DRAWS = int(DRAWS)

dbutils.widgets.text("tune", defaultValue=str(TUNE))
TUNE1 = dbutils.widgets.get("tune")
if TUNE1 != "na":
    TUNE = int(TUNE1)

dbutils.widgets.text("cores", defaultValue=str(CORES))
CORES1 = dbutils.widgets.get("cores")
if CORES1 != "na":
    CORES = int(CORES1)

dbutils.widgets.text("chains", defaultValue=str(CHAINS))
CHAINS1 = dbutils.widgets.get("chains")
if CHAINS1 != "na":
    CHAINS = int(CHAINS1)

# COMMAND ----------

# import sys
# sys.exit(0)

# COMMAND ----------

# dbutils.widgets.removeAll()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Set MLFLOW EXP

# COMMAND ----------

# set experiment
ml_exp = ml.set_experiment(experiment_id=EXP_ID)
ml_exp

# COMMAND ----------

ml_run = ml.start_run(run_name=RUN_NAME)

# COMMAND ----------

# log global params
global_dict = {
    "EXP_ID":EXP_ID,
    "RUN_NAME":RUN_NAME,
    "CODE":CODE,
    "TIME_SCALE":TIME_SCALE,
    "BALANCE": BALANCE,
    "SAMPLE_TRN":SAMPLE_TRN,
    "SAMPLE_TST":SAMPLE_TST,
    "SEED":np_seed
 }

model_dict_main = {
    "TREES":TREES,
    "SPLIT_RULES":SPLIT_RULES,
    "DRAWS":DRAWS,
    "TUNE":TUNE,
    "CORES":CORES,
    "CHAINS":CHAINS
}

ml.log_dict(global_dict, f"{CODE}_global_dict.json")
ml.log_dict(model_dict_main, f"{CODE}_model_dict.json")
print(global_dict)
print(model_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

# ccsr_l = spark.table("cdh_premier_exploratory.twj8_pat_ccsr_long_f_06")
# Load ccsr 
ccsr_s = spark.table("cdh_premier_exploratory.twj8_pat_ccsr_short_f_06")
cov = spark.table("cdh_premier_exploratory.twj8_pat_covariates_f_07")
# codes = [x[0] for x in ccsr_s.select("code_sm").distinct().collect()]
# print(codes)

# Get the datasets data
cc1, cc_name = bmb.get_sk_sp(ccsr_s, cov, CODE)

# COMMAND ----------

# get the dataset for specific code
# def get_sk_sp(df, cov, code):
#     df_code = df.filter(F.col("code_sm") == code)
#     mg = (
#         df_code
#         .join(cov, on="medrec_key", how="left")
#         .drop("medrec_key", "code_sm")
#         .withColumn("ccsr_tt_p3", F.col("ccsr_tt_p3")-30) # adjust date by -30
#     )
#     names = mg.columns
#     out = [[x[i] for i in range(len(names))] for x in mg.collect() ]
#     # out = [[x[0], x[1], x[2]] for x in mg]
#     return np.array(out, dtype="int"), names

# COMMAND ----------

# MAGIC %md
# MAGIC # Prep and Cut Datasets

# COMMAND ----------


# Group low freq categoricals
cats = ["pat_type", "ms_drg", "std_payor", "race", "hispanic_ind"]
tmp = cc1
# quick regroup of categroical variables to 4
for i in range(0,17):
    tb = np.unique(tmp[:,i], return_counts=True, return_index=True)
    if cc_name[i] in cats:
        val = tb[0][np.argsort(-tb[2])]
        cnt = tb[2][np.argsort(-tb[2])]
        if len(cnt) > 4:
            print(cc_name[i])
            cn_val = cnt[4]
            repl = val[[c<=cn_val for c in cnt]]
            tt = np.where(np.in1d(tmp[:,i], repl), -99, tmp[:,i])
            print(np.unique(tt, return_counts=True))
            tmp[:,i] = tt
cc1 = tmp
tmp = []

# COMMAND ----------

# adjust time and y_sk
t_event_scale = bmb.get_time_transform(cc1[:,1], time_scale=TIME_SCALE)
y_sk = bmb.get_y_sklearn(cc1[:,0], t_event_scale)

# COMMAND ----------

# get train
trn = bmb.get_coh(y_sk, cc1, sample_n = SAMPLE_TRN, balance=BALANCE, train=True, idx=None, seed = np_seed)
# get test
tst = bmb.get_coh(y_sk, cc1, sample_n = SAMPLE_TST, balance=False, train=False, idx=trn["idx"], seed = np_seed, resample=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model

# COMMAND ----------

# # intitialize models
model_dict = {"trees": TREES,
    "split_rules": SPLIT_RULES
}
sampler_dict = {
            "draws": DRAWS,
            "tune": TUNE,
            "cores": CORES,
            "chains": CHAINS,
            "compute_convergence_checks": False
        }

# initialize bart
bart_model = bmb.BartSurvModel(model_config=model_dict, sampler_config=sampler_dict)

# COMMAND ----------

# fit model
bart_model.fit(trn["coh_y"], trn["coh_x"], trn["coh_w"], trn["coh_coords"])
# sample posterior
post = bart_model.sample_posterior_predictive(trn["x_tst"], trn["tst_coords"], extend_idata=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Posterior Analysis

# COMMAND ----------

trn_val = bmb.get_sv_prob(post)
trn_mq = bmb.get_sv_mean_quant(trn_val["sv"],trn["msk"])

ml.log_dict(dict([(k, trn_mq[k].tolist()) for k in trn_mq.keys()]), f"{CODE}_trn_sv_cov_ncov_mq.json")

# COMMAND ----------

fig = ut.sv_plot(
    sv= trn_mq, 
    y_sk_coh = trn["y_sk_coh"], 
    msk=trn["msk"], 
    y_sk=y_sk, 
    cc1=cc1, 
    strat=True, 
    cred_int=True, 
    kpm_all=True, 
    kpm_sample=True,
    whr = "mid"
)
ml.log_figure(fig, f"{CODE}_trn_sv_prob.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## C-Index, Brier Score

# COMMAND ----------

# CI and BRIER in-sample
ci = sks.metrics.concordance_index_censored(
    trn["y_sk_coh"]["Status"], 
    trn["y_sk_coh"]["Survival_in_days"], 
    trn_val["prob"].mean(axis=0)[:,3]
    )
bs = sks.metrics.brier_score(
    trn["y_sk_coh"], 
    trn["y_sk_coh"], 
    trn_val["sv"].mean(axis=0)[:,1:4], 
    np.arange(1,4))
ibs = sks.metrics.integrated_brier_score(
    trn["y_sk_coh"], 
    trn["y_sk_coh"], 
    trn_val["sv"].mean(axis=0)[:,1:4], 
    np.arange(1,4))

cbsibs_dict = {
    "ci":np.round(ci[0],4),
    "bs":(bs[0].tolist(), bs[1].tolist()),
    "ibs": ibs
}
ml.log_dict(cbsibs_dict, f"{CODE}_trn_sv_ci_bs.json")
cbsibs_dict

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Calibration 

# COMMAND ----------

trn_calib = ut.calib_metric(trn_val["sv"], trn["y_sk_coh"], q = np.arange(0,1,0.1))
fig = ut.plot_calib_diff(trn_calib)

ml.log_figure(fig, f"{CODE}_trn_sv_calib_diff_time_rnk.png")

# COMMAND ----------

diff_dict = {}
for idx,i in enumerate(trn_calib["diff"].mean(0)):
    diff_dict[f"p_{idx+1}"] = np.round(i,3).tolist()

print(diff_dict)
ml.log_dict(diff_dict, f"{CODE}_trn_sv_calib_diff_time_rnk.json")

# COMMAND ----------

# MAGIC %md 
# MAGIC # OOO

# COMMAND ----------

# sample posterior
tst_post = bart_model.sample_posterior_predictive(tst["x_tst_test"], tst["x_tst_coords_test"], extend_idata=False) 

# COMMAND ----------

tst_val = bmb.get_sv_prob(tst_post)
tst_mq = bmb.get_sv_mean_quant(tst_val["sv"],tst["msk_test"])
ml.log_dict(dict([(k, tst_mq[k].tolist()) for k in tst_mq.keys()]), f"{CODE}_tst_sv_cov_ncov_mq.json")

# COMMAND ----------

fig = ut.sv_plot(
    sv= tst_mq, 
    y_sk_coh = tst["y_sk_coh"], 
    msk=tst["msk_test"], 
    y_sk=y_sk, 
    cc1=cc1, 
    strat=True, 
    cred_int=True, 
    kpm_all=True, 
    kpm_sample=True,
    whr = "mid"
)
ml.log_figure(fig, f"{CODE}_tst_sv_prob.png")



# COMMAND ----------

# MAGIC %md
# MAGIC ## C-Index, Brier Score

# COMMAND ----------

# CI and BRIER in-sample
ci = sks.metrics.concordance_index_censored(
    tst["y_sk_coh"]["Status"], 
    tst["y_sk_coh"]["Survival_in_days"], 
    tst_val["prob"].mean(axis=0)[:,3]
    )
bs = sks.metrics.brier_score(
    trn["y_sk_coh"], 
    tst["y_sk_coh"], 
    tst_val["sv"].mean(axis=0)[:,1:4], 
    np.arange(1,4))
ibs = sks.metrics.integrated_brier_score(
    trn["y_sk_coh"], 
    tst["y_sk_coh"], 
    tst_val["sv"].mean(axis=0)[:,1:4], 
    np.arange(1,4))

cbsibs_dict = {
    "ci":np.round(ci[0],4),
    "bs":(bs[0].tolist(), bs[1].tolist()),
    "ibs": ibs
}
ml.log_dict(cbsibs_dict, f"{CODE}_tst_sv_ci_bs.json")
cbsibs_dict

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Calibration

# COMMAND ----------

tst_calib = ut.calib_metric(tst_val["sv"], tst["y_sk_coh"])
fig = ut.plot_calib_diff(tst_calib)
ml.log_figure(fig, f"{CODE}_tst_sv_calib_diff_time_rnk.png")

# COMMAND ----------

diff_dict = {}
for idx,i in enumerate(tst_calib["diff"].mean(0)):
    diff_dict[f"p_{idx+1}"] = np.round(i,3).tolist()

print(diff_dict)
ml.log_dict(diff_dict, f"{CODE}_tst_sv_calib_diff_time_rnk.json")

# COMMAND ----------

# MAGIC %md
# MAGIC # Variable Importance

# COMMAND ----------

vmean = bart_model.idata.sample_stats.variable_inclusion.values[:,:,1:].mean((0,1))
vname = np.array(cc_name[2:])
vname = vname[np.argsort(-vmean)]

var_dict = tuple(zip(vname, 
                    [int(x) for x in np.argsort(-vmean)],
                    -np.sort(-vmean))
    )

ml.log_dict({"var_rank":var_dict}, f"{CODE}_naive_var_ranks.json")

# COMMAND ----------

# MAGIC %md
# MAGIC # PDP

# COMMAND ----------

# MAGIC %md
# MAGIC ## COVID

# COMMAND ----------

# covid pdp
trn_cov_pdp = bmb.pdp_eval(
    trn["x_sk_coh"], 
    bart_model = bart_model, 
    var_col = [5], 
    values = [[0,1]],
    var_name="covid_icd_lab", 
    sample_n=None, 
    uniq_times=bart_model.uniq_times
    )

tst_cov_pdp = bmb.pdp_eval(
    tst["x_sk_coh"], 
    bart_model=bart_model,
    var_col = [5], 
    values = [[0,1]], 
    var_name="covid_icd_lab", 
    sample_n=None, 
    uniq_times=bart_model.uniq_times)

# COMMAND ----------

out = {}
for n,s in [("trn_rr",trn_cov_pdp["pdp_rr"]), ("tst_rr",tst_cov_pdp["pdp_rr"]), ("trn_diff",trn_cov_pdp["pdp_diff"]), ("tst_diff",tst_cov_pdp["pdp_diff"])]:
    tmp = {}
    for k in s.keys():
        tmp[k] = s[k].tolist()
    out[n] = tmp
ml.log_dict(out, f"{CODE}_pdp_covid_trn_tst.json")
out

# COMMAND ----------

# MAGIC %md
# MAGIC ## All Vars PDP Quick
# MAGIC

# COMMAND ----------

# get all pdps
pdp_dict = {}
pdp_summ = {}
for v in var_dict:
    uniq = np.unique(tst["x_sk_coh"][:,v[1]], return_counts=True)
    if len(uniq[0]) <= 4:
        vq = uniq[0][np.argsort(-uniq[1])][0:2]
    else:
        vq = uniq[0][0:2]
    pdp = bmb.pdp_eval(trn["x_sk_coh"], 
                       bart_model=bart_model,
                      var_col = [v[1]], 
                      values = [vq], 
                      var_name=v[0], 
                      sample_n=1000, 
                      uniq_times=bart_model.uniq_times
                      )
    pdp_dict[v[0]] = [pdp["pdp_diff"], pdp["pdp_rr"]]
    pdp_summ[v[0]] = {
        "val": [int(vq[0]),int(vq[1])],
        "diff_m": np.round(pdp["pdp_diff"]["diff_m"].mean(),3).tolist(),
        "diff_ql": np.round(pdp["pdp_diff"]["diff_q"].mean(1)[0],3).tolist(),
        "diff_qh": np.round(pdp["pdp_diff"]["diff_q"].mean(1)[1],3).tolist(),
        "rr_m": np.round(pdp["pdp_rr"]["rr_m"].mean(),3).tolist(),
        "rr_ql": np.round(pdp["pdp_rr"]["rr_q"].mean(1)[0],3).tolist(),
        "rr_qh": np.round(pdp["pdp_rr"]["rr_q"].mean(1)[1],3).tolist()
    }



# COMMAND ----------

ml.log_dict(pdp_summ, f"{CODE}_all_pdp_sample_summary.json")

# COMMAND ----------

# MAGIC %md 
# MAGIC # CPH
# MAGIC - need to onehot this

# COMMAND ----------

tmp = cc1[trn["idx"]["sample_idx"]]
tmp = pd.get_dummies(pd.DataFrame(tmp, columns=cc_name), columns=["pat_type", "std_payor", "ms_drg", "race", "hispanic_ind", "i_o_ind"], drop_first=True, dtype="int")

# COMMAND ----------

cph = ll.CoxPHFitter(penalizer=0.0001)
c = cph.fit(
    tmp,
    event_col = "ccsr_ind_p3", 
    duration_col = "ccsr_tt_p3", 
    fit_options = {"step_size":0.1}
    )
# c.print_summary()

# COMMAND ----------

ml.log_dict(c.summary.T.to_dict(), f"{CODE}_trn_cph_result.json")

# COMMAND ----------

ml.log_dict({"cindex":c.concordance_index_}, f"{CODE}_trn_cph_cindex.json")

# COMMAND ----------

# Full dataset
tmp = cc1
tmp = pd.get_dummies(pd.DataFrame(tmp, columns=cc_name), columns=["pat_type", "std_payor", "ms_drg", "race", "hispanic_ind", "i_o_ind"], drop_first=True, dtype="int")

cph = ll.CoxPHFitter(penalizer=0.0001)
c2 = cph.fit(tmp, 
             event_col = "ccsr_ind_p3", 
             duration_col = "ccsr_tt_p3", 
             fit_options = {"step_size":0.1})

# COMMAND ----------

ml.log_dict(c2.summary.T.to_dict(), f"{CODE}_all_cph_result.json")
ml.log_dict({"cindex":c.concordance_index_}, f"{CODE}_all_cph_cindex.json")

# COMMAND ----------

ml.end_run()
