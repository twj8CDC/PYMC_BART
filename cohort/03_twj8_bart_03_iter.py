# Databricks notebook source
# MAGIC %pip install scikit-survival pymc pymc_experimental matplotlib colorcet pymc_bart lifelines mlflow psutil

# COMMAND ----------


import psutil
import os

def mem_chk():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{mem_info.rss/1_000_000_000} Gb")
    print(f"{mem_info.vms/1_000_000_000} Gb")

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

# # importlib.reload(bmb)
# Set Seed
np_seed = int(np.ceil(time.time()))

# COMMAND ----------

# MAGIC %md
# MAGIC # Globals

# COMMAND ----------

# GLOBALS
CODE = "CIR"
RUN_NAME = f"{CODE}_run_01"
EXP_ID = 502851330942627
TIME_SCALE = 91.5
BALANCE = True
SAMPLE_TRN = 10_000
SAMPLE_TST = 10_000

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
DRAWS = 600
TUNE = 600
CORES = 4
CHAINS = 4
PDP_ALL = True
WEIGHT = 1
RUN_NUM = 1



# COMMAND ----------

def get_param(widget, PARAM, ptype):
    dbutils.widgets.text(widget, defaultValue=str(PARAM))
    PARAM1 = dbutils.widgets.get(widget)
    if PARAM1 != "na":
        if ptype == "int":
            PARAM = int(PARAM1)
        if ptype == "str":
            PARAM = str(PARAM1)
        if ptype == "bool":
            PARAM = eval(PARAM1)
        if ptype == "eval":
            PARAM = eval(PARAM1)
        if ptype == "float":
            PARAM = float(PARAM1)
    return PARAM

# COMMAND ----------

np_seed = get_param("seed", np_seed, "int")
CODE = get_param("code", CODE, "str")
EXP_ID = get_param("exp_id", EXP_ID, "int")
TIME_SCALE = get_param("time_scale", TIME_SCALE, "float")
BALANCE = get_param("balance", BALANCE, "eval")
SAMPLE_TRN = get_param("sample_trn", SAMPLE_TRN, "int")
SAMPLE_TST = get_param("sample_tst", SAMPLE_TST, "int")
TREES = get_param("trees", TREES, "int")
SPLIT_RULES = get_param("split_rules", SPLIT_RULES, "eval")
DRAWS = get_param("draws", DRAWS, "int")
TUNE = get_param("tune", TUNE, "int")
CORES = get_param("cores", CORES, "int")
CHAINS = get_param("chains", CHAINS, "int")
RUN_NUM = get_param("run_num", RUN_NUM, "int")
PDP_ALL = get_param("pdp_all", PDP_ALL, "bool")
WEIGHT = get_param("weight", WEIGHT, "int")

RUN_NAME = "pcc_" + CODE + "_" + str(RUN_NUM)

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
    "CHAINS":CHAINS,
    "WEIGHT":WEIGHT
}

ml.log_dict(global_dict, f"{CODE}_global_dict.json")
ml.log_dict(model_dict_main, f"{CODE}_model_dict.json")
print(global_dict)
print(model_dict_main)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

# ccsr_l = spark.table("cdh_premier_exploratory.twj8_pat_ccsr_long_f_06")
# Load ccsr 
ccsr_s = spark.table("cdh_premier_exploratory.twj8_pat_ccsr_short_f_06")
cov = spark.table("cdh_premier_exploratory.twj8_pat_covariates_f_07")

mem_chk()

# Get the datasets data
cc1, cc_name = bmb.get_sk_sp(ccsr_s, cov, CODE)

mem_chk()

# COMMAND ----------

# codes = [x[0] for x in ccsr_s.select("code_sm").distinct().collect()]
# print(codes)

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
del tmp

mem_chk()

# COMMAND ----------

# adjust time and y_sk
t_event_scale = bmb.get_time_transform(cc1[:,1], time_scale=TIME_SCALE)
y_sk = bmb.get_y_sklearn(cc1[:,0], t_event_scale)

mem_chk()

# COMMAND ----------

# get train
trn = bmb.get_coh(y_sk, cc1, sample_n = SAMPLE_TRN, balance=BALANCE, train=True, idx=None, seed = np_seed, prop=WEIGHT)
# get test
tst = bmb.get_coh(y_sk, cc1, sample_n = SAMPLE_TST, balance=False, train=False, idx=trn["idx"], seed = np_seed, resample=False, prop=1)

# COMMAND ----------

trn_counts = np.unique(trn["x_sk_coh"][:,5], return_counts=True)
tst_counts = np.unique(tst["x_sk_coh"][:,5], return_counts=True)
counts_dict = {
    "trn":{"ncov":int(trn_counts[1][0]),
           "cov":int(trn_counts[1][1]),
           "prop_cov": float(trn_counts[1][1]/SAMPLE_TRN)
           },
    "tst":{"ncov":int(tst_counts[1][0]),
           "cov":int(tst_counts[1][1]),
           "prop_cov": float(tst_counts[1][1]/SAMPLE_TST)
           }
}
ml.log_dict(counts_dict, f"{CODE}_samples_counts.json")

mem_chk()

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

# # fit model
# bart_model.fit(trn["coh_y"], trn["coh_x"], trn["coh_w"], trn["coh_coords"])
# # sample posterior
# post = bart_model.sample_posterior_predictive(trn["x_tst"], trn["tst_coords"], extend_idata=True)

# fit model
bart_model.fit(trn["coh_y"].astype(np.float32),
               trn["coh_x"].astype(np.float32), 
               trn["coh_w"].astype(np.float32), 
               trn["coh_coords"].astype(np.float32)
               )
# sample posterior
post = bart_model.sample_posterior_predictive(trn["x_tst"], trn["tst_coords"], extend_idata=False)

# COMMAND ----------

mem_chk()

# COMMAND ----------

# MAGIC %md
# MAGIC # Posterior Analysis

# COMMAND ----------

trn_val = bmb.get_sv_prob(post)
trn_mq = bmb.get_sv_mean_quant(trn_val["sv"],trn["msk"])

ml.log_dict(dict([(k, trn_mq[k].tolist()) for k in trn_mq.keys()]), f"{CODE}_trn_sv_cov_ncov_mq.json")

# COMMAND ----------

trn_val

# COMMAND ----------

title = f"{CODE}_trn_sv_prob.png"
fig = ut.sv_plot(
    sv= trn_mq, 
    y_sk_coh = trn["y_sk_coh"], 
    msk=trn["msk"], 
    y_sk=y_sk,
    title =title, 
    cc1=cc1, 
    strat=True, 
    cred_int=True, 
    kpm_all=True, 
    kpm_sample=True,
    whr = "mid"
)
ml.log_figure(fig, title)

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

# title = f"{CODE}_trn_sv_calib_diff_time_rnk.png"
# trn_calib = ut.calib_metric(
#     trn_val["sv"], 
#     trn["y_sk_coh"], 
#     q = np.arange(0,1,0.1)
#     )
# fig = ut.plot_calib_diff(trn_calib)
# ml.log_figure(fig, title)

# COMMAND ----------

cb = ut.calib_metric_bart(
    trn_val["sv"], 
    trn["y_sk_coh"], 
    t=4, q = np.arange(0,1,0.1), 
    single_time=True
)
title = f"{CODE}_trn_sv_calib_plot.png"
try:
    fig = ut.plot_calib_prob(cb, title)
    ml.log_figure(fig, title)
except:
    print("No Plot Available")

# COMMAND ----------

try:
    cb_dict = {
        "exp":cb["obs"].tolist(),
        "pred":cb["pred"].tolist(),
        "diff":cb["diff"].tolist(),
        "qt": cb["qtile"].tolist()
    }
    ml.log_dict(cb_dict, f"{CODE}_trn_sv_calib.json")
    print(cb_dict)  
except:
    print("Failed to save")

# COMMAND ----------

# diff_dict = {}
# for idx,i in enumerate(trn_calib["diff"].mean(0)):
#     diff_dict[f"p_{idx+1}"] = np.round(i,3).tolist()

# print(diff_dict)
# ml.log_dict(diff_dict, f"{CODE}_trn_sv_calib_diff_time_rnk.json")

# COMMAND ----------

# clear  trn_val
mem_chk()
del trn_val
del post
mem_chk()

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

title = f"{CODE}_tst_sv_prob.png"
fig = ut.sv_plot(
    sv= tst_mq, 
    y_sk_coh = tst["y_sk_coh"], 
    msk=tst["msk_test"], 
    y_sk=y_sk, 
    cc1=cc1, 
    title = title,
    strat=True, 
    cred_int=True, 
    kpm_all=True, 
    kpm_sample=True,
    whr = "mid"
)
ml.log_figure(fig, title)



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

# tst_calib = ut.calib_metric(tst_val["sv"], tst["y_sk_coh"])
# fig = ut.plot_calib_diff(tst_calib)
# ml.log_figure(fig, f"{CODE}_tst_sv_calib_diff_time_rnk.png")

# COMMAND ----------

cb = ut.calib_metric_bart(
    tst_val["sv"], 
    tst["y_sk_coh"], 
    t=4, q = np.arange(0,1,0.1), 
    single_time=True
)

title = f"{CODE}_tst_sv_calib_plot.png"
try:
    fig = ut.plot_calib_prob(cb, title)
    ml.log_figure(fig, title)
except:
    print("No Plot Available")

# COMMAND ----------

try:
    cb_dict = {
        "exp":cb["obs"].tolist(),
        "pred":cb["pred"].tolist(),
        "diff":cb["diff"].tolist(),
        "qt": cb["qtile"].tolist()
    }
    print(cb_dict)
    ml.log_dict(cb_dict, f"{CODE}_tst_sv_calib.json")
except:
    print("Failed to save")

# COMMAND ----------

# diff_dict = {}
# for idx,i in enumerate(tst_calib["diff"].mean(0)):
#     diff_dict[f"p_{idx+1}"] = np.round(i,3).tolist()

# print(diff_dict)
# ml.log_dict(diff_dict, f"{CODE}_tst_sv_calib_diff_time_rnk.json")

# COMMAND ----------

# drop tst_val
mem_chk()
del tst_val
del tst_post
mem_chk()

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

var_dict

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
    sample_n=10_000, 
    uniq_times=bart_model.uniq_times,
    return_all = True
    )

tst_cov_pdp = bmb.pdp_eval(
    tst["x_sk_coh"], 
    bart_model=bart_model,
    var_col = [5], 
    values = [[0,1]], 
    var_name="covid_icd_lab", 
    sample_n=10_000, 
    uniq_times=bart_model.uniq_times,
    return_all=True
    )

# COMMAND ----------

np.quantile(x3[:,3], [0.025,0.975])


# COMMAND ----------

x = trn_cov_pdp["pdp_val"]["prob"][:,SAMPLE_TRN:,:]/trn_cov_pdp["pdp_val"]["prob"][:,:SAMPLE_TRN,:]
x2 = np.quantile(x, [0.25, 0.5, 0.75], 1)
x3 = x.mean(1)
x4 = np.quantile(x3[:,3], [0.025, 0.11,0.89, 0.975])


fig, ax = plt.subplots(1, figsize=(10,10))
# ax.hist(x3[:,3], alpha=0.3, label="mean", bins=50)
# ax.hist(x2[0,:,3], alpha=0.2, label ="25qtile", bins=50)
# ax.hist(x2[1,:,3], alpha=0.2, label = "50qtile", bins=50)
# ax.hist(x2[2,:,3], alpha=0.1, label = "75qtile", bins=50)
# plt.axvline(x4[0], color = "red", label = "95%")
# plt.axvline(x4[3], color = "red")
# plt.axvline(x4[1], color = "blue", label = "89%")
# plt.axvline(x4[2], color = "blue")
# plt.axvline(x3[:,3].mean(0), color = "green", label="mean")

binss = 25
ax.hist(x3[:,3], alpha=0.3, label="mean", bins=binss)
ax.hist(x2[0,:,3], alpha=0.2, label ="25qtile", bins=binss)
# ax.hist(x2[1,:,3], alpha=0.2, label = "50qtile", bins=binss)
ax.hist(x2[2,:,3], alpha=0.1, label = "75qtile", bins=binss)
ax.axvline(x4[0], color = "lightblue", label = "mean 95%")
ax.axvline(x4[3], color = "lightblue")
ax.axvline(x4[1], color = "blue", label = "mean 89%", alpha=0.7)
ax.axvline(x4[2], color = "blue", alpha=0.7)
ax.axvline(x3[:,3].mean(0), color = "darkblue", label="mean mean")
ax.axvline(x2[2,:,3].mean(0), color = "green", label="mean 75th", alpha=0.8)
ax.axvline(x2[0,:,3].mean(0), color = "orange", label="mean 25th", alpha=0.8)
ax.set_title(f"Distribution of draws ({DRAWS * CHAINS}) of calculated RR")
ax.set_xlabel("RR")
ax.legend()

title = f"{CODE}_trn_pdp_sv_rr.png"
ml.log_figure(fig, title)

# COMMAND ----------

x = trn_cov_pdp["pdp_val"]["sv"][:,SAMPLE_TRN:,:] - trn_cov_pdp["pdp_val"]["sv"][:,:SAMPLE_TRN,:]
x2 = np.quantile(x, [0.25, 0.5, 0.75], 1)
x3 = x.mean(1)
x4 = np.quantile(x3[:,3], [0.025, 0.11,0.89, 0.975])


fig, ax = plt.subplots(1, figsize=(10,10))
binss = 30
ax.hist(x3[:,3], alpha=0.3, label="mean", bins=binss)
ax.hist(x2[0,:,3], alpha=0.2, label ="25qtile", bins=binss)
# ax.hist(x2[1,:,3], alpha=0.2, label = "50qtile", bins=binss)
ax.hist(x2[2,:,3], alpha=0.1, label = "75qtile", bins=binss)
ax.axvline(x4[0], color = "lightblue", label = "mean 95%")
ax.axvline(x4[3], color = "lightblue")
ax.axvline(x4[1], color = "blue", label = "mean 89%", alpha=0.7)
ax.axvline(x4[2], color = "blue", alpha=0.7)
ax.axvline(x3[:,3].mean(0), color = "darkblue", label="mean mean")
ax.axvline(x2[2,:,3].mean(0), color = "green", label="mean 75th", alpha=0.8)
ax.axvline(x2[0,:,3].mean(0), color = "orange", label="mean 25th", alpha=0.8)
ax.set_title(f"Distribution of draws ({DRAWS * CHAINS}) of calculated SV diff")
ax.set_xlabel("SV diff (COV - NCOV)")
# ax.set_xticks(np.arange(-0.3,0.2, .05))
ax.legend()

title = f"{CODE}_trn_pdp_sv_distribution.png"
ml.log_figure(fig, title)

# COMMAND ----------

title = f"{CODE}_trn_pdp_sv_plot.png"
fig = ut.sv_plot(
    sv= trn_cov_pdp["pdp_mq"], 
    y_sk_coh = trn["y_sk_coh"], 
    msk=trn["msk"], 
    y_sk=y_sk, 
    title = title,
    cc1=cc1, 
    strat=True, 
    cred_int=True, 
    kpm_all=False, 
    kpm_sample=False,
    whr = "mid"
)
ml.log_figure(fig, title)

# COMMAND ----------

title = f"{CODE}_tst_pdp_sv_plot.png"
fig = ut.sv_plot(
    sv= tst_cov_pdp["pdp_mq"], 
    y_sk_coh = tst["y_sk_coh"], 
    msk=tst["msk_test"], 
    y_sk=y_sk, 
    title = title,
    cc1=cc1, 
    strat=True, 
    cred_int=True, 
    kpm_all=False, 
    kpm_sample=False,
    whr = "mid"
)
ml.log_figure(fig, title)

# COMMAND ----------

out = {}
for n,s in [("trn_rr",trn_cov_pdp["pdp_rr"]), ("tst_rr",tst_cov_pdp["pdp_rr"]), 
            ("trn_diff",trn_cov_pdp["pdp_diff"]), ("tst_diff",tst_cov_pdp["pdp_diff"])]:
    tmp = {}
    for k in s.keys():
        tmp[k] = s[k].tolist()
    out[n] = tmp
ml.log_dict(out, f"{CODE}_pdp_covid_trn_tst.json")
out

# COMMAND ----------

# drop cov pdp
mem_chk()

del trn_cov_pdp
del tst_cov_pdp

mem_chk()


# COMMAND ----------

# MAGIC %md
# MAGIC ## All Vars PDP Quick
# MAGIC

# COMMAND ----------

# get all pdps
if PDP_ALL:
    try:
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
            del pdp
        ml.log_dict(pdp_summ, f"{CODE}_all_pdp_sample_summary.json")
        del pdp_summ
        del pdp_dict
    except:
        print("Error in pdp all")
    mem_chk()


# COMMAND ----------

# MAGIC %md 
# MAGIC # CPH
# MAGIC - need to onehot this

# COMMAND ----------

# pull of trn idx and tst idx
trn_idx = trn["idx"]["sample_idx"]
tst_idx = tst["idx_test"]

# drop the other large memory items
mem_chk()
del trn
del tst 
del bart_model
mem_chk()

# COMMAND ----------

# tmp = cc1[trn["idx"]["sample_idx"]]
tmp = cc1[trn_idx]
tmp = pd.get_dummies(pd.DataFrame(tmp, columns=cc_name), columns=["pat_type", "std_payor", "ms_drg", "race", "hispanic_ind", "i_o_ind"], drop_first=True, dtype="int")

# COMMAND ----------

# tmp_tst = cc1[tst["idx_test"]]
tmp_tst = cc1[tst_idx]
tmp_tst = pd.get_dummies(pd.DataFrame(tmp_tst, columns=cc_name), columns=["pat_type", "std_payor", "ms_drg", "race", "hispanic_ind", "i_o_ind"], drop_first=True, dtype="int")

# COMMAND ----------

mem_chk()

# COMMAND ----------

cph = ll.CoxPHFitter(penalizer=0.0001)
c = cph.fit(
    tmp,
    event_col = "ccsr_ind_p3", 
    duration_col = "ccsr_tt_p3", 
    fit_options = {"step_size":0.1}
    )
c.print_summary()

# COMMAND ----------

cph_sv = c.predict_survival_function(tmp, 365).T.to_numpy().squeeze()
tmp_sky = bmb.get_y_sklearn(tmp["ccsr_ind_p3"], tmp["ccsr_tt_p3"])
cb2 = ut.calib_metric_cph(cph_sv, 
                               tmp_sky, 
                               t=4, 
                               q = np.arange(0,1,0.1), 
                               single_time=True)
title =f"{CODE}_trn_cph_sv_calib_plot.png"
try:
    fig = ut.plot_calib_prob(cb2, title)
    ml.log_figure(fig, title)
except:
    print("No Plot Available")

# COMMAND ----------

try:
    cb_dict = {
        "exp":cb2["obs"].tolist(),
        "pred":cb2["pred"].tolist(),
        "diff":cb2["diff"].tolist(),
        "qt": cb2["qtile"].tolist()
    }
    print(cb_dict)
    ml.log_dict(cb_dict, f"{CODE}_cph_trn_sv_calib.json")
except:
    print("Failed to save")

# COMMAND ----------

cph_sv = c.predict_survival_function(tmp_tst, 365).T.to_numpy().squeeze()
tmp_sk_y = bmb.get_y_sklearn(tmp_tst["ccsr_ind_p3"], tmp_tst["ccsr_tt_p3"])
cb2 = ut.calib_metric_cph(cph_sv, 
                               tmp_sk_y, 
                               t=4, 
                               q = np.arange(0,1,0.1), 
                               single_time=True)
title =f"{CODE}_tst_cph_sv_calib_plot.png"
try:
    fig = ut.plot_calib_prob(cb2, title)
    ml.log_figure(fig, title)
except:
    print("No Plot Available")

# COMMAND ----------

try:
    cb_dict = {
        "exp":cb2["obs"].tolist(),
        "pred":cb2["pred"].tolist(),
        "diff":cb2["diff"].tolist(),
        "qt": cb2["qtile"].tolist()
    }
    print(cb_dict)
    ml.log_dict(cb_dict, f"{CODE}_cph_tst_sv_calib.json")
except:
    print("Failed to save")

# COMMAND ----------

ml.log_dict(c.summary.T.to_dict(), f"{CODE}_trn_cph_result.json")
ml.log_dict({"cindex":c.concordance_index_}, f"{CODE}_trn_cph_cindex.json")

# COMMAND ----------

mem_chk()

del tmp
del tmp_tst

mem_chk()

# COMMAND ----------

import gc
gc.collect()
dbutils.notebook.exit("exit notebook task")

# COMMAND ----------

# Full dataset
# tmp = cc1
# cc1 = pd.get_dummies(pd.DataFrame(cc1, columns=cc_name), columns=["pat_type", "std_payor", "ms_drg", "race", "hispanic_ind", "i_o_ind"], drop_first=True, dtype="int")

# tmp = cc1
# cc1 = pd.get_dummies(pd.DataFrame(cc1, columns=cc_name), columns=["pat_type", "std_payor", "ms_drg", "race", "hispanic_ind", "i_o_ind"], drop_first=True, dtype="int")
# cph = ll.CoxPHFitter(penalizer=0.0001)
# c2 = cph.fit(cc1, 
#              event_col = "ccsr_ind_p3", 
#              duration_col = "ccsr_tt_p3", 
#              fit_options = {"step_size":0.1})

# COMMAND ----------

# cph = ll.CoxPHFitter(penalizer=0.0001)
# c2 = cph.fit(cc1, 
#              event_col = "ccsr_ind_p3", 
#              duration_col = "ccsr_tt_p3", 
#              fit_options = {"step_size":0.1})

# c2.print_summary()

# COMMAND ----------

# ml.log_dict(c2.summary.T.to_dict(), f"{CODE}_all_cph_result.json")
# ml.log_dict({"cindex":c2.concordance_index_}, f"{CODE}_all_cph_cindex.json")

# COMMAND ----------

# ml.end_run()

# COMMAND ----------

# import gc
# gc.collect()

# COMMAND ----------

# mem_chk()

# COMMAND ----------

# dbutils.notebook.exit("exit notebook task")
