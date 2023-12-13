# Databricks notebook source
# MAGIC %pip install mlflow

# COMMAND ----------

import mlflow as ml
import pandas as pd
import numpy as np

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

# MAGIC %md
# MAGIC # Metrics

# COMMAND ----------

p1 = pd.DataFrame(artf["tst_sv_ci_bs"]).T.add_prefix("tst_")
p2 = pd.DataFrame(artf["trn_sv_ci_bs"]).T.add_prefix("trn_")
p3 = pd.DataFrame(artf["all_cph_cindex"]).T. add_prefix("cph_all_")
p4 = pd.DataFrame(artf["trn_cph_cindex"]).T.add_prefix("cph_trn_")
#
p5 = pd.DataFrame(artf["tst_sv_calib"]).T.add_prefix("tst_")
p6 = pd.DataFrame(artf["cph_trn_sv_calib"]).T.add_prefix("cph_trn_")
p7 = pd.DataFrame(artf["cph_tst_sv_calib"]).T.add_prefix("cph_tst_")

metrics = p1.join(p2).join(p3).join(p4).join(p5).join(p6).join(p7)

metrics2 = metrics.drop(["tst_bs", 
              "trn_bs", 
              "tst_exp", 
              "tst_pred", 
              "tst_qt", 
              "cph_trn_exp",
              "cph_trn_pred",
              "cph_trn_qt",
              "cph_tst_exp",
              "cph_tst_pred",
              "cph_tst_qt",
              "cph_all_cindex"
              ], axis=1)

# get abs sum of diff
for i in ["tst_diff", "cph_tst_diff", "cph_trn_diff"]:
    metrics2[i] = metrics2[i].apply(lambda x: np.abs(x).mean())
 

# COMMAND ----------


metrics2

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

# plot
fig,ax = plt.subplots(1, figsize=(10,10))
ax.plot(metrics2.index, metrics2.tst_ci, label = "tst_bart")
ax.plot(metrics2.index, metrics2.trn_ci , label = "trn_bart")
ax.plot(metrics2.index, metrics2.cph_trn_cindex, label = "trn_cph")
ax.set_xticklabels(metrics2.index, rotation=90)
ax.set_title("C-Index")
ax.legend()


# COMMAND ----------

fig,ax = plt.subplots(1, figsize=(10,10))
ax.plot(metrics2.index, metrics2.tst_ibs, label = "tst_bart")
ax.plot(metrics2.index, metrics2.trn_ibs , label = "trn_bart")
ax.set_xticklabels(metrics2.index, rotation=90)
ax.set_title("Integrated Brier Score")
ax.legend()

# COMMAND ----------

fig,ax = plt.subplots(1, figsize=(10,10))
ax.plot(metrics2.index, metrics2.tst_diff, label = "tst_bart")
ax.plot(metrics2.index, metrics2.cph_trn_diff, label = "trn_cph")
ax.plot(metrics2.index, metrics2.cph_tst_diff, label = "tst_cph")
ax.set_xticklabels(metrics2.index, rotation=90)
ax.set_title("Calibration Difference (by last time)")
ax.set_ylabel("Mean Absolute Error")
ax.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC # Results

# COMMAND ----------

# MAGIC %md
# MAGIC ## HR

# COMMAND ----------

[print(k) for k in artf.keys()]

# COMMAND ----------

p1 =pd.DataFrame(artf["pdp_covid_trn_tst"]).T.add_prefix("bart_")
covid_all = {k:v["covid_icd_lab"] for k,v in artf["all_cph_result"].items()}
covid_trn = {k:v["covid_icd_lab"] for k,v in artf["trn_cph_result"].items()}
p2 = pd.DataFrame(covid_all).T.add_prefix("cph_all_")
p3 = pd.DataFrame(covid_trn).T.add_prefix("cph_trn_")

res = p1.join(p2).join(p3)

# COMMAND ----------

[print(c) for c in res.columns]

# COMMAND ----------

res_rr = res[[
    "bart_trn_rr",
    "bart_tst_rr",
    "cph_trn_exp(coef)",
    "cph_trn_exp(coef) lower 95%",
    "cph_trn_exp(coef) upper 95%",
    "cph_all_exp(coef)",
    "cph_all_exp(coef) lower 95%",
    "cph_all_exp(coef) upper 95%",
    "cph_trn_p",
    "cph_all_p"
]]
res_rr["bart_rr_ml"] = res_rr["bart_trn_rr"].apply(lambda x: x["rr_m"])
res_rr["bart_rr_ll"] = res_rr["bart_trn_rr"].apply(lambda x: x["rr_q"][0])
res_rr["bart_rr_ul"] = res_rr["bart_trn_rr"].apply(lambda x: x["rr_q"][1])

res_rr["bart_rr_m"] = res_rr["bart_trn_rr"].apply(lambda x: np.mean(x["rr_m"]))
res_rr["bart_rr_l"] = res_rr["bart_trn_rr"].apply(lambda x: np.mean(x["rr_q"][0]))
res_rr["bart_rr_u"] = res_rr["bart_trn_rr"].apply(lambda x: np.mean(x["rr_q"][1]))
res_rr["cph_trn_sig"] = res_rr["cph_trn_p"].apply(lambda x: 3 if x < 0.05 else np.NAN)
res_rr["cph_all_sig"] = res_rr["cph_all_p"].apply(lambda x: 2.9 if x < 0.05 else np.NAN)


# COMMAND ----------

res_rr.columns

# COMMAND ----------

ind = res_rr.index.to_list()
print(ind)
inds = np.arange(1,len(ind)+1)
print(inds)
fig,ax = plt.subplots(1, figsize=(10,10))
# ax.plot(ind, res_rr.bart_rr_m)
ax.errorbar(y = inds, 
            x = res_rr.bart_rr_m,
            xerr = (
                res_rr.bart_rr_m - res_rr.bart_rr_l,
                res_rr.bart_rr_u - res_rr.bart_rr_m
            ),
            fmt = "o",
            label = "bart"
            )
ax.errorbar(y = inds + .11, 
            x = res_rr["cph_all_exp(coef)"],
            xerr = (
                res_rr["cph_all_exp(coef)"] - res_rr["cph_all_exp(coef) lower 95%"],
                res_rr["cph_all_exp(coef) upper 95%"] - res_rr["cph_all_exp(coef)"], 
            ),
            fmt = "o",
            label = "cph_all"
            )
ax.errorbar(y =inds + .15, 
            x = res_rr["cph_trn_exp(coef)"],
            xerr = (
                res_rr["cph_trn_exp(coef)"] - res_rr["cph_trn_exp(coef) lower 95%"],
                res_rr["cph_trn_exp(coef) upper 95%"] - res_rr["cph_trn_exp(coef)"], 
            ),
            fmt = "o",
            label = "cph_trn"
            )
ax.plot(res_rr["cph_trn_sig"], inds+.1, "*", color = "green")
ax.plot(res_rr["cph_all_sig"], inds+.1, "*", color = "darkorange")
ax.set_yticks(inds)
ax.set_yticklabels(ind)
ax.plot(np.repeat(1,len(inds)), inds, "--")
ax.legend()
ax.set_title("Hazard Ratios COVID Condition")
ax.set_xlabel("HR")

# ax.set_xlim((, 2))



# COMMAND ----------

# MAGIC %md
# MAGIC ## Diff

# COMMAND ----------

[print(k) for k in artf.keys()]

# COMMAND ----------

def get_diff(artf, time):
    t = time-1
    p1 = pd.DataFrame(artf["pdp_covid_trn_tst"]).T
    p1["trn_m"] = p1["trn_diff"].apply(lambda x: x["diff_m"][t] * -1)
    p1["trn_l"] = p1["trn_diff"].apply(lambda x: x["diff_q"][0][t] * -1)
    p1["trn_h"] = p1["trn_diff"].apply(lambda x: x["diff_q"][1][t] * -1)
    p1["tst_m"] = p1["tst_diff"].apply(lambda x: x["diff_m"][t] * -1)
    p1["tst_l"] = p1["tst_diff"].apply(lambda x: x["diff_q"][0][t] * -1)
    p1["tst_h"] = p1["tst_diff"].apply(lambda x: x["diff_q"][1][t] * -1)
    return p1

def get_diff_mean_time(artf, time):
    t = time -1
    p2 = pd.DataFrame(artf["tst_sv_cov_ncov_mq"]).T
    p3 =  np.round(np.hstack([
        np.array([x[t] for x in p2.mt_m.to_numpy()]).reshape(-1,1),
        np.array([x[t] for x in p2.mf_m.to_numpy()]).reshape(-1,1)
    ]).mean(1), 2).astype("str")
    return p3

def plt_diff(p1, p3, time):
    t = time * 91.25
    ind = p1.index.to_list()
    print(ind)
    inds = np.arange(1,len(ind)+1)
    print(inds)
    fig,ax = plt.subplots(1, figsize=(10,10))
    # ax.plot(ind, res_rr.bart_rr_m)
    ax.errorbar(y = inds, 
                x = p1.trn_m,
                xerr = (
                    p1.trn_m - p1.trn_l,
                    p1.trn_h - p1.trn_m
                ),
                fmt = "o",
                label = "bart_trn"
                )
    ax.errorbar(y = inds + .2, 
                x = p1.tst_m,
                xerr = (
                    p1.tst_m - p1.tst_l,
                    p1.tst_h - p1.tst_m
                ),
                fmt = "o",
                label = "bart_tst"
                )
    # ax.plot(res_rr["cph_trn_sig"], inds+.1, "*", color = "green")
    # ax.plot(res_rr["cph_all_sig"], inds+.1, "*", color = "darkorange")
    ax.set_yticks(inds)
    ax.set_yticklabels(ind)
    ax.plot(np.repeat(0,len(inds)), inds, "--")
    print(np.repeat(-0.6, len(inds)))
    print(inds)

    ax.legend()
    ax.set_title(f"Marginal Distribution COVID Condition ({t} days)")
    ax.set_xlabel("Survival Difference (Covid - Non-Covid)")
    for idx, i in enumerate(p3):
        ax.text(y = inds[idx], x = -0.075, s=i)
    return fig


# COMMAND ----------

time = 2
p1 = get_diff(artf, time)
p3 = get_diff_mean_time(artf, time)
plt_diff(p1,p3, time)


# COMMAND ----------

time = 4
p1 = get_diff(artf, time)
p3 = get_diff_mean_time(artf, time)
plt_diff(p1,p3, time)


# COMMAND ----------

# MAGIC %md 
# MAGIC ## All Vars

# COMMAND ----------

[print(k) for k in artf.keys()]

# COMMAND ----------

covid_all = {k:v["covid_icd_lab"] for k,v in artf["all_cph_result"].items()}
covid_trn = {k:v["covid_icd_lab"] for k,v in artf["trn_cph_result"].items()}
p2 = pd.DataFrame(covid_all).T.add_prefix("cph_all_")
p3 = pd.DataFrame(covid_trn).T.add_prefix("cph_trn_")

# COMMAND ----------



# COMMAND ----------

def all_hr(artf):
    p1 = pd.DataFrame(artf["all_pdp_sample_summary"]).T
    cols = p1.columns
    vals = [x for x in artf["all_pdp_sample_summary"]["SYM"]["p3_sum"].keys()]
    for c in cols:
        for v in vals:
            p1[c + "_" + v] = p1[c].apply(lambda x: x[v])
    cols = [c for c in p1.columns if "rr_m" in c]
    p2 = p1[cols]
    return p2

# p1

# COMMAND ----------

p2 = all_hr(artf)

# COMMAND ----------

colors = plt.cm.tab20_r(np.arange(0,20))
# print(p2.columns)
idx = [i for i in p2.index]
# print(idx)

fig, ax = plt.subplots(1, figsize = (10,10))
for idxx,c in enumerate(cols):
    ax.plot(idx, p2[c], label = c, color = colors[idxx,:])

plt.legend(bbox_to_anchor = (1.3,1))
ax.set_xticklabels(idx, rotation=90)
ax.set_ylim(0,5)
ax.set_yticks(np.arange(0,5,.5))
ax.grid(alpha=0.2)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## All HR COX

# COMMAND ----------

artf.keys()

# COMMAND ----------

[x for x in artf["all_cph_result"]["SYM"]["p3_sum"].keys()]

# COMMAND ----------

def all_hr_cph(artf):
    p1 = pd.DataFrame(artf["all_cph_result"]).T
    cols = p1.columns
    print(cols)
    vals = [x for x in artf["all_cph_result"]["SYM"]["p3_sum"].keys()]
    print(vals)
    for c in cols:
        p1[c + "_" + "HR"] = p1[c].apply(lambda x: x["exp(coef)"])
    cols = [c for c in p1.columns if "HR" in c]
    p2 = p1[cols]
    return p2

p2 = all_hr_cph(artf)

# COMMAND ----------

# len(p1.columns)
cols = p2.columns
colors = plt.cm.tab20_r(np.arange(0,20))
# print(p2.columns)
idx = [i for i in p2.index]
# print(idx)

fig, ax = plt.subplots(1, figsize = (10,10))
for idxx,c in enumerate(cols):
    if idxx > 19:
        break
    ax.plot(idx, p2[c], label = c, color = colors[idxx,:])

plt.legend(bbox_to_anchor = (1.3,1))
ax.set_xticklabels(idx, rotation=90)
ax.set_ylim(0,5)
ax.set_yticks(np.arange(0,5,.5))
ax.grid(alpha=0.2)
