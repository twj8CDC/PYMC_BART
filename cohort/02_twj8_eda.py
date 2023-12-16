# Databricks notebook source
# MAGIC %pip install scikit-survival

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.window as W
from pyspark.storagelevel import StorageLevel
import numpy as np

import sksurv as sks
from sksurv import nonparametric
import matplotlib.pyplot as plt


# COMMAND ----------

# MAGIC %md
# MAGIC # Data

# COMMAND ----------

ccsr_l = spark.table("cdh_premier_exploratory.twj8_pat_ccsr_long_f_06")
ccsr_s = spark.table("cdh_premier_exploratory.twj8_pat_ccsr_short_f_06")
cov = spark.table("cdh_premier_exploratory.twj8_pat_covariates_f_06")

# COMMAND ----------

codes = [x[0] for x in ccsr_s.select("code_sm").distinct().collect()]
print(codes)

# COMMAND ----------

ccsr_l.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Prevalence

# COMMAND ----------


def get_prev(df, code):
    sum = df.filter(F.col("code_sm")==code).count()
    prev = (
        df.filter(F.col("code_sm") == code)
        .groupBy("ccsr_ind_p3").agg(F.count(F.col("medrec_key")).alias("count"))
        .select(
            "*",
            F.lit(code).alias("code"),
            F.round(F.col("count")/F.lit(sum),3).alias("perc")
        )
    )
    return prev


# COMMAND ----------

for i in codes:
    get_prev(ccsr_s, i).show()

# COMMAND ----------

# MAGIC %md
# MAGIC # KM estimates

# COMMAND ----------

def get_sk_sp(df, cov, code):
    df_code = df.filter(F.col("code_sm") == code)
    mg = (
        df_code
        .join(cov, on="medrec_key", how="left")
        .select("ccsr_ind_p3", "ccsr_tt_p3", "covid_icd_lab")
    ).collect()
    out = [[x[0], x[1], x[2]] for x in mg]
    return out

def kpm_plot(cc):
    msk = cc[:,2] == 1
    cc_kp = sks.nonparametric.kaplan_meier_estimator(cc[:,0].astype("bool"), 
                                                        cc[:,1]-30)

    cc_kp_c = sks.nonparametric.kaplan_meier_estimator(cc[:,0].astype("bool")[msk], 
                                                        cc[:,1][msk]-30)
    cc_kp_nc = sks.nonparametric.kaplan_meier_estimator(cc[:,0].astype("bool")[~msk], 
                                                        cc[:,1][~msk]-30)

    plt.step(cc_kp[0], cc_kp[1], label="all")
    plt.step(cc_kp_c[0], cc_kp_c[1], label = "covid")
    plt.step(cc_kp_nc[0], cc_kp_nc[1], label = "non-covid")
    plt.legend()

# COMMAND ----------

cc = np.array(get_sk_sp(ccsr_s, cov, codes[0]))
print(codes[0])
kpm_plot(cc)


# COMMAND ----------

cc = np.array(get_sk_sp(ccsr_s, cov, codes[1]))
print(codes[1])
kpm_plot(cc)

# COMMAND ----------

N = 2
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 3
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 4
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 5
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 6
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 7
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 8
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 9
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])     
kpm_plot(cc)

# COMMAND ----------

N = 10
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 11
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 12
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 13
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 14
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 15
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 16
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 17
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 18
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 19
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)

# COMMAND ----------

N = 20
cc = np.array(get_sk_sp(ccsr_s, cov, codes[N]))
print(codes[N])
kpm_plot(cc)
