# Databricks notebook source
# MAGIC %md
# MAGIC # Set-up

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.window as W
from pyspark.storagelevel import StorageLevel
# import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt
# import scipy.stats as ana

# from pyspark.sql.functions import col, concat, expr, lit
from util.conditions import *

DISPLAY = False
CACHE = False
SAMPLE = False
CHECK = False
WRITE = True

# COMMAND ----------

# MAGIC %md
# MAGIC # Data

# COMMAND ----------

# Sample down the dataset
patdemo=spark.read.table("cdh_premier_v2.patdemo")
if SAMPLE:
    patdemo2 = (
        patdemo
        .select("medrec_key")
        .distinct()
        .sample(.001)
    )
    patdemo = (
        patdemo2
        .join(patdemo, on="medrec_key", how="left")
    )
    patdemo.write.mode("overwrite").format("delta").saveAsTable("cdh_premier_exploratory.twj8_patdemo_sample_01")
    patdemo = spark.table("cdh_premier_exploratory.twj8_patdemo_sample_01")
    patdemo.count()

paticd=spark.read.table("cdh_premier_v2.paticd_diag")
if SAMPLE:
    paticd = (
        patdemo
        .select("pat_key")
        .join(paticd, on="pat_key", how="inner")
    )

patcpt=spark.read.table("cdh_premier_v2.patcpt")
if SAMPLE:
    patcpt = (
        patdemo
        .select("pat_key")
        .join(patcpt, on="pat_key", how = "inner")
    )

patlabres=spark.read.table("cdh_premier_v2.lab_res")
if SAMPLE:
    patlabres = (
        patdemo
        .select("pat_key")
        .join(patlabres, on = "pat_key", how = "inner")
    )

genlab=spark.read.table("cdh_premier_v2.genlab")
if SAMPLE:
    genlab = (
        patdemo
        .select("pat_key")
        .join(genlab, on = "pat_key", how = "inner")
    )

patbill=spark.read.table("cdh_premier_v2.patbill")
if SAMPLE:
    patbill = (
        patdemo
        .select("pat_key")
        .join(patbill, on = "pat_key", how = "inner")
    )


# Look-up tables
chgmstr=spark.read.table("cdh_premier_v2.chgmstr")
hospchg = spark.table("cdh_premier_v2.hospchg")
disstat=spark.read.table("cdh_premier_v2.disstat")
pattype=spark.read.table("cdh_premier_v2.pattype")
providers=spark.read.table("cdh_premier_v2.providers")
zip3=spark.read.table("cdh_premier_v2.prov_digit_zip")
prov=providers.join(zip3,"prov_id","inner")
payor = spark.table("cdh_premier_v2.payor")
icd_desc=spark.read.table("cdh_premier_v2.icdcode")
cptcode = spark.table("cdh_premier_v2.cptcode")

# CCSR Table
ccsr_lookup = spark.table("cdh_reference_data.icd10cm_diagnosis_codes_to_ccsr_categories_map")


# if SAMPLE:
#     # patdemo.persist(StorageLevel.DISK_ONLY)
#     # patdemo.count()
#     paticd.persist(StorageLevel.DISK_ONLY)
#     print(paticd.count())
#     patcpt.persist(StorageLevel.DISK_ONLY)
#     print(patcpt.count())
#     patlabres.persist(StorageLevel.DISK_ONLY)
#     print(patlabres.count())
#     genlab.persist(StorageLevel.DISK_ONLY)
#     print(genlab.count())
#     patbill.persist(StorageLevel.DISK_ONLY)
#     print(patbill.count())

# COMMAND ----------

# MAGIC %md # Cohort

# COMMAND ----------

# MAGIC %md
# MAGIC ## General Range Inclusion

# COMMAND ----------

# identify patients with a record in the time range
inc_range = (patdemo
 .select("pat_key", "medrec_key", "admit_date", "discharge_date")
 .filter((F.col("admit_date") > "2020-03-01") &
         (F.col("admit_date") < "2021-11-01")
         )
 .withColumn("inc_range", F.lit(1))
 # ensure encounters are of an appropriate type
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Covid Patients

# COMMAND ----------

covid_icd = (
    paticd
    .select("pat_key", "icd_code")
    .withColumn("icd_code", F.regexp_replace(F.col("icd_code"), "\.", ""))
    .withColumn("covid_icd", F.when(F.col("icd_code").isin(["U071", "B9729"]), 1).otherwise(0))
    .filter(F.col("covid_icd")== 1)
)


######################## Labs
EXCLUSION_LAB = [
    "SARS coronavirus 2 RNA:PrThr:Pt:Respiratory:Ord:Probe.amp.tar",
    "SARS coronavirus 2 RNA:PrThr:Pt:XXX:Ord:Probe.amp.tar"
]
    
covid_lab = (
    patlabres
    .select("pat_key", "lab_test_desc", "observation")
    .withColumn("covid_lab",
        F.when(
            (F.col("lab_test_desc").isin(EXCLUSION_LAB))
            &
            (F.col("observation")=="positive")
        ,1).otherwise(0)
    )
    .filter(F.col("covid_lab") == 1)
)

# join covid icd and covid lab
# filter the covid encounter
# identify the first covid encounter
# filter the first covid encounter
pat_covid_tf = (
    inc_range
    .join(covid_icd, on = "pat_key", how = "left")
    .join(covid_lab, on = "pat_key", how = "left")
    .withColumn("covid_icd_lab", F.when(
        (F.col("covid_icd") == 1) | (F.col("covid_lab") == 1)
        ,1).otherwise(0)
    )
    .filter(F.col("covid_icd_lab") == 1)
    .withColumn("covid_min_date",
                F.min(F.col("admit_date")).over(W.Window.partitionBy(F.col("medrec_key")))
                )
    .withColumn("is_covid_min_date",
                F.when(F.col("admit_date") == F.col("covid_min_date"), 1).otherwise(0)
                )
    .filter(F.col("is_covid_min_date") == 1)
    .distinct()
)

TABLE = "cdh_premier_exploratory.twj8_pat_covid_tf_f_02"
if WRITE:
    (
        pat_covid_tf
        .write
        .mode("overwrite")
        .format("delta")
        .saveAsTable(TABLE)
    )
    pat_covid_tf = spark.table(TABLE)
    pat_covid_tf.count()


# check that there is just 1 date per patient
# if CHECK:
#     (
#         pat_covid_tf 
#         .select("medrec_key", "admit_date")
#         .distinct()
#         .withColumn("n", F.count(F.col("medrec_key")).over(W.Window.partitionBy(F.col("medrec_key"))))
#         .filter(F.col("n") > 1)
#     ).display()

# no results returned so we know we have a dataset where the first covid date has been identified

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Non Covid Patients

# COMMAND ----------

# get distinct covid patients
pat_cov_tmp = (
    pat_covid_tf
    .select("medrec_key", "covid_icd_lab")
    .distinct()
)

# drop the covid patients
pat_noncovid_enc = (
    inc_range
    .join(pat_cov_tmp, how="left", on="medrec_key")
    .filter(F.col("covid_icd_lab").isNull())
)

# rejoin the patdemo data
pat_noncov_temp = (
    pat_noncovid_enc
    .select("pat_key")
    .join(patdemo, on="pat_key", how="left")
)

# select only select visit-type encountors for the random encounter set
pat_noncov_typeres = (
    pat_noncov_temp
    .filter(F.col("pat_type").isin([28,8,90,34,29]))
)

# Select non-covid encounters
pat_noncov_rand_evnt = (
    pat_noncov_typeres
    .withColumn("rand_n", 
        F.row_number()
        .over(
            W.Window
            .partitionBy('medrec_key')
            .orderBy(F.rand(99))
        )
    )  
    .withColumn("n", F.count(F.col("medrec_key")).over(W.Window.partitionBy("medrec_key")))
    .select("pat_key", "medrec_key", "admit_date", "rand_n",'n')
    .filter(F.col("rand_n") == 1)
)

if WRITE:
    TABLE = "cdh_premier_exploratory.twj8_pat_noncovid_f_03"
    pat_noncov_rand_evnt.write.mode("overwrite").format("delta").saveAsTable(TABLE)
    pat_noncov_rand_evnt = spark.table(TABLE)
    pat_noncov_rand_evnt.count()

# COMMAND ----------

# check that covid and noncovid are exclusive sets
# CHECK=True
if CHECK:
    cov_medrec = [r[0] for r in pat_covid_tf.select("medrec_key").collect()]
    noncov_medrec = [r[0] for r in pat_noncov_rand_evnt.select("medrec_key").collect()]
    print(any(check in cov_medrec for check in noncov_medrec))

if CHECK:
    print(pat_covid_tf.select("medrec_key").distinct().count())
    n_noncov = pat_noncov_rand_evnt.count()
    n_noncov_dist = pat_noncov_rand_evnt.select("medrec_key").distinct().count()
    print(n_noncov)
    print(n_noncov_dist)

# COMMAND ----------

# join the covid and noncovid patients
pat_cov_noncov = (
    pat_covid_tf.select("medrec_key", "admit_date", "covid_icd_lab").distinct()
    .union(
        pat_noncov_rand_evnt.select("medrec_key","admit_date").distinct().withColumn("covid_icd_lab", F.lit(0))
    )
)

# rejoin the patdemo data
pat_cov_noncov_dt = (
    pat_cov_noncov
    .join(patdemo, on = ["medrec_key", "admit_date"], how="left")
)

if WRITE:
    TABLE = "cdh_premier_exploratory.twj8_pat_cov_noncov_f_04"
    pat_cov_noncov_dt.write.mode("overwrite").format("delta").saveAsTable(TABLE)
    pat_cov_noncov_dt = spark.table(TABLE)
    pat_cov_noncov_dt.count()

# COMMAND ----------

if CHECK:
    check1 = pat_cov_noncov.count()
    check2 = pat_cov_noncov.select("medrec_key").distinct().count()
    check3 = pat_covid_tf.select("medrec_key").distinct().count()
    check4 = pat_noncov_rand_evnt.select("medrec_key").distinct().count()
    print(check1)
    print(check2)
    print(check3)
    print(check4)
    print(check3 + check4)

if CHECK:
    check = pat_cov_noncov_dt.count()
    print(check)

# COMMAND ----------

# MAGIC %md
# MAGIC # Exclusion

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Get Periods

# COMMAND ----------

# Get periods
pat_period = (
    pat_cov_noncov_dt
    .withColumn("init_date", F.col("admit_date"))
    .select("medrec_key", "init_date")
    .distinct()
    .withColumn("p1_str", F.col("init_date") - 365)
    .withColumn("p1_end", F.col("init_date") - 8)
    .withColumn("p2_str", F.col("init_date") - 7)
    .withColumn("p2_end", F.col("init_date") + 30)
    .withColumn("p3_str", F.col("init_date") + 31)
    .withColumn("p3_end", F.col("init_date") + 395)
)

# Get encounters in the periods
pat_period2 = (
    pat_period
    .join(patdemo.select("medrec_key", "pat_key", "admit_date")
          , on = "medrec_key", how = "left")
    .withColumn("p1", 
        F.when(
            (F.col("admit_date") >= F.col("p1_str")) & (F.col("admit_date") <= F.col("p1_end"))
            ,1
        ).otherwise(0)   
    )
    .withColumn("p2", 
        F.when(
            (F.col("admit_date") >= F.col("p2_str")) & (F.col("admit_date") <= F.col("p2_end"))
            ,1
        ).otherwise(0)   
    )
    .withColumn("p3", 
        F.when(
            (F.col("admit_date") >= F.col("p3_str")) & (F.col("admit_date") <= F.col("p3_end"))
            ,1
        ).otherwise(0)   
    )
    .withColumn(
        "p1_sum",
        F.sum(F.col("p1")).over(W.Window.partitionBy("medrec_key"))
    )
    .withColumn(
        "p2_sum",
        F.sum(F.col("p2")).over(W.Window.partitionBy("medrec_key"))
    )
    .withColumn(
        "p3_sum",
        F.sum(F.col("p3")).over(W.Window.partitionBy("medrec_key"))
    )
)  

if CHECK:
    print(pat_period2.count())
    print(pat_period2.select("medrec_key").distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Period Exclusions

# COMMAND ----------

# drop patients that have no p1 obs
# drop patients that have no p3 obs
pat_period3 = (
    pat_period2
    .filter(F.col("p1_sum") != 0)
    .filter(F.col("p3_sum") != 0)
)
# Remove the encounters not in p1, p2, or p3
pat_period4 = (
    pat_period3
    .filter(
        ~((F.col("p1")==0) & (F.col("p2")==0) & (F.col("p3")==0))
    )
)

if CHECK:
    print(pat_period3.select("medrec_key").distinct().count())
    print(pat_period3.count())
    print(pat_period4.count())
    print(pat_period4.select("medrec_key").distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## NonCovid Exclusion because of later Covid
# MAGIC

# COMMAND ----------

# join the covid indicator to get just the noncovid patients
# the join to pat_period4 should give
exc_base = (
    pat_cov_noncov_dt
    .select("medrec_key","covid_icd_lab")
    .filter(F.col("covid_icd_lab")==0)
    .join(pat_period4, on="medrec_key", how="inner")
)


if CHECK:
    print(exc_base.select("medrec_key").distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### exclusion meds

# COMMAND ----------

# get codes for meds
DRUGS = ["PAXLOVID", "MOLNUPIR","EVUSHELD","TIXAGEVIMAB","CILGAVIMAB","BEBTELOVIMA","SOTROVIMAB","BAMLANIVIMAB","ETESEVIMAB","CASIRIVIMAB","IMDEVIMAB","DEXAMETHASONE","TOFACITINIB", "TOCILIZUMAB","SARILUMAB","BARICITINIB","REMDESIVIR"]
expression = "+".join(DRUGS)

# hospchg drugs
med_search1 = (
    hospchg
    .select("*", *[F.col("hosp_chg_desc").contains(x).alias(x).cast("int") for x in DRUGS])
    .withColumn("ismed1", F.expr(expression))
    .filter(F.col("ismed1") > 0)
    .drop(*DRUGS)
    .distinct()
)

# cpt drugs
med_search2 = (
    cptcode
    .select("*", *[F.col("cpt_desc").contains(x).alias(x).cast("int") for x in DRUGS])
    .withColumn("ismed2", F.expr(expression))
    .filter(F.col("ismed2")>0)
    .drop(*DRUGS)
    .distinct()
)

# std_chg drugs
med_search3 = (
    chgmstr
    .select("std_chg_code", "std_chg_desc")
    .select("*", *[(F.col("std_chg_desc")).contains(x).alias(x).cast("int") for x in DRUGS])
    .withColumn("ismed3", F.expr(expression))
    .filter(F.col("ismed3")>0)
    .drop(*DRUGS)
    # .select("hosp_chg_id", "hosp_chg_desc", "ismed3")
    # .select("*", F.lit(1).alias("drug_ind3"))
)

##########################################################################################################
# get patbill exclusion drugs
drug_chg_excl = (
    patbill
    .select("pat_key", "std_chg_code", "hosp_chg_id")
    .distinct()
    .join(med_search1, on ="hosp_chg_id", how = "left")
    .join(med_search3, on = "std_chg_code", how = "left")
    .filter((F.col("ismed1")==1) |( F.col("ismed3")==1))
)

# merge to exc_base to identify the patients to exclude
# exc_base has all the pat_keys in the time range, so we want to identify the patients to exclude based on the pat_keys in the ranges having these codes.
excl1 = (
    exc_base
    .select("pat_key", "medrec_key")
    .join(drug_chg_excl, on="pat_key", how="inner")
)

# cpt drug exclusion
drug_cpt_excl = (
    patcpt 
    .select("pat_key", "cpt_code")
    .join(med_search2, on = "cpt_code", how="left")
    .filter(F.col("ismed2") == 1)
)

# merge to exc_base to identify patients to be excluded
excl2 = (
    exc_base
    .select("pat_key", "medrec_key")
    .join(drug_cpt_excl, on = "pat_key", how="inner")
)

if CHECK:
    print(excl1.select("medrec_key").distinct().count())
    print(excl2.select("medrec_key").distinct().count())


# COMMAND ----------

# MAGIC %md
# MAGIC ### exclusion icd 

# COMMAND ----------

# get the covid icd exclusions
EXCLUSION_DIAG_CODES = [
        "U071",
        "U072",
        "U099",
        "B342",
        "B9721",
        "B9729",
        "J1281",
        "J1282",
        "M3581",
        "Z8616",
        "B948",
        "B949"   
    ]
icd_excl = (
    paticd
    .select("pat_key", "icd_code")
    .withColumn("isicd", F.translate(F.col("icd_code"),".","").isin(EXCLUSION_DIAG_CODES).cast("int"))
    .filter(F.col("isicd") == 1)
)

excl3 = (
    exc_base
    .select("pat_key", "medrec_key", "admit_date", "init_date", "covid_icd_lab", "p1","p2","p3")
    .join(icd_excl, on = "pat_key", how="inner")
)

# excl3.display()
# any of these patients that have a covid dx are covid following the inclusion period or using a covid code not used in the inclusion set

# COMMAND ----------

# MAGIC %md
# MAGIC ### exclusion labs

# COMMAND ----------


EXCLUSION_LAB = [
    "SARS coronavirus 2 RNA:PrThr:Pt:Respiratory:Ord:Probe.amp.tar",
    "SARS coronavirus 2 RNA:PrThr:Pt:XXX:Ord:Probe.amp.tar"
]

lab_excl = (
    patlabres
    .select("pat_key", "lab_test_desc", "observation")
    .filter(
        (F.col("lab_test_desc").isin(EXCLUSION_LAB)) 
        & 
        (F.col("observation") == "positive")
    )
    .select("pat_key", F.lit("lab").alias("islab"))
)

excl4 = (
    exc_base
    .select("pat_key", "medrec_key", "admit_date", "init_date", "covid_icd_lab", "p1","p2","p3")
    .join(lab_excl, on = "pat_key", how="inner")
)



# COMMAND ----------

# MAGIC %md
# MAGIC ### exclusion cpt

# COMMAND ----------

EXCLUSION_PROCEDURE_CODES = [
    #unchanged from HV
    "Q0240",
    "Q0243",
    "Q0244",
    "M0240",
    "M0241",
    "M0243",
    "M0244",
    "Q0245",
    "M0245",
    "M0246",
    "Q0247",
    "M0247",
    "M0248",
    "Q0222",
    "M0222",
    "M0223",
    "J0248"
]
cpt_excl = (
    patcpt
    .select("pat_key","cpt_code")
    .filter(
        F.col("cpt_code").isin(EXCLUSION_PROCEDURE_CODES)
    )
    .select("pat_key", F.lit("cpt").alias("iscpt"))
)    

excl5 = (
    exc_base
    .select("pat_key", "medrec_key", "admit_date", "init_date", "covid_icd_lab", "p1","p2","p3")
    .join(cpt_excl, on = "pat_key", how="inner")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### all exclusions medrecs 

# COMMAND ----------

# join all the exclusion
exclusions = (
    excl1.select("medrec_key").distinct()
    .union(
        excl2.select("medrec_key").distinct()
    )
    .union(
        excl3.select("medrec_key").distinct()
    )
    .union(
        excl4.select("medrec_key").distinct()
    )
    .union(
        excl5.select("medrec_key").distinct()
    )
    .distinct()
    .select("*", F.lit(1).alias("exclude"))
)

# join exclusions to the pat_cov_noncov dataset
pat_cov_noncov_2 = (
    pat_period4
    .join(exclusions, on="medrec_key", how="left")
    .join(pat_cov_noncov.select("medrec_key", "covid_icd_lab"), on = "medrec_key", how="left")
)

if CHECK:
    pat_cov_noncov_2.groupby(["covid_icd_lab", "exclude"]).count().display()
    (
        pat_cov_noncov_2
        .select("medrec_key", "covid_icd_lab", "exclude")
        .distinct()
        .groupby(["covid_icd_lab", "exclude"])
        .count()
        .display()
    )

# remove the exclusion patients
pat_cov_noncov_3 = (
    pat_cov_noncov_2
    .filter(F.col("exclude").isNull())
)

if WRITE:
    TABLE = "cdh_premier_exploratory.twj8_pat_cov_noncov_excl_f_04"
    pat_cov_noncov_3.write.mode("overwrite").format("delta").saveAsTable(TABLE)
    pat_cov_noncov_3 = spark.table(TABLE)
    pat_cov_noncov_3.count()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Demographic Exclusions

# COMMAND ----------

# exclude on age and gender
# inclusions identified by the full pat_cov_noncov_dt and then used in reducing the inclusion from other exclusions

dem_inc = (
    pat_cov_noncov_dt
    .filter(
        (F.col("age") >= 18)
        &
        (F.col("age") < 65)
    )
    .filter(
        (F.col("gender").isin(["M","F"]))
    )
    .select("medrec_key")
    .distinct()
) 

if CHECK:
    print(dem_inc.count())
    print(dem_inc.select("medrec_key").distinct().count())

############################################################################
# apply demographic exclusions
pat_cov_noncov_4 = (
    pat_cov_noncov_3
    .join(dem_inc, on="medrec_key", how="inner")
)

if CHECK:
    print(pat_cov_noncov_4.select("medrec_key").distinct().count())

if CHECK:
    pat_cov_noncov_4.select("medrec_key","covid_icd_lab").distinct().groupby("covid_icd_lab").count().display()
###############################################################################
if WRITE:
    table = "cdh_premier_exploratory.twj8_pat_cov_noncov_final_f_05"
    pat_cov_noncov_4.write.mode("overwrite").format("delta").saveAsTable(table)
    pat_cov_noncov_4 = spark.table(table)
    pat_cov_noncov_4.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # CCSR

# COMMAND ----------

# Join the icd codes and corresponding ccsrs to each observation
ccsr1 = (
    pat_cov_noncov_4
    .join(
        paticd
        .select("pat_key", "icd_code")
        .withColumn("icd_code", F.translate(F.col("icd_code"), ".", ""))
        ,on = "pat_key"
        ,how = "left"
    )
    .join(
        # ccsr_code
        ccsr_lookup
        .withColumnRenamed("icd-10-cm_code", "icd_code")
        .select("icd_code", "ccsr_category")
        ,on = "icd_code"
        ,how = "left"
    )
)

####################################################################################
# Prepare CCSR code sets
ccsr_codes = (
    ccsr_lookup
    .select("ccsr_category")
    .distinct()
)


# COMMAND ----------

# Use ccsr_codes as a frame to merge all observations
# identify observations with ccsr code
# subset p1,p2,p3 and get ind for each ccsr code and min time
cc1 = (
    ccsr_codes
    .select(F.col("ccsr_category").alias("code"))
    .join(
        ccsr1
        # .limit(10)
    )
    .select(
        "*",
        F.when(
            F.col("code") == F.col("ccsr_category"),
            1
        )
        .otherwise(0)
        .alias("ccsr_ind"),
        F.when(
            F.col("code") == F.col("ccsr_category"),
            F.datediff(F.col("admit_date"),F.col("init_date"))
        )
        .otherwise(
            F.datediff(F.col("p3_end"),F.col("init_date"))
        )
        .alias("ccsr_tt")
    )
)

# Get the indicator for each period
cc2_p3 = (
    cc1
    .filter(F.col("p3") == 1)
    .groupBy(["medrec_key", "code"])
    .agg(
        F.max(F.col("ccsr_ind")).alias("ccsr_ind_p3"),
        F.min(F.col("ccsr_tt")).alias("ccsr_tt_p3")
    )
)
cc2_p2 = (
    cc1
    .filter(F.col("p2") == 1)
    .groupBy(["medrec_key", "code"])
    .agg(
        F.max(F.col("ccsr_ind")).alias("ccsr_ind_p2"),
    )
)
cc2_p1 = (
    cc1
    .filter(F.col("p1") == 1)
    .groupBy(["medrec_key", "code"])
    .agg(
        F.max(F.col("ccsr_ind")).alias("ccsr_ind_p1"),
    )
)

# join datasets
cc3_long = (
    cc2_p3
    .join(cc2_p2, on = ["medrec_key","code"], how = "left")
    .join(cc2_p1, on = ["medrec_key","code"], how = "left")
)

# get the short code dataset
cc3_short = (
    cc3_long
    .withColumn("code_sm", F.substring(F.col("code"), 0,3))
    .groupBy(["medrec_key", "code_sm"])
    .agg(
        F.max(F.col("ccsr_ind_p3")).alias("ccsr_ind_p3"),
        F.min(F.col("ccsr_tt_p3")).alias("ccsr_tt_p3"),
        F.max(F.col("ccsr_ind_p2")).alias("ccsr_ind_p2"),
        F.max(F.col("ccsr_ind_p1")).alias("ccsr_ind_p1"),
    )
)

if CHECK:
    cc3_short.groupby("code_sm").count().display()
    cc3_long.groupby("code").count().display()


if WRITE:
    table = "cdh_premier_exploratory.twj8_pat_ccsr_long_f_06"
    cc3_long.write.mode("overwrite").format("delta").saveAsTable(table)
    cc3_long = spark.table(table)
    cc3_long.count()
    table = "cdh_premier_exploratory.twj8_pat_ccsr_short_f_06"
    cc3_short.write.mode("overwrite").format("delta").saveAsTable(table)
    cc3_short = spark.table(table)
    cc3_short.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Covariate 
# MAGIC

# COMMAND ----------

# Co
cov1 = (
    pat_cov_noncov_4
    .select("medrec_key","init_date", "p1_sum", "p2_sum", "p3_sum", "covid_icd_lab")
    .distinct()
    .select(
        "*",
        F.date_format(F.col("init_date"), "yyyyMM").alias("init_date_ym")
    )
    .drop("init_date")
)

cov2 = (
    pat_cov_noncov_4
    .select("medrec_key", "init_date")
    .distinct()
    .join(
        patdemo.select("*", F.col("admit_date").alias("init_date")) , 
        on=["medrec_key", "init_date"], 
        how = "left"
    )
    .select("medrec_key", "i_o_ind", "pat_type", "ms_drg", "std_payor", "los", "age", "race", "hispanic_ind")
    .distinct()
)
cov2.count()

# simple fix for multiple obs on the index date is to randomly take the values from one of them
# a more holistic fix will require determ
cov3 = (
    cov2
    .select(
        "*",
        # F.count(F.col("medrec_key")).over(W.Window.partitionBy("medrec_key")).alias("c"),
        # F.random.over(W.Window.partitionBy("medrec_key")).alias("c") 
        F.row_number().over(W.Window.partitionBy('medrec_key').orderBy(F.rand(99))).alias("r")
    )
    .filter(F.col("r")==1)
    .drop("r")
)


# COMMAND ----------

# Recoder
def get_cat_recode(df, var):
    val = (
        df
        .groupby(var).count()
        .withColumn("temp", F.lit(1))
        .withColumn("val", F.row_number().over(W.Window.partitionBy("temp").orderBy(F.col("count").desc())))
        .drop("temp")
    )

    out = (
        df
        .join(val.select(var, "val"), on=var, how="left")
        .drop(var)
        .select("*", F.col("val").alias(var))
        .drop("val")
    )

    val = (
        val
        .withColumn("var", F.lit(var))
        .withColumnRenamed(var, "val_og")
    )
    return out, val

cov4, race_dict = get_cat_recode(cov3, "race")
cov5, ethn_dict = get_cat_recode(cov4, "hispanic_ind")
cov6, i_o_dict = get_cat_recode(cov5, "i_o_ind")

# COMMAND ----------

vdict = race_dict.union(ethn_dict).union(i_o_dict)
if WRITE:
    table = "cdh_premier_exploratory.twj8_pat_var_dict_06"
    vdict.write.mode("overwrite").format("delta").saveAsTable(table)

# COMMAND ----------

cov_final = (
    cov1
    .join(cov6, on = "medrec_key", how="left")
)

if CHECK:
    print(cov_final.count())
    print(cov_final.select("medrec_key").distinct().count())

if WRITE:
    table = "cdh_premier_exploratory.twj8_pat_covariates_f_07"
    cov_final.write.mode("overwrite").option("overwriteSchema","true").format("delta").saveAsTable(table)
    cov_final = spark.table(table)
    cov_final.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # FINAL
# MAGIC - Run notebook in next notebook
# MAGIC - Need:
# MAGIC   -   twj8_pat_covariates_06
# MAGIC   -   twj8_pat_ccsr_short_06
# MAGIC   -   twj8_pat_ccsr_long_06
# MAGIC   -   twj8_pat_var_dict_06
