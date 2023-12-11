# Databricks notebook source
# GLOBALS
# CODE = "EAR"
# RUN_NAME = f"{CODE}_run_02"
EXP_ID = 502851330942627
TIME_SCALE = 91.5
BALANCE = True
SAMPLE_TRN = 10_000
SAMPLE_TST = 20_000

TREES = 30
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

DRAWS = 300
TUNE = 300
CORES = 8
CHAINS = 8
PDP_ALL = True
WEIGHT = 1
RUN_NUM = 1

# COMMAND ----------

# dbutils.widgets.text("seed", defaultValue = str(np_seed))
# np_seed1 = dbutils.widgets.get("seed")
# if np_seed1 != "na":
#     np_seed = int(np_seed1)

# dbutils.widgets.text("code", defaultValue=str(CODE))
# CODE1 = dbutils.widgets.get("code")
# if CODE1 != "na":
#     CODE = CODE1

# dbutils.widgets.text("run_name", defaultValue=str(RUN_NAME))
# RUN_NAME1 = dbutils.widgets.get("run_name")
# if RUN_NAME1 != "na":
#     RUN_NAME = RUN_NAME1

dbutils.widgets.text("exp_id", defaultValue=str(EXP_ID))
EXP_ID1 = dbutils.widgets.get("exp_id")
if EXP_ID1 != "na":
    EXP_ID = int(EXP_ID1)
dbutils.jobs.taskValues.set(key = 'exp_id', value = EXP_ID)

dbutils.widgets.text("time_scale", defaultValue=str(TIME_SCALE))
TIME_SCALE1 = dbutils.widgets.get("time_scale")
if TIME_SCALE1 != "na":
    TIME_SCALE = float(TIME_SCALE1)
dbutils.jobs.taskValues.set(key = 'time_scale', value = TIME_SCALE)


dbutils.widgets.text("balance", defaultValue=str(BALANCE))
BALANCE1 = dbutils.widgets.get("balance")
if BALANCE1 != "na":
    BALANCE = eval(BALANCE1)
dbutils.jobs.taskValues.set(key = 'balance', value = BALANCE)


dbutils.widgets.text("sample_trn", defaultValue=str(SAMPLE_TRN))
SAMPLE_TRN1 = dbutils.widgets.get("sample_trn")
if SAMPLE_TRN1 != "na":
    SAMPLE_TRN = int(SAMPLE_TRN1)
dbutils.jobs.taskValues.set(key = 'sample_trn', value = SAMPLE_TRN)


dbutils.widgets.text("sample_tst", defaultValue=str(SAMPLE_TST))
SAMPLE_TST1 = dbutils.widgets.get("sample_tst")
if SAMPLE_TST1 != "na":
    SAMPLE_TST = int(SAMPLE_TST1)
dbutils.jobs.taskValues.set(key = 'sample_tst', value = SAMPLE_TRN)


dbutils.widgets.text("trees", defaultValue=str(TREES))
TREES1 = dbutils.widgets.get("trees")
if TREES1 != "na":
    TREES = int(TREES1)
dbutils.jobs.taskValues.set(key = 'trees', value = TREES)


dbutils.widgets.text("split_rules", defaultValue=str(SPLIT_RULES))
SPLIT_RULES1 = dbutils.widgets.get("split_rules")
if SPLIT_RULES1 != "na":
    SPLIT_RULES =  SPLIT_RULES1
dbutils.jobs.taskValues.set(key = 'split_rules', value = SPLIT_RULES)


dbutils.widgets.text("draws", defaultValue=str(DRAWS))
DRAWS1 = dbutils.widgets.get("draws")
if DRAWS1 != "na":
    DRAWS = int(DRAWS1)
dbutils.jobs.taskValues.set(key = 'draws', value = DRAWS)


dbutils.widgets.text("tune", defaultValue=str(TUNE))
TUNE1 = dbutils.widgets.get("tune")
if TUNE1 != "na":
    TUNE = int(TUNE1)
dbutils.jobs.taskValues.set(key = 'tune', value = TUNE)


dbutils.widgets.text("cores", defaultValue=str(CORES))
CORES1 = dbutils.widgets.get("cores")
if CORES1 != "na":
    CORES = int(CORES1)
dbutils.jobs.taskValues.set(key = 'cores', value = CORES)


dbutils.widgets.text("chains", defaultValue=str(CHAINS))
CHAINS1 = dbutils.widgets.get("chains")
if CHAINS1 != "na":
    CHAINS = int(CHAINS1)
dbutils.jobs.taskValues.set(key = 'chain', value = CHAINS)


dbutils.widgets.text("pdp_all", defaultValue=str(PDP_ALL))
PDP_ALL1 = dbutils.widgets.get("pdp_all")
if PDP_ALL1 != "na":
    PDP_ALL = eval(PDP_ALL1)
dbutils.jobs.taskValues.set(key = 'pdp_all', value = PDP_ALL)


dbutils.widgets.text("weight", defaultValue=str(WEIGHT))
WEIGHT1 = dbutils.widgets.get("weight")
if WEIGHT1 != "na":
    WEIGHT = int(WEIGHT1)
dbutils.jobs.taskValues.set(key = 'weight', value = WEIGHT)


dbutils.widgets.text("run_num", defaultValue=str(RUN_NUM))
RUN_NUM1 = dbutils.widgets.get("run_num")
if RUN_NUM1 != "na":
    RUN_NUM = RUN_NUM1
dbutils.jobs.taskValues.set(key = 'run_num', value = RUN_NUM)

