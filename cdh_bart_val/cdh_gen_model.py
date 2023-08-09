# Databricks notebook source
import sksurv
import sksurv.linear_model
import sksurv.ensemble
import pymc as pm
import pymc_bart as pmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import sklearn as skl
import scipy.stats as sp
import simsurv_func as ssf
import mlflow
import pyspark

# COMMAND ----------

import importlib
importlib.reload(ssf)

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages("mlflow")
# MAGIC install.packages("SparkR")

# COMMAND ----------

# MAGIC %r
# MAGIC library(mlflow)
# MAGIC library(SparkR)
# MAGIC library(BART)

# COMMAND ----------

plt.ioff()


# COMMAND ----------

experiment_id=dbutils.jobs.taskValues.get("cdh-ml-init",
                                          "experiment_id",
                                          debugValue=2256023545555400)

run_name = dbutils.jobs.taskValues.get("cdh-ml-init", 
                                         "run_name", 
                                         debugValue="test2")

run_id = dbutils.jobs.taskValues.get("cdh-ml-run",
                                    "run_id_main",
                                    debugValue = "5c4b0bab2668466ea9ac022e482adc35")

ALPHA = dbutils.jobs.taskValues.get("cdh-ml-init",
                            "alpha", 
                            debugValue=3)
ALPHA_F = dbutils.jobs.taskValues.get("cdh-ml-init",
                            "alpha_f",
                            debugValue= None)
LAMBDA = dbutils.jobs.taskValues.get("cdh-ml-init", 
                            "lambda", 
                            debugValue="np.exp(2 + 0.4*(x_mat[:,0] + x_mat[:,1]))")
N = dbutils.jobs.taskValues.get("cdh-ml-init",
                            "n", 
                            debugValue=100)
X_VARS = dbutils.jobs.taskValues.get("cdh-ml-init",
                            "x_vars", 
                            debugValue=2)
CENS_IND = dbutils.jobs.taskValues.get("cdh-ml-init",
                            "cens_ind", 
                            debugValue=False)
CENS_SCALE = dbutils.jobs.taskValues.get("cdh-ml-init",
                            "cens_scale", 
                            debugValue=60)

# COMMAND ----------

dbutils.widgets.text("sub_run_name", "sub_run")
sub_run_name = dbutils.widgets.get("sub_run_name")
# dbutils.jobs.taskValues.set("sub_run_name", sub_run_name)
# sub_run_name = dbutils.jobs.taskValues.get("cdh-gen-model","sub_run_name")
print(sub_run_name)

dbutils.widgets.text("seed", "99")
seed = dbutils.widgets.get("seed")
# dbutils.jobs.taskValues.set("seed", seed)
# seed = dbutils.jobs.taskValues.get("cdh-gen-model", "seed", debugValue="99")
print(int(seed))
np.random.seed(int(seed))

# COMMAND ----------

mlflow.set_experiment(experiment_id=experiment_id)

# COMMAND ----------

# # run_info = mlflow.active_run()
# # run_id = run_info.info.run_id
# OUTPUTS = "outputs"
# ALPHA = 3
# # ALPHA_F = "1 + (1.5 * x_mat[:,0]) + x_mat[:,1]"
# LAMBDA = "np.exp(2 + 0.4*(x_mat[:,0] + x_mat[:,1]))"
# # LAMBDA = "np.exp(1 + .2*x_mat[:,0] + .3*x_mat[:,1] + 0.8*np.sin(x_mat[:,0] * x_mat[:,1]) + np.power((x_mat[:,2] - 0.5),2))"
# TRAIN_CSV = "outputs/train.csv"
# RBART_CSV = "outputs/rbart_surv.csv"
# N = 100
# X_VARS = 2
# CENS_IND = False
# CENS_SCALE = 60

# COMMAND ----------

# MAGIC %md 
# MAGIC # Data Gen

# COMMAND ----------

with mlflow.start_run(experiment_id=experiment_id, run_id = run_id) as run:
    ###########################################################################
    with mlflow.start_run(experiment_id=experiment_id, run_name = sub_run_name, nested=True) as sub:
        sub_run_id = sub.info.run_id
        # Simulate data
        sv_mat, x_mat, lmbda, a, tlat, cens, t_event, status, T = ssf.sim_surv(
                        N=N, 
                        # T=T,
                        x_vars=X_VARS,
                        a = ALPHA,
                        # alpha_f = ALPHA_F,
                        lambda_f = LAMBDA,
                        cens_scale=CENS_SCALE,
                        cens_ind = CENS_IND,
                        err_ind = False)
        # log param alpha
        mlflow.log_param("alpha", ALPHA)
        # log param labmda
        mlflow.log_param("lambda", LAMBDA)
        # log param N
        mlflow.log_param("N", N)
        # log param T (# timepoint probabilites generated)
        mlflow.log_param("T", T)
        # log param X_VARS
        mlflow.log_param("X_VARS", X_VARS)
        # log parm CENS_SCALE
        mlflow.log_param("CENS_SCALE", CENS_SCALE)
        # log parm CENS_IND
        mlflow.log_param("CENS_IND", CENS_IND)
        # log param x_info
        x_out, x_idx, x_cnt = ssf.get_x_info(x_mat)
        try:
            mlflow.log_param("X_INFO", str(list(zip(x_out, x_cnt))))
        except:
            print("error")
        # mlflow.log_dict("X_INFO", dict(zip(x_out, x_cnt)))

        # log metric cen percent calculated
        # log metric status event calculated
        event_calc, cens_calc = ssf.get_status_perc(status)
        mlflow.log_metric("EVENT_PERC", event_calc)
        mlflow.log_metric("CENS_PERC", cens_calc)

        # log metric t_event mean
        # log metric t_event max
        t_mean, t_max = ssf.get_event_time_metric(t_event)
        mlflow.log_metric("T_EVENT_MEAN", t_mean)
        mlflow.log_metric("T_EVENT_MAX", t_max)


        ########################################

        # log artif train dataset
        train = ssf.get_train_matrix(x_mat, t_event, status)
        # mlflow.log_table(train.to_dict(), "train")
        temp = spark.createDataFrame(pd.DataFrame(train))
        pyspark.sql.DataFrame.createOrReplaceTempView(temp, "test")

        # log artif plot curves
        title = "actual_survival"
        fig = ssf.plot_sv2(x_mat, sv_mat, T, title=title, save = False)
        # mlflow.log_artifact(f"{OUTPUTS}/{title}.png")
        mlflow.log_figure(fig, f"{title}.png")
        
        # get sklearn components
        y_sk = ssf.get_y_sklearn(status, t_event)
        x_sk = train.iloc[:,2:]


# COMMAND ----------

# MAGIC %md
# MAGIC # COX, RSF and PYMC-BART

# COMMAND ----------

with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:
    with mlflow.start_run(experiment_id=experiment_id, run_id = sub_run_id, nested=True) as sub:
        ###########################################################################
        # model cph
        cph = sksurv.linear_model.CoxPHSurvivalAnalysis()
        cph.fit(x_sk, y_sk)
        # log metri coeff
        for i in np.arange(len(cph.coef_)):
            mlflow.log_metric(f"cph_coef_{i}", cph.coef_[i])
            # log metri exp(coef)
            mlflow.log_metric(f"cph_exp_coef_{i}", np.exp(cph.coef_[i]))
        # predic cph
        cph_surv = cph.predict_survival_function(pd.DataFrame(x_out))

        # get plotable data
        # cph_sv_val = [sf(np.arange(T)) for sf in cph_surv]
        cph_sv_t = cph_surv[0].x
        cph_sv_val = [sf(cph_sv_t) for sf in cph_surv]
        cph_sv_t = np.concatenate([np.array([0]), cph_sv_t])
        cph_sv_val = [np.concatenate([np.array([1]), sv]) for sv in cph_sv_val]

        # log artif plot curves
        title = "cph_surv_pred"
        fig = ssf.plot_sv2(x_mat, cph_sv_val, t=cph_sv_t, title = title, save=False)
        mlflow.log_figure(fig, f"{title}.png")
        # mlflow.log_artifact(f"outputs/{title}.png")
        # log model cph
        # idk how to do

        ###################################################################################
        #  model rsf
        rsf = sksurv.ensemble.RandomSurvivalForest(
            n_estimators=1000, min_samples_split=0.5, min_samples_leaf=15, n_jobs=-1, random_state=20
        )
        rsf.fit(x_sk, y_sk)
        # predict rsf
        rsf_surv = rsf.predict_survival_function(pd.DataFrame(x_out))
        # get plotable predictions
        # rsf_sv_val = [sf(np.arange(T)) for sf in rsf_surv]
        rsf_sv_t = rsf_surv[0].x
        rsf_sv_val = [sf(rsf_sv_t) for sf in rsf_surv]
        rsf_sv_t = np.concatenate([np.array([0]), rsf_sv_t])
        rsf_sv_val = [np.concatenate([np.array([1]), sv]) for sv in rsf_sv_val]
        # log artif plot curves
        title = "rsf_surv_pred"
        fig = ssf.plot_sv2(x_mat, rsf_sv_val, t=rsf_sv_t, title=title, save=False)
        mlflow.log_figure(fig, f"{title}.png")
        # ml.log_artifact(f"outputs/{title}.png")
        # log model resf

        ################################################################################
        # BART
        M = 50 # number of trees
        DRAWS = 400
        TUNE = 400
        CORES = 4
        mlflow.log_param("n_tree", M)
        mlflow.log_param("draws", DRAWS)
        mlflow.log_param("tune", TUNE)

        # tranform data long-form
        b_tr_t, b_tr_delta, b_tr_x = ssf.surv_pre_train2(x_sk, y_sk)
        # b_te_t, b_te_x = surv_pre_test(x_sk, y_sk)
        b_te_x = ssf.get_bart_test(x_out, np.unique(b_tr_t))
        off = sp.norm.ppf(np.mean(b_tr_delta))
        # model bart

        with pm.Model() as bart:
            x_data = pm.MutableData("x", b_tr_x)
            f = pmb.BART("f", X=x_data, Y=b_tr_delta, m=M)
            z = pm.Deterministic("z", f + off)
            mu = pm.Deterministic("mu", pm.math.invprobit(z))
            y_pred = pm.Bernoulli("y_pred", p=mu, observed=b_tr_delta, shape=x_data.shape[0])
            bdata = pm.sample(random_seed=2, draws=DRAWS, tune = TUNE, cores=CORES)

        with bart:
        # pm.set_data({"x":pd.DataFrame(test_x), "off":off_test})
            pm.set_data({"x":pd.DataFrame(b_te_x)})
            pp = pm.sample_posterior_predictive(bdata, var_names = ["y_pred", "f", "z", "mu"])
        
        # transform to survival
        bart_sv_fx = ssf.get_sv_fx(pp, x_out)
        # bart_svt
        bart_sv_t = np.unique(b_tr_t)

        # add a time 0 with prob 1 
        bart_sv_t = np.concatenate([np.array([0]), bart_sv_t])
        bart_sv_val = [np.concatenate([np.array([1]), sv]) for sv in bart_sv_fx]
        
        # log artif plot curves
        title = "bart_surv_pred"
        fig = ssf.plot_sv2(x_mat, bart_sv_val, t=bart_sv_t, title=title, save=False)
        mlflow.log_figure(fig, f"{title}.png")



# COMMAND ----------

# MAGIC %md
# MAGIC # RBART

# COMMAND ----------

# MAGIC %r
# MAGIC # get the test dataset
# MAGIC test = SparkR::sql("select * from test")
# MAGIC test = SparkR::collect(test)
# MAGIC # test = read.csv(input)
# MAGIC
# MAGIC # set values
# MAGIC delta = test$status
# MAGIC times = test$time
# MAGIC
# MAGIC # get x matrix
# MAGIC test_names = names(test) 
# MAGIC x_names = grep("X[0-9]+", test_names)
# MAGIC x_mat = as.matrix(test[x_names])
# MAGIC
# MAGIC # train 
# MAGIC post = mc.surv.bart(x.train = x_mat, times=times, delta=delta, mc.cores=8, seed=99)
# MAGIC
# MAGIC # get unique times
# MAGIC tst_times = sort(unique(times))
# MAGIC
# MAGIC # order the x matrix to match python
# MAGIC tst_x = unique(x_mat)
# MAGIC sq_x = seq(1, ncol(tst_x),1)
# MAGIC ord_x = eval(parse(text = paste0('order(', paste0("tst_x[,", sq_x, "]", collapse=","), ")")))
# MAGIC tst_x = tst_x[ord_x,]
# MAGIC
# MAGIC # set N
# MAGIC N = length(tst_times)
# MAGIC
# MAGIC # fake matrix is N * nrow by ncol + 1
# MAGIC out_x = matrix(nrow = N * nrow(tst_x), ncol = ncol(tst_x) + 1)
# MAGIC names = c("t")
# MAGIC # create test matrix blocks and assign to the out_x
# MAGIC for(i in 1:nrow(tst_x)){
# MAGIC     g = matrix(tst_times)
# MAGIC     
# MAGIC     for(j in 1:ncol(tst_x)){
# MAGIC         o = rep(tst_x[i,j], N)
# MAGIC         g = cbind(g, o)
# MAGIC         if(i == 1){
# MAGIC             names = append(names, paste0("x",j))
# MAGIC         }
# MAGIC     }
# MAGIC     # assign
# MAGIC     r1 = 1+((i-1)*N)
# MAGIC     r2 = i*N
# MAGIC     out_x[r1:r2,] = g
# MAGIC }
# MAGIC # set names for out_x
# MAGIC dimnames(out_x)[[2]] = names
# MAGIC
# MAGIC # get predictions
# MAGIC pred = predict(post, newdata = out_x, mc.cores=8)
# MAGIC
# MAGIC # create the final csv to output
# MAGIC df = cbind(data.frame(out_x), pred$surv.test.mean)
# MAGIC names(df) = c(names, "surv")
# MAGIC
# MAGIC # get id
# MAGIC lbl_mrg = names(df)[grep("x", names(df))]
# MAGIC mm = paste0("df$",lbl_mrg, collapse=", ")
# MAGIC id = eval(parse(text = paste0("paste0(",mm, ")")))
# MAGIC df["id"] = paste0("i",id)
# MAGIC
# MAGIC
# MAGIC out = SparkR::as.DataFrame(df)
# MAGIC SparkR::createOrReplaceTempView(out, "out")
# MAGIC # write.csv(df, output)

# COMMAND ----------

with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:
    with mlflow.start_run(experiment_id=experiment_id, run_id = sub_run_id, nested=True) as sub:
        # get Rbart data
        rbart_df = spark.sql("select * from out").toPandas()
        rb_mat, rb_x, rb_idx, rb_sv_t, rb_sv_val = ssf.get_rbart_data(rbart_df)
        # get survival plot
        title = "rbart_surv_pred"
        fig = ssf.plot_sv2(rb_mat, rb_sv_val, t=rb_sv_t, title=title, 
                        show = False, save=False, dir="outputs")
        mlflow.log_figure(fig, f"{title}.png")
        # mlflow.log_artifact(f"outputs/{title}.png")

# COMMAND ----------

# get metrics rmse, bias
rsf_rmse, rsf_bias, t_quant = ssf.get_metrics( f_t = rsf_sv_val, f = sv_mat[x_idx], T = rsf_sv_t[rsf_sv_t <T])

cph_rmse, cph_bias, t_quant = ssf.get_metrics( f_t = cph_sv_val, f = sv_mat[x_idx], T = cph_sv_t[cph_sv_t < T])

bart_rmse, bart_bias, t_quant = ssf.get_metrics(f_t = bart_sv_val, f = sv_mat[x_idx], T = bart_sv_t[bart_sv_t < T])

rb_rmse, rb_bias, t_quant = ssf.get_metrics(f_t = rb_sv_val, f = sv_mat[x_idx], T = rb_sv_t)

t_q = np.arange(t_quant.shape[0])

with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:
    rmse_dict = dict({"time":t_quant.tolist(), 
                      "t_q": t_q.tolist(),
                        "rsf":rsf_rmse.tolist()[0], 
                        "cph":cph_rmse.tolist()[0],
                        "bart":bart_rmse.tolist()[0],
                        "rbart":rb_rmse.tolist()[0]})
    bias_dict = dict({"time":t_quant.tolist(), 
                      "t_q": t_q.tolist(),
                        "rsf":rsf_bias.tolist()[0], 
                        "cph":cph_bias.tolist()[0],
                        "bart":bart_bias.tolist()[0],
                        "rbart":rb_bias.tolist()[0]})
    mlflow.log_table(rmse_dict, "rmse_dict")
    mlflow.log_table(bias_dict, "bias_dict")
