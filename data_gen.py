# Databricks notebook source
import sksurv
# import pymc
# import pymc_bart
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import sklearn as skl
import scipy.stats as sp
import simsurv_func as ssf
import mlflow

# COMMAND ----------

plt.ioff()
np.random.seed(99)

# COMMAND ----------

experiment_id=dbutils.jobs.taskValues.get("ml-head",
                                          "experiment_id",
                                          debugValue=2256023545555400)

run_name = dbutils.jobs.taskValues.get("ml-head", 
                                         "run_name", 
                                         debugValue="test2")

# COMMAND ----------

mlflow.set_experiment(experiment_id=experiment_id)

# COMMAND ----------

with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
    run_info = mlflow.active_run()
    OUTPUTS = "outputs"
    ALPHA = 3
    # ALPHA_F = "1 + (1.5 * x_mat[:,0]) + x_mat[:,1]"
    LAMBDA = "np.exp(2 + 0.4*(x_mat[:,0] + x_mat[:,1]))"
    # LAMBDA = "np.exp(1 + .2*x_mat[:,0] + .3*x_mat[:,1] + 0.8*np.sin(x_mat[:,0] * x_mat[:,1]) + np.power((x_mat[:,2] - 0.5),2))"
    TRAIN_CSV = "outputs/train.csv"
    RBART_CSV = "outputs/rbart_surv.csv"
    N = 100
    X_VARS = 2
    CENS_IND = False
    CENS_SCALE = 60
    
    ###########################################################################
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
    # try:
    #     mlflow.log_param("X_INFO", str(list(zip(x_out, x_cnt))))
    # except:
    #     print("error")
    mlflow.log_dict("X_INFO", dict(zip(x_out, x_cnt)))

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

    # log artif train dataset
    train = ssf.get_train_matrix(x_mat, t_event, status)
    mlflow.log_table(train.to_dict(), "train")
    # data = mlflow.data.from_pandas(train)
    # mlflow.log_input(data)

    # model cph
    cph = sksurv.linear_model.CoxPHSurvivalAnalysis()
    cph.fit(x_sk, y_sk)
    # log metri coeff
    for i in np.arange(len(cph.coef_)):
        ml.log_metric(f"cph_coef_{i}", cph.coef_[i])
        # log metri exp(coef)
        ml.log_metric(f"cph_exp_coef_{i}", np.exp(cph.coef_[i]))
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
    ssf.plot_sv(x_mat, cph_sv_val, t=cph_sv_t, title = title, save=True, dir="outputs")
    ml.log_artifact(f"outputs/{title}.png")
    # log model cph
    # idk how to do

# COMMAND ----------

# import mlflow.data
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10))
    mlflow.log_figure(fig, "plt.png")

# COMMAND ----------

fig,ax = plt.subplots()
ax.plot(np.arange(10), np.arange(10))


# COMMAND ----------

run = mlflow.get_run(mlflow.last_active_run().info.run_id)

# COMMAND ----------

mlflow.load_table("train", run_ids=[mlflow.last_active_run().info.run_id])

# COMMAND ----------

    # train.to_csv(TRAIN_CSV)
    # mlflow.log_artifact("outputs/train.csv")
