# Databricks notebook source
import mlflow
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
plt.ioff()

# COMMAND ----------

experiment_id=dbutils.jobs.taskValues.get("cdh-ml-init",
                                          "experiment_id",
                                          debugValue=2256023545555400)

run_name = dbutils.jobs.taskValues.get("cdh-ml-init", 
                                         "run_name", 
                                         debugValue="test1")

run_id_main = dbutils.jobs.taskValues.get("cdh-ml-run", 
                                          "run_id_main",
                                          debugValue = "957cd8b5d5f74732b0c51ca176298166")

ALPHA = dbutils.jobs.taskValues.get("cdh-ml-init",
                            "alpha", 
                            debugValue=3)
ALPHA_F = dbutils.jobs.taskValues.get("cdh-ml-init",
                            "alpha_f",
                            debugValue= "")
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

exp = mlflow.set_experiment(experiment_id=experiment_id)

# COMMAND ----------

rmse = mlflow.load_table("rmse_dict", run_ids = [run_id_main])
bias = mlflow.load_table("bias_dict", run_ids = [run_id_main])

# COMMAND ----------


def get_boxplots(dtst, title):
    t = np.unique(dtst["t_q"])
    t_size = t.shape[0]
    dim0 = int(dtst.shape[0]/t_size)
    labels = ["5", "25", "50", "75", "95"]
    # max_t = 
    fig = plt.figure()
    mod = ["rsf", "cph", "bart", "rbart"]
    colors = list(mcolors.TABLEAU_COLORS.values())
    adj = [-.3, -.1, .1, .3]

    for idx,i in enumerate(mod):
        n = dtst[i].to_numpy().reshape(dim0,t_size)
        p1 = plt.boxplot(n, positions = np.arange(1,t_size+1,1)+adj[idx], widths=0.1)
        plt.setp(p1["boxes"], color=colors[idx])
        plt.plot([], c=colors[idx], label=i)


    plt.legend()
    plt.tight_layout()
    # ticks
    tick = np.arange(1, t_size+1)
    plt.xticks(tick, labels=labels)
    plt.xlabel("Percentiles")
    plt.title(f"{title}: Reps={dim0}", y=1.05)
    plt.suptitle(f"N:{N}, Alpha:{ALPHA}, Alpha_F:{ALPHA_F}, Lambda:{LAMBDA}, \n Cens_ind:{CENS_IND}, cens_scale:{CENS_SCALE}, x_vars:{X_VARS}", fontsize=6, y=1.02)
    plt.close(fig)
    return fig

# COMMAND ----------

with mlflow.start_run(experiment_id = experiment_id, run_id = run_id_main) as run:
    fig = get_boxplots(rmse, "RMSE")
    mlflow.log_figure(fig, "rmse.png")
    fig = get_boxplots(bias, "BIAS")
    mlflow.log_figure(fig, "bias.png")

# COMMAND ----------

# fig = get_boxplots(rmse, "RMSE")
# fig
