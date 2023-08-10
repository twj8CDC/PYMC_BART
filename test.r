x = 1

train = read.csv("train.csv")

library("BART")

train[c("X0", "X1")]

pre = surv.pre.bart(train$time, train$status, x.train=train[c("X0", "X1")])

str(pre)

pre$tx.train
