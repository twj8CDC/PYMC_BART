library(BART)


run = function(input, output) {
    test = read.csv(input)

    # set values
    delta = test$status
    times = test$time

    x_mat = as.matrix(test[c("sex", "age", "ph.karno")])

    # train 
    post = mc.surv.bart(x.train = x_mat, 
                        times=times, 
                        delta=delta, 
                        x.test = x_mat, 
                        mc.cores=8, 
                        seed=99)
    
    
    names = unlist(dimnames(post$tx.test))
    # create the final csv to output
    df = cbind(data.frame(post$tx.test), post$prob.test.mean)
    df = cbind(df, post$surv.test.mean)
    names(df) = c(names, "prob", "surv")
    
    
    write.csv(df, output)
}

args = commandArgs(trailingOnly=TRUE)
try(run(args[1], args[2]))



