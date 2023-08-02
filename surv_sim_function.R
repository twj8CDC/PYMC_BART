library(BART)


run = function(input, output) {
    test = read.csv(input)

    # set values
    delta = test$status
    times = test$time

    # get x matrix
    test_names = names(test) 
    x_names = grep("X[0-9]+", test_names)
    x_mat = as.matrix(test[x_names])

    # train 
    post = mc.surv.bart(x.train = x_mat, times=times, delta=delta, mc.cores=8, seed=99)
    
    # get unique times
    tst_times = sort(unique(times))

    # order the x matrix to match python
    tst_x = unique(x_mat)
    sq_x = seq(1, ncol(tst_x),1)
    ord_x = eval(parse(text = paste0('order(', paste0("tst_x[,", sq_x, "]", collapse=","), ")")))
    tst_x = tst_x[ord_x,]
    
    # set N
    N = length(tst_times)

    # fake matrix is N * nrow by ncol + 1
    out_x = matrix(nrow = N * nrow(tst_x), ncol = ncol(tst_x) + 1)
    names = c("t")
    # create test matrix blocks and assign to the out_x
    for(i in 1:nrow(tst_x)){
        g = matrix(tst_times)
        
        for(j in 1:ncol(tst_x)){
            o = rep(tst_x[i,j], N)
            g = cbind(g, o)
            if(i == 1){
                names = append(names, paste0("x",j))
            }
        }
        # assign
        r1 = 1+((i-1)*N)
        r2 = i*N
        out_x[r1:r2,] = g
    }
    # set names for out_x
    dimnames(out_x)[[2]] = names

    # get predictions
    pred = predict(post, newdata = out_x, mc.cores=8)
    
    # create the final csv to output
    df = cbind(data.frame(out_x), pred$surv.test.mean)
    names(df) = c(names, "surv")

    # get id
    lbl_mrg = names(df)[grep("x", names(df))]
    mm = paste0("df$",lbl_mrg, collapse=", ")
    id = eval(parse(text = paste0("paste0(",mm, ")")))
    df["id"] = paste0("i",id)

    write.csv(df, output)
}

args = commandArgs(trailingOnly=TRUE)


try(run(args[1], args[2]))

