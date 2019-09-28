# PlotCD

library(scmamp)
library(dplyr)

input <- read.csv("./results.csv", sep='\t')
data = input %>% group_by(dataset) %>% summarise(auc_challenger = mean(auc_challenger), auc_baseline = mean(auc_baseline), auc_offline = mean(auc_offline))
plotCD(input[,4:6], alpha=0.05)

