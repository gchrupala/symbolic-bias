library(jsonlite)
library(dplyr)
library(ggplot2)
data <- fromJSON("valid_runs.json") %>% mutate(s2i=factor(s2i)) %>% filter()

# Summarize recall and speaker id
result <- data %>% group_by(cond, tasks, s, t, s2i, t2s, t2i) %>% summarize(recall=mean(`recall@10`), speaker_id=round(mean(speaker_id),3))

# tab:speaker-inv
write.csv(result, file="speaker-inv.csv")

 
