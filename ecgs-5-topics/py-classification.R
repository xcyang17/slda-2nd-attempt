
setwd("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/ecgs-5-topics")
library(RcppCNPy)
library(ggplot2)

## compute the classification result using phi (which is theta in our report's setup) 
## from the python output
phi_5_topics = npyLoad("phi-ecgs-5-topics-mean-of-last-300-iters.npy")

colnames(phi_5_topics) = 1:5

max_idx = apply(phi_5_topics, 1, which.max) 


# load the true labels for these documents
setwd("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/unsupervised-2-topics/R_output/CRIME_EDUCATION_SPORTS_RELIGION_MEDIA")
true_label = read.table("category.txt")
true_label = as.character(true_label$V1)
table(true_label)


#=================
# in the python output, the label-topic relationship is as follows:
# topic1 = crime
# topic2 = media
# topic3 = sports
# topic4 = religion
# topic5 = education

# classification accuracy
mean(max_idx[which(true_label == "CRIME")] == 1) # 0.8948571

mean(max_idx[which(true_label == "MEDIA")] == 2) # 0.8094984

mean(max_idx[which(true_label == "SPORTS")] == 3) # 0.7488038

mean(max_idx[which(true_label == "RELIGION")] == 4) # 0.8026135

mean(max_idx[which(true_label == "EDUCATION")] == 5) # 0.9015152






