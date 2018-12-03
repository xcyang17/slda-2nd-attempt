
library(wordcloud)
library(RcppCNPy)
library(ggplot2)

setwd("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/ecgs-5-topics")
## compute the classification result using phi (which is theta in our report's setup)
theta_5_topics = npyLoad("theta-ecgs-5-topics-mean-of-last-300-iters.npy")
theta_5_topics_flattened = as.numeric(t(phi_5_topics))

# read in the terms
setwd("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/unsupervised-2-topics/R_output/CRIME_EDUCATION_SPORTS_RELIGION_MEDIA/")
terms_py = read.table("Terms.txt")
terms_py = as.character(terms_py$V1)
terms_py[which(terms_py == "polouse")] = "police"

# generate all_content as in lda-5-topics.R
all_contents_py = data.frame(topic = rep(1:5, nrow(theta_5_topics)), term =  rep(terms_py, each = 5),
                             beta = theta_5_topics_flattened)

#plot words of each topic
#=================
# in the python output, the label-topic relationship is as follows:
# topic1 = crime
# topic2 = media
# topic3 = sports
# topic4 = religion
# topic5 = education

par(mfrow = c(2, 3))
for (i in 1:5) {
  topicToViz <- i
  top40terms <- order(all_contents_py$beta[all_contents_py$topic==topicToViz], decreasing=TRUE)[1:20]
  words <- all_contents_py$term[all_contents_py$topic==topicToViz][top40terms]
  # extract the probabilites of each of the 40 terms
  probabilities <- all_contents_py$beta[all_contents_py$topic==topicToViz][top40terms]
  # visualize the terms as wordcloud
  mycolors <- brewer.pal(8, "Dark2")
  wordcloud(words, probabilities, random.order = FALSE, color = mycolors)
}



