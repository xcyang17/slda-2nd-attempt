library("rjson")
library("jsonlite")
library("tidyverse")
library("topicmodels")
library("broom")
library("dplyr")
library("tidyr")
library("tidytext")
setwd("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/news_category")
out <- lapply(readLines("News_Category_Dataset.json"), fromJSON)
missing_id = c(102405, 77834, 105740, 93455, 90134, 97302, 88856, 95001, 118557, 
               69179, 116796, 113471, 90950, 100426, 72277, 120153, 120410, 82270, 
               120926, 101479, 118891, 56945, 50804, 118900, 102272, 87180, 110989, 
               91299, 122807, 112572, 37830, 94408, 40910, 124879, 95445, 89050, 
               92381, 92127, 96233, 119028, 123134) + 1
out2 = list(0)
keep_idx = (1:length(out))[-missing_id]
for (j in 1:length(keep_idx)) {
  out2[[j]] = out[[keep_idx[j]]]
}
out.orig = out
out = out2

n_obs=length(out)
description=character(n_obs)
headline=character(n_obs)
category=character(n_obs)
for(i in 1:n_obs){
  description[i]=out[i][[1]]$short_description
  headline[i]=out[i][[1]]$headline
  category[i]=out[i][[1]]$category
}
#31 categories
#124989 news

news_data1=tibble(news_id=rep(1:n_obs,times=2), 
                  text=c(description,headline), 
                  indicator=rep(c("desp","hdl"),each=n_obs,times=1),
                  category=rep(category,times=2))

news_data<-mutate(news_data1, 
                  news_id_indicator=unite_(news_data1,"news_id_indicator",c("news_id","indicator"))$news_id_indicator)


by_news_word <- news_data %>% unnest_tokens(word,text)

words_count <- by_news_word %>% anti_join(stop_words) %>% count(news_id,word,sort=TRUE)%>%ungroup()

news_id_dtm<- words_count %>% cast_dtm(news_id,word,n)
news_id_dtm
str(news_id_dtm)

# save output and use it in python
# minus 1 for zero indexing in python
options(scipen = 999)
setwd("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/news_category/R_output")
write.table(news_id_dtm$i-1,file="i-2.txt", row.names = F, col.names = F)
write.table(news_id_dtm$j-1,file="j-2.txt", row.names = F, col.names = F)
write.table(news_id_dtm$v,file="v-2.txt", row.names = F, col.names = F)
write.table((as.numeric(news_id_dtm$dimnames$Docs)-1),file="Docs-2.txt", row.names = F, col.names = F)
write.table(news_id_dtm$dimnames$Terms,file="Terms-2.txt", row.names = F, col.names = F)

# examine whether my python code works
(news_id_dtm$i-1)[1]+1
(news_id_dtm$j-1)[1]+1
(as.numeric(news_id_dtm$dimnames$Docs)-1)[1]

# some inspection
news_id_dtm$j[news_id_dtm$i == 1556]
news_id_dtm$dimnames$Terms[news_id_dtm$j[news_id_dtm$i == 1556]]
news_id_dtm$dimnames$Docs[1556]
out[[as.numeric(news_id_dtm$dimnames$Docs[1556])]]

# doc 120487+1 (6335 in i)
news_id_dtm$j[news_id_dtm$i == 6335]
news_id_dtm$dimnames$Terms[news_id_dtm$j[news_id_dtm$i == 6335]]
news_id_dtm$dimnames$Docs[6335]
out[[as.numeric(news_id_dtm$dimnames$Docs[120488])]]




# modeling code continued
news_id_lda<-LDA(news_id_dtm,k=31,method="Gibbs",control = list(seed=1234))
news_id_lda
str(news_id_lda) # 2000 iterations

news_topics <- tidy(news_id_lda,matrix="beta")

top_terms <- news_topics %>% group_by(topic) %>% top_n(5,beta) %>% ungroup() %>% arrange(topic,-beta)

a=news_topics %>% group_by(topic) %>% top_n(5,beta)

library(ggplot2)

top_terms %>% mutate(term=reorder(term,beta)) %>% 
  ggplot(aes(term,beta,fill=factor(topic)))+
  geom_col(show.legend = FALSE)+facet_wrap(~topic,scales="free")+
  coord_flip()


news_id_gamma <- tidy(news_id_lda,matrix="gamma")
news_id_gamma$document=as.integer(news_id_gamma$document)
#left_join(news_id_gamma,news_data[,c(1,4)],by=c("document"="news_id"))

plot(news_id_gamma %>% filter(document==1)%>%select(topic,gamma))

class<-news_id_gamma %>% group_by(document) %>% top_n(1,gamma) %>% arrange(document)

left_join(class,news_data[1:n_obs,],by=c("document"="news_id"))

#consider the headline and description as the replications of a document


index=2
doc=news_id_dtm$dimnames$Docs[news_id_dtm$i[index]] 
#^^^ 1556-th row of the "matrix" described by the `news_id_dtm` corresponds to 
# news chapter 20820
term=news_id_dtm$dimnames$Terms[news_id_dtm$j[index]]
K = 31
z_temp=sample(1:K,size=news_id_dtm$v[index],replace = TRUE)
z[[c(doc,term)]]=z_temp

# news_id for which there seems no doc exists?
missing_id = c(102405, 77834, 105740, 93455, 90134, 97302, 88856, 95001, 118557, 
               69179, 116796, 113471, 90950, 100426, 72277, 120153, 120410, 82270, 
               120926, 101479, 118891, 56945, 50804, 118900, 102272, 87180, 110989, 
               91299, 122807, 112572, 37830, 94408, 40910, 124879, 95445, 89050, 
               92381, 92127, 96233, 119028, 123134) + 1
description[missing_id] # plus one because the missing_id above is copy-pasted
# from python output, which uses zero-indexing
# so these news have no short description, but do have headline?
headline[missing_id] # however, some of them does not even have a headline....

both_missing = missing_id[which(headline[missing_id] == "")]
for (i in both_missing) {
  print(paste0(out[[i]]))
}
# 82271 exists on internet, others are 404

# data cleaning - remove these a few news articles from `out`?


