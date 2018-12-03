
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
category[category=="THE WORLDPOST"]="WORLDPOST"
category[category=="ARTS & CULTURE"]="ARTS"

news_id_for_category=1:n_obs

#============

news_data1=tibble(news_id=rep(1:n_obs,times=2),text=c(description,headline),
                  indicator=rep(c("desp","hdl"),each=n_obs,times=1),category=rep(category,times=2))
news_data<-mutate(news_data1,news_id_indicator=unite_(news_data1,"news_id_indicator",c("news_id","indicator"))$news_id_indicator)
by_news_word <- news_data %>% unnest_tokens(word,text)
words_count <- by_news_word %>% anti_join(stop_words) %>% count(news_id,word,sort=TRUE)%>%ungroup()
news_id_dtm <- words_count %>% cast_dtm(news_id,word,n)
news_id_dtm
#words_count$news_id is 1:124948 and category is also length 124948

#keep documents with term over 10

#=======================
count_term=table(news_id_dtm$i)#124948
length(which(count_term>10))/length(count_term)#0.58
plot(density(count_term))

which_news_id_keep=as.integer(names(table(words_count$news_id)))[which(table(words_count$news_id)>10)] #which document to keep:72968 docs
words_count1=words_count[words_count$news_id %in% which_news_id_keep,]#1137520 rows
#which(table(words_count1$news_id)<11)

news_id_for_category1=which_news_id_keep
category1=category[which_news_id_keep]#72968 length

news_id_dtm1 <- words_count1 %>% cast_dtm(news_id,word,n)
news_id_dtm1 

#=========================================

Terms_to_delete=news_id_dtm1$dimnames$Terms[which(table(news_id_dtm1$j)==1)]#Terms to delete:24305
words_count2=words_count1[-which(words_count1$word %in% Terms_to_delete),]#delete 26589 Terms

news_id_dtm2 <- words_count2 %>% cast_dtm(news_id,word,n)
news_id_dtm2 

#which(table(news_id_dtm2$j)==1) 
#min(table(news_id_dtm2$i)) :min of number of Term in a document is 6
#length(table(news_id_dtm2$i)) #72968

category2=category1#length 72968
news_id_for_category2=news_id_for_category1


#delete plural forms
#=============================

load("word_count2_nound_modified.RData",verbose = TRUE)
words_count2_again=words_count2_nouns_modified
news_id_dtm2_again <- words_count2_again %>% cast_dtm(news_id,word,n)
news_id_dtm2_again

target_category=c("CRIME","EDUCATION","SPORTS","RELIGION","MEDIA")
target_index=which(category2 %in% target_category[1:5])
#target_index=1:length(category2)
category3=category2[target_index]
news_id_for_category3=news_id_for_category2[target_index]

words_count3=words_count2_again[words_count2_again$news_id %in% news_id_for_category3,]
news_id_dtm3 <- words_count3 %>% cast_dtm(news_id,word,n)
news_id_dtm3 

# now run LDA on news_id_dtm3?
# modeling code continued
news_id_lda<-LDA(news_id_dtm3, k=5, method="Gibbs",control = list(seed=1234))
news_id_lda
str(news_id_lda) # 2000 iterations

news_topics <- tidy(news_id_lda,matrix="beta")
news_topics$term[which(news_topics$term == "polouse")] = "police"

#top_terms <- news_topics %>% group_by(topic) %>% top_n(5,beta) %>% ungroup() %>% arrange(topic,-beta)
top_terms <- news_topics %>% group_by(topic) %>% top_n(10,beta) %>% ungroup() %>% arrange(topic,-beta)
top_terms$term[which(top_terms$term == "polouse")] = "police"

all_content=news_topics %>% group_by(topic)

#a=news_topics %>% group_by(topic) %>% top_n(5,beta)
a=news_topics %>% group_by(topic) %>% top_n(10,beta)

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


# generate classification accuracy
#=============================
# in this R LDA package output, the label-topic relationship is as follows:
# topic1 = education
# topic2 = crime
# topic3 = politics
# topic4 = sports
# topic5 = religion


if(TRUE){
  target=which(category3=="CRIME")
  length(which(apply((news_id_gamma[news_id_gamma$document %in% target,]%>%spread(key=topic,value=gamma))[,-1],1,which.max)==2))/length(target)
  
  
  target=which(category3=="EDUCATION")#792
  length(which(apply((news_id_gamma[news_id_gamma$document %in% target,]%>%spread(key=topic,value=gamma))[,-1],1,which.max)==1))/length(target)#predict CRIME|given crime 0.8
  
  
  target=which(category3=="MEDIA")#2090
  #length(target)
  length(which(apply((news_id_gamma[news_id_gamma$document %in% target,]%>%
                        spread(key=topic,value=gamma))[,-1],1,which.max)==3))/length(target)#predict CRIME|given crime 0.8
  
  target=which(category3=="RELIGION")#792
  length(which(apply((news_id_gamma[news_id_gamma$document %in% target,]%>%spread(key=topic,value=gamma))[,-1],1,which.max)==5))/length(target)#predict CRIME|given crime 0.8
  
  
  target=which(category3=="SPORTS")#2090
  #length(target)
  length(which(apply((news_id_gamma[news_id_gamma$document %in% target,]%>%
                        spread(key=topic,value=gamma))[,-1],1,which.max)==4))/length(target)#predict CRIME|given crime 0.8
  
}




