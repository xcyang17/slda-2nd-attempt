library("rjson")
library("jsonlite")
library("tidyverse")
library("topicmodels")
library("broom")
library("dplyr")
library("tidyr")
library("tidytext")
library(stringr)
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

news_id_for_category=1:n_obs
#names(category)=1:n_obs#n_obs is 124948
#31 categories
#124948 news
#============
description=gsub(pattern = "'s ",replacement = " ",description)
description=gsub(pattern = "'s",replacement = "",description)
headline=gsub(pattern = "'s ",replacement = " ",headline)
headline=gsub(pattern = "'s",replacement = "",headline)

for(i in 1:n_obs){
  out[i][[1]]$short_description=description[i]
  out[i][[1]]$headline=headline[i]
  out[i][[1]]$category=category[i]
}
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


#which(table(news_id_dtm1$i)<11)#check whether 0
#=================================


#delete Terms only appear once in the corpus:some news_id is also deleted here
#!!later delete Terms in description 
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

# now try replacing plural of nouns with singular in `out`
load("singularized-unique-terms-news_id_dtm2.Rdata", verbose = T)
load("unique-terms-news_id_dtm2.Rdata", verbose = T)

names(sigularized.terms) = unique.terms

# now try modifying words_count2
head(sigularized.terms)
head(unique.terms)

dff = sigularized.terms[sigularized.terms!=unique.terms]
names(dff) = unique.terms[sigularized.terms!=unique.terms]

words_count2_modified = words_count2
for (i in 1:length(dff)) {
  words_count2_modified$word[which(words_count2_modified$word == names(dff)[i])] = dff[i]
}

length(unique(words_count2_modified$word)) == length(unique(sigularized.terms))
words_count2_modified = as.data.frame(words_count2_modified)

# now merge rows if (news_id_1, term_1) == (news_id_2, term_2)
tmp0 = paste(words_count2$news_id, words_count2$word, sep = " ")
tmp = paste(words_count2_modified$news_id, words_count2_modified$word, sep = " ")

tmp[which(duplicated(tmp))] # these are rows to merge/remove
tmp[which(!duplicated(tmp))] # these are rows to keep

length(unique(tmp)) == length(tmp[which(!duplicated(tmp))])

words_count2_modified_keep = words_count2_modified[which(!duplicated(tmp)),]

# now merge one by one
to_merge = tmp[which(duplicated(tmp))]
for (i in 1:length(to_merge)) {
  if(i %% 1000 == 0) {
    print(paste0("current iter: ", i))
  }
  news.id = as.numeric(unlist(strsplit(to_merge[i], split = " "))[1])
  wd = unlist(strsplit(to_merge[i], split = " "))[2]
  to_keep_row_idx = which(words_count2_modified_keep$news_id == news.id
                         & words_count2_modified_keep$word == wd)
  tpo_merge_row_idx = which(words_count2_modified$news_id == news.id
                            & words_count2_modified$word == wd)
  words_count2_modified_kee[to_keep_row_idx,"n"] = sum(words_count2_modified[tpo_merge_row_idx,"n"])
}

# now check if there are duplicates in the (news_id, term) pair
which((duplicated(paste(words_count2_modified_keep$news_id, words_count2_modified_keep$word, sep = " "))))
# no duplicates this time

# now cast `words_count2_modified_keep` back to the original class of 
words_count2_nouns_modified = tibble(news_id = words_count2_modified_keep$news_id, 
                                     word = words_count2_modified_keep$word,
                                     n = words_count2_modified_keep$n)

str(words_count2_nouns_modified)
save(words_count2_nouns_modified, file = "word_count2_nound_modified.RData")




dff2 = tmp[tmp0 != tmp]
names(dff2) = tmp0[tmp0 != tmp]

dff2[which(duplicated(dff2))]




words_count2_modified_2 = as.data.frame(words_count2_modified)

# stores the index of dff2 where the dataframe actually has two duplicate columns
# after replacing the words (plural to singular)
idx_dff2_duplicates = numeric(0) 
for (i in 10000:length(dff2)) {
  if (i %% 10000 == 0) {
    print(paste0("current iter: ", i))
  }
  tmpp = unlist(strsplit(dff2[i], split = " "))
  news_id = as.numeric(tmpp[1])
  wrd = tmpp[2]
  subset_df = subset(words_count2_modified_2, 
                      subset = (words_count2_modified_2$news_id == news_id) && 
                       (words_count2_modified_2$word == wrd))
  if (nrow(subset_df) > 1) {
    print(i)
    idx_dff2_duplicates = c(idx_dff2_duplicates, i)
  }
}

# now remove the duplicated rows and add the word counts
dff2[idx_dff2_duplicates]
words_count2_modified_3 = words_count2_modified_2

dup_news_id = numeric(length(idx_dff2_duplicates))
dup_wrd = character(length(idx_dff2_duplicates))
idx_to_remove = numeric(0)
for (j in 1:length(idx_dff2_duplicates)) {
  dup_news_id[j] = as.numeric(unlist(strsplit(dff2[j], split = " "))[1])
  dup_wrd[j] = (unlist(strsplit(dff2[j], split = " "))[2])
  tmp_df = subset(words_count2_modified_2, 
                  subset = (words_count2$news_id == dup_news_id[j]) & 
                    (words_count2_modified_2$word == dup_wrd[j]))
  if (nrow(tmp_df) > 1) {
    print(tmp_df)  
  }
  words_count2_modified_3[as.numeric(rownames(tmp_df)[1]), "n"] = sum(tmp_df$n)
  idx_to_remove = c(idx_to_remove, as.numeric(rownames(tmp_df)[-1]))
}

words_count2_modified_4 = words_count2_modified_3[-idx_to_remove,]

# now examine if there are duplicates
words_count2_modified_4[(duplicated(paste0(words_count2_modified_4$news_id, words_count2_modified_4$word))),]

words_count2_modified_4[which(words_count2_modified_4$word == "matter"
                              & words_count2_modified_4$news_id == 87065),]



as.data.frame(words_count2)[which(words_count2$news_id == 86807),]


for (j in 1:length(idx_dff2_duplicates)) {
  dup_news_id[j] = as.numeric(unlist(strsplit(dff2[j], split = " "))[1])
  dup_wrd[j] = (unlist(strsplit(dff2[j], split = " "))[2])
  tmp_df = subset(words_count2_modified_2, 
                  subset = (words_count2$news_id == dup_news_id[j]) & 
                    (words_count2_modified_2$word == dup_wrd[j]))
  print(tmp_df)
}


# should contain no duplicate (news_id, term) pair any more - not true









cbind(dup_news_id, dup_wrd)

subset(words_count2_modified_2, 
       subset = (words_count2_modified_2$news_id == 66055) && 
         (words_count2_modified_2$word == "body"))

subset(words_count2_modified_2, 
       subset = (words_count2$news_id == 66055))

# this one
subset(as.data.frame(words_count2), 
       subset = (words_count2$news_id == 66055) & (words_count2_modified_2$word == "body"))

subset(words_count2_modified_2, 
       subset = (words_count2$news_id == 66055) & (words_count2_modified_2$word == "body"))






words_count2_modified_2[(words_count2_modified_2$news_id == 124938) & 
                          (words_count2_modified_2$word == "million"),]
words_count2_modified_2[(words_count2_modified_2$news_id == 124938),]
words_count2_modified_2[(words_count2$news_id == 124938),]


words_count2_modified_2[(words_count2_modified_2$news_id == 8875) & 
                          (words_count2_modified_2$word == "black"),]

words_count2_modified_2[(words_count2_modified_2$news_id == 8875) & 
                          (words_count2_modified_2$word == "blacks"),]






if(FALSE){
  setwd("R_output_6")# first keep news_id with more than 10 terms and then delete Terms with one occurance
  write.table(news_id_dtm2$i-1,file="i-6.txt", row.names = F, col.names = F)
  write.table(news_id_dtm2$j-1,file="j-6.txt", row.names = F, col.names = F)
  write.table(news_id_dtm2$v,file="v-6.txt", row.names = F, col.names = F)
  write.table((as.numeric(news_id_dtm2$dimnames$Docs)-1),file="Docs-6.txt", row.names = F, col.names = F)
  write.table(news_id_dtm2$dimnames$Terms,file="Terms-6.txt", row.names = F, col.names = F)
  write.table(category2,file="category-6.txt", row.names = F, col.names = F)
  setwd("..")
}
#========================================


#select a small sample
#========================
#CRIME and Education and entertainment
#crime and education and sports
target_index=which(category2=="CRIME"|category2=="EDUCATION")
#target_index=1:length(category2)
category3=category2[target_index]
news_id_for_category3=news_id_for_category2[target_index]

words_count3=words_count2[words_count2$news_id %in% news_id_for_category3,]
news_id_dtm3 <- words_count3 %>% cast_dtm(news_id,word,n)
news_id_dtm3 


words_count3_for_py<-words_count3
fake_news_id_for_category3_py=1:length(news_id_for_category3)#2524 and category3 is the same order
#true news id is news_id_for_category3
temp=numeric(nrow(words_count3_for_py))
for (i in 1:length(temp)) {
  temp[i]=fake_news_id_for_category3_py[which(news_id_for_category3==words_count3$news_id[i])]
}
words_count3_for_py$news_id=temp

#words_count3$news_id[1:5]
#words_count3_for_py$news_id[1:5]
#news_id_for_category3[fake_news_id_for_category3_py==words_count3_for_py$news_id[3]]

words_count4=words_count3_for_py
news_id_dtm4 <- words_count4 %>% cast_dtm(news_id,word,n)
news_id_dtm4 
category4=category3# fake_news_id_for_category3_py

if(FALSE){
  setwd("./R_output_7/")
  write.table(news_id_dtm4$i-1,file="i-7.txt", row.names = F, col.names = F)
  write.table(news_id_dtm4$j-1,file="j-7.txt", row.names = F, col.names = F)
  write.table(news_id_dtm4$v,file="v-7.txt", row.names = F, col.names = F)
  write.table((as.numeric(news_id_dtm4$dimnames$Docs)-1),file="Docs-7.txt", row.names = F, col.names = F)
  write.table(news_id_dtm4$dimnames$Terms,file="Terms-7.txt", row.names = F, col.names = F)
  write.table(category4,file="category-7.txt", row.names = F, col.names = F)
  setwd("..")
}

#max(as.integer(news_id_dtm4$dimnames$Docs))
#============================

news_id_dtm<-news_id_dtm4
news_id_lda<-LDA(news_id_dtm,k=30,method="Gibbs",control = list(seed=1234))
news_id_lda

news_topics <- tidy(news_id_lda,matrix="beta")

top_terms <- news_topics %>% group_by(topic) %>% top_n(15,beta) %>% ungroup() %>% arrange(topic,-beta)

all_content=news_topics %>% group_by(topic)

a=news_topics %>% group_by(topic) %>% top_n(15,beta)

library(ggplot2)
top_terms %>% mutate(term=reorder(term,beta)) %>% 
  ggplot(aes(term,beta,fill=factor(topic)))+
  geom_col(show.legend = FALSE)+facet_wrap(~topic,scales="free")+
  coord_flip()

news_id_gamma <- tidy(news_id_lda,matrix="gamma")
news_id_gamma$document=as.integer(news_id_gamma$document)
#left_join(news_id_gamma,news_data[,c(1,4)],by=c("document"="news_id"))

#for more than 2 category
if(TRUE){
target=which(category3=="CRIME")
length(which(apply((news_id_gamma[news_id_gamma$document %in% target,]%>%spread(key=topic,value=gamma))[,-1],1,which.max)==3))/length(target)
#predict CRIME|given crime is 0.768

target=which(category3=="EDUCATION")#792
length(which(apply((news_id_gamma[news_id_gamma$document %in% target,]%>%spread(key=topic,value=gamma))[,-1],1,which.max)==1))/length(target)#predict CRIME|given crime 0.8
#predict education|given education is 0.9267677

target=which(category3=="ENTERTAINMENT")#2090
length(target)
length(which(apply((news_id_gamma[news_id_gamma$document %in% target,]%>%
                      spread(key=topic,value=gamma))[,-1],1,which.max)==2))/length(target)#predict CRIME|given crime 0.8
#predict sports|given sports is 0.688

}

#for 2 category
#nrow(news_id_gamma[news_id_gamma$document %in% target,]%>%filter(topic==1)%>%filter(gamma>0.5))/length(target)#predict crime|given crime=0.76

if(TRUE){
target=which(category3=="ARTS")#792
nrow(news_id_gamma[news_id_gamma$document %in% target,]%>%filter(topic==2)%>%filter(gamma>0.5))/length(target)#predict education|given education=0.93

}



#plot(news_id_gamma %>% filter(document==1)%>%select(topic,gamma))
#class<-news_id_gamma %>% group_by(document) %>% top_n(1,gamma) %>% arrange(document)
#left_join(class,news_data[1:n_obs,],by=c("document"="news_id"))
#consider the headline and description as the replications of a document

#plot words of each topic
#=================
library(wordcloud)
topicToViz <- 2
top40terms <- order(all_content$beta[all_content$topic==topicToViz], decreasing=TRUE)[1:40]
words <- all_content$term[all_content$topic==topicToViz][top40terms]
# extract the probabilites of each of the 40 terms
probabilities <- all_content$beta[all_content$topic==topicToViz][top40terms]
# visualize the terms as wordcloud
mycolors <- brewer.pal(8, "Dark2")
wordcloud(words, probabilities, random.order = FALSE, color = mycolors)
#=============


table(category)
which(category=="THE WORLDPOST")[1:5]
out[which(category=="THE WORLDPOST")[1]]
out[which(category=="WORLDPOST")[1]]

















news_id_gamma <- tidy(news_id_lda,matrix="gamma")
news_id_gamma$document=as.integer(news_id_gamma$document)

news_id_gamma[news_id_gamma$document %in% 6:10,]
category4[6:10]
#left_join(news_id_gamma,news_data[,c(1,4)],by=c("document"="news_id"))

plot(news_id_gamma %>% filter(document==1)%>%select(topic,gamma))

class<-news_id_gamma %>% group_by(document) %>% top_n(1,gamma) %>% arrange(document)#maybe the same gamma

all_class<-left_join(class,news_data[1:n_obs,],by=c("document"="news_id"))
temp<-all_class[,c(2,3,6)] %>%count(topic,category)

ggplot(all_class,aes(x=topic,color=category))+geom_histogram()

d=news_id_gamma[news_id_gamma$document==1,]




news_id_gamma_cat=mutate(news_id_gamma,category=category[news_id_gamma$document])
gamma_by_news_id<-news_id_gamma_cat %>% mutate(title=reorder(category,gamma*topic)) 

index=which(gamma_by_news_id$category %in% category[c(21,27,28)])
gamma_by_news_id[index,] %>% ggplot(aes(factor(topic),gamma))+geom_boxplot()+facet_wrap(~title)

#crime-topic 23
#entertainment-topic 10,28
#world news:21
#impact
#politics
#
#plot crime document's gamma weights on every topics
category_document=gamma_by_news_id %>% filter(title=="CRIME")
temp=spread(gamma_by_news_id,key="topic",value="gamma")

temp_crime=temp%>%filter(temp$category=="CRIME")
layout(matrix(1:10,2,5))
plot(1:31,temp_crime[1,4:34],type="l",ylim=c(0,0.15))
points(1:31,temp_crime[1,4:34],col="red")
for(i in 2:10){
  plot(1:31,temp_crime[i,4:34],type="l",ylim=c(0,0.15))
  points(1:31,temp_crime[i,4:34],col="red")
}

cat="CRIME"
temp_cat=temp%>%filter(temp$category==cat)
layout(matrix(1:10,2,5))
plot(1:31,temp_cat[1,4:34],type="l",ylim=c(0,0.15),main=cat)
points(1:31,temp_cat[1,4:34],col="red")
for(i in 2:10){
  plot(1:31,temp_cat[i,4:34],type="l",ylim=c(0,0.15))
  points(1:31,temp_cat[i,4:34],col="red")
}

data_temp_cat=as.matrix(temp_cat[,4:34])
order(data_temp_cat[10,],decreasing = TRUE)[1:5]
length(which(apply(data_temp_cat,1,which.max)==23))



install.packages("lda")
demo("lda")

try <- c("I am the very model of a model major general general",
         "I have a major headache")
lexicalize(try)



temp=table(words_count$news_id)
no_word_news_id=which(1:124948 %in% names(temp)==FALSE)
description[no_word_news_id]#no description
headline[no_word_news_id]#
category[no_word_news_id]

unique(category)




#data cleaning
#===================
word_content=news_id_dtm$dimnames$Terms #69464
num_term=length(word_content)
word_index=news_id_dtm$j
word_index_count=table(word_index)#name is word_index, value is count

word_content(1:num_term)[word_index_count==1]

temp=as.integer(names(word_index_count))
#which(temp-1:length(temp)!=0)

news_id_dtm$dimnames$Terms[word_index[word_index_count==1]]

#69464 terms
word_content[(1:num_term)[word_index_count==1]]#how many words has one occurence in documents
which(word_content=="vampire")#
which(word_content=="vampire")#

word_content[grep("vampire",x=word_content)]
word_content[grep("tripadvisor",x=word_content)]
word_content[grep("linkedin",x=word_content)]
length(word_content[grep("'s",x=word_content)])/length(word_content)


word_content[as.integer(word_content)==word_content]#how many words are pure integer
whether_integer=which(as.integer(word_content)==word_content)
length(word_content[whether_integer])#678

grep(pattern="'s",x=description)

i=7
description[i]
gsub(pattern="'s",replacement="",x=description[i])


words_count%>%filter(news_id==82350)
