
# generate top 10 words for each topic using python output of ECGS
setwd("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/ecgs-5-topics")

# use output copy pasted from python

# topic 0 
topic0_top10_terms = c('police', 'shooting', 'officer', 'found', 'woman', 
                       'death', 'suspect', 'killed', 'shot', 'cop')
topic0_top10_theta = c(0.01973074, 0.01087409, 0.00829795, 0.00678337, 0.00677206,
                       0.00660594, 0.00651424, 0.00629406, 0.0059808 , 0.00585864)
topicc = 1:5
topic0_df = data.frame(term = topic0_top10_terms, beta = topic0_top10_theta, 
                       topic = rep(1, length(topic0_top10_terms)))
topic0_df = topic0_df[order(topic0_df$beta, decreasing = T),]

# topic 1
topic1_top10_terms = c('trump', 'news', 'media', 'fox', 'donald', 
                       'time', 'president', 'journalist', 'host', 'reporter')
topic1_top10_theta = c(0.02291053, 0.01944898, 0.0110341 , 0.00989401, 0.00931047,
                       0.00773853, 0.00658899, 0.00620122, 0.00616948, 0.0058063)
topic1_df = data.frame(term = topic1_top10_terms, beta = topic1_top10_theta, 
                       topic = rep(2, length(topic1_top10_terms)))
topic1_df = topic1_df[order(topic1_df$beta, decreasing = T),]

# topic 2
topic2_top10_terms = c('game', 'player', 'olympic', 'team', 'nfl', 'football', 
                       'sport', 'win', 'world', 'fan')
topic2_top10_theta = c(0.0125392 , 0.01168263, 0.01124833, 0.01093903, 0.01022369,
                       0.00798028, 0.00717584, 0.00715711, 0.00682215, 0.00531254)
topic2_df = data.frame(term = topic2_top10_terms, beta = topic2_top10_theta, 
                       topic = rep(3, length(topic2_top10_terms)))
topic2_df = topic2_df[order(topic2_df$beta, decreasing = T),]

# try 
topic012_df = rbind(topic0_df, topic1_df, topic2_df)

ggplot(data = topic012_df, aes(x = reorder(term, beta), y = beta,fill=factor(topic)))+
  geom_col(show.legend = FALSE)+facet_wrap(~topic,scales="free")+
  coord_flip()

# good, continue on the next two topics
topic3_top10_terms = c('person', 'muslim', 'church', 'pope', 'christian', 
                       'daily', 'life', 'american', 'god', 'religiou')
topic3_top10_theta = c(0.00952875, 0.00890832, 0.00876033, 0.00836259, 0.00773183,
                       0.00705255, 0.00694546, 0.0068444 , 0.00663467, 0.00640849)
topic3_df = data.frame(term = topic3_top10_terms, beta = topic3_top10_theta, 
                       topic = rep(4, length(topic3_top10_terms)))
topic3_df = topic3_df[order(topic3_df$beta, decreasing = T),]

# topic 4
topic4_top10_terms = c('school', 'student', 'education', 'teacher', 'college',
                       'time', 'child', 'public', 'person', 'day')
topic4_top10_theta = c(0.02112328, 0.0147641 , 0.01278014, 0.01015008, 0.00676419,
                       0.00648682, 0.00640858, 0.00629343, 0.00498013, 0.00378099)
topic4_df = data.frame(term = topic4_top10_terms, beta = topic4_top10_theta, 
                       topic = rep(5, length(topic4_top10_terms)))
topic4_df = topic4_df[order(topic4_df$beta, decreasing = T),]

# barplot
topic01234_df = rbind(topic0_df, topic1_df, topic2_df, topic3_df, topic4_df)

ggplot(data = topic01234_df, aes(x = reorder(term, beta), y = beta,fill=factor(topic)))+
  geom_col(show.legend = FALSE)+facet_wrap(~topic,scales="free")+
  coord_flip()



