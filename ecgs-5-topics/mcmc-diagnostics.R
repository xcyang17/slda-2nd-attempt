
setwd("/Files/documents/ncsu/fa18/ST740/ST740-FA18-Final/ecgs-5-topics")
library(RcppCNPy)

# this is actually \phi_{term, topic} in our report's setup
theta_topic1 = npyLoad("theta-unsupervised-ecgs-5-topics-topic0.npy")

# now produce the traceplot
theta_topic1_term0 = theta_topic1[1,]
quantile(theta_topic1_term0, probs = (c(0.03, 0.97)))

# plot 4 traceplots
par(mfrow = c(2,4))
plot(theta_topic1[72,201:500], type = "l", 
     ylim = quantile(theta_topic1[72,201:500], probs = (c(0.00, 0.99))), 
     main = "phi_{term_72, topic_1} = phi_{police, crime}")
plot(theta_topic1[145,201:500], type = "l", 
     ylim = quantile(theta_topic1[145,201:500], probs = (c(0.00, 0.99))), 
     main = "phi_{term_145, topic_1} = phi_{shooting, crime}")
plot(theta_topic1[161,201:500], type = "l", 
     ylim = quantile(theta_topic1[161,201:500], probs = (c(0.00, 0.99))), 
     main = "phi_{term_161, topic_1} = phi_{officer, crime}")
plot(theta_topic1[55,201:500], type = "l", 
     ylim = quantile(theta_topic1[55,201:500], probs = (c(0.00, 0.99))), 
     main = "phi_{term_55, topic_1} = phi_{found, crime}")
acf(theta_topic1[72,201:500], plot = T, 
    main = "phi_{term_72, topic_1} = phi_{police, crime}")
acf(theta_topic1[145,201:500], plot = T, 
    main = "phi_{term_145, topic_1} = phi_{shooting, crime}")
acf(theta_topic1[161,201:500], plot = T, 
    main = "phi_{term_161, topic_1} = phi_{officer, crime}")
acf(theta_topic1[55,201:500], plot = T, 
    main = "phi_{term_55, topic_1} = phi_{found, crime}")

