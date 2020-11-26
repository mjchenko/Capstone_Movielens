#####load libraries#####
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(lubridate)
library(knitr)
library(kableExtra)
library(data.table)


####create data####
# Create edx set, validation set (final hold-out test set)
# Note: this process could take a couple of minutes
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



####Data visualization####
#visualizing the sparse data
users <- sample(unique(edx$userId), 100)
rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")
title("Sparse Ratings")


#visualizing Movies by number of ratings
edx %>% 
  group_by(movieId) %>%
  summarize(n = n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "blue", alpha = 0.75) + 
  scale_x_log10() + 
  ggtitle("Grouping Movies by Number of Ratings") +
  xlab("Number of Ratings") + ylab("Number of Movies")

#visualize 5 movies with the most ratings
tab <- edx %>% 
  group_by(title) %>%
  summarize(Number_of_Ratings = n(), Average = mean(rating)) %>%
  arrange(desc(Number_of_Ratings)) %>%
  slice_head(n = 5)

colnames(tab) <- c("Title", "Number of Ratings", "Average Rating")

kable(tab, caption = "Movies with the Most Ratings", booktabs = T, linesep = "") %>% 
  kable_styling(latex_options = "striped")

#grouping movies by average rating
edx %>% group_by(title) %>% 
summarize(Average = mean(rating)) %>%
  ggplot(aes(x = Average)) + 
  geom_histogram(bins = 30, color = "black", fill = "blue", alpha = 0.75) + 
  ggtitle("Grouping Movies by Average Rating") +
  xlab("Average Rating") + ylab("Number of Movies")

#grouping users by number of ratings
edx %>%
  group_by(userId) %>% 
  summarize(n = n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "green", alpha = 0.75) + 
  scale_x_log10() +
  ggtitle("Grouping Users by Number of Ratings") +
  xlab("Number of Ratings") + ylab("Number of Users")

#grouping users by average rating
edx %>% group_by(userId) %>% 
  summarize(Average = mean(rating)) %>%
  ggplot(aes(x = Average)) + 
  geom_histogram(bins = 30, color = "black", fill = "green", alpha = 0.75) + 
  ggtitle("Grouping Users by Average Rating") +
  xlab("Average Rating") + ylab("Number of Users")

#time effect
edx %>% 
  mutate(date = as_datetime(timestamp)) %>%
  mutate(week = round_date(date, unit = "week")) %>% 
  group_by(week) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(week, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Time Effect") +
  xlab("Date") + ylab("Rating")

#genre effect
edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), stdev = sd(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n > 100000) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() + 
  geom_errorbar(width=0.75, colour="black", alpha=0.75, size=0.25) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Genre Effect") +
  xlab("Genres") + ylab("Rating")



####Evaluation Function####
#RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
} 


##########################Non Reg Models#######################################
####Naive Bayes####
#getting the average overall
avg_rating_overall <- edx %>% summarize(mu = mean(rating)) %>% pull(mu)

#baseline of naive rmse i.e assuming average rating is prediction
naive_rmse <- RMSE(validation$rating, avg_rating_overall)
naive_rmse



####Movie Bias Model####
#calculate the bias for each movie rating by subtracting the rating - mu
#because some movies are generally higher rated, and other movies are generally
#lower rated. Cannot run lm because dataset too large
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - avg_rating_overall))

pred_movies <- avg_rating_overall + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

movies_rmse <- RMSE(validation$rating, pred_movies)
movies_rmse

####Movie + User Bias####
#we also need to calculate the bias for each user
#we do this by taking the rating and subtracting the average and movie bias
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - avg_rating_overall - b_i))

pred_user_movie <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = avg_rating_overall + b_i + b_u) %>%
  .$pred

movie_user_rmse <- RMSE(validation$rating, pred_user_movie)
movie_user_rmse

####Movie + User + Genre####
genre_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - avg_rating_overall - b_i - b_u))

pred_movie_user_genre <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = avg_rating_overall + b_i + b_u + b_g) %>%
  .$pred

movie_user_genre_rmse <- RMSE(validation$rating, pred_movie_user_genre)
movie_user_genre_rmse

####Movie + User + Genre + Time####
edx <- edx %>% mutate(date = as_datetime(timestamp)) %>% 
  mutate(week = round_date(date, unit = "week"))

validation <- validation %>% mutate(date = as_datetime(timestamp)) %>% 
  mutate(week = round_date(date, unit = "week"))

week_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  group_by(week) %>%
  summarize(b_w = mean(rating - avg_rating_overall - b_i - b_u - b_g))

pred_movie_user_genre_week <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(week_avgs, by='week') %>%
  mutate(pred = avg_rating_overall + b_i + b_u + b_g + b_w) %>%
  .$pred

movie_user_genre_week_rmse <- RMSE(validation$rating, pred_movie_user_genre_week)
movie_user_genre_week_rmse



####Results####
result_no_reg <- data.frame("Model" = c("Naive Bayes","Movie Bias",
                                        "Movie + User Bias", 
                                        "Movie + User + Genre Bias", 
                                        "Movie + User + Genre + Time Bias"), 
                            "RMSE" = c(naive_rmse, movies_rmse, 
                                       movie_user_rmse,
                                       movie_user_genre_rmse,
                                       movie_user_genre_week_rmse))
kable(result_no_reg, caption = "Summarized Results", booktabs = T, linesep = "") %>% 
  kable_styling(latex_options = "striped")

result_no_reg


##########################Regularized Models#######################################
#show effects of bi with small sample size - 
#random movies with small number of ratings have highest bi
movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

tab <- edx %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10)
colnames(tab) <- c("Title", "bi", "Number of Ratings")

kable(tab, caption = "Movie Effects", booktabs = T, linesep = "") %>% 
  kable_styling(latex_options = "striped")

#show benefits of regularizing bi
#popular movies with a lot of ratings have a high bi
lambda <- 1
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - avg_rating_overall)/(n()+lambda), n_i = n()) 

data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

tab <- edx %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10)

colnames(tab) <- c("Title", "bi", "Number of Ratings")

kable(tab, caption = "Movie Effects with Regularization", booktabs = T, linesep = "") %>% 
  kable_styling(latex_options = "striped")

####Regularized model coarse lambda to save time####
#run model with lambda spaced ever 1 to visualize where to focus in on
ls <- seq(0, 10, 1)

reg_rmses <- sapply(ls, function(l) {
  
  # Calculate the regularized biases for movie, user, genre, and time
  
  movie_avgs_reg <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - avg_rating_overall) / (n() + l))
  
  user_avgs_reg <- edx %>%
    left_join(movie_avgs_reg, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - avg_rating_overall - b_i) / (n() + l))
  
  genre_avgs_reg <- edx %>%
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - avg_rating_overall - b_i - b_u) / (n() + l))
  
  week_avgs_reg <- edx %>% 
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%
    left_join(genre_avgs_reg, by='genres') %>%
    group_by(week) %>%
    summarize(b_w = sum(rating - avg_rating_overall - b_i - b_u - b_g) / (n() + l))
  
  
  pred_movie_user_genre_week_reg <- edx %>%
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%
    left_join(genre_avgs_reg, by='genres') %>%
    left_join(week_avgs_reg, by='week') %>%
    mutate(pred = avg_rating_overall + b_i + b_u + b_g + b_w) %>%
    .$pred
  
  RMSE(edx$rating, pred_movie_user_genre_week_reg)
})

# pick the lambda to focus region around
plot(ls, reg_rmses)
l_first <- ls[which.min(reg_rmses)]


####regularized model with fine lambda to choose final lambda####
#focusing region based on coarse lambda.
ls_focused <- seq(l_first - 1, l_first + 1, 0.05)

reg_rmses_focused <- sapply(ls_focused, function(l) {
  
  # Calculate the regularized biases for movie, user, genre, and time
  
  movie_avgs_reg <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - avg_rating_overall) / (n() + l))
  
  user_avgs_reg <- edx %>%
    left_join(movie_avgs_reg, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - avg_rating_overall - b_i) / (n() + l))
  
  genre_avgs_reg <- edx %>%
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - avg_rating_overall - b_i - b_u) / (n() + l))
  
  week_avgs_reg <- edx %>% 
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%
    left_join(genre_avgs_reg, by='genres') %>%
    group_by(week) %>%
    summarize(b_w = sum(rating - avg_rating_overall - b_i - b_u - b_g) / (n() + l))
  
  #predict the movie ratings on edx set
  pred_movie_user_genre_week_reg <- edx %>%
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%
    left_join(genre_avgs_reg, by='genres') %>%
    left_join(week_avgs_reg, by='week') %>%
    mutate(pred = avg_rating_overall + b_i + b_u + b_g + b_w) %>%
    .$pred
  
  #calculate RMSE
  RMSE(edx$rating, pred_movie_user_genre_week_reg)
})

# pick the final lambda
plot(ls_focused, reg_rmses_focused)
l_final <- ls_focused[which.min(reg_rmses_focused)]
l_final


#################Predictions using final lambda##############
# rerun the regularization on the validation set with final lamda to generate the RMSE
reg_rmse_validation <- sapply(l_final, function(l) {
  
  # Calculate the regularized biases for movie, user, genre, and time
  movie_avgs_reg <- validation %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - avg_rating_overall) / (n() + l))
  
  user_avgs_reg <- validation %>%
    left_join(movie_avgs_reg, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - avg_rating_overall - b_i) / (n() + l))
  
  genre_avgs_reg <- validation %>%
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - avg_rating_overall - b_i - b_u) / (n() + l))
  
  week_avgs_reg <- validation %>% 
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%
    left_join(genre_avgs_reg, by='genres') %>%
    group_by(week) %>%
    summarize(b_w = sum(rating - avg_rating_overall - b_i - b_u - b_g) / (n() + l))
  
  #predict the movie ratings on validation set
  pred_movie_user_genre_week_reg <- validation %>%
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%
    left_join(genre_avgs_reg, by='genres') %>%
    left_join(week_avgs_reg, by='week') %>%
    mutate(pred = avg_rating_overall + b_i + b_u + b_g + b_w) %>%
    .$pred
  
  #calculate RMSE
  RMSE(validation$rating, pred_movie_user_genre_week_reg)
})

results <- result_no_reg %>% add_row("Model" = "Regularized Movie + User + Genre + Time Bias", "RMSE" = reg_rmse_validation)
kable(results, caption = "Movies with the Most Ratings", booktabs = T, linesep = "") %>% 
  kable_styling(latex_options = "striped")
results
