set.seed(111)
imdb <- read.csv("data-raw/imdb.csv")

imdb <- imdb[sample(1:nrow(imdb), 500), ]
imdb$sentiment <- ifelse(imdb$sentiment == "positive", 1, 0)

names(imdb) <- c("text", "y")

write.csv(imdb, "data-raw/imdb_sample.csv", row.names = FALSE)
usethis::use_data(imdb, internal = TRUE)
