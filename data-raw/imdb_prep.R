# -----------------------------------------------------------------------------
# Sampling a small subset from the Large Movie Review Dataset for package
# testing and examples
#
# Citation: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang,
# Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for
# Sentiment Analysis. The 49th Annual Meeting of the Association for
# Computational Linguistics (ACL 2011).
# -----------------------------------------------------------------------------

set.seed(111)

### Sample from the individual review text files in the training set.
n <- 100
pos_files <- list.files("data-raw/aclimdb/train/pos", full.names = TRUE)
pos_files <- pos_files[sample(1:length(pos_files), n)]
neg_files <- list.files("data-raw/aclimdb/train/neg", full.names = TRUE)
neg_files <- neg_files[sample(1:length(neg_files), n)]

imdb <- sapply(c(pos_files, neg_files), function(f) {
  "text" = readLines(f, warn = FALSE)
})
imdb <- data.frame("text" = imdb, "y" = rep(c(1, 0), each = n))


# Remove non-ASCII characters from the text column.
imdb$text <- stringi::stri_trans_general(imdb$text, "latin-ascii")

write.csv(imdb, "data-raw/imdb_sample.csv", row.names = FALSE)
usethis::use_data(imdb)
