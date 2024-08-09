# -----------------------------------------------------------------------------
# Sampling a small subset from the Large Movie Review Dataset for package
# testing and examples. Saving the full data set for interactive testing.
#
# Citation: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang,
# Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for
# Sentiment Analysis. The 49th Annual Meeting of the Association for
# Computational Linguistics (ACL 2011).
# -----------------------------------------------------------------------------

set.seed(111)

pos_files <- list.files("data-raw/aclimdb/train/pos", full.names = TRUE)
neg_files <- list.files("data-raw/aclimdb/train/neg", full.names = TRUE)

imdb <- sapply(c(pos_files, neg_files), function(f) {
  "text" = readLines(f, warn = FALSE)
})
imdb <- data.frame("text" = imdb, "y" = rep(c(1, 0), each = 12500))

# Remove non-ASCII characters from the text column.
imdb$text <- stringi::stri_trans_general(imdb$text, "latin-ascii")

write.csv(imdb, "data-raw/imdb_full.csv", row.names = FALSE)

### Sample from the individual review text files in the training set.
n <- 100
imdb_pos <- imdb[imdb$y == 1, ][sample(1:length(pos_files), n), ]
imdb_neg <- imdb[imdb$y == 0, ][sample(1:length(neg_files), n), ]

write.csv(rbind(imdb_pos, imdb_neg), "data-raw/imdb_sample.csv",
          row.names = FALSE)
usethis::use_data(imdb)
