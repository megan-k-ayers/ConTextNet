#' IMDB Review Data
#'
#' A small subset of movie reviews with sentiment labels from the Large Movie
#' Review Dataset.
#'
#' The full citation for this dataset is:
#' Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
#'
#' @format ## `imdb`
#' A data frame with 250 rows and 2 columns:
#' \describe{
#'   \item{text}{Text of the movie review}
#'   \item{y}{The sentiment label for the review (1 = positive, 0 = negative)}
#' }
#' @source <https://ai.stanford.edu/~amaas/data/sentiment/>
"imdb"
