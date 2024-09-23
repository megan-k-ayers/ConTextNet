###############################################################################
###           GET TOP PHRASES THAT ACTIVATE ON EACH CONTEXTNET FILTER
###
### Runs:         Locally and on HPC cluster.
### Status:       Almost complete - mainly tests and documentation remaining.
### Priority:     Medium.
### User facing:  Yes.
###############################################################################


#' Get Text Phrase from Document Tokens
#'
#' Retrieve a phrase given the starting character within `doc_tokens`, the
#' phrase length `k`, and the `vocab` list from the tokenizer.
#'
#' @param doc_tokens Tokens for the entire document that the phrase is within
#' @param phrase_id Position of the starting character of the phrase
#' @param k Phrase length
#' @param vocab Vocabulary list from the tokenizer.
#'
#' @return String containing the phrase.
#'
#' @examples \dontrun{
#' get_phrase(imdb_embed$tokens[1, ], 6, 4, imdb_embed$vocab)
#' }
get_phrase <- function(doc_tokens, phrase_id, k, vocab) {
  these_tokens <- doc_tokens[phrase_id:(phrase_id + k - 1)]
  return(paste(vocab[these_tokens + 1], collapse = " "))
}


#' Get Highest Activated Phrases
#'
#' @param phrase_acts Data frame with phrase activations from phrase_acts()
#' @param tokens Data frame with tokens corresponding to `dat$text`
#' @param params List of model parameters used in training `model`
#' @param vocab Named list, where values are token values and names are the
#'        corresponding text
#' @param m Number of top phrases to pull per filter.
#'
#' @return Data frame with top phrases and their activations.
#'
#' @examples \dontrun{
#' model <- train_model(imdb_embed$dat, imdb_embed$embeds, imdb_embed$params)
#' tokens <- imdb_embed$tokens
#' embeds <- imdb_embed$embeds[imdb_embed$dat$fold == "train", , ]
#' dat <- imdb_embed$dat[imdb_embed$dat$fold == "train", ]
#' params <- imdb_embed$params
#' vocab <- imdb_embed$vocab
#' phrase_acts <- get_phrase_acts(model, embeds, params, dat)
#' get_top_phrases(phrase_acts, tokens, params, vocab)
#' }
get_top_phrases <- function(phrase_acts, tokens, params, vocab, m = 10) {

  ### Retrieve phrases corresponding to the input embedding sequences.
  if (m == "all") {  # If we want to do this for all activations...
    p <- phrase_acts[, c("sample_id", "phrase_id", "k")]
    p <- p[!duplicated(p), ]
    p$text <- sapply(1:nrow(p), function(i) {
      get_phrase(tokens[p$sample_id[i], ], p$phrase_id[i], p$k[i], vocab)})
    phrase_acts <- merge(phrase_acts, p, by = c("sample_id", "phrase_id", "k"))
    return(phrase_acts)

  } else {  # Otherwise, get top m phrases for each CNN filter...
    res <- data.frame("filter" = character(), "activation" = numeric(),
                      "text" = character())
    for (f in unique(phrase_acts$filter)) {
      these <- phrase_acts[phrase_acts$filter == f, ]
      these <- these[order(these$activation, decreasing = TRUE), ][1:m, ]
      these$text <- sapply(1:nrow(these), function(i){
        get_phrase(tokens[these$sample_id[i], ], these$phrase_id[i],
                   these$k[i], vocab)})
      res <- rbind(res, these)
    }
    return(res)
  }
}


#' Get Quick Summary of Highest Activated Filters
#'
#' @return A string listing the top 3 phrases most highly associated with the
#'         top 4 convolutional filters with the highest output layer weight.
#' @export
#' @inheritParams get_top_phrases
#'
#' @examples
get_top_phrases_quick <- function(phrase_acts, tokens, params, vocab) {

  acts <- get_top_phrases(phrase_acts, tokens, params, vocab, m = 3)
  acts <- acts[, c("filter", "wt", "text")]
  acts <- stats::aggregate(acts$text,
                           by = list(filters = acts$filter, wt = acts$wt),
                           paste0, collapse = ", ")
  acts <- acts[order(abs(acts$wt), decreasing = TRUE), ]

  phrases <- paste(paste0(round(acts$wt[1:3], 3), ": [", acts$x[1:3], "]"), collapse = ", ")
  return(phrases)

}

