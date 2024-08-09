#' Get Phrase Filter Activations (single layer)
#'
#' @param model R Keras trained model
#' @param input Model input (either embedding matrix or list with embedding
#'        matrix and covariate matrix)
#' @param k Kernel size (indicating which convolutional layer to consider)
#' @param n_filts The number of filters in this convolutional layer
#' @param max_length Number of tokens
#'
#' @return A data frame of phrase filter activations for this convolutional
#'         layer.
#'
#' @examples
get_phrase_acts <- function(model, input, k, n_filts, max_length) {

  # Define intermediate CNN model from the trained model, pull its predictions.
  this <- paste0("conv1d_", k)
  int_model <- model$input %>%
    keras::keras_model(outputs = keras::get_layer(model, this)$output)
  acts <- stats::predict(int_model, input)

  # Flatten by removing the token dimension (stack all tokens from the same
  # sample together). Keep track via phrase and sample IDs corresponding to
  # ascending order in the original data set/phrase.
  acts <- lapply(1:dim(acts)[1], function(i) {
    d <- as.data.frame(acts[i, , ])
    d$phrase_id <- 1:nrow(d)
    return(d)})
  acts <- do.call(rbind, acts)
  acts$sample_ind <- sort(rep(1:dim(input)[1], max_length - k + 1))

  # Clean up formatting, names.
  acts <- as.data.frame(acts)
  names(acts) <- c(paste0("CNN", k, "_", "F", 1:n_filts),
                   "phrase_id", "sample_id")
  acts <- tidyr::pivot_longer(acts, cols = tidyr::all_of(1:n_filts),
                              names_to = "filter", values_to = "activation")
  return(acts[, c("filter", "activation", "sample_id", "phrase_id")])

}


#' Get Phrase Level Filter Activation Data Frame
#'
#' @param model R Keras trained model
#' @param embeds Matrix of word embeddings to get activations for
#' @param params Model parameter list
#' @param dat Original data set (only necessary if covariates are used in model)
#'
#' @return Data frame with convolutional layer activations for all input word
#'         embeddings.
#' @export
#'
#' @examples \dontrun{
#' model <- train_model(imdb_embed)
#' embeds <- imdb_embed$embeds[imdb_embed$dat$fold == "train", , ]
#' dat <- imdb_embed$dat[imdb_embed$dat$fold == "train", ]
#' params <- imdb_embed$params
#' }
get_phrase_acts_df <- function(model, embeds, params, dat = NULL) {
  ### TODO: Test function to check dimensions, order (by intermed modeling one)?

  ### Unpack frequently used parameters.
  kern_sizes <- params$kern_sizes
  n_filts <- params$n_filts
  max_length <- params$n_tokens
  if (!is.null(params$covars)) covar_flag <- TRUE else covar_flag <- FALSE

  ### Prep inputs.
  if (covar_flag) {
    input <- list(embeds, as.matrix(dat[, params$covars]))
  } else input <- embeds

  ### Create data frame of phrase activations on each filter from each CNN.
  phrase_act <- data.frame("filter" = character(), "activation" = numeric(),
                           "sample_id" = integer(), "phrase_id" = integer())
  filter_names <- paste0("F", as.numeric(sapply(n_filts, function(n) 1:n)))
  filter_names <- paste0("CNN", rep(kern_sizes, n_filts), "_", filter_names)
  for (k in kern_sizes) {
    phrase_act <- rbind(phrase_act, get_phrase_acts(model, input, k, n_filts,
                                                    max_length))
  }

  return(phrase_act)

}


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
  ### TODO: Should be an easy function to write tests for for my peace of mind.
  these_tokens <- doc_tokens[phrase_id:(phrase_id + k - 1)]
  return(paste(vocab[these_tokens + 1], collapse = " "))
}


#' Get Highest Activated Filters
#'
#' @param model Trained Keras model
#' @param tokens Data frame with tokens corresponding to `dat$text`
#' @param embeds Data frame with embeddings for `tokens`
#' @param dat Data frame with text samples to assess activations for
#' @param params List of model parameters used in training `model`
#' @param vocab Named list, where values are token values and names are the
#'        corresponding text
#' @param m Number of top phrases to pull per filter.
#'
#' @return Data frame with top phrases and their activations.
#'
#' @examples \dontrun{
#' model <- train_model(imdb_embed)
#' tokens <- imdb_embed$tokens
#' embeds <- imdb_embed$embeds[imdb_embed$dat$fold == "train", , ]
#' dat <- imdb_embed$dat[imdb_embed$dat$fold == "train", ]
#' params <- imdb_embed$params
#' vocab <- imdb_embed$vocab
#' get_top_phrases(model, tokens, embeds, dat, params, vocab)
#' }
get_top_phrases <- function(model, tokens, embeds, dat, params, vocab, m = 10) {
  ### TODO: Documentation!
  ### TODO: Removing options for Chinese for now, consider reintroducing later.
  ### TODO: Faster loop for m = "all" option - parallel?

  ### Get data frame of all phrase activations on each filter from each CNN.
  acts <- as.data.frame(get_phrase_acts_df(model, embeds, params))
  acts$k <- as.numeric(gsub("CNN([0-9]+)_F[0-9]+", "\\1", acts$filter))

  ### Attach output layer weights associated with each filter.
  out_wts <- model$get_layer("output")$get_weights()[[1]]
  filt_names <- as.character(sapply(params$kern_sizes, function(k) {
    paste0("CNN", k, "_", "F", 1:params$n_filts)}))
  out_wts <- data.frame(filter = filt_names, wt = out_wts)
  acts <- merge(acts, out_wts, by = "filter")

  ### Retrieve phrases corresponding to the input embedding sequences.
  if (m == "all") {  # If we want to do this for all activations...
    p <- acts[, c("sample_id", "phrase_id", "k")]
    p <- p[!duplicated(p), ]
    p$text <- sapply(1:nrow(p), function(i) {
      get_phrase(tokens[p$sample_id[i], ], p$phrase_id[i], p$k[i], vocab)})
    acts <- merge(acts, p, by = c("sample_id", "phrase_id", "k"))
    return(acts)

  } else {  # Otherwise, get top m phrases for each CNN filter...
    res <- data.frame("filter" = character(), "activation" = numeric(),
                      "text" = character())
    for (f in unique(acts$filter)) {
      these <- acts[acts$filter == f, ]
      these <- these[order(these$activation, decreasing = TRUE), ][1:m, ]
      these$text <- sapply(1:nrow(these), function(i){
        get_phrase(tokens[these$sample_id[i], ], these$phrase_id[i],
                   these$k[i], vocab)})
      res <- rbind(res, these)
    }
    return(res)
  }
}

