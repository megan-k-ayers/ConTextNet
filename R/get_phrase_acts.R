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
get_cnn_layer_acts <- function(model, input, k, n_filts, max_length) {

  # Define intermediate CNN model from the trained model, pull its predictions.
  this <- paste0("conv1d_", k)
  int_model <- model$input %>%
    keras::keras_model(outputs = model$get_layer(this)$output)
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
  acts <- as.data.frame(acts)
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
#'
#' @examples \dontrun{
#' model <- train_model(imdb_embed$dat, imdb_embed$embeds, imdb_embed$params)
#' embeds <- imdb_embed$embeds[imdb_embed$dat$fold == "train", , ]
#' dat <- imdb_embed$dat[imdb_embed$dat$fold == "train", ]
#' params <- imdb_embed$params
#' res <- get_phrase_acts(model, embeds, params, dat)
#' }
get_phrase_acts <- function(model, embeds, params, dat = NULL) {
  ### TODO: Test function to check dimensions, order (by intermed modeling one)?

  ### Unpack frequently used parameters.
  kern_sizes <- params$kern_sizes
  n_filts <- params$n_filts
  max_length <- params$n_tokens

  ### Prep inputs.
  if (!is.null(params$covars)) {
    input <- list(embeds, as.matrix(dat[, params$covars]))
  } else input <- embeds

  ### Create data frame of phrase activations on each filter from each CNN.
  phrase_act <- data.frame("filter" = character(), "activation" = numeric(),
                           "sample_id" = integer(), "phrase_id" = integer())
  filter_names <- paste0("F", as.numeric(sapply(n_filts, function(n) 1:n)))
  filter_names <- paste0("CNN", rep(kern_sizes, n_filts), "_", filter_names)
  for (k in kern_sizes) { # One CNN per kernel size
    phrase_act <- rbind(phrase_act, get_cnn_layer_acts(model, input, k, n_filts,
                                                       max_length))
  }

  ### Attach output layer weights associated with each filter.
  out_wts <- get_output_wts(model, params$kern_sizes, params$n_filts)
  phrase_act <- merge(phrase_act, out_wts, by = "filter")

  ### Pull out kernel widths.
  phrase_act$k <- as.numeric(gsub("CNN([0-9]+)_F[0-9]+", "\\1", phrase_act$filter))

  return(phrase_act)
}
