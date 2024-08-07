#' Get Phrase Filter Activations (single layer)
#'
#' @param model R Keras trained model
#' @param k Kernel size (indicating which convolutional layer to consider)
#' @param input Model input (either embedding matrix or list with embedding
#'        matrix and covariate matrix)
#' @param n_filts The number of filters in this convolutional layer
#' @param max_length Number of tokens
#'
#' @return A data frame of phrase filter activations for this convolutional
#'         layer.
#' @export
#'
#' @examples
get_phrase_acts <- function(model, k, input, n_filts, max_length) {

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
    phrase_act <- rbind(phrase_act, get_phrase_acts(model, k, input, n_filts,
                                                    max_length))
  }

  return(phrase_act)

}


# get_top_phrases <- function() {
#   # If we want to save something, but this model doesn't have a folder yet,
#   # create it.
#   if (!is.null(save_path)) {
#     if (!file.exists(paste0(save_path, "/"))){
#       dir.create(paste0(save_path, "/"))
#     }
#   }
#
#   # Deduce needed parameters from model passed in (could also pass params but
#   # maybe this is more direct/good sanity check?)
#   layer_names <- sapply(1:length(model$layers), function(i) {
#     model$layers[[i]]$name })
#   conv_ind <- which(grepl("conv1d", layer_names))
#
#   kernel_sizes <- sapply(conv_ind, function(i) {
#     as.numeric(model$layers[[i]]$kernel_size) })
#   n_filts <- sapply(conv_ind, function(i) {
#     as.numeric(model$layers[[i]]$filters) })
#   max_length <- ncol(test$tokens)
#
#   if ("covars" %in% layer_names) covars_flag <- TRUE else covars_flag <- FALSE
#
#
#   # Get data frame of all phrase activations on each filter from each CNN.
#   filter_names <- paste0("F", as.numeric(sapply(n_filts, function(n) 1:n)))
#   phrase_act <- as.data.frame(get_phrase_acts(model, test, params))
#
#   # Get word index from fitted tokenizer to more easily get words from tokens
#   vocab <- names(tokenizer$get_vocab())  # Just need to double check indexing
#
#
#   # Function to retrieve phrase given the text sample index, the start index
#   # of the phrase from the CNN window, and the kernel size k
#   get_phrase <- function(sample_ind, phrase_start, k) {
#
#     doc_tokens <- test$tokens[sample_ind, ]
#     these_tokens <- doc_tokens[phrase_start:(phrase_start + k - 1)]
#
#     collapse <- if (params$language == "english") " " else ""
#     return(paste(vocab[these_tokens + 1], collapse = collapse))
#   }
#
#
#   if (m == "all") {  # If we want to link all phrases to activations:
#
#     m <- nrow(phrase_act)
#     res <- phrase_act
#     res$text <- sapply(1:nrow(res), function(i) {
#       get_phrase(res$sample_id[i], res$phrase_id[i],
#                  as.numeric(gsub("CNN.([0-9]+) F[0-9]+", "\\1",
#                                  res$filter[i])))})
#
#   } else { # Get top m phrases for each CNN filter.
#     top_filt_phrases <- vector("list")
#     for (f in unique(phrase_act$filter)) {
#
#       these <- phrase_act[phrase_act$filter == f, ]
#       k <- gsub("CNN.([0-9]+) F[0-9]+", "\\1", f)
#
#       these <- these[order(these$activation, decreasing = TRUE), ]
#       these <- these[1:m, ]
#
#       # Collecting top m phrases for this filter and recording in the list
#       # top_filt_phrases.
#       these$phrase <- sapply(1:nrow(these), function(i){
#         get_phrase(these[i, "sample_id"], these[i, "phrase_id"],
#                    as.numeric(k))})
#       top_filt_phrases[[f]] <- paste0(these$phrase, ":   ",
#                                       round(these[, "activation"], 4))
#       top_filt_phrases[[paste0(f, "_id")]] <- these$sample_id
#
#     }
#
#     res <- as.data.frame(top_filt_phrases)
#
#   }
#
#
#   if (!is.null(save_path)) {
#     for (f in unique(phrase_act$filter)) {
#       Encoding(res[, gsub(" ", ".", f)]) <- "UTF8"
#     }
#     readr::write_excel_csv(res, paste0(save_path, "/top_", m, "_phrases.csv"))
#   }
#
#   return(res)
#
# }





