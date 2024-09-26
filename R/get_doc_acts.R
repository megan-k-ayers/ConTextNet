###############################################################################
###                   GET MAX POOLED FILTER ACTIVATIONS FROM
###                   TRAINED CONTEXTNET MODEL + INPUT DATA
###
### Runs:         Locally and on HPC cluster.
### Status:       Almost complete - only tests and documentation remaining.
### Priority:     Medium.
### User facing:  Yes.
###############################################################################


#' Get Document Level Filter Activation Data Frame
#'
#' @param model R Keras trained model
#' @param embeds Matrix of word embeddings to get activations for
#' @param params Model parameter list
#' @param dat Original data set
#'
#' @return Data frame with convolutional layer activations for all input word
#'         embeddings.
#'
#' @examples \dontrun{
#' model <- train_model(imdb_embed$dat, imdb_embed$embeds, imdb_embed$params)
#' embeds <- imdb_embed$embeds[imdb_embed$dat$fold == "train", , ]
#' dat <- imdb_embed$dat[imdb_embed$dat$fold == "train", ]
#' params <- imdb_embed$params
#' res <- get_doc_acts(model, embeds, params, dat)
#' }
get_doc_acts <- function(model, embeds, params, dat) {

  ### Unpack frequently used parameters.
  kern_sizes <- params$kern_sizes
  n_filts <- params$n_filts
  filt_names <- as.character(sapply(params$kern_sizes, function(k) {
    paste0("CNN", k, "_", "F", 1:params$n_filts)}))

  ### Prep inputs.
  if (!is.null(params$covars)) {
    input <- list(embeds, as.matrix(dat[, params$covars]))
  } else input <- embeds

  ### Create data frame of max-pooled filter activations. Each col corresponds
  ### with a different filter.
  acts <- matrix(nrow = nrow(dat), ncol = n_filts*length(kern_sizes))
  acts <- as.data.frame(acts)
  names(acts) <- filt_names
  for (j in 1:length(params$kern_sizes)) {  # One CNN per kernel size
    k <- params$kern_sizes[j]
    this <- paste0("max_pool_", k)
    int_model <- keras::keras_model(inputs = model$input,
                                    outputs = model$get_layer(this)$output)
    these_acts <- stats::predict(int_model, input, verbose = 0)
    these_acts <- as.data.frame(these_acts)
    names(these_acts) <- paste0("CNN", k, "_", "F", 1:params$n_filts)
    acts[, names(these_acts)] <- these_acts
  }

  acts$sample_id <- 1:nrow(acts)
  acts <- tidyr::pivot_longer(acts, cols = grep("CNN", names(acts)),
                              names_to = "filter", values_to = "activation")
  acts <- as.data.frame(acts)

  ### Attach output layer weights associated with each filter and the label.
  out_wts <- get_output_wts(model, params)
  out_wts <- out_wts[grepl("^CNN", out_wts$feature), ]
  acts <- merge(acts, out_wts, by.x = "filter", by.y = "feature")

  dat$sample_id <- 1:nrow(dat)
  acts <- merge(acts, dat, by = "sample_id")

  return(acts[order(acts$sample_id), ])
}
