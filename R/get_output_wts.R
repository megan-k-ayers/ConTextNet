###############################################################################
###            GET FINAL LAYER WEIGHTS FROM TRAINED CONTEXTNET MODEL
###
### Runs:         Locally and on HPC cluster.
### Status:       Almost complete - mainly tests and documentation remaining.
### Priority:     Medium.
### User facing:  Yes.
###############################################################################


#' Get Output Layer Weights
#'
#' @param model Keras model with a final dense layer named "output"
#' @param params Model parameters
#'
#' @return Data frame with output layer weights corresponding to each
#'         convolutional layer filter (and potentially covariates)
#' @export
#'
#' @examples \dontrun{
#' model <- train_model(imdb_embed$dat, imdb_embed$embeds, imdb_embed$params)
#' params <- imdb_embed$params
#' get_output_wts(model, params)
#' }
get_output_wts <- function(model, params) {
  out_wts <- model$get_layer("output")$get_weights()[[1]]
  filt_names <- as.character(sapply(params$kern_sizes, function(k) {
    paste0("CNN", k, "_", "F", 1:params$n_filts)}))

  # Covariate handling
  if (!is.null(params$covars)) {
    out_wts <- data.frame(feature = c(filt_names, params$covars),
                          wt = out_wts)
  } else {
    out_wts <- data.frame(feature = filt_names, wt = out_wts)
  }

  return(out_wts)
}
