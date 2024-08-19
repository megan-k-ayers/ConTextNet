#' Get Output Layer Weights
#'
#' @param model Keras model with a final dense layer named "output"
#' @param kern_sizes Array of convolutional layer kernel sizes in the model
#' @param f Integer, number of filters per convolutional layer
#'
#' @return Data frame with output layer weights corresponding to each
#'         convolutional layer filter (and potentially covariates)
#' @export
#'
#' @examples \dontrun{
#' model <- train_model(imdb_embed$dat, imdb_embed$embeds, imdb_embed$params)
#' params <- imdb_embed$params
#' get_output_wts(model, params$kern_sizes, params$n_filts)
#' }
get_output_wts <- function(model, kern_sizes, f) {
  ### TODO: Fix for when covariates are included.
  out_wts <- model$get_layer("output")$get_weights()[[1]]
  filt_names <- as.character(sapply(kern_sizes, function(k) {
    paste0("CNN", k, "_", "F", 1:f)}))
  out_wts <- data.frame(filter = filt_names, wt = out_wts)
  return(out_wts)
}
