#' Activation Correlation Regularizer
#'
#' Custom regularizer for CNN layers that applies a penalty proportional to the
#' max correlation between activations of filters within a convolutional layer.
#' Note: the nested function structure is necessary for passing in weight and
#' params.
#'
#' @param weight Weight to apply to this penalty.
#' @param params List of model parameters.
#'
#' @return The penalty value.
#' @export
#'
#' @examples
corr_regularizer <- function(weight, params) {
  function(w) {

    # Penalty is at token sequence activation level (stack all token sequences)
    w <- tf$reshape(w, shape = list(tf$multiply(tf$shape(w)[1], tf$shape(w)[2]),
                                    tf$shape(w)[3]))
    corrs <- tfp$stats$correlation(w)

    # Remove correlations of 1 on the diagonal, set negative correlations to 0.
    corrs <- tf$subtract(corrs, diag(1, nrow = params$n_filts))
    corrs <- tf$maximum(corrs, 0)

    return(weight * tf$reduce_max(corrs))  # Take max correlation as penalty

  }
}


#' Initialize ConTextNet Model
#'
#' @param params List of model parameters - from tune_prep() or directly from
#'        data_prep().
#'
#' @return A compiled keras model.
#' @export
#'
#' @examples
#' \dontrun{
#' model_params <- list("n_filts" = list(2), "kern_sizes" = list(c(3, 5)),
#'                      "lr" = list(0.0001), "lambda_cnn" = list(0),
#'                      "lambda_corr" = list(0), "lambda_out" = list(0),
#'                      "epochs" = list(20), "batch_size" = list(32),
#'                      "covars" = list(NULL))
#' input_list <- prep_data(x = imdb, y_name = "y", text_name = "text",
#'                         model_params = model_params, task = "class",
#'                         folder_name = "example")
#' res <- embed(input_list)
#' init_model(res$params)
#' }
init_model <- function(params) {
  ### TODO: Eventually, examples should use intermediate files and not reference
  ### other functions.

  ### Create input layer.
  input_layer <- keras::layer_input(shape = list(params$n_tokens,
                                                 params$embed_dim),
                                    name = "input")

  ### Create covariate layer and concatenate training inputs (if needed).
  if (!is.null(params$covars)) {
    input_covars_layer <- keras::layer_input(shape = length(params$covars),
                                             name = "covars")
    train_input <- list(input_layer, input_covars_layer)
  } else {
    train_input <- input_layer
  }

  ### 1D CNN layers and corresponding max pooled layers - one per kern_sizes
  ### given in params.
  conv_layers <- vector("list", length(params$kern_sizes))
  names(conv_layers) <- params$kern_sizes
  for (size in params$kern_sizes) {

    kernel_reg <- keras::regularizer_l2(params$lambda_cnn)
    activity_reg <- corr_regularizer(params$lambda_corr, params)

    cnn_name <- paste0("conv1d_", size)
    mp_name <- paste0("max_pool_", size)
    conv1d_layer <- input_layer %>%
      keras::layer_conv_1d(filters = params$n_filts,
                           kernel_size = size,
                           activation = "sigmoid",
                           kernel_regularizer = kernel_reg,
                           activity_regularizer = activity_reg,
                           name = cnn_name)
    pooling_layer <- conv1d_layer %>%
      keras::layer_global_max_pooling_1d(name = mp_name)
    conv_layers[[as.character(size)]] <- pooling_layer

  }

  ### Concatenate max pooled layers from multiple CNN layers together with
  ### covariates, if given.
  if (!is.null(params$covars)) {
    conv_layers <- c(conv_layers, input_covars_layer)
    concat_cnns <- tf$keras$layers$Concatenate()(unname(conv_layers))
  } else if (length(params$kern_sizes) > 1) {
    concat_cnns <- tf$keras$layers$Concatenate()(unname(conv_layers))
  } else {
    concat_cnns <- conv_layers[[1]]
  }

  ### Define loss and activation functions.
  if (params$task == "class") {
    loss_type <- "binary_crossentropy"
    out_act <- "sigmoid"
  } else if (params$task == "reg") {
    loss_type <- "mse"
    out_act <- "linear"
  }

  ### Final dense layer.
  out_reg <- keras::regularizer_l1(params$lambda_out)
  output_layer <- keras::layer_dense(units = 1,
                                     activation = out_act,
                                     kernel_regularizer = out_reg,
                                     name = "output")(concat_cnns)

  ### Create and compile model.
  optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate = params$lr)
  model <- keras::keras_model(inputs = train_input,
                              outputs = output_layer)
  model %>% keras::compile(optimizer = optimizer,
                           loss = loss_type,
                           metrics = "mean_absolute_error")

  return(model)

}
