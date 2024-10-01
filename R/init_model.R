###############################################################################
###                         INITIALIZE CONTEXTNET MODEL
###
### Runs:         On HPC cluster.
### Status:       Almost complete - only tests and documentation remaining.
### Priority:     Medium.
### User facing:  No.
###############################################################################


#' Activation Correlation Regularizer
#'
#' Custom regularizer for CNN layers that applies a penalty proportional to the
#' max correlation between activations of filters within a convolutional layer.
#' Note: the nested function structure is necessary for passing in additional
#' parameters.
#'
#' @param weight Weight to apply to this penalty.
#' @param f Number of filters for this convolutional layer.
#'
#' @return The penalty value.
#'
#' @examples
corr_regularizer <- function(weight, f) {
  function(w) {

    # Penalty is at token sequence activation level (stack all token sequences)
    w <- tf$reshape(w, shape = list(tf$multiply(tf$shape(w)[1], tf$shape(w)[2]),
                                    tf$shape(w)[3]))

    # Calculating correlation manually - tensorflow_probability would be much
    # easier, but experiencing some version control issues with tensorflow
    # version that R Keras expects.
    mean_w <- tf$reduce_mean(w, axis = as.integer(0), keepdims = TRUE)
    covs <- tf$divide(tf$matmul(tf$transpose(tf$subtract(w, mean_w)),
                                tf$subtract(w, mean_w)),
                      tf$cast(tf$shape(w)[1], tf$float32))
    stdev <- tf$sqrt(tf$linalg$diag_part(covs))
    stdev <- tf$tensordot(stdev, stdev, axes = as.integer(0))
    corrs <- tf$divide(covs, stdev)

    # Remove correlations of 1 on the diagonal.
    corrs <- tf$subtract(corrs, diag(1, nrow = f))

    # Take max correlation as penalty (reminder that this will be 0 if all
    # correlations are negative because the variance diag was set to 0)
    return(weight * tf$reduce_max(corrs))

  }
}


#' Create 1D CNN and Max Pooled Layers for ConTextNet
#'
#' @param k Kernel size (int)
#' @param f Number of filters (int)
#' @param l_cnn Kernel regularizer weight (numeric)
#' @param l_corr Correlation regularizer weight (numeric)
#' @param input_layer Layer corresponding to word embedding input
#'
#' @return R Keras model layer.
#'
#' @examples
cnn_mp_layer <- function(k, f, l_cnn, l_corr, input_layer) {

  kernel_reg <- keras::regularizer_l2(l_cnn)  # CNN kernel regularizer
  activity_reg <- corr_regularizer(l_corr, f)  # Correlation regularizer
  conv1d_layer <- input_layer %>%  # 1D Convolutional layer
    keras::layer_conv_1d(filters = f,
                         kernel_size = k,
                         activation = "sigmoid",
                         kernel_regularizer = kernel_reg,
                         activity_regularizer = activity_reg,
                         name = paste0("conv1d_", k))
  pooling_layer <- conv1d_layer %>%  # Max pooled layer (within document)
    keras::layer_global_max_pooling_1d(name = paste0("max_pool_", k))

  return(pooling_layer)
}


#' Initialize ConTextNet Model
#'
#' @param params List of model parameters - from tune_prep() or directly from
#'        data_prep().
#' @param train_dat Embedding *training* data set, as a list also including
#'        covariates as second element if they are used.  Used to set
#'        normalization layers.
#'
#' @return A compiled keras model.
#' @export
#'
#' @examples
#' \dontrun{
#' init_model(imdb_embed$params)
#' }
init_model <- function(params, train_dat) {

  ### Create input layer.
  input <- keras::layer_input(shape = list(params$n_tokens, params$embed_dim),
                              name = "input")

  ### Create covariate layer and concatenate training inputs (if needed).
  if (!is.null(params$covars)) {
    input_covars <- keras::layer_input(shape = length(params$covars),
                                       name = "covars")
    train_input <- list(input, input_covars)

    # Normalize covariates - use training data to adapt the normalization
    # layer.
    normed_covars <- tf$keras$layers$Normalization(axis = as.integer(-1),
                                                   trainable = FALSE)
    normed_covars$adapt(as.matrix(train_dat[[2]]))
    normed_covars <- normed_covars(input_covars)

  } else train_input <- input


  ### Create 1D CNN layers and corresponding max pooled layers - one per
  ### kern_sizes given in params.
  conv_layers <- lapply(params$kern_sizes, cnn_mp_layer, f = params$n_filts,
                        l_cnn = params$lambda_cnn, l_corr = params$lambda_corr,
                        input_layer = input)
  names(conv_layers) <- params$kern_sizes

  ### Concatenate max pooled layers from multiple CNN layers together with
  ### covariates, if given.
  if (!is.null(params$covars)) {
    conv_layers <- c(conv_layers, normed_covars)
    concat_cnns <- tf$keras$layers$Concatenate()(unname(conv_layers))
  } else if (length(params$kern_sizes) > 1) {
    concat_cnns <- tf$keras$layers$Concatenate()(unname(conv_layers))
  } else concat_cnns <- conv_layers[[1]]

  ### Define loss and activation functions.
  loss_type <- if (params$task == "class") "binary_crossentropy" else "mse"
  out_act <- if (params$task == "class") "sigmoid" else "linear"

  ### Final dense layer.
  out_reg <- keras::regularizer_l1(params$lambda_out)
  output_layer <- keras::layer_dense(units = 1, activation = out_act,
                                     kernel_regularizer = out_reg,
                                     name = "output")(concat_cnns)

  ### Create and compile model.
  optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate = params$lr)
  model <- keras::keras_model(inputs = train_input, outputs = output_layer)
  model %>% keras::compile(optimizer = optimizer, loss = loss_type,
                           metrics = "mean_absolute_error")

  return(model)

}
