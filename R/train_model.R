# ### Custom regularizer for CNNs - penalty when there is a high correlation
# ### between the outputs of each filter.
# corr_regularizer <- function(weight) {
#   function(w) {
#
#     # Penalty is at token sequence activation level (stack all token sequences)
#     w <- tf$reshape(w, shape = list(tf$multiply(tf$shape(w)[1], tf$shape(w)[2]),
#                                     tf$shape(w)[3]))
#
#     corrs <- tfp$stats$correlation(w)
#     corrs <- tf$subtract(corrs,  # Remove diagonal correlations of 1
#                          diag(1, nrow = params$num_filters))
#     corrs <- tf$maximum(corrs, 0)  # Set negative correlations to 0
#
#     if (params$corr_reg_fun == "max") {
#       penalty <- tf$reduce_max(corrs)
#     } else if (params$corr_reg_fun == "pos_sum") {
#       penalty <- tf$reduce_sum(corrs)
#     }
#
#     return(weight * penalty)
#
#   }
# }
#
#
# initialize_model <- function(params) {
#
#   ### Input layer and covariate layer (if given).
#   input_layer <- layer_input(shape = list(params$n_tokens, params$embed_dim),
#                              name = "input")
#
#   if (!is.null(params$covars)) {
#     input_covars_layer <- layer_input(shape = length(params$covars),
#                                       name = "covars")
#   }
#
#   if (!is.null(params$covars)) {
#     train_input <- list(input_layer, input_covars_layer)
#   } else {
#     train_input <- input_layer
#   }
#
#
#   ### 1D CNN layers and corresponding max pooled layers - one per kernel_size
#   ### given in params.
#   conv_layers <- vector("list", length(params$kernel_sizes))
#   names(conv_layers) <- params$kernel_sizes
#   for (size in params$kernel_sizes) {
#
#     kernel_reg <- regularizer_l2(params$lambda_cnn)
#     activity_reg <- corr_regularizer(params$corr_reg_weight)
#     name <- paste0("max_pool_", size)
#
#     conv1d_layer <- layer_conv_1d(filters = params$num_filters,
#                                   kernel_size = size,
#                                   activation = "sigmoid",
#                                   kernel_regularizer = kernel_reg,
#                                   activity_regularizer = activity_reg,
#                                   name = paste0("conv1d_", size))(input_layer)
#     pooling_layer <- layer_global_max_pooling_1d(name = name)(conv1d_layer)
#     conv_layers[[as.character(size)]] <- pooling_layer
#
#   }
#
#
#   ### Concatenate max pooled layers from multiple CNN layers together with
#   ### covariates, if given.
#   if (!is.null(params$covars)) {
#     conv_layers <- c(conv_layers, input_covars_layer)
#     concat_cnns <- tf$keras$layers$Concatenate()(unname(conv_layers))
#   } else if (length(params$kernel_sizes) > 1) {
#     concat_cnns <- tf$keras$layers$Concatenate()(unname(conv_layers))
#   } else {
#     concat_cnns <- conv_layers[[1]]
#   }
#
#
#   if (is.null(params$outcome_type)) {
#     loss_type <- "binary_crossentropy"
#     params$outcome_activation <- "sigmoid"
#     # activation <- "sigmoid"
#   } else if (params$outcome_type == "binary") {
#     loss_type <- "binary_crossentropy"
#     # activation <- "sigmoid"
#   } else if (params$outcome_type == "continuous") {
#     loss_type <- "mse"
#     # activation <- "linear"
#   }
#
#   ### Final dense layer.
#   out_reg <- regularizer_l1(params$lambda_out)
#   output_layer <- layer_dense(units = 1,
#                               activation = params$outcome_activation,
#                               kernel_regularizer = out_reg,
#                               name = "output")(concat_cnns)
#
#
#   ### Create and compile model.
#   lr <- if (is.null(params$lr)) 0.001 else params$lr
#   model <- keras_model(inputs = train_input,
#                        outputs = output_layer)
#   model %>% compile(optimizer = tf$keras$optimizers$legacy$Adam(learning_rate = lr),
#                     loss = loss_type,
#                     metrics = "mean_absolute_error")
#
#   return(model)
#
# }
