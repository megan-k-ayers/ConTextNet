# -----------------------------------------------------------------------------
# Generating toy examples of intermediate data sets to use for package testing
# and examples.
# -----------------------------------------------------------------------------

load_all()

# Specify toy example parameters, use with IMDB data.
model_params <- list("n_filts" = list(2), "kern_sizes" = list(c(3, 5)),
                     "lr" = list(0.0001), "lambda_cnn" = list(0),
                     "lambda_corr" = list(0), "lambda_out" = list(0),
                     "epochs" = list(20), "batch_size" = list(32),
                     "patience" = 15,  "covars" = list(NULL))
imdb_input_list <- prep_data(x = imdb, y_name = "y", text_name = "text",
                             model_params = model_params, task = "class",
                             folder_name = "example",
                             embed_instr = list(max_length = 50))

imdb_embed <- embed(imdb_input_list)

usethis::use_data(imdb_input_list, imdb_embed, internal = TRUE, overwrite = TRUE)
