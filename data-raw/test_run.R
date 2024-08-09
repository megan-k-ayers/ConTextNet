# -----------------------------------------------------------------------------
# Testing out package functionality so far.
# *** This script should not stay with the package once dev is done ***
# -----------------------------------------------------------------------------

rm(list = ls())
load_all()

imdb_full <- read.csv("data-raw/imdb_full.csv")
imdb_full <- imdb_full[sample(1:nrow(imdb_full), 2000), ]

model_params <- list("n_filts" = list(2), "kern_sizes" = list(c(3, 5)),
                     "lr" = list(0.0001), "lambda_cnn" = list(0),
                     "lambda_corr" = list(0), "lambda_out" = list(0),
                     "epochs" = list(100), "batch_size" = list(32),
                     "patience" = 20,  "covars" = list(NULL))
# inputs <- prep_data(x = imdb_full, y_name = "y", text_name = "text",
#                     model_params = model_params, task = "class",
#                     folder_name = "example",
#                     embed_instr = list(max_length = 50))
# input_embeds <- embed(inputs)
# saveRDS(input_embeds, "data-raw/imdb_embeds_temp.RDS")

input_embeds <- readRDS("data-raw/imdb_embeds_temp.RDS")
model <- train_model(input_embeds)

p_acts <- get_phrase_acts(model, input_embeds$embeds, input_embeds$params)
get_top_phrases(p_acts, input_embeds$tokens, input_embeds$params,
                input_embeds$vocab)

d_acts <- get_doc_acts(model, input_embeds$embeds, input_embeds$params,
                       input_embeds$dat)
plot_doc_acts(d_acts)

