# -----------------------------------------------------------------------------
# Testing out package functionality so far.
# *** This script should not stay with the package once dev is done ***
# -----------------------------------------------------------------------------

rm(list = ls())
library(devtools)
load_all()

imdb_full <- read.csv("data-raw/imdb_full.csv")
imdb_full <- imdb_full[sample(1:nrow(imdb_full), 50), ]

model_params <- list("n_filts" = list(2), "kern_sizes" = list(c(3, 5)),
                     "lr" = list(0.00001), "lambda_cnn" = list(0.01),
                     "lambda_corr" = list(0), "lambda_out" = list(0.01),
                     "epochs" = list(100), "batch_size" = list(25),
                     "patience" = 20,  "covars" = list(NULL))
inputs <- prep_data(x = imdb_full, y_name = "y", text_name = "text",
                    model_params = model_params, task = "class",
                    folder_name = "example",
                    embed_instr = list(max_length = 50))
input_embeds <- embed(inputs)
tf$keras$backend$clear_session()
gc()
saveRDS(input_embeds, "data-raw/imdb_embeds_temp.RDS")

these <- readRDS("data-raw/imdb_embeds_temp.RDS")
model <- train_model(these)
eval_model(model, these$embeds, these$dat$y,
           metrics = c("mse", "f1", "accuracy"))

p_acts <- get_phrase_acts(model, these$embeds, these$params)
get_top_phrases(p_acts, these$tokens, these$params,
                these$vocab)

d_acts <- get_doc_acts(model, these$embeds, these$params,
                       these$dat)
plot_doc_acts(d_acts)

