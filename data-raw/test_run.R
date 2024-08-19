# -----------------------------------------------------------------------------
# Testing out package functionality so far.
# *** This script should not stay with the package once dev is done ***
# -----------------------------------------------------------------------------

rm(list = ls())
library(devtools)
load_all()
set.seed(123)
tensorflow::set_random_seed(123)

imdb_full <- read.csv("data-raw/imdb_full.csv")
imdb_full <- imdb_full[sample(1:nrow(imdb_full), 100), ]

model_params <- list("n_filts" = list(4, 8),
                     "kern_sizes" = list(c(5), c(3), c(3, 5)),
                     "lr" = list(0.001, 0.0001),
                     "lambda_cnn" = list(0.001, 0.0001),
                     "lambda_corr" = list(0), "lambda_out" = list(0.005, 0.01),
                     "epochs" = list(100), "batch_size" = list(32),
                     "patience" = 30,  "covars" = list(NULL))
inputs <- prep_data(x = imdb_full, y_name = "y", text_name = "text",
                    model_params = model_params, task = "class",
                    folder_name = "example", tune_method = "local",
                    embed_instr = list(max_length = 100), folds = 3)
input_embeds <- embed(inputs)


dat <- input_embeds$dat; embeds <- input_embeds$embeds
meta_params <- input_embeds$params; grid <- input_embeds$grid;
tokens <- input_embeds$tokens; vocab <- input_embeds$vocab

tune_res <- tune_model(dat, embeds, meta_params, grid, tokens, vocab)




model <- train_model(input_embeds$dat, input_embeds$embeds, input_embeds$params)

test_embeds <- input_embeds$embeds[input_embeds$dat$fold == "test", , ]
test_y <- input_embeds$dat$y[input_embeds$dat$fold == "test"]
eval_model(model, test_embeds, test_y, metrics = c("mse", "f1", "accuracy"))

p_acts <- get_phrase_acts(model, test_embeds, input_embeds$params)
get_top_phrases(p_acts, input_embeds$tokens[input_embeds$dat$fold == "test", ],
                input_embeds$params, input_embeds$vocab)
get_top_phrases_quick(p_acts,
                      input_embeds$tokens[input_embeds$dat$fold == "test", ],
                      input_embeds$params, input_embeds$vocab)

d_acts <- get_doc_acts(model, test_embeds, input_embeds$params,
                       input_embeds$dat[input_embeds$dat$fold == "test", ])
plot_doc_acts(d_acts)

