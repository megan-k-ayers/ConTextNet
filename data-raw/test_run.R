# -----------------------------------------------------------------------------
# Testing out package functionality so far.
# *** This script should not stay with the package once dev is done ***
# -----------------------------------------------------------------------------

rm(list = ls())
library(devtools)
library(dplyr)
load_all()
set.seed(123)
tensorflow::set_random_seed(123)

### Setup
n <- 50
# x <- read.csv("data-raw/imdb_full.csv")
x <- read.csv("data-raw/beer_reviews_cleaned.csv")[, c("text", "taste_score")]
names(x) <- c("text", "y")
x <- x[sample(1:nrow(x), n), ]

# Create fake covariate(s) to test
x$cov1 <- rnorm(n)
x$cov2 <- rnorm(n, mean = x$y, sd = 0.4)


model_params <- list("n_filts" = list(8),
                     "kern_sizes" = list(5),
                     "lr" = list(0.001),
                     "lambda_cnn" = list(0.001),
                     "lambda_corr" = list(0), "lambda_out" = list(0.005),
                     "epochs" = list(100), "batch_size" = list(32),
                     "patience" = 30,
                     "covars" = list(NULL, "cov1", "cov2"))

### Prep and embed
inputs <- prep_data(x = x, y_name = "y", text_name = "text",
                    model_params = model_params, task = "reg",
                    folder_name = "example", tune_method = "local",
                    embed_instr = list(max_length = 100), folds = 3)
input_embeds <- embed(inputs)

### Run parameter tuning, take best ones w.r.t. validation MSE
dat <- input_embeds$dat; embeds <- input_embeds$embeds
meta_params <- input_embeds$params; grid <- input_embeds$grid;
tokens <- input_embeds$tokens; vocab <- input_embeds$vocab

# tune_res <- tune_model(dat, embeds, meta_params, grid, tokens, vocab)
#
# # Aggregating over runs of the same setting.
# tune_res$act_range_avg <- apply(do.call(rbind, strsplit(tune_res$act_range, "|",
#                                                         fixed = TRUE)),
#                                 1, function(a) mean(as.numeric(a)))
# quant_cols <- grep("train|val|max_corr|avg", names(tune_res), value = TRUE)
# tune_res_agg <- tune_res %>%
#   group_by(id) %>%
#   summarise(across(all_of(quant_cols), ~ mean(.x, na.rm = TRUE)),
#             id = unique(id))
# tune_res_agg <- tune_res_agg[order(tune_res_agg$val_mse), ]
#
# best_params <- c(meta_params, get_row_list(tune_res[tune_res_agg$id[1], ]))
# input_embeds$params <- best_params

best_params <- c(meta_params, get_row_list(grid[9, ]))

### Train final model with the "best" parameters
model <- train_model(dat, embeds, best_params)

### Assess the model quantitatively on the test set.
test_embeds <- embeds[dat$fold == "test", , ]
test_y <- dat$y[dat$fold == "test"]
# test_cov <- as.matrix(dat[dat$fold == "test", best_params$covars])
# eval_model(model, list(test_embeds), test_y,
#            metrics = c("mse", "f1", "accuracy"))
eval_model(model, list(test_embeds), test_y,
           metrics = c("mse"))

### Basic interpretation
p_acts <- get_phrase_acts(model, test_embeds, best_params,
                          dat[dat$fold == "test", ])
get_top_phrases(p_acts, tokens[dat$fold == "test", ], best_params, vocab,
                m = 5)

d_acts <- get_doc_acts(model, test_embeds, best_params,
                       dat[dat$fold == "test", ])
plot_doc_acts(d_acts)

