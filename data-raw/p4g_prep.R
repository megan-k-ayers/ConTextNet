rm(list = ls())
library(ggplot2)
library(dplyr)

# Names for data frames is coming from the readme.md file that comes with the
# Persuasion for Good data.
x <- read.csv("data-raw/persuasion_for_good/full_info.csv")
names(x)[1:5] <- c("dial_id", "user_id", "role", "donation", "n_turns")
x$role <- ifelse(x$role == 0, "persuader", "persuadee")
head(x)
length(unique(x$user_id))
length(unique(x$dial_id[x$role == "persuadee"]))

# Some people both persuaded and were persuadees... Only 584 "pure" persuadees.
temp <- x %>%
  group_by(user_id) %>%
  summarize(persuader = sum(role == "persuader"),
            persuadee = sum(role == "persuadee"))
table("persuader" = temp$persuader > 0, "persuadee" = temp$persuadee > 0)

# Ignoring repeat users/roles, focus only on donations by persuadees.
x <- x[x$role == "persuadee", ]
table(table(x$dial_id))
table(table(x$user_id))
table(x$donation)  # Veerrry skewed.
hist(x$donation)
hist(log(x$donation))  # This drops the $0's - log1p not going to help much.
hist(x$donation[x$donation < 5], breaks = 100)

# Simplify and look at just whether or not the persuadee donated. Losing
# information -- can also try modeling donation directly. Or if we really
# wanted to get into it, a 2 part model where first we model whether someone
# donated, and then model continuous donation value for those that did.
table(x$donated)
x$donated <- x$donation > 0

# Merging persuadee donation info with dialogue text.
y <- read.csv("data-raw/persuasion_for_good/full_dialog.csv")[, -1]
names(y)[1:4] <- c("text", "turn", "role", "dial_id")
y$role <- ifelse(y$role == 0, "persuader", "persuadee")

y <- merge(x[, c("dial_id", "donation", "donated", "n_turns")],
           y, by.x = "dial_id", by.y = "dial_id")
length(unique(y$dial_id))

# Some conversations are flagged as "BAD" - drop them.
View(y[grep("BAD", y$dial_id), ])
length(unique(y$dial_id[grep("BAD", y$dial_id)]))

# Also filtering to only the persuaders' dialogue.
y <- y[y$role == "persuader" & !grepl("BAD", y$dial_id), ]
length(unique(y$dial_id))

# Approach 1: Stitch entire persuader text together (omitting persuadee
# responses).
y_all <- y %>%
  group_by(dial_id) %>%
  summarize(text = paste(text, collapse = " "),
            donation = unique(donation),
            donated = unique(donated),
            n_turns = unique(n_turns))
hist(stringr::str_count(y_all$text, " "))
y_all <- y_all[, c("text", "donation", "donated")]
write.csv(y_all, "data-raw/persuasion_for_good/all-cleaned.csv",
          row.names = FALSE)

# Approach 2: Take only the first three persuaders' statements (first 1-2 is
# often just "Hi" and a brief intro).
y_first <- y[y$turn %in% 0:2, ] %>%
  group_by(dial_id) %>%
  summarize(text = paste(text, collapse = " "),
            donation = unique(donation),
            donated = unique(donated),
            n_turns = unique(n_turns))
hist(stringr::str_count(y_first$text, " "))
y_first <- y_first[, c("text", "donation", "donated")]
write.csv(y_first, "data-raw/persuasion_for_good/first-cleaned.csv",
          row.names = FALSE)

# Approach 3: Remove initial statements with less than 5 words (intros),
# but leave each remaining statement as its own case. Major independence
# violations between cases but maybe the model will be able to find more
# patterns when looking at larger number of shorter statements...
intro_rows <- which(y$turn == 0 & stringr::str_count(y$text, " ") < 5)
table(y$text[intro_rows])
y_indiv <- y[-intro_rows, c("text", "donation", "donated")]
write.csv(y_indiv, "data-raw/persuasion_for_good/indiv-cleaned.csv",
          row.names = FALSE)


