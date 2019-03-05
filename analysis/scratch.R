library(gridExtra)
top <- data  %>%   ggplot(aes(y=`recall@10`, x=speaker_id, color=s2i, shape=s2i)) + geom_point(size=5) + #geom_smooth(method="lm") +
  theme(aspect.ratio=2/3, text=element_text(size=20)) +
  ylim(0.19, 0.3) + 
  xlab("Accuracy of speaker ID") +
  ylab("Recall@10")

bot <- data %>%   ggplot(aes(y=`recall@10`, x=speaker_id, color=s2t, shape=s2t)) + geom_point(size=5) + #geom_smooth(method="lm") +
  theme(aspect.ratio=2/3, text=element_text(size=20)) +
  ylim(0.19, 0.3) + 
  xlab("Accuracy of speaker ID") +
  ylab("Recall@10") 
grid.arrange(top, bot, nrow=2)
ggsave("~/repos/Perceptual-and-symbolic-correlates-of-spoken-language/spkrinv-grid2.pdf", arrangeGrob(top, bot), 
       units='cm', width=20, height=20)


# Both panels
summary(lm(`recall@10` ~ speaker_id, data=data))$r.squared

# Raw R2
# Top panel
summary(lm(speaker_id ~ s2i, data=data))$r.squared
summary(lm(`recall@10` ~ s2i, data=data))$r.squared

#Bottom panel
summary(lm(speaker_id ~ s2t, data=data))$r.squared
summary(lm(`recall@10` ~ s2t, data=data))$r.squared

# Coefficient of partial determination

r2.part <- function(f.full, f.reduced, data) {
  full <- lm(f.full, data=data)
  red <- lm(f.reduced, data=data)
  r <- (sum(red$residuals^2) - sum(full$residuals^2))/sum(red$residuals^2)
  return(r)
}

# Controlling for the other var
# Top panel
r2.part(speaker_id ~ s2i + s + t + t2i + s2t, speaker_id ~ s + t + t2i + s2t, data=data)
r2.part(`recall@10` ~  s2i + s + t + t2i + s2t, `recall@10` ~ s + t + t2i + s2t, data=data)


# Bottom panel
r2.part(speaker_id ~ s2t + s + t + t2i + s2i, speaker_id ~ s + t + t2i + s2i, data=data)
r2.part(`recall@10` ~  s2t + s + t + t2i + s2i, `recall@10` ~ s + t + t2i + s2i, data=data)


