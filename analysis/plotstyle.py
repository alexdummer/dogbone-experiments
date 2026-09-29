import seaborn as sns

sns.set_theme(
    context="paper",
    style="ticks",
    font_scale=1,
    rc={
        "lines.linewidth": 1.0,
        "text.usetex": True,
        "font.family": "cm",
        "legend.frameon": False,
        "legend.fontsize": "small",
    },
)

sns.set_palette("colorblind")

colors = sns.color_palette()

linestyles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5, 1, 5))]

# figure sizes
figsize_single = (8 / 2.54, 7 / 2.54)
figsize_double = (19 / 2.54, 7 / 2.54)
