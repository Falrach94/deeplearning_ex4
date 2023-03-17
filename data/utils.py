

def get_distribution(df, label_provider):
    om = label_provider.get_output_mode()
    label_provider.set_output_mode('idx')

    label_series = df.apply(
        lambda row: label_provider.get_label_from_row(row),
        axis=1
    )

    categories = label_series.value_counts()
    for i in range(label_provider.class_count(False)):
        if i not in categories.index:
            categories[i] = 0

    categories = list(categories.sort_index())
    label_provider.output_mode = om
    return categories


def split_df_by_category(df):
    id_col = 'label_id'
    cat_cnt = max(df[id_col])+1
    return [df[df[id_col]==i] for i in range(cat_cnt)]