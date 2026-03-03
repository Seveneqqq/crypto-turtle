from sklearn.model_selection import train_test_split


def split_data(X, y, split_type="random"):
    if split_type == "random":
        return train_test_split(X, y, test_size=0.2, random_state=42)

    elif split_type == "time":
        split_index = int(len(X) * 0.8)
        return (
            X[:split_index],
            X[split_index:],
            y[:split_index],
            y[split_index:]
        )

    else:
        raise ValueError("Unknown split type")