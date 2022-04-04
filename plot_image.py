def visualize_predictions(
    models,
    epoch=1,
    path="",
    downsample_size=(256, 256),
    preprocessed_folder: str = "",
    dataset="kmni",
):
    plt.clf()
    with t.no_grad():
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        merge_nodes = issubclass(UnetModel, type(model))
        loader, _, _ = get_loaders(
            train_batch_size=2,
            test_batch_size=2,
            preprocessed_folder=preprocessed_folder,
            device=device,
            downsample_size=downsample_size,
            dataset=dataset,
            merge_nodes=False,
            shuffle=False,
        )
        loader, _, _ = get_loaders(
            train_batch_size=2,
            test_batch_size=2,
            preprocessed_folder=preprocessed_folder,
            device=device,
            downsample_size=downsample_size,
            dataset=dataset,
            merge_nodes=True,
            shuffle=False,
        )
        for key, val in models.items():
            pass
        model.eval()
        N_COLS = 4  # frames
        N_ROWS = 3  # x, y, preds
        plt.title(f"Epoch {epoch}")
        _fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS)
        for x, y in loader:
            for k in range(len(x)):
                raininess = t.sum(x[k] != 0) / t.prod(t.tensor(x[k].shape))
                if raininess >= 0.5:
                    preds = model(x)
                    to_plot = [x[k], y[k], preds[k]]
                    for i, row in enumerate(ax):
                        for j, col in enumerate(row):
                            # ipdb.set_trace()
                            col.imshow(
                                to_plot[i].cpu().detach().numpy()[:, :, j, 1]
                                if not merge_nodes
                                else to_plot[i]
                                .cpu()
                                .detach()
                                .numpy()[j, : downsample_size[0], : downsample_size[1],]
                            )

                    row_labels = ["x", "y", "preds"]
                    for ax_, row in zip(ax[:, 0], row_labels):
                        ax_.set_ylabel(row)

                    col_labels = ["frame1", "frame2", "frame3", "frame4"]
                    for ax_, col in zip(ax[0, :], col_labels):
                        ax_.set_title(col)

                    plt.savefig(os.path.join(path, f"pred_{epoch}.png"))
                    plt.close()
                    model.train()
                    return


if __name__ == "__main__":
    visualize_predictions({})
