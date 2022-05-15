def Gaus2D(img_w: int, img_h: int, center_coord_x, center_coord_y):
    center_x = center_coord_x + img_w / 2
    center_y = img_h / 2 - center_coord_y
    x_range = np.arange(0, img_w, 1)
    y_range = np.arange(0, img_h, 1)
    # xx, yy = np.meshgrid(x_range, y_range, sparse=True)
    sigma = min(img_w, img_h) / 4
    x_mean = center_x
    y_mean = center_y
    gx = np.exp(-(x_range - x_mean) ** 2 / (2 * sigma ** 2))
    gy = np.exp(-(y_range - y_mean) ** 2 / (2 * sigma ** 2))
    g = 1 - np.outer(gx, gy)
    return g


def retrieve_depth(X_coord: float, Y_coord: float, depth_image):
    X_cam = depth_image.shape[1] / 2 + X_coord
    Y_cam = depth_image.shape[0] / 2 - Y_coord
    # print(f"depth found {depth_image[int(Y_cam), int(X_cam)]}")
    return depth_image[int(Y_cam), int(X_cam)]


def step(objective, obstacles, neighbors) -> Tuple[float, float, float]:
    img_w = 320
    img_h = 240
    depth = obstacles
    depth = np.where(depth > 0.8, 1, depth)
    # print(depth.shape)
    mask = -Gaus2D(img_h, img_w, objective[0], objective[1])

    minimum_val = np.min(mask)
    mask = (mask - minimum_val) / (- minimum_val)
    non_bin_mask = mask * depth

    maximum_val = np.amax(non_bin_mask)
    maximum_index = np.where(non_bin_mask == maximum_val)
    maximum_index = (maximum_index[0][0], maximum_index[1][0])
    best_index = list(maximum_index)
    print(f"objective {objective}")
    # print(maximum_val)
    # print(best_index)

    z = best_index[0]
    y = best_index[1]
    try:
        window = 20
        sub = non_bin_mask[best_index[0] - window: best_index[0] + window,
              best_index[1] - window: best_index[1] + window]

        M = cv2.moments(sub)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        z += (round(cy) - window)
        y += (round(cx) - window)
    except (ValueError, ZeroDivisionError):
        pass

    # print(non_bin_mask.shape)
    # plt.imshow(non_bin_mask)
    # plt.scatter(y, z, c="red", marker="o")
    # plt.show()
    y = -(y - img_w / 2)
    z = img_h / 2 - z
    x = retrieve_depth(-y, z, depth)
    print(f"final coordinates {x} {y} {z}")
    return x, y, z