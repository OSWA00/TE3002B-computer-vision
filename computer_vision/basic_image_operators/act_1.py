import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path: str) -> np.ndarray:
    """Load image using cv2.imread"""
    image = cv2.imread(path)
    return image


def plot_cv2_to_plt(img_array: list, title: str, gray: bool) -> None:
    """Plot array with CV2 image arrays."""
    fig, ax = plt.subplots(5, 2)

    for index, img in enumerate(img_array):

        row = index % 5
        column = index % 2

        if gray:
            ax[row][column].imshow(img, interpolation="nearest", cmap="gray")
        else:
            ax[row][column].imshow(img, interpolation="nearest")
        ax[row][column].axis("off")

    plt.savefig(f"data/act_1/plots/{title}_plot.png")
    plt.show()


def save_img(img_array: list, file_path: str, base_file_name: str) -> None:
    """Save image array as a png file."""
    for index, img in enumerate(img_array):
        cv2.imwrite(f"data/act_1/{file_path}/{base_file_name}_{index}.png", img)


def add_gaussian_noise(img):
    gauss = np.random.normal(0, 1, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype("uint8")
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(img, gauss)
    return img_gauss


if __name__ == "__main__":

    # Initial configuration
    base_path = "data/road_sign_detection/images"

    # Exercise 1
    img_cv2 = [load_image(f"{base_path}/road{i}.png") for i in range(10)]
    img_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_cv2]
    img_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_cv2]

    plot_cv2_to_plt(img_gray, "act1_gray", gray=True)
    save_img(img_gray, "gray", "gray")
    plot_cv2_to_plt(img_rgb, "act1_rgb", gray=False)
    save_img(img_cv2, "color", "bgr")

    # Exercise 2
    img_binary = []
    for img in img_gray:
        ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        img_binary.append(thresh)
    plot_cv2_to_plt(img_binary, "act1_binary", gray=True)
    save_img(img_binary, "binary", "binary")

    # Exercise 3
    # Green HSV values
    lower_color_bound = (36, 25, 25)
    upper_color_bound = (86, 255, 255)

    img_binary_color = []
    for index, img in enumerate(img_cv2):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color_bound, upper_color_bound)
        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        img_binary_color.append(green)

    plot_cv2_to_plt(img_binary_color, "act1_binary_color", gray=True)
    save_img(img_binary_color, "binary_color", "binary_color")

    # Exercise 4
    img_gaussian = [add_gaussian_noise(img) for img in img_cv2]
    plot_cv2_to_plt(img_gaussian, "act1_gaussian", gray=True)
    save_img(img_gaussian, "gaussian", "gaussian")

    img_gaussian_with_smooth = [
        cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT) for img in img_gaussian
    ]
    plot_cv2_to_plt(img_gaussian_with_smooth, "act1_gaussian_smooth", gray=False)
    save_img(img_gaussian_with_smooth, "gaussian_smooth", "gaussian_smooth")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
