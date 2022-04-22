import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path: str) -> np.ndarray:
    """Load image using cv2.imread"""
    image = cv2.imread(path)
    return image


def save_img(img_array: list, file_path: str, base_file_name: str) -> None:
    """Save image array as a png files."""
    for index, img in enumerate(img_array):
        cv2.imwrite(f"data/act_1/{file_path}/{base_file_name}_{index}.png", img)


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


if __name__ == "__main__":
    base_path = "data/buildings/building_"

    img_array = []

    for i in range(1, 5 + 1):
        img_array.append(load_image(f"{base_path}{i}.png"))

    gray_img_array = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_array]

    img_blur = [cv2.GaussianBlur(img, (3, 3), 0) for img in gray_img_array]

    # Sobel edge detection
    img_sobel = []

    for img in img_blur:
        sobel = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        img_sobel.append(sobel)

    save_img(img_sobel, "building_sobel", "building_sobel")

    # Canny edge detection
    img_canny = []
    for img in img_blur:
        canny = cv2.Canny(image=img, threshold1=100, threshold2=200)
        img_canny.append(canny)

    save_img(img_canny, "building_canny", "building_canny")
