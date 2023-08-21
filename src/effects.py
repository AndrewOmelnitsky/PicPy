import skimage
from matplotlib import pyplot as plt
import numpy as np
import math
from numba import njit, prange


# from functools import wraps
# import time
#
# def timeit(func):
#     @wraps(func)
#     def timeit_wrapper(*args, **kwargs):
#         start_time = time.perf_counter()
#         result = func(*args, **kwargs)
#         end_time = time.perf_counter()
#         total_time = end_time - start_time
#         print(f'Function {func.__name__} Took {total_time:.4f} seconds')
#         return result
#     return timeit_wrapper


class ImageEffect:
    def __init__(self, mask=None):
        self.mask = mask

    @staticmethod
    @njit(cache=True)
    def use_mask_njit(image, res_image, mask):
        a = np.multiply(res_image, mask)
        b = np.multiply(image, 1 - mask)
        return a + b

    def use_mask(self, image, res_image, mask):
        if mask is not None:
            res = self.use_mask_njit(image, res_image, mask)
            return np.array(res, dtype=np.uint8)
        return res_image

    def set_mask(self, new_mask):
        self.mask = new_mask

    def __call__(self, input_image):
        """
        return image after make effects
        ex:
        res = self.effect_func(input_image) -> effectedImage

        self.use_mask(input_image, res, self.mask)
        """
        return input_image


class ChromaticAberrationEffect(ImageEffect):
    def __init__(self, dx=5, dy=5, *args, **kwargs):
        self.set_delta_x(dx)
        self.set_delta_y(dy)
        super().__init__(*args, **kwargs)

    def set_delta_x(self, dx: int):
        self.dx = dx

    def set_delta_y(self, dy: int):
        self.dy = dy

    def hrom(self, image):
        h, w, _ = image.shape
        dx = self.dx
        dy = self.dy
        crop_h = h - dy
        crop_w = w - dx

        image[:crop_h, :crop_w, 0] = image[dy:h, dx:w, 0]
        image[dy:h, dx:w, 2] = image[:crop_h, :crop_w, 2]

        return image

    def __call__(self, input_image):
        res = self.hrom(input_image.copy())
        return self.use_mask(input_image, res, self.mask)


class MargeEffect(ImageEffect):
    CONVERT_TYPES = ["Lab", "RGB", "HSV", "XYZ", "YIQ"]

    def __init__(self, ref_image, convert_type="Lab", *args, **kwargs):
        self.ref_image = ref_image
        self.set_type(convert_type)
        super().__init__(*args, **kwargs)

    def _count_ref_image_params(self, ref_image=None):
        if ref_image is None:
            ref_image = self.ref_image

        ref_image = ref_image.transpose(2, 0, 1)
        self.E_r = [self.count_E(ref_image[i]) for i in range(ref_image.shape[0])]
        self.D_r = [
            self.count_D(ref_image[i], self.E_r[i]) for i in range(ref_image.shape[0])
        ]

    def set_type(self, convert_type):
        self.convert_type = convert_type
        ref = self.change_rgb2type(self.ref_image.copy(), self.convert_type)
        self._count_ref_image_params(ref)

    def change_rgb2type(self, image, convert_type):
        if convert_type == "Lab":
            return skimage.color.rgb2lab(image)
        if convert_type == "HSV":
            return skimage.color.rgb2hsv(image)
        if convert_type == "XYZ":
            return skimage.color.rgb2xyz(image)
        if convert_type == "YIQ":
            return skimage.color.rgb2yiq(image)
        if convert_type == "RGB":
            return image

    def change_type2rgb(self, image, convert_type):
        if convert_type == "Lab":
            return skimage.color.lab2rgb(image) * 255
        if convert_type == "HSV":
            return skimage.color.hsv2rgb(image) * 255
        if convert_type == "XYZ":
            return skimage.color.xyz2rgb(image) * 255
        if convert_type == "YIQ":
            return skimage.color.yiq2rgb(image) * 255
        if convert_type == "RGB":
            return image

    @staticmethod
    @njit(cache=True)
    def count_D(array, E=None):
        n = array.shape[0] * array.shape[1]
        if E is None:
            E = np.sum(array) / n
        return np.power(np.sum(np.power(array - E, 2)) / n, 0.5)

    @staticmethod
    @njit(cache=True)
    def count_E(array):
        return np.sum(array) / (array.shape[0] * array.shape[1])

    def merge_images(self, image):
        image = image.transpose(2, 0, 1)

        E_i = [self.count_E(image[i]) for i in range(image.shape[0])]
        D_i = [self.count_D(image[i], E_i[i]) for i in range(image.shape[0])]

        E_r = self.E_r
        D_r = self.D_r

        image = image.transpose(1, 2, 0)
        result = np.multiply(image - E_i, np.divide(D_r, D_i)) + E_r

        return result

    def merge_images_by_type(self, image):
        image = self.change_rgb2type(image, self.convert_type)
        result = self.merge_images(image)

        result = self.change_type2rgb(result, self.convert_type)

        dev = max(int(np.max(result)) + 1, 255)
        result = skimage.img_as_ubyte(result / dev)
        return result

    def __call__(self, input_image):
        res = self.merge_images_by_type(input_image)
        return self.use_mask(input_image, res, self.mask)


class FilterEffect(ImageEffect):
    default_kernel = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    default_color_shift = 0
    default_divider = 1

    def __init__(
        self,
        kernel=None,
        divider=None,
        color_shift=None,
        channels=("R", "G", "B"),
        *args,
        **kwargs
    ):
        if kernel is None:
            kernel = self.default_kernel
        self.kernel = kernel
        self.color_shift = color_shift or self.default_color_shift
        self.divider = divider or self.default_divider
        self.channels = channels
        super().__init__(*args, **kwargs)

    def set_kernel(self, kernel):
        self.kernel = kernel

    def set_channels(self, channels):
        self.channels = channels

    def mul_by_channels(self, channels):
        R, G, B = 0, 0, 0
        if "R" in channels:
            R = 1
        if "G" in channels:
            G = 1
        if "B" in channels:
            B = 1

        return (R, G, B)

    def image_preparation(self, image):
        k_h, k_w = self.kernel.shape

        for i in range(k_h // 2):
            image = np.insert(image, 0, image[:, 0, :], axis=1)
            image = np.insert(image, -1, image[:, -1, :], axis=1)

        for j in range(k_w // 2):
            image = np.insert(image, 0, image[0, :, :], axis=0)
            image = np.insert(image, -1, image[-1, :, :], axis=0)

        return image

    @staticmethod
    @njit(cache=True)
    def filter_nopython(image, kernel):
        h, w, _ = image.shape
        k_h, k_w, _ = kernel.shape

        result = np.zeros(image.shape)

        k_h_2 = k_h // 2
        k_w_2 = k_w // 2
        n_h = h + k_h_2 * 2
        n_w = w + k_w_2 * 2

        for i in range(-k_h_2, k_h_2 + 1):
            for j in range(-k_w_2, k_w_2 + 1):
                m = np.multiply(image, kernel[i][j])
                result[
                    max(0, i) : min(n_h, n_h + i), max(0, j) : min(n_w, n_w + j)
                ] += m[max(0, -i) : min(n_h, n_h - i), max(0, -j) : min(n_w, n_w - j)]

        return result[k_h_2:-k_h_2, k_w_2:-k_w_2]

    @staticmethod
    def filter_python(image, kernel_m):
        h, w, _ = image.shape
        k_h, k_w, _ = kernel_m.shape
        result = np.zeros(image.shape)
        for i in range(h):
            for j in range(w):
                k_h_i = k_h - i
                k_w_j = k_w - j
                c_h = max(0, k_h_i - 1)
                c_w = max(0, k_w_j - 1)
                im_m = np.multiply(
                    image[
                        i : None if k_h_i > 0 else -k_h_i : -1,
                        j : None if k_w_j > 0 else -k_w_j : -1,
                    ],
                    kernel_m[c_h:, c_w:],
                )
                result[i][j] = np.sum(
                    im_m.reshape(((k_h - c_h) * (k_w - c_w), 3)), axis=0
                )

        return result

    def filter_image(self, image):
        R, G, B = self.mul_by_channels(self.channels)
        kernel = self.kernel / self.divider
        k_h, k_w = kernel.shape
        kernel_def = np.zeros(kernel.shape)
        kernel_def[k_h // 2][k_w // 2] = 1

        kernel_m = np.array(
            [
                kernel if R else kernel_def,
                kernel if G else kernel_def,
                kernel if B else kernel_def,
            ]
        ).transpose(1, 2, 0)

        image_p = self.image_preparation(image / 255)
        result = self.filter_nopython(image_p, kernel_m) + self.color_shift / 255
        # print(max((np.max(result), abs(np.min(result)), 1)))
        dev = max((np.max(result), abs(np.min(result)), 1))

        return skimage.img_as_ubyte(result / dev)

    def __call__(self, input_image):
        res = self.filter_image(input_image.copy())
        return self.use_mask(input_image, res, self.mask)


class BlurEffect(FilterEffect):
    def __init__(self, a=1, *args, **kwargs):
        n = a * 2 + 1
        kernel = np.zeros((n, n))
        for i in range(n):
            k = a - abs(a - i) + 1
            for j in range(n):
                kernel[i][j] = (a - abs(j - a) + 1) * k

        divider = np.sum(kernel)
        super().__init__(kernel=kernel, divider=divider, *args, **kwargs)


class SharpeningEffect(FilterEffect):
    def __init__(self, a=1, b=10, *args, **kwargs):
        n = a * 2 + 1
        kernel = np.zeros((n, n))
        for i in range(n):
            k = a - abs(a - i) + 1
            for j in range(n):
                kernel[i][j] = -(a - abs(j - a) + 1) * k

        kernel[a][a] = 0
        kernel[a][a] = -np.sum(kernel) + b

        super().__init__(kernel=kernel, divider=b * args, **kwargs)


class MedianEffect(FilterEffect):
    def __init__(self, a=1, *args, **kwargs):
        n = a * 2 + 1
        kernel = np.ones((n, n))
        divider = np.sum(kernel)
        super().__init__(kernel=kernel, divider=divider, *args, **kwargs)


class EdgeDetectionCentralEffect(FilterEffect):
    default_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ]
    )


class EdgeDetectionVerticalEffect(FilterEffect):
    default_kernel = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]
    )


class EdgeDetectionHorizontalEffect(FilterEffect):
    default_kernel = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ]
    )


class FirstEmbossingEffect(FilterEffect):
    default_kernel = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 0],
        ]
    )
    default_color_shift = 128


class SecondEmbossingEffect(FilterEffect):
    default_kernel = np.array(
        [
            [0, 1, 0],
            [1, 0, -1],
            [0, -1, 0],
        ]
    )
    default_color_shift = 128


class GammaCorrectionEffect(ImageEffect):
    def __init__(self, correction_array, *args, **kwargs):
        self.correction_array = correction_array
        super().__init__(*args, **kwargs)

    @staticmethod
    @njit(cache=True, parallel=True)
    def correction_njit(image, correction_array):
        h, w, c = image.shape
        for i in prange(h):
            for j in prange(w):
                for c_i in prange(c):
                    image[i][j][c_i] = correction_array[int(image[i][j][c_i])]

        return image

    def correction(self, input_image):
        return self.correction_njit(input_image, self.correction_array)

    def __call__(self, input_image):
        res = self.correction(input_image.copy())
        return self.use_mask(input_image, res, self.mask)


def main():
    im1 = skimage.io.imread("5.jpg")
    im2 = skimage.io.imread("1.jpg")
    test = skimage.io.imread("test.png")
    test = im1

    # ef = MargeEffect(im2, 'RGB')
    # res = ef(im1)
    # blur
    # k = np.array([
    #     [1, 2, 1],
    #     [2, 4, 2],
    #     [1, 2, 1],
    # ]) * (1/16)
    # res
    k = np.array(
        [
            [-1, -2, -1],
            [-2, 22, -2],
            [-1, -2, -1],
        ]
    ) * (1 / 10)
    # print(np.sum(k))
    # edge
    k = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ]
    )  # * 10
    k = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]
    )
    # k = np.array([
    #     [1, 1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1],
    # ]) * (1/9)

    # print(k)

    skimage.io.imshow(test)
    plt.show()
    plt.close()

    blur = BlurEffect(10)
    sharp = SharpeningEffect(2, 100)
    median = MedianEffect(2)
    embossing = FirstEmbossingEffect()
    # ef = FilterEffect(k)
    # res = ef(test)
    # res = sharp(test)
    # res = blur(test)
    res = median(test)
    # res = embossing(test)

    skimage.io.imshow(res)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
