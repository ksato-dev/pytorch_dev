import torch
import glob
import PIL
# from PIL import Image
import torchvision


# class SquareResize:
#     """
#     ref: https://axa.biopapyrus.jp/deep-learning/object-classification/dataloader.html
#     comment: 解像度の揃っていないデータセットを与えると、データローダーからドローする時にエラーが出ることがあり、その対策として使う。
#     """

#     def __init__(self, shape=224, bg_color=[0, 0, 0]):
#         self.shape = shape
#         self.bg_color = bg_color

#     def __call__(self, img):
#         w, h = img.size
#         img_square = None

#         if w == h:
#             img_square = img
#         elif w > h:
#             img_square = PIL.Image.new(img.mode, (w, w), self.bg_color)
#             img_square.paste(img, (0, (w - h) // 2))
#         else:
#             img_square = PIL.Image.new(img.mode, (h, h), self.bg_color)
#             img_square.paste(img, ((h - w) // 2, 0))

#         img_square = img_square.resize((self.shape, self.shape))
#         return img_square


class MyTransfroms(object):

    def __init__(self, resize, mean, std):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(resize, resize)),
            # SquareResize(resize, (0, 0, 0)),
            torchvision.transforms.RandomResizedCrop(
                resize, scale=(0.5, 1.0)
            ),
            torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        ret_img = self.transforms(img)
        # vis_img = ret_img.numpy().transpose((1, 2, 0))
        # vis_img = np.clip(vis_img, 0, 1)
        # plt.imshow(vis_img)
        # plt.show()
        return ret_img


class HymenopteraDataSet(torch.utils.data.Dataset):

    def __init__(self, data_dir_path, transform, ext=""):
        if ext == "":
            self.ext = "jpg"
        else:
            self.ext = ext

        self.img_path_list = glob.glob(
            data_dir_path + "/**/*." + self.ext, recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        curr_img_path = self.img_path_list[idx]
        split_str_list = curr_img_path.split("/")
        # print(split_str_list)
        label = None
        if split_str_list[-2] == "ants":
            label = 0
        elif split_str_list[-2] == "bees":
            label = 1

        pil_img = PIL.Image.open(curr_img_path)
        # print(np.array(pil_img))
        torch_img = self.transform(pil_img)

        return (torch_img, label)
