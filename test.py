from recognition import Recognition
import time
if __name__ == '__main__':
    regcon = Recognition()
    links = ["/Users/phaihoang/Documents/datasets/satudora-product/all/product01_back_2_0001.jpg",
"/Users/phaihoang/Documents/datasets/satudora-product/all/product01_back_4_0001.jpg",
"/Users/phaihoang/Documents/datasets/satudora-product/all/product01_front_4_0001.jpg",
"/Users/phaihoang/Documents/datasets/satudora-product/all/product01_side1_2_0001.jpg",
"/Users/phaihoang/Documents/datasets/satudora-product/all/product01_side3_3_0001.jpg",
"/Users/phaihoang/Documents/datasets/satudora-product/all/product01_side4_3_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product02_back_1_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product02_side1_3_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product02_side4_1_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product02_side4_2_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product03_back_4_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product03_front_1_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product03_side1_3_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product03_side1_4_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product03_side2_3_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product03_side2_4_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product04_back_2_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product04_side2_4_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product04_side3_2_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product05_back_4_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product05_front_2_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product05_front_4_0001.jpg",
             "/Users/phaihoang/Documents/datasets/satudora-product/all/product05_side2_1_0001.jpg",
             ]
    for image in links:
        start = time.time()
        print(regcon.predict_image(image))
        end = time.time()
        print(end - start)
    # top_5 = regcon.predict_image("/Users/phaihoang/Documents/datasets/satudora-product/all/product03_side1_4_0001.jpg")
    # print(top_5)