"""
学習の推移を動画化
潜在空間zや席取りLossの機能確認
"""

import cv2
import glob
import os
import argparse

def images_to_video(folder, output="output", idx=0, fps=10):
    todolist = ["PCA_min", "PCA_use", "histgram", "worst_histgram_sorted"]
    handlist = ["zl", "zr"]
    for todo in todolist:
        for hand in handlist:
            path_folder = os.path.join(folder, todo, hand)
            # フォルダ内のpng画像を取得（ソート必須）
            files = sorted(glob.glob(os.path.join(path_folder, "*.png")))
            if not files:
                print("画像が見つかりませんでした。")
                break
            # 1枚目の画像サイズを取得
            frame = cv2.imread(files[0])
            height, width, layers = frame.shape

            # 動画ライター作成
            path = os.path.join(output, f"{todo}_{hand}.mp4")
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            for file in files:
                img = cv2.imread(file)
                out.write(img)

            out.release()
            print(f"動画を保存しました: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="Log/trained_model", help='Path to the folder containing images')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video file name')
    parser.add_argument('--idx', type=int, default=0, help='Index of the image to start from (not used in this version)')
    opt = parser.parse_args()
    images_to_video(opt.folder, opt.output, idx=opt.idx, fps=10)