import cv2
from PIL import Image
import os

def mp4_to_gif(input_file, output_file=None, fps=10, resize_factor=0.5, quality=50):
    """
    MP4ファイルをGIFに変換する
    
    Parameters:
    -----------
    input_file : str
        入力するMP4ファイルのパス
    output_file : str, optional
        出力するGIFファイルのパス（指定しない場合は自動生成）
    fps : int, optional
        GIFのフレームレート（デフォルト: 10）
    resize_factor : float, optional
        リサイズの倍率（デフォルト: 0.5 = 50%）
    quality : int, optional
        画質（1-100、デフォルト: 50）
    """
    
    # 出力ファイル名が指定されていない場合は自動生成
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.gif"
    
    print(f"変換中: {input_file} -> {output_file}")
    
    # 動画を開く
    cap = cv2.VideoCapture(input_file)
    
    # 元の動画のFPSを取得
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(original_fps / fps))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # フレームをスキップしてFPSを調整
        if frame_count % frame_skip == 0:
            # BGRからRGBに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # リサイズ
            if resize_factor != 1.0:
                new_width = int(frame_rgb.shape[1] * resize_factor)
                new_height = int(frame_rgb.shape[0] * resize_factor)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # PILイメージに変換
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        
        frame_count += 1
    
    cap.release()
    
    if frames:
        # GIFとして保存
        duration = int(1000 / fps)  # ミリ秒単位
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
            quality=quality
        )
        
        print(f"変換完了: {output_file}")
        print(f"フレーム数: {len(frames)}")
        print(f"ファイルサイズ: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    else:
        print("エラー: フレームを読み込めませんでした")


# 使用例
if __name__ == "__main__":
    # 基本的な使い方
    mp4_to_gif("demo/collecting_dataset.mp4")
    
    # 詳細設定の例
    # mp4_to_gif(
    #     "input_video.mp4",
    #     output_file="output.gif",
    #     fps=15,  # より滑らかなGIF
    #     resize_factor=0.3,  # より小さいファイルサイズ
    #     quality=40  # 画質を下げてサイズを削減
    # )