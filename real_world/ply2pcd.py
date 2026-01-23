import open3d as o3d
import numpy as np
import os
import glob
from pathlib import Path

def downsample_to_target_points(pcd, target_points):
    """
    点群を指定した点数にダウンサンプリング
    
    Args:
        pcd: Open3Dの点群オブジェクト
        target_points: ターゲットの点数
    
    Returns:
        ダウンサンプリングされた点群オブジェクト
    """
    current_points = len(pcd.points)
    
    if current_points <= target_points:
        print(f"  現在の点数({current_points}) がターゲット以下のため、ダウンサンプリングなし")
        return pcd
    
    # ランダムサンプリング
    indices = np.random.choice(current_points, target_points, replace=False)
    downsampled_pcd = pcd.select_by_index(indices)
    
    print(f"  ダウンサンプリング: {current_points:,} → {target_points:,} 点")
    
    return downsampled_pcd

def convert_ply_to_pcd(input_file, output_file=None, binary=True, save_csv=False, target_points=None):
    """
    PLYファイルをPCD形式に変換
    
    Args:
        input_file: 入力PLYファイルのパス
        output_file: 出力PCDファイルのパス（Noneの場合は自動生成）
        binary: Trueならバイナリ形式、Falseならアスキー形式
        save_csv: Trueなら頂点座標をCSVでも保存
        target_points: ターゲットの点数（Noneの場合はダウンサンプリングなし）
    
    Returns:
        出力ファイルのパス
    """
    # PLYファイルを読み込み
    print(f"読み込み中: {input_file}")
    pcd = o3d.io.read_point_cloud(input_file)
    
    if len(pcd.points) == 0:
        raise ValueError(f"エラー: {input_file} に点群データがありません")
    
    # ダウンサンプリング
    if target_points is not None:
        pcd = downsample_to_target_points(pcd, target_points)
    
    # 点群の情報を表示
    print(f"  点数: {len(pcd.points):,}")
    print(f"  カラー情報: {'あり' if pcd.has_colors() else 'なし'}")
    print(f"  法線情報: {'あり' if pcd.has_normals() else 'なし'}")
    
    # 出力ファイル名の生成
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.pcd'))
    
    # PCDファイルとして保存
    print(f"保存中: {output_file}")
    success = o3d.io.write_point_cloud(
        output_file, 
        pcd, 
        write_ascii=not binary,
        compressed=False
    )
    
    if success:
        # ファイルサイズを表示
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  保存完了: {file_size:.2f} MB")
        print(f"  形式: {'バイナリ' if binary else 'アスキー'}")
        
        # CSV保存（ヘッダーなし、頂点座標のみ）
        if save_csv:
            csv_file = str(Path(output_file).with_suffix('.csv'))
            points = np.asarray(pcd.points)
            np.savetxt(csv_file, points, delimiter=',', fmt='%.6f')
            csv_size = os.path.getsize(csv_file) / (1024 * 1024)
            print(f"  CSV保存: {csv_file} ({csv_size:.2f} MB)")
            print(f"  形式: x,y,z (ヘッダーなし)")
        
        return output_file
    else:
        raise RuntimeError(f"エラー: {output_file} の保存に失敗しました")

def batch_convert(input_pattern="*.ply", output_dir=None, binary=True, save_csv=False, target_points=None):
    """
    複数のPLYファイルを一括変換
    
    Args:
        input_pattern: 入力ファイルのパターン（例: "*.ply", "data/*.ply"）
        output_dir: 出力ディレクトリ（Noneの場合は入力ファイルと同じ場所）
        binary: バイナリ形式で保存するか
        save_csv: CSVも保存するか
        target_points: ターゲットの点数（Noneの場合はダウンサンプリングなし）
    """
    # ファイルを検索
    ply_files = glob.glob(input_pattern)
    
    if len(ply_files) == 0:
        print(f"警告: {input_pattern} にマッチするファイルが見つかりません")
        return
    
    print(f"\n{len(ply_files)}個のPLYファイルが見つかりました")
    print("=" * 60)
    
    # 出力ディレクトリの作成
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # 変換処理
    converted_files = []
    failed_files = []
    
    for i, ply_file in enumerate(ply_files, 1):
        print(f"\n[{i}/{len(ply_files)}] 処理中...")
        
        try:
            # 出力ファイル名の生成
            if output_dir is not None:
                filename = os.path.basename(ply_file)
                output_file = os.path.join(output_dir, Path(filename).with_suffix('.pcd'))
            else:
                output_file = None
            
            # 変換実行
            result = convert_ply_to_pcd(ply_file, output_file, binary, save_csv, target_points)
            converted_files.append(result)
            
        except Exception as e:
            print(f"  エラー: {e}")
            failed_files.append(ply_file)
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("変換完了")
    print("=" * 60)
    print(f"成功: {len(converted_files)}個")
    print(f"失敗: {len(failed_files)}個")
    
    if failed_files:
        print("\n失敗したファイル:")
        for f in failed_files:
            print(f"  - {f}")
    
    return converted_files

def compare_formats(ply_file):
    """PLYとPCDのファイルサイズを比較"""
    print("\n形式比較:")
    print("=" * 60)
    
    # PLYファイルのサイズ
    ply_size = os.path.getsize(ply_file) / (1024 * 1024)
    print(f"PLY (オリジナル): {ply_size:.2f} MB")
    
    # PCDバイナリ形式
    pcd_binary = str(Path(ply_file).with_suffix('.pcd'))
    convert_ply_to_pcd(ply_file, pcd_binary, binary=True)
    pcd_binary_size = os.path.getsize(pcd_binary) / (1024 * 1024)
    print(f"PCD (バイナリ):   {pcd_binary_size:.2f} MB ({pcd_binary_size/ply_size*100:.1f}%)")
    
    # PCDアスキー形式
    pcd_ascii = str(Path(ply_file).stem) + "_ascii.pcd"
    convert_ply_to_pcd(ply_file, pcd_ascii, binary=False)
    pcd_ascii_size = os.path.getsize(pcd_ascii) / (1024 * 1024)
    print(f"PCD (アスキー):   {pcd_ascii_size:.2f} MB ({pcd_ascii_size/ply_size*100:.1f}%)")
    
    print("=" * 60)
    print("推奨: バイナリ形式（ファイルサイズが小さく読み込みが高速）")

def visualize_pcd(pcd_file):
    """PCDファイルを可視化"""
    print(f"\n可視化: {pcd_file}")
    pcd = o3d.io.read_point_cloud(pcd_file)
    
    print(f"点数: {len(pcd.points):,}")
    
    # 統計情報
    points = np.asarray(pcd.points)
    print(f"範囲:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # 3D表示
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="PCD Viewer",
        width=1024,
        height=768
    )

if __name__ == "__main__":
    print("=" * 60)
    print("PLY → PCD 変換プログラム")
    print("=" * 60)
    print("\n必要なライブラリ:")
    print("- open3d")
    print("- numpy")
    print("\nインストール方法:")
    print("pip install open3d numpy")
    print("=" * 60)

    # 3000点にダウンサンプリング（単一ファイル）
    print("\n=== 3000点にダウンサンプリング（単一ファイル） ===")
    
    ply_files = glob.glob("*.ply")
    if ply_files:
        print(f"\n現在のディレクトリのPLYファイル:")
        for i, f in enumerate(ply_files, 1):
            size = os.path.getsize(f) / (1024 * 1024)
            print(f"{i}. {f} ({size:.2f} MB)")
        
        try:
            file_idx = int(input("\nファイル番号を入力（または0で手動入力）: "))
            if file_idx > 0 and file_idx <= len(ply_files):
                input_file = ply_files[file_idx - 1]
            else:
                input_file = input("PLYファイルのパスを入力: ").strip()
        except ValueError:
            input_file = input("PLYファイルのパスを入力: ").strip()
    else:
        input_file = input("PLYファイルのパスを入力: ").strip()
    
    output_file = input("出力ファイル名（Enter=自動生成）: ").strip() or None
    
    csv_choice = input("頂点座標をCSVでも保存しますか？ (y/n, デフォルト: n): ").strip().lower()
    save_csv = csv_choice == 'y'
    
    try:
        result = convert_ply_to_pcd(input_file, output_file, binary=True, save_csv=save_csv, target_points=3000)
        print(f"\n✓ 変換成功: {result}")
        
        # 可視化するか確認
        view = input("\n変換したPCDファイルを表示しますか？ (y/n): ").strip().lower()
        if view == 'y':
            visualize_pcd(result)
            
    except Exception as e:
        print(f"\n✗ エラー: {e}")
