# BiRefNet-WebUI vX.Y.Z

BiRefNet / ToonOut ベースの背景切り抜きを、ローカルWeb UIで使うための配布版です。

Repository:
- https://github.com/rumda500/Automatic-image-cropping-tool

## 主な機能

- 画像から透過PNGを生成
- 入力画像の既存アルファ保護
- UI上で保護マスクを直接描画（白=保護）
- Linux/macOS/Windows 起動スクリプト同梱

## 使い方

### Clone

```bash
git clone https://github.com/rumda500/Automatic-image-cropping-tool.git
cd BiRefNet-WebUI
```

### Linux / macOS

```bash
chmod +x run_ui.sh
./run_ui.sh
```

### Windows

- `run_ui.bat` を実行

起動後、ブラウザで `http://127.0.0.1:7860` を開いてください。

## 同梱ライセンス

- `LICENSE.BiRefNet-ToonOut` (MIT)
- `NOTICE.txt`
- `THIRD_PARTY_LICENSES.md`

※ ToonOutデータセットを再配布する場合のみ `LICENSE.ToonOut-Dataset.CC-BY-4.0` の同梱が必要です。

## 謝辞

- Upstream: https://github.com/MatteoKartoon/BiRefNet
