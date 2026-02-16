# BiRefNet Web UI (配布用1ディレクトリ版)

BiRefNet / ToonOut ベースの背景切り抜きを、ローカルWeb UIで簡単に使うための最小パッケージです。
GitHub配布しやすいように、重みファイルは初回起動時に自動ダウンロードします。

## できること

- 画像から透過PNGを生成
- 透過保護（入力画像の既存アルファ保護）
- UI上で保護マスクを直接描画（白=保護、消しゴム対応）
- Linux/macOS/Windows の起動スクリプト同梱

## 含まれるもの

- `app.py` : Web UI本体（Gradio）
- `cutout_tool.py` : CLI本体
- `fetch_checkpoint.py` : 重みファイル取得
- `birefnet/` : 推論に必要なモデルコード
- `requirements.txt` : 依存
- `run_ui.sh` : セットアップ兼起動スクリプト

## 起動

このREADMEはリポジトリ直下で実行する前提です。

```bash
git clone https://github.com/rumda500/Automatic-image-cropping-tool.git
cd Automatic-image-cropping-tool
chmod +x run_ui.sh
./run_ui.sh
```

起動後、ブラウザで `http://127.0.0.1:7860` を開いてください。

### Windows の場合

```bat
git clone https://github.com/rumda500/Automatic-image-cropping-tool.git
cd Automatic-image-cropping-tool
run_ui.bat
```

### 起動スクリプト

- Linux/macOS 直接起動: `run_ui.sh`
- Windows 直接起動: `run_ui.bat`
- Linux/macOS メニュー起動: `launcher.sh`
- Windows メニュー起動: `launcher.bat`

初回起動時は `birefnet_finetuned_toonout.pth` が自動取得されます。

Linux/macOSでメニュー起動する場合:

```bash
chmod +x launcher.sh
./launcher.sh
```

## 透過保護

- 入力画像の既存アルファを保護（デフォルトON）
- 画像上に直接描ける保護マスクキャンバス（白=保護、消しゴムあり）

## 配布時の注意

このディレクトリを再配布する場合は、以下を同梱してください。

- `LICENSE.BiRefNet-ToonOut`（MIT）
- `NOTICE.txt`
- `THIRD_PARTY_LICENSES.md`

ToonOutデータセットを再配布する場合のみ、以下も同梱してください。

- `LICENSE.ToonOut-Dataset.CC-BY-4.0`（CC BY 4.0）

## GitHub Releases用テンプレ

GitHubのRelease本文に使えるテンプレートは以下に同梱しています。

- `GITHUB_RELEASE_TEMPLATE.md`

## GitHub公開向けメモ

- `.gitignore` で `.venv/` と重みファイル（`.pth`）を除外済みです。
- 重みをリポジトリに含めたい場合は Git LFS を利用してください。

