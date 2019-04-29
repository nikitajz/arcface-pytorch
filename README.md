# ArcFace Pytorch

Arcface の pytorch 実装

オリジナル実装: [https://github.com/ronghuaiyang/arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch) :pray:

## Requierments

* docker
* docker-compose

## SetUp

```bash
# 環境変数のコピー. 自分でいい感じにデータセットへのパスを書き換えてください
cp project.env .env

# image の build
docker-compose build

# jupyter server の container が起動
docker-compose up -d

# コードの実行は container 内部でね
docker exec -it arc-face-pytorch bash
# or docker exec -it arc-face-pytorch zsh
```

## データセットの準備

学習には CASIA Dataset を使い, テスト時に lfw dataset を使います.  
なぜデータセットが二個必要かというと，検証時には学習時に見ていない人の同一性判定を行う必要があるためです．

普通は train data を train/valid に分割して loss の値を見たりします．

arcface のような metric learning の目的は当然 validation data に対する loss を下げることも必要ですが, 対象となるデータをうまく表すことのできる特徴ベクトルを獲得することが一番の目的です．
その検証のためには，データセットには無い画像で検証を行う必要があるため本実装でもデータセットを2つ用意して検証を行っています．

### CASIA

以下のリンクからダウンロードできます(2019/04/29現在)

* [https://drive.google.com/file/d/1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz/view](https://drive.google.com/file/d/1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz/view)

### lfw dataset

[http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/) からダウンロードできます．
いくつかバージョンがありますが顔の位置を修正したもの ([All images aligned with deep funneling](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz))を使うのがいいと思います．

### データを置く場所

上記のデータをダウンロードしたら `.env` で `DATASET_DIR` に指定したディレクトリに展開してください. このとき CASIA は `CASIA-WebFace` に, lfw は `lfw` という名前のディレクトリに展開してください.

たとえば `DATASET_DIR=arcface/` であれば以下のようになります

```
tree -L 2

├── arcface
│   ├── CASIA-WebFace
│   ├── CASIA-WebFace.zip
│   └── lfw
```

### メタデータの作成

配置が終わったら, 画像データのメタデータを作成します. `notebooks/create_metadata` を実行してください. CASIA データセットのメタデータ csv が生成されます.

lfw データセットのメタデータはプロジェクトルートディレクトリにあります．これは `{DATASET_DIR}/lfw` に配置してください.

## 学習

`train.py` が実行ファイルです

## おまけ

docker 内に入っているライブラリ一覧

```
pip freeze 

alembic==1.0.10
asn1crypto==0.24.0
atomicwrites==1.3.0
attrs==19.1.0
autopep8==1.4.4
backcall==0.1.0
bleach==3.1.0
boto==2.49.0
boto3==1.9.137
botocore==1.12.137
certifi==2019.3.9
cffi==1.11.5
chardet==3.0.4
cliff==2.14.1
cmd2==0.9.12
colorama==0.4.1
colorlog==4.0.2
conda==4.6.14
cryptography==2.4.2
cycler==0.10.0
decorator==4.4.0
defusedxml==0.6.0
docutils==0.14
entrypoints==0.3
gensim==3.7.2
graphviz==0.10.1
idna==2.8
ipykernel==5.1.0
ipython==7.5.0
ipython-genutils==0.2.0
ipywidgets==7.4.2
jedi==0.13.3
Jinja2==2.10.1
jmespath==0.9.4
jsonschema==3.0.1
jupyter==1.0.0
jupyter-client==5.2.4
jupyter-console==6.0.0
jupyter-contrib-core==0.3.3
jupyter-contrib-nbextensions==0.5.1
jupyter-core==4.4.0
jupyter-highlight-selected-word==0.2.0
jupyter-latex-envs==1.4.6
jupyter-nbextensions-configurator==0.4.1
kiwisolver==1.1.0
lightgbm==2.2.4
llvmlite==0.28.0
lxml==4.3.3
Mako==1.0.9
MarkupSafe==1.1.1
matplotlib==3.0.3
mecab-python3==0.996.2
mistune==0.8.4
mkl-fft==1.0.10
mkl-random==1.0.2
more-itertools==7.0.0
nbconvert==5.5.0
nbformat==4.4.0
notebook==5.7.8
numba==0.43.1
numpy==1.16.3
olefile==0.46
opencv-python==4.1.0.25
optuna==0.10.0
pandas==0.24.2
pandas-profiling==1.4.2
pandocfilters==1.4.2
parso==0.4.0
pbr==5.2.0
pexpect==4.7.0
pickleshare==0.7.5
Pillow==6.0.0
pluggy==0.9.0
prettytable==0.7.2
prometheus-client==0.6.0
prompt-toolkit==2.0.9
ptyprocess==0.6.0
py==1.8.0
pycodestyle==2.5.0
pycosat==0.6.3
pycparser==2.19
Pygments==2.3.1
pyOpenSSL==18.0.0
pyparsing==2.4.0
pyperclip==1.7.0
pyrsistent==0.15.1
PySocks==1.6.8
pytest==4.4.1
python-dateutil==2.8.0
python-editor==1.0.4
pytz==2019.1
PyYAML==5.1
pyzmq==18.0.1
qtconsole==4.4.3
requests==2.21.0
ruamel-yaml==0.15.46
s3transfer==0.2.0
scikit-learn==0.20.3
scipy==1.2.1
seaborn==0.9.0
Send2Trash==1.5.0
six==1.12.0
smart-open==1.8.3
SQLAlchemy==1.3.3
stevedore==1.30.1
terminado==0.8.2
testpath==0.4.2
torch==1.0.1
torchfile==0.1.0
torchvision==0.2.2
tornado==5.1.1
tqdm==4.31.1
traitlets==4.3.2
typing==3.6.6
urllib3==1.24.1
visdom==0.1.8.8
wcwidth==0.1.7
webencodings==0.5.1
websocket-client==0.56.0
widgetsnbextension==3.4.2
xgboost==0.82
```
