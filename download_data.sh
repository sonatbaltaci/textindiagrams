set -e
mkdir -p data

gdown "https://drive.google.com/uc?export=download&id=1m2JXwgop-GR6PnLZSvvy7LLSVrtIRWhI" -O diagrams.zip
unzip diagrams.zip && rm diagrams.zip
mv EIDA/ data/
mv EIDALatin/ data/
rm -rf __MACOSX
