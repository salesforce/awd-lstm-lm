echo "=== Acquiring datasets ==="
echo "---"
mkdir -p data
cd data

echo "- Downloading Penn Treebank (PTB)"
mkdir -p penn
cd penn
wget --quiet --continue https://github.com/pytorch/examples/raw/master/word_language_model/data/penn/train.txt
wget --quiet --continue https://github.com/pytorch/examples/raw/master/word_language_model/data/penn/valid.txt
wget --quiet --continue https://github.com/pytorch/examples/raw/master/word_language_model/data/penn/test.txt
cd ..

echo "- Downloading WikiText-2 (WT2)"
wget --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -q wikitext-2-v1.zip
cd wikitext-2
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt

echo "---"
echo "Happy language modeling :)"
